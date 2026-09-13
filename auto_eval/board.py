"""Block 6: the comparison board.

Block 5 asks *is this agent good enough*. This asks a different question - *which
of these systems is better at this job* - and the difference in question changes
everything about the shape of the answer.

A board is one dataset, many systems, one row each. What makes it readable is
what is held constant: the items, the model that reads results, and the judge.
Change any of those between rows and the board is measuring that instead. What
makes it honest is the row nobody asks for: the **model-only baseline**, the same
model with no endpoint at all. On a dataset of facts published after the model's
cutoff that row should be near zero. When it is, every other row is retrieval.
When it is not, the board is partly a memory test and says so at the top.

Three shapes of row, and each is scored on what it actually produced:

* `search_api` - the endpoint alone. A query goes out, results come back, and the
  score is whether the answer is in them and how far down. No model involved, so
  this is retrieval quality and nothing else.
* `model_only` - the model answering from memory. The baseline.
* `model_with_tool` - the model with that endpoint as a tool. The agent case: the
  score includes whether the model used what it was handed.

Rows are ordered alphabetically and never by score. A board sorted by its own
headline invites the reader to treat that column as the answer, and with
accuracy, latency and cost on the same table it usually is not.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, computed_field

from .classifier import ClassifierError, _build_client
from .cohort import Cohort, System, SystemKind, with_baseline
from .config import Settings, get_settings
from .contamination import ContaminationReport, Cutoff
from .contamination import for_dataset as contamination_for_dataset
from .dataset import Dataset, Item
from .endpoint import Hit, Response, call
from .fetch import check_url
from .judge import judge_reply
from .score import Summary, Verdict, percentile, score_hits, score_reply, summarise

# Items in flight per system. Vendors rate-limit; this is deliberately modest.
DEFAULT_CONCURRENCY = 4

# Steps a tool-using row is allowed before the item is abandoned.
MAX_TOOL_STEPS = 6

BOARD_FILENAME = "board.json"

Progress = Callable[[int, int, str], None]

ANSWER_PROMPT = (
    "Answer the question as briefly as the answer allows - a name, a number, a date, a phrase. "
    "No preamble, no explanation, no hedging. If you do not know the answer, reply exactly: unknown."
)

TOOL_PROMPT = (
    "Answer the question using the search tool. Search first; do not answer from memory. "
    "When you have the answer, reply with it as briefly as the answer allows, and nothing else. "
    "If the searches do not turn it up, reply exactly: unknown."
)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web and return ranked results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class BoardError(ClassifierError):
    """Raised when a board cannot be run at all."""


class Row(BaseModel):
    """One system's line on the board."""

    label: str
    vendor: str = ""
    kind: SystemKind = SystemKind.SEARCH_API
    configuration: str = ""
    docs_url: Optional[str] = None

    n: int = 0
    correct: int = 0
    errors: int = 0
    judged: int = 0

    seconds_p50: Optional[float] = None
    seconds_p95: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = Field(default=0, description="Billable endpoint calls made.")

    usd: Optional[float] = Field(default=None, description="Spend, only when priced.")
    price_note: Optional[str] = None
    price_source: Optional[str] = None

    skipped: Optional[str] = Field(
        default=None, description="Why this row was not run, if it was not."
    )
    note: Optional[str] = None
    verdicts: List[Verdict] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scored(self) -> int:
        return self.n - self.errors

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accuracy(self) -> Optional[float]:
        return self.correct / self.scored if self.scored else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall_at_1(self) -> Optional[float]:
        ranked = [v for v in self.verdicts if v.rank is not None]
        if not ranked or not self.scored:
            return None
        return sum(1 for v in ranked if v.rank == 1) / self.scored

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recall_at_5(self) -> Optional[float]:
        ranked = [v for v in self.verdicts if v.rank is not None]
        if not ranked or not self.scored:
            return None
        return (
            sum(1 for v in ranked if v.rank is not None and v.rank <= 5) / self.scored
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def usd_per_1k_correct(self) -> Optional[float]:
        """What a thousand right answers cost. The number a buyer actually needs."""
        if self.usd is None or not self.correct:
            return None
        return round(self.usd / self.correct * 1000, 4)


class Board(BaseModel):
    """One dataset, many systems, and the terms the comparison holds under."""

    board_id: str
    task: str = ""
    started_at: str
    finished_at: str = ""
    seconds: float = 0.0

    dataset: str
    dataset_digest: str
    dataset_source: Optional[str] = None
    items: int = 0

    cohort: str = ""
    model: Optional[str] = Field(
        default=None, description="Held constant across every row that uses a model."
    )
    judge_model: Optional[str] = None

    rows: List[Row] = Field(default_factory=list)
    contamination: Optional[ContaminationReport] = None
    methodology: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def baseline(self) -> Optional[Row]:
        return next((r for r in self.rows if r.kind is SystemKind.MODEL_ONLY), None)

    @property
    def measured(self) -> List[Row]:
        return [
            r
            for r in self.rows
            if r.skipped is None and r.kind is not SystemKind.MODEL_ONLY
        ]

    def leader(self) -> Optional[Row]:
        """The row with the best accuracy, for the one sentence at the top."""
        scored = [r for r in self.measured if r.accuracy is not None]
        return max(scored, key=lambda r: r.accuracy or 0) if scored else None


# --------------------------------------------------------------------------
# Calling one system on one item
# --------------------------------------------------------------------------


def _hits_as_text(hits: List[Hit]) -> str:
    if not hits:
        return "No results."
    return "\n".join(
        f"{hit.rank}. {hit.title} — {hit.url}\n{hit.snippet[:600]}" for hit in hits
    )


def _ask(
    client: Any, model: str, system: str, question: str, max_tokens: int = 600
) -> Tuple[str, int, int, Optional[str]]:
    """One model call: (reply, input tokens, output tokens, error)."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            max_completion_tokens=max_tokens,
        )
    except Exception as exc:
        return "", 0, 0, f"{type(exc).__name__}: {exc}"
    message = completion.choices[0].message
    usage = getattr(completion, "usage", None)
    return (
        (message.content or "").strip(),
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
        None,
    )


def _search_loop(
    client: Any, model: str, item: Item, system: System, http: Any
) -> Tuple[str, int, int, int, Optional[str]]:
    """The model with one endpoint as a tool. (reply, in, out, searches, error)."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": TOOL_PROMPT},
        {"role": "user", "content": item.query},
    ]
    tokens_in = tokens_out = searches = 0

    for _ in range(MAX_TOOL_STEPS):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[SEARCH_TOOL],
                tool_choice="auto",
                max_completion_tokens=1200,
            )
        except Exception as exc:
            return "", tokens_in, tokens_out, searches, f"{type(exc).__name__}: {exc}"

        usage = getattr(completion, "usage", None)
        tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)

        message = completion.choices[0].message
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            return (
                (message.content or "").strip(),
                tokens_in,
                tokens_out,
                searches,
                None,
            )

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments or "{}",
                        },
                    }
                    for c in calls
                ],
            }
        )
        for tool_call in calls:
            try:
                query = json.loads(tool_call.function.arguments or "{}").get(
                    "query", item.query
                )
            except ValueError:
                query = item.query
            assert system.endpoint is not None  # guaranteed by readiness()
            response = call(system.endpoint, str(query), client=http)
            searches += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": response.error or _hits_as_text(response.hits),
                }
            )

    return "", tokens_in, tokens_out, searches, "The run never stopped searching."


class _Outcome(BaseModel):
    """What one item cost this row, alongside its verdict."""

    verdict: Verdict
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


def _run_item(
    item: Item,
    system: System,
    *,
    client: Any,
    http: Any,
    model: str,
    judge_model: Optional[str],
) -> _Outcome:
    """Put one item to one system and score what came back."""
    started = time.monotonic()

    if system.kind is SystemKind.SEARCH_API:
        assert system.endpoint is not None
        response: Response = call(system.endpoint, item.query, client=http)
        verdict = score_hits(
            item.id,
            item.answers,
            response.hits,
            seconds=response.seconds,
            error=response.error,
        )
        return _Outcome(verdict=verdict, calls=1)

    if system.kind is SystemKind.MODEL_ONLY:
        reply, tokens_in, tokens_out, error = _ask(
            client, model, ANSWER_PROMPT, item.query
        )
        searches = 0
    else:
        reply, tokens_in, tokens_out, searches, error = _search_loop(
            client, model, item, system, http
        )

    elapsed = round(time.monotonic() - started, 3)
    verdict = score_reply(item.id, item.answers, reply, seconds=elapsed, error=error)

    # An item with no gold answer cannot be matched; a judge settles it, and only
    # if one was configured.
    if not error and item.judged and judge_model:
        ruling = judge_reply(
            item.query,
            reply,
            client=client,
            model=judge_model,
            reference=item.note,
        )
        verdict = Verdict(
            item_id=item.id,
            correct=ruling.passed,
            seconds=elapsed,
            evidence=ruling.quote or " ".join(reply.split())[:200],
            note=ruling.reason,
            judged=True,
        )
    return _Outcome(
        verdict=verdict,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        calls=searches,
    )


# --------------------------------------------------------------------------
# One row, and then the board
# --------------------------------------------------------------------------


def _price(system: System, row: Row) -> Optional[float]:
    """Spend for this row at the vendor's published list price. Never estimated."""
    price = system.price
    if not price.stated:
        return None
    total = 0.0
    if price.per_call_usd is not None:
        total += price.per_call_usd * row.calls
    if price.per_million_input_usd is not None:
        total += price.per_million_input_usd * row.input_tokens / 1_000_000
    if price.per_million_output_usd is not None:
        total += price.per_million_output_usd * row.output_tokens / 1_000_000
    return round(total, 6)


def run_row(
    system: System,
    items: List[Item],
    *,
    client: Any,
    http: Any,
    model: str,
    judge_model: Optional[str],
    concurrency: int = DEFAULT_CONCURRENCY,
    check_address: bool = True,
    progress: Optional[Progress] = None,
) -> Row:
    """Run one system over every item."""
    row = Row(
        label=system.label,
        vendor=system.vendor,
        kind=system.kind,
        configuration=system.configuration,
        docs_url=system.docs_url,
        price_note=system.price.note,
        price_source=system.price.source,
        note=system.note,
        n=len(items),
    )

    ok, why = system.readiness()
    if not ok:
        return row.model_copy(update={"skipped": why, "n": 0})

    # Checked once per row rather than once per item: the address does not change
    # between items, and a refusal should cost one DNS lookup, not three hundred.
    if check_address and system.endpoint is not None:
        allowed, refusal = check_url(system.endpoint.url)
        if not allowed:
            return row.model_copy(update={"skipped": refusal, "n": 0})

    model_for_row = system.model or model
    outcomes: Dict[str, _Outcome] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _run_item,
                item,
                system,
                client=client,
                http=http,
                model=model_for_row,
                judge_model=judge_model,
            ): item
            for item in items
        }
        for done, future in enumerate(futures, 1):
            item = futures[future]
            outcomes[item.id] = future.result()
            if progress:
                progress(done, len(items), system.label)

    ordered = [outcomes[item.id] for item in items if item.id in outcomes]
    row.verdicts = [o.verdict for o in ordered]
    row.input_tokens = sum(o.input_tokens for o in ordered)
    row.output_tokens = sum(o.output_tokens for o in ordered)
    row.calls = sum(o.calls for o in ordered)

    summary: Summary = summarise(row.verdicts)
    row.correct = summary.correct
    row.errors = summary.errors
    row.judged = summary.judged

    latencies = [v.seconds for v in row.verdicts if not v.error]
    row.seconds_p50 = percentile(latencies, 0.5)
    row.seconds_p95 = percentile(latencies, 0.95)
    row.usd = _price(system, row)
    return row


def _methodology(board: Board, cohort: Cohort, dataset: Dataset) -> List[str]:
    return [
        f"{board.items} item(s) from {dataset.name} (`{board.dataset_digest}`), the same items for every row.",
        f"The model is held constant at {board.model or 'none - no row uses one'}; "
        f"the judge at {board.judge_model or 'none - nothing was judged'}. "
        "Only the system under test changes between rows.",
        "Rows are alphabetical. No column decides the order, because no single column decides the answer.",
        "An endpoint call that failed is counted as an error, not as a wrong answer: accuracy is over "
        "the items that came back.",
        "Spend is the vendor's published list price applied to the calls this board actually made. "
        "Rows without a price on file are left blank rather than estimated.",
        *cohort.notes,
    ]


def _warnings(board: Board, dataset: Dataset) -> List[str]:
    out: List[str] = []
    baseline = board.baseline

    if baseline is None:
        out.append(
            "No model-only row, so nothing here shows whether the answers could simply be recalled. "
            "Add one: it is the control the rest of the board rests on."
        )
    elif baseline.skipped:
        out.append(f"The model-only baseline did not run: {baseline.skipped}")
    elif baseline.accuracy is not None and baseline.accuracy > 0.1:
        out.append(
            f"The model-only baseline scored {baseline.accuracy:.0%} with no endpoint at all. "
            "That much of this board is a memory test, not a retrieval test - every other row is "
            "inflated by roughly that amount, and the dataset needs items the model cannot already know."
        )
    elif baseline.accuracy is not None:
        out.append(
            f"The model-only baseline scored {baseline.accuracy:.0%}: the answers cannot be recalled, "
            "so the rows below are measuring retrieval."
        )

    skipped = [r for r in board.rows if r.skipped]
    if skipped:
        out.append(
            f"{len(skipped)} system(s) did not run and are not on the board: "
            + "; ".join(f"{r.label} ({r.skipped})" for r in skipped)
        )

    erroring = [r for r in board.rows if r.n and r.errors / r.n > 0.1]
    if erroring:
        out.append(
            "Error rate over 10% on: "
            + ", ".join(f"{r.label} ({r.errors}/{r.n})" for r in erroring)
            + ". Read those rows as a measurement of the day, not of the product."
        )

    unpriced = [r for r in board.measured if r.usd is None]
    if unpriced:
        out.append(
            f"{len(unpriced)} row(s) carry no published price, so their cost columns are blank: "
            + ", ".join(r.label for r in unpriced)
        )

    if dataset.judged_share and not board.judge_model:
        out.append(
            f"{dataset.judged_share:.0%} of the items have no gold answer and no judge was configured, "
            "so they are scored as wrong for every row. Set a judge or add gold answers."
        )

    if board.items < 30:
        out.append(
            f"{board.items} items is too few to separate close rows. Treat the ordering as a hint and "
            "the gaps under a few points as noise."
        )
    return out


def run_board(
    dataset: Dataset,
    cohort: Cohort,
    *,
    client: Optional[Any] = None,
    http: Optional[Any] = None,
    model: Optional[str] = None,
    judge_model: Optional[str] = None,
    limit: Optional[int] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    cutoff: Optional[Cutoff] = None,
    baseline: bool = True,
    progress: Optional[Progress] = None,
    settings: Optional[Settings] = None,
) -> Board:
    """Run every system in `cohort` over `dataset`, and build the board."""
    items = dataset.sample(limit)
    if not items:
        raise BoardError("Nothing to run: the dataset is empty.")

    settings = settings or get_settings()
    cohort = with_baseline(cohort) if baseline else cohort
    held_model = model or cohort.model or settings.effective_subject_model
    held_judge = judge_model or cohort.judge_model

    needs_model = any(s.calls_a_model for s in cohort.systems)
    if client is None and needs_model:
        client = _build_client(settings)
    # Whether this module opened the transport decides whether it is this
    # module's job to vet the addresses going through it.
    own_transport = http is None
    if own_transport:
        try:
            import httpx

            http = httpx.Client(timeout=30.0, follow_redirects=True)
        except ImportError:  # pragma: no cover - httpx is a dependency
            http = None

    started = datetime.now(timezone.utc)
    board = Board(
        board_id=f"{cohort.name or 'board'}-{started.strftime('%Y%m%d-%H%M%S')}",
        task=cohort.task or dataset.task,
        started_at=started.isoformat(timespec="seconds"),
        dataset=dataset.name,
        dataset_digest=dataset.digest,
        dataset_source=dataset.source,
        items=len(items),
        cohort=cohort.name,
        model=held_model if needs_model else None,
        judge_model=held_judge,
    )

    for system in cohort.ordered():
        board.rows.append(
            run_row(
                system,
                items,
                client=client,
                http=http,
                model=held_model,
                judge_model=held_judge,
                concurrency=concurrency,
                check_address=own_transport,
                progress=progress,
            )
        )

    finished = datetime.now(timezone.utc)
    board.finished_at = finished.isoformat(timespec="seconds")
    board.seconds = round((finished - started).total_seconds(), 2)
    board.contamination = contamination_for_dataset(
        dataset.model_copy(update={"items": items}), held_model, cutoff=cutoff
    )
    board.methodology = _methodology(board, cohort, dataset)
    board.warnings = [*_warnings(board, dataset), *board.contamination.warnings]
    return board


# --------------------------------------------------------------------------
# Writing it down
# --------------------------------------------------------------------------


class BoardIndexEntry(BaseModel):
    """One line of the board list."""

    board_id: str
    path: str
    started_at: str
    task: str = ""
    dataset: str = ""
    items: int = 0
    rows: int = 0
    leader: Optional[str] = None
    leader_accuracy: Optional[float] = None
    baseline_accuracy: Optional[float] = None


def write_board(board: Board, out_dir: Path) -> Path:
    target = out_dir / board.board_id
    target.mkdir(parents=True, exist_ok=True)
    (target / BOARD_FILENAME).write_text(
        board.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return target


def load_board(path: Path) -> Board:
    target = path / BOARD_FILENAME if path.is_dir() else path
    if not target.exists():
        raise BoardError(f"No board at {target}.")
    try:
        return Board.model_validate_json(target.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BoardError(f"{target} is not a readable board: {exc}") from exc


def list_boards(boards_dir: Path) -> List[BoardIndexEntry]:
    """Every board under `boards_dir`, newest first. Unreadable ones are skipped."""
    if not boards_dir.is_dir():
        return []
    entries: List[BoardIndexEntry] = []
    for candidate in sorted(boards_dir.iterdir(), reverse=True):
        path = candidate / BOARD_FILENAME
        if not path.is_file():
            continue
        try:
            board = Board.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        leader = board.leader()
        baseline = board.baseline
        entries.append(
            BoardIndexEntry(
                board_id=board.board_id,
                path=str(candidate),
                started_at=board.started_at,
                task=board.task,
                dataset=board.dataset,
                items=board.items,
                rows=len([r for r in board.rows if r.skipped is None]),
                leader=leader.label if leader else None,
                leader_accuracy=leader.accuracy if leader else None,
                baseline_accuracy=baseline.accuracy if baseline else None,
            )
        )
    return sorted(entries, key=lambda e: e.started_at, reverse=True)


__all__ = [
    "BOARD_FILENAME",
    "DEFAULT_CONCURRENCY",
    "MAX_TOOL_STEPS",
    "Board",
    "BoardError",
    "BoardIndexEntry",
    "Row",
    "list_boards",
    "load_board",
    "run_board",
    "run_row",
    "write_board",
]

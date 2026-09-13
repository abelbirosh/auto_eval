"""Block 7: the loop that keeps a board from saturating.

A board whose leader scores 100% has stopped answering the question it was
built for. Every system above the ceiling is tied, the gaps that matter are
invisible, and the honest reading of the number is not *this vendor is perfect*
but *these items ran out of difficulty*. The number to fix is the dataset.

So this module closes the loop the board leaves open:

```text
run the board -> find what saturated -> rewrite those items harder -> run again
```

Two readings of "saturated", and both are used. A **row** at 100% is the
headline: some system got everything it was scored on right. The **items** that
row answered correctly are what held its score up, and the ones every other row
also got right are the ones with no difficulty left in them at all. Those are
rewritten first.

A rewrite is the one place a model is allowed to invent something here, so it is
also the one place nothing it says is believed. The model gets the item's own
source document and is asked for a *different, harder* fact from that same page.
What comes back is then checked the way the board checks an answer: the new gold
answer has to appear, at a word boundary, in the text actually fetched from the
cited URL. A proposal that fails any check is dropped and the original item
stands - a suite that quietly fills with unverifiable items is worse than a
saturated one.

The loop stops when no row is at the ceiling, when a round rewrites nothing, or
when it runs out of rounds. It never stops because it decided it was finished.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .board import Board, BoardError, run_board, write_board
from .classifier import _build_client
from .cohort import Cohort, SystemKind
from .config import Settings, get_settings
from .dataset import Dataset, Item
from .fetch import Fetched, fetch_text
from .score import contains, normalise

# A row at or above this is what the loop exists to break. It is a parameter
# rather than a constant because a board that has been hardened once usually
# wants the bar lowered - a 95% row is saturating too, just more slowly.
SATURATED = 1.0

# How many items one round rewrites. Rewriting everything at once throws away
# the comparison with the round before it: nothing is held constant.
DEFAULT_REWRITES = 3

# Rounds by default. Two is enough to see whether hardening moved the ceiling.
DEFAULT_ROUNDS = 2

# How much of the source document the model is shown. Enough for the fact to be
# in it, short enough that a long page does not become the whole prompt.
SOURCE_CHARS = 24_000

REWRITE_SYSTEM_PROMPT = """\
You are hardening a factual-lookup benchmark whose items have become too easy: \
every search system under test answers this one correctly, so it no longer \
separates them.

You are given one item and the full text of the source document it was written \
from. Write a replacement item that asks about a DIFFERENT fact from that SAME \
document - one that is harder to retrieve with a single query.

What makes an item hard, in order of how much it matters:

1. The answer sits inside the document - a table cell, a caption, a mid-section \
clause - rather than in its title, abstract, or headline.
2. A near-identical competing value sits nearby: another number of the same \
magnitude and units, another date, another name in the same list. A system that \
retrieves the right page and reads the wrong line must get it wrong.
3. The question is unambiguous anyway. Exactly one value in the document answers \
it, and a careful reader who has the document cannot argue for a second one.

Hard rules, and a proposal that breaks one is thrown away:

- The answer must appear in the document text you were given, spelled exactly as \
you write it in `answers`.
- Never put the answer, or any part of it, in the query.
- Do not reuse the previous question or the previous answer.
- `answers` lists every wording that should count as correct - the bare value and \
the value with its unit, for instance. Do not pad it with near-misses that are \
not the answer.
- Keep the question to one sentence. It is a query someone will type.

`note` is one line for the suite's author: what makes this one hard, and what \
the competing value on the page is.\
"""

REWRITE_TEMPLATE = """\
TASK: {task}

THE ITEM THAT SATURATED
id: {item_id}
question: {query}
answers: {answers}
why it is being replaced: every scored system answered it correctly

SOURCE DOCUMENT ({url})
---
{source}
---
"""


class SaturationError(BoardError):
    """Raised when the loop cannot run at all."""


class Proposal(BaseModel):
    """What the model suggests for one item. Nothing here is trusted yet."""

    query: str = Field(description="The replacement question, one sentence.")
    answers: List[str] = Field(
        default_factory=list,
        description="Every wording that counts as correct, spelled as in the source.",
    )
    note: str = Field(default="", description="What makes this one hard.")


class ItemScore(BaseModel):
    """One item, across every system that was scored on it."""

    item_id: str
    correct: int = 0
    scored: int = 0

    @property
    def accuracy(self) -> Optional[float]:
        return self.correct / self.scored if self.scored else None

    @property
    def saturated(self) -> bool:
        """True when every system that got an answer back got it right."""
        return self.scored > 0 and self.correct == self.scored


class RowScore(BaseModel):
    """One row's headline, kept so a round can be compared with the next."""

    label: str
    kind: SystemKind = SystemKind.SEARCH_API
    correct: int = 0
    scored: int = 0
    accuracy: Optional[float] = None


class Rewrite(BaseModel):
    """One attempt to replace an item, kept whether or not it was accepted."""

    item_id: str
    accepted: bool = False
    why: str = Field(default="", description="Why it was accepted or dropped.")
    was_query: str = ""
    now_query: str = ""
    was_answers: List[str] = Field(default_factory=list)
    now_answers: List[str] = Field(default_factory=list)
    source: Optional[str] = Field(
        default=None, description="The URL the new answer was verified against."
    )
    evidence: str = Field(
        default="", description="The span of the source the answer was found in."
    )


class Round(BaseModel):
    """One pass of the loop: a board, what it showed, and what was rewritten."""

    index: int
    started_at: str
    board_id: str = ""
    dataset_digest: str = ""
    items: int = 0

    rows: List[RowScore] = Field(default_factory=list)
    item_scores: List[ItemScore] = Field(default_factory=list)
    baseline: Optional[float] = Field(
        default=None, description="The model-only row, which says what is retrieval."
    )
    rewrites: List[Rewrite] = Field(default_factory=list)

    @property
    def saturated_rows(self) -> List[str]:
        return [
            r.label
            for r in self.rows
            if r.accuracy is not None and r.accuracy >= SATURATED
        ]

    @property
    def saturated_items(self) -> List[str]:
        return [i.item_id for i in self.item_scores if i.saturated]

    @property
    def accepted(self) -> List[Rewrite]:
        return [r for r in self.rewrites if r.accepted]


class LoopReport(BaseModel):
    """Every round, and why the loop stopped."""

    dataset: str = ""
    cohort: str = ""
    started_at: str = ""
    finished_at: str = ""
    threshold: float = SATURATED
    rounds: List[Round] = Field(default_factory=list)
    stopped: str = ""
    dataset_path: Optional[str] = None

    @property
    def rewritten(self) -> int:
        return sum(len(r.accepted) for r in self.rounds)


# --------------------------------------------------------------------------
# Reading a board
# --------------------------------------------------------------------------


def measured_rows(board: Board) -> List[Any]:
    """The rows a ceiling can be read off: run, and not the model-only control."""
    return [
        row
        for row in board.rows
        if row.skipped is None and row.kind is not SystemKind.MODEL_ONLY
    ]


def row_scores(board: Board) -> List[RowScore]:
    return [
        RowScore(
            label=row.label,
            kind=row.kind,
            correct=row.correct,
            scored=row.scored,
            accuracy=row.accuracy,
        )
        for row in board.rows
        if row.skipped is None
    ]


def item_scores(board: Board) -> List[ItemScore]:
    """Per item, how many systems got it right out of how many were scored.

    An item a system errored on is not counted against it - the same rule the
    board itself scores by. An item nothing came back for is not saturated; it
    is unmeasured, and rewriting it would be guessing.
    """
    tally: Dict[str, ItemScore] = {}
    for row in measured_rows(board):
        for verdict in row.verdicts:
            if verdict.error:
                continue
            score = tally.setdefault(
                verdict.item_id, ItemScore(item_id=verdict.item_id)
            )
            score.scored += 1
            score.correct += 1 if verdict.correct else 0
    return list(tally.values())


def saturated(board: Board, *, threshold: float = SATURATED) -> List[RowScore]:
    """The rows at or above the ceiling. Empty means the board still separates."""
    return [
        score
        for score in row_scores(board)
        if score.kind is not SystemKind.MODEL_ONLY
        and score.accuracy is not None
        and score.accuracy >= threshold
    ]


def targets(
    board: Board, *, threshold: float = SATURATED, limit: int = DEFAULT_REWRITES
) -> List[str]:
    """The items propping up a saturated row, hardest-to-defend first.

    Only items the saturated rows actually got right are candidates: an item a
    100% row never answered is not what made it 100%. Among those, the ones the
    rest of the board also got right come first - they have the least difficulty
    left in them.
    """
    rows = saturated(board, threshold=threshold)
    if not rows:
        return []

    labels = {row.label for row in rows}
    propping = {
        verdict.item_id
        for row in measured_rows(board)
        if row.label in labels
        for verdict in row.verdicts
        if verdict.correct
    }
    candidates = [score for score in item_scores(board) if score.item_id in propping]
    candidates.sort(key=lambda s: (-(s.accuracy or 0), -s.scored, s.item_id))
    return [score.item_id for score in candidates[: max(0, limit)]]


# --------------------------------------------------------------------------
# Rewriting one item
# --------------------------------------------------------------------------


def _source_of(item: Item, reader: Callable[[str], Fetched]) -> Fetched:
    if not item.url:
        return Fetched(url="", ok=False, error="the item cites no source to reread")
    return reader(item.url)


def _propose(
    client: Any, model: str, *, task: str, item: Item, source: str
) -> Proposal:
    message = REWRITE_TEMPLATE.format(
        task=task or "factual lookup",
        item_id=item.id,
        query=item.query,
        answers=", ".join(item.answers) or "(judged, no gold answer)",
        url=item.url or "",
        source=source[:SOURCE_CHARS],
    )
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        response_format=Proposal,
        max_completion_tokens=2000,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise SaturationError("The model returned no proposal.")
    return parsed


def _check(proposal: Proposal, item: Item, source: str) -> Optional[str]:
    """Why this proposal cannot be accepted, or None when it can be.

    Deterministic on purpose. The model is the only part of this loop that can
    invent an item, so it is the only part whose output is not taken on trust.
    """
    answers = [a.strip() for a in proposal.answers if a.strip()]
    if not proposal.query.strip():
        return "no question"
    if not answers:
        return "no gold answer"
    if normalise(proposal.query) == normalise(item.query):
        return "the question did not change"
    if any(contains(proposal.query, answer) for answer in answers):
        return "the question gives the answer away"
    # Any overlap at all, not just a full match: an answers list that still
    # accepts the old value leaves the old fact scorable under a new question.
    if any(
        contains(old, new) or contains(new, old)
        for new in answers
        for old in item.answers
    ):
        return "the answer is the one the item already had"
    missing = [answer for answer in answers if not contains(source, answer)]
    if missing:
        return f"not in the source: {', '.join(missing[:3])}"
    return None


def _evidence(source: str, answer: str, width: int = 220) -> str:
    flat = " ".join(source.split())
    position = normalise(flat).find(normalise(answer))
    if position < 0:
        return flat[:width]
    start = max(0, min(position - width // 3, len(flat) - width))
    return ("…" if start else "") + flat[start : start + width] + "…"


def harden_item(
    item: Item,
    *,
    client: Any,
    model: str,
    task: str = "",
    reader: Callable[[str], Fetched] = fetch_text,
    attempts: int = 2,
) -> tuple[Item, Rewrite]:
    """Replace one item with a harder one from its own source, or leave it alone.

    Returns the item to use and the record of what happened, accepted or not.
    The original item is returned whenever the replacement cannot be verified:
    a saturated item that is real beats a hard one that might not be.
    """
    record = Rewrite(
        item_id=item.id,
        was_query=item.query,
        was_answers=list(item.answers),
        source=item.url,
    )

    fetched = _source_of(item, reader)
    if not fetched.ok or not fetched.text.strip():
        record.why = (
            f"could not reread the source: {fetched.error or 'it came back empty'}"
        )
        return item, record

    last = ""
    for _ in range(max(1, attempts)):
        try:
            proposal = _propose(
                client, model, task=task, item=item, source=fetched.text
            )
        except Exception as exc:  # a failed proposal is a value, not a crash
            last = f"{type(exc).__name__}: {exc}"
            continue

        problem = _check(proposal, item, fetched.text)
        if problem:
            last = problem
            continue

        answers = [a.strip() for a in proposal.answers if a.strip()]
        record.accepted = True
        record.why = "verified against the source"
        record.now_query = proposal.query.strip()
        record.now_answers = answers
        record.evidence = _evidence(fetched.text, answers[0])
        harder = item.model_copy(
            update={
                "query": proposal.query.strip(),
                "answers": answers,
                "note": proposal.note.strip() or item.note,
            }
        )
        return harder, record

    record.why = f"dropped: {last or 'no usable proposal'}"
    return item, record


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

BoardRunner = Callable[..., Board]
Announce = Callable[[Round, Board], None]


def _round_from(board: Board, index: int, started: str) -> Round:
    return Round(
        index=index,
        started_at=started,
        board_id=board.board_id,
        dataset_digest=board.dataset_digest,
        items=board.items,
        rows=row_scores(board),
        item_scores=item_scores(board),
        baseline=board.baseline.accuracy if board.baseline else None,
    )


def run_loop(
    dataset: Dataset,
    cohort: Cohort,
    *,
    rounds: int = DEFAULT_ROUNDS,
    threshold: float = SATURATED,
    rewrites: int = DEFAULT_REWRITES,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    settings: Optional[Settings] = None,
    reader: Callable[[str], Fetched] = fetch_text,
    board_runner: BoardRunner = run_board,
    boards_dir: Optional[Path] = None,
    announce: Optional[Announce] = None,
    **board_kwargs: Any,
) -> tuple[LoopReport, Dataset]:
    """Run the board, rewrite what saturated, run it again. Up to `rounds` times.

    Every board is reported as it finishes through `announce`, because the point
    of the loop is the sequence of boards and not the last one.
    """
    if rounds < 1:
        raise SaturationError("A loop of fewer than one round runs nothing.")

    settings = settings or get_settings()
    started = datetime.now(timezone.utc)
    report = LoopReport(
        dataset=dataset.name,
        cohort=cohort.name,
        started_at=started.isoformat(timespec="seconds"),
        threshold=threshold,
    )

    current = dataset
    for index in range(1, rounds + 1):
        began = datetime.now(timezone.utc).isoformat(timespec="seconds")
        board = board_runner(current, cohort, settings=settings, **board_kwargs)
        if boards_dir is not None:
            write_board(board, boards_dir)

        this_round = _round_from(board, index, began)
        report.rounds.append(this_round)
        if announce:
            announce(this_round, board)

        if not saturated(board, threshold=threshold):
            report.stopped = (
                f"no row reached {threshold:.0%} in round {index}: "
                "the board separates its systems again"
            )
            break

        if index == rounds:
            report.stopped = f"ran the {rounds} round(s) it was given"
            break

        chosen = targets(board, threshold=threshold, limit=rewrites)
        if not chosen:
            report.stopped = (
                "a row saturated but no item it answered could be rewritten"
            )
            break

        if client is None:
            client = _build_client(settings)
        rewriter = model or settings.model

        replaced: List[Item] = []
        for item in current.items:
            if item.id not in chosen:
                replaced.append(item)
                continue
            harder, record = harden_item(
                item,
                client=client,
                model=rewriter,
                task=current.task,
                reader=reader,
            )
            this_round.rewrites.append(record)
            replaced.append(harder)

        if not this_round.accepted:
            report.stopped = (
                f"round {index} could not verify a single replacement, "
                "so the dataset was left as it was"
            )
            break

        current = current.model_copy(update={"items": replaced})
        current = _redigest(current)

    if not report.stopped:
        report.stopped = f"ran the {rounds} round(s) it was given"
    report.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report, current


def _redigest(dataset: Dataset) -> Dataset:
    """A rewritten dataset is a different dataset, and says so in its digest."""
    from .dataset import _digest

    return dataset.model_copy(update={"digest": _digest(dataset.items)})


def write_dataset(dataset: Dataset, path: Path) -> Path:
    """Write the items back out as JSONL, in the order they were run in."""
    lines = []
    for item in dataset.items:
        row = item.model_dump(mode="json", exclude_none=True)
        if not row.get("meta"):
            row.pop("meta", None)
        lines.append(json.dumps(row, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_REWRITES",
    "DEFAULT_ROUNDS",
    "SATURATED",
    "ItemScore",
    "LoopReport",
    "Proposal",
    "Rewrite",
    "Round",
    "RowScore",
    "SaturationError",
    "harden_item",
    "item_scores",
    "measured_rows",
    "row_scores",
    "run_loop",
    "saturated",
    "targets",
    "write_dataset",
]

"""Pull the actual ground truth out of the sources the identifier found.

`auto_eval.ground_truth` says where the labelled data and the published numbers
are. This module goes and gets them, and hands back cases an eval harness can
run on: an input, the answer it should produce, and where that came from.

Two routes in, because public ground truth comes in two shapes:

* **Datasets** - rows are pulled from the Hugging Face datasets server, which
  serves real rows of public datasets over an open API. The model is asked one
  narrow question, which column is the input and which holds the answer, and
  the cases are then built from the rows themselves. No example here is
  model-written.
* **Pages** - a benchmark page, a spec, a vendor's numbers. The model extracts,
  but every value it reports must come with a span copied from the page, and
  `verify_page` checks that span against the text we fetched. Anything that
  does not match is dropped and counted.

So the rule from the rest of the pipeline holds here too, in its strongest
form: nothing reaches the output because a model said it. Either it came out of
a dataset API, or it was found in a page we read ourselves.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .classifier import _as_classifier_error, _build_client
from .config import get_settings
from .fetch import DEFAULT_ROW_LIMIT, Fetched, Rows, fetch_rows, fetch_text, hf_dataset_id
from .ground_truth import Fit, GroundTruthError, GroundTruthReport, SourceKind
from .prompts import (
    FIELD_MAPPING_SYSTEM_PROMPT,
    PAGE_EXTRACTION_SYSTEM_PROMPT,
    build_field_mapping_message,
    build_ground_truth_message,
    build_page_extraction_message,
)
from .schema import TaskSpec

DEFAULT_MAX_TOKENS = 8000

# Enough cases to build a first eval on and still read through by hand. Raise it
# on the call when you want a full split.
DEFAULT_MAX_EXAMPLES = 20

# Page text sent to the model. Beyond this a page is a document dump, and the
# numbers worth having are near the top of it.
MAX_PAGE_CHARS = 40_000

# Rows shown to the mapper. It only has to recognise the columns.
SAMPLE_ROWS = 3
MAX_SAMPLE_CHARS = 1200


# --------------------------------------------------------------------------
# What comes out
# --------------------------------------------------------------------------


class Origin(str, Enum):
    DATASET_ROWS = "dataset_rows"    # verbatim from a dataset API
    PAGE_QUOTE = "page_quote"        # copied from a page we fetched and checked


class GroundTruthExample(BaseModel):
    """One labelled case: what goes in, and the answer that should come out."""

    input: str
    expected: str
    kpi: Optional[str] = None
    source: str = Field(description="Name of the source it came from.")
    url: str
    origin: Origin
    split: Optional[str] = Field(default=None, description="Dataset split, where there is one.")


class CitedBaseline(BaseModel):
    """A published number, with the span of page it was copied from."""

    metric: str
    value: str
    system: Optional[str] = None
    as_of: Optional[str] = None
    quote: str = Field(description="Verbatim text from the page, checked against it.")
    source: str
    url: str


class Outcome(str, Enum):
    EXTRACTED = "extracted"
    NOTHING_FOUND = "nothing_found"
    UNREACHABLE = "unreachable"
    UNUSABLE = "unusable"            # reachable, but not ground truth for this task
    SKIPPED = "skipped"              # contextual fit, or the budget was spent


class SourceOutcome(BaseModel):
    """What came of trying one source - including why nothing did."""

    source: str
    url: str
    outcome: Outcome
    detail: str = ""
    examples: int = 0
    baselines: int = 0
    discarded: int = Field(
        default=0, description="Items dropped because they were not found in the source."
    )


class GroundTruthSet(BaseModel):
    """The extracted ground truth: what an eval harness would actually run on."""

    subject: str
    examples: List[GroundTruthExample] = Field(default_factory=list)
    baselines: List[CitedBaseline] = Field(default_factory=list)
    outcomes: List[SourceOutcome] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.examples or self.baselines)


# --------------------------------------------------------------------------
# What the model returns
# --------------------------------------------------------------------------


class FieldMapping(BaseModel):
    """Which column is the question and which is the answer."""

    usable: bool
    input_columns: List[str] = Field(default_factory=list)
    expected_column: str = ""
    kpi: Optional[str] = None
    reason: str = ""


class QuotedBaseline(BaseModel):
    metric: str
    value: str
    system: Optional[str] = None
    as_of: Optional[str] = None
    quote: str = Field(description="Copied from the page character for character.")


class QuotedExample(BaseModel):
    input: str = Field(description="Copied from the page character for character.")
    expected: str = Field(description="Copied from the page character for character.")
    kpi: Optional[str] = None


class PageExtraction(BaseModel):
    baselines: List[QuotedBaseline] = Field(default_factory=list)
    examples: List[QuotedExample] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Verification - the deterministic half
# --------------------------------------------------------------------------


def _flatten(text: str) -> str:
    """Compare on words, not whitespace: markup stripping moves spaces around."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _appears(needle: str, haystack: str) -> bool:
    needle = _flatten(needle)
    return bool(needle) and needle in haystack


def _cell(value: Any) -> str:
    """A row value as text, without losing structure that a grader may need."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def _match_kpi(claimed: Optional[str], kpi_names: Sequence[str]) -> Optional[str]:
    if not claimed:
        return None
    wanted = re.sub(r"[^a-z0-9]+", " ", claimed.lower()).strip()
    for name in kpi_names:
        if re.sub(r"[^a-z0-9]+", " ", name.lower()).strip() == wanted:
            return name
    return None


def examples_from_rows(
    rows: Rows,
    mapping: FieldMapping,
    *,
    source: str,
    url: str,
    kpi_names: Sequence[str],
    limit: int,
) -> Tuple[List[GroundTruthExample], str]:
    """Build cases out of real rows. Returns the cases and why there are none."""
    if not mapping.usable:
        return [], mapping.reason or "The mapper judged this dataset unusable for the task."

    columns = set(rows.columns)
    unknown = [c for c in (*mapping.input_columns, mapping.expected_column) if c not in columns]
    if unknown or not mapping.input_columns or not mapping.expected_column:
        # A column that is not in the data means the mapping was guessed.
        return [], f"Mapping named columns the dataset does not have: {', '.join(unknown) or 'none given'}."

    kpi = _match_kpi(mapping.kpi, kpi_names)
    built: List[GroundTruthExample] = []
    for row in rows.rows[:limit]:
        expected = _cell(row.get(mapping.expected_column))
        parts = [(column, _cell(row.get(column))) for column in mapping.input_columns]
        if not expected.strip() or not any(value.strip() for _, value in parts):
            continue  # an unlabelled row is not ground truth
        text = (
            parts[0][1]
            if len(parts) == 1
            else "\n".join(f"{column}: {value}" for column, value in parts)
        )
        built.append(
            GroundTruthExample(
                input=text,
                expected=expected,
                kpi=kpi,
                source=source,
                url=url,
                origin=Origin.DATASET_ROWS,
                split=rows.split,
            )
        )
    if not built:
        return [], "Every row was missing an input or an answer."
    return built, ""


def verify_page(
    extraction: PageExtraction,
    page: str,
    *,
    source: str,
    url: str,
    kpi_names: Sequence[str],
    limit: int,
) -> Tuple[List[GroundTruthExample], List[CitedBaseline], int]:
    """Keep only what the fetched page actually says. Returns what was dropped."""
    haystack = _flatten(page)
    discarded = 0

    baselines: List[CitedBaseline] = []
    for item in extraction.baselines:
        # The quote has to be on the page, and the number has to be in the quote,
        # so a real quote cannot be used to carry an invented figure.
        if not _appears(item.quote, haystack) or not _appears(item.value, _flatten(item.quote)):
            discarded += 1
            continue
        baselines.append(
            CitedBaseline(
                metric=item.metric,
                value=item.value,
                system=item.system,
                as_of=item.as_of,
                quote=item.quote.strip(),
                source=source,
                url=url,
            )
        )

    examples: List[GroundTruthExample] = []
    for item in extraction.examples[: max(limit, 0)]:
        if not _appears(item.input, haystack) or not _appears(item.expected, haystack):
            discarded += 1
            continue
        examples.append(
            GroundTruthExample(
                input=item.input.strip(),
                expected=item.expected.strip(),
                kpi=_match_kpi(item.kpi, kpi_names),
                source=source,
                url=url,
                origin=Origin.PAGE_QUOTE,
            )
        )

    return examples, baselines, discarded


# --------------------------------------------------------------------------
# The model calls
# --------------------------------------------------------------------------


def _parse(client: Any, *, model: str, instructions: str, message: str, schema, max_tokens: int):
    return client.responses.parse(
        model=model,
        instructions=instructions,
        input=message,
        text_format=schema,
        max_output_tokens=max_tokens,
    )


def _map_fields(
    client: Any, *, model: str, task: str, rows: Rows, max_tokens: int
) -> FieldMapping:
    sample = json.dumps(rows.rows[:SAMPLE_ROWS], ensure_ascii=False, indent=1)[:MAX_SAMPLE_CHARS]
    response = _parse(
        client,
        model=model,
        instructions=FIELD_MAPPING_SYSTEM_PROMPT,
        message=build_field_mapping_message(
            task, rows.dataset, rows.split or "", rows.columns, sample
        ),
        schema=FieldMapping,
        max_tokens=max_tokens,
    )
    return getattr(response, "output_parsed", None) or FieldMapping(
        usable=False, reason="The mapper returned nothing parsable."
    )


def _read_page(
    client: Any, *, model: str, task: str, url: str, text: str, max_tokens: int
) -> PageExtraction:
    response = _parse(
        client,
        model=model,
        instructions=PAGE_EXTRACTION_SYSTEM_PROMPT,
        message=build_page_extraction_message(task, url, text[:MAX_PAGE_CHARS]),
        schema=PageExtraction,
        max_tokens=max_tokens,
    )
    return getattr(response, "output_parsed", None) or PageExtraction(
        notes=["The extractor returned nothing parsable for this page."]
    )


# --------------------------------------------------------------------------
# The pass over the sources
# --------------------------------------------------------------------------

TextFetcher = Callable[[str], Fetched]
RowFetcher = Callable[..., Rows]


def extract(
    spec: TaskSpec,
    report: GroundTruthReport,
    *,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    text_fetcher: TextFetcher = fetch_text,
    row_fetcher: RowFetcher = fetch_rows,
) -> GroundTruthSet:
    """Fetch the identified sources and return the ground truth found in them.

    Every source is either read from a dataset API or fetched and checked; what
    could not be confirmed is dropped and counted in `outcomes`, so a thin
    result says which sources let us down rather than going quiet.
    """
    settings = get_settings(model=model)
    client = client or _build_client(settings)
    search_model = settings.effective_search_model
    task = build_ground_truth_message(spec)
    kpi_names = [kpi.name for kpi in spec.kpis]

    examples: List[GroundTruthExample] = []
    baselines: List[CitedBaseline] = []
    outcomes: List[SourceOutcome] = []
    notes: List[str] = []

    for source in report.sources:
        remaining = max_examples - len(examples)
        if source.fit is Fit.CONTEXTUAL and source.kind not in (
            SourceKind.LEADERBOARD,
            SourceKind.PUBLISHED_RESULT,
            SourceKind.VENDOR_CLAIM,
        ):
            # Contextual reading material, with no numbers to lift either.
            outcomes.append(
                SourceOutcome(
                    source=source.name,
                    url=source.url,
                    outcome=Outcome.SKIPPED,
                    detail="Contextual fit; nothing here to extract.",
                )
            )
            continue

        try:
            outcome, found, numbers = _extract_one(
                source,
                client=client,
                model=search_model,
                task=task,
                kpi_names=kpi_names,
                limit=max(remaining, 0),
                max_tokens=max_tokens,
                text_fetcher=text_fetcher,
                row_fetcher=row_fetcher,
            )
        except Exception as exc:
            raise _as_extraction_error(exc, settings, search_model) from exc

        examples.extend(found)
        baselines.extend(numbers)
        outcomes.append(outcome)

    if len(examples) >= max_examples:
        notes.append(
            f"Stopped at the {max_examples}-example cap; raise --max-examples for a full split."
        )

    return GroundTruthSet(
        subject=spec.subject.name,
        examples=examples,
        baselines=baselines,
        outcomes=outcomes,
        notes=notes,
    )


def _extract_one(
    source,
    *,
    client: Any,
    model: str,
    task: str,
    kpi_names: Sequence[str],
    limit: int,
    max_tokens: int,
    text_fetcher: TextFetcher,
    row_fetcher: RowFetcher,
) -> Tuple[SourceOutcome, List[GroundTruthExample], List[CitedBaseline]]:
    outcome = SourceOutcome(source=source.name, url=source.url, outcome=Outcome.NOTHING_FOUND)

    dataset = hf_dataset_id(source.url)
    if dataset and limit > 0:
        rows = row_fetcher(dataset, max(min(limit, DEFAULT_ROW_LIMIT), 1))
        if not rows.ok:
            outcome.outcome = Outcome.UNREACHABLE
            outcome.detail = rows.error or "The dataset server would not serve this dataset."
            return outcome, [], []

        mapping = _map_fields(client, model=model, task=task, rows=rows, max_tokens=max_tokens)
        found, why = examples_from_rows(
            rows,
            mapping,
            source=source.name,
            url=source.url,
            kpi_names=kpi_names,
            limit=limit,
        )
        if not found:
            outcome.outcome = Outcome.UNUSABLE
            outcome.detail = why
            return outcome, [], []

        outcome.outcome = Outcome.EXTRACTED
        outcome.examples = len(found)
        outcome.detail = (
            f"{len(found)} row(s) from the {rows.split or 'default'} split, "
            f"input {', '.join(mapping.input_columns)} -> answer {mapping.expected_column}."
        )
        return outcome, found, []

    page = text_fetcher(source.url)
    if not page.ok:
        outcome.outcome = Outcome.UNREACHABLE
        outcome.detail = page.error or "Could not be read."
        return outcome, [], []

    extraction = _read_page(
        client, model=model, task=task, url=source.url, text=page.text, max_tokens=max_tokens
    )
    found, numbers, discarded = verify_page(
        extraction,
        page.text,
        source=source.name,
        url=source.url,
        kpi_names=kpi_names,
        limit=limit,
    )
    outcome.examples = len(found)
    outcome.baselines = len(numbers)
    outcome.discarded = discarded
    if found or numbers:
        outcome.outcome = Outcome.EXTRACTED
        outcome.detail = "; ".join(extraction.notes) or "Read from the page and checked against it."
    else:
        outcome.detail = (
            f"Nothing on the page survived checking ({discarded} discarded)."
            if discarded
            else "The page states no examples or numbers for this task."
        )
    return outcome, found, numbers


def _as_extraction_error(exc: Exception, settings, model: str) -> GroundTruthError:
    mapped = _as_classifier_error(exc, settings)
    return GroundTruthError(str(mapped).replace(repr(settings.model), repr(model)))


def to_jsonl(examples: Sequence[GroundTruthExample]) -> str:
    """The cases as JSON lines - what a harness loads."""
    return "\n".join(example.model_dump_json() for example in examples)


__all__ = [
    "CitedBaseline",
    "DEFAULT_MAX_EXAMPLES",
    "FieldMapping",
    "GroundTruthExample",
    "GroundTruthSet",
    "Origin",
    "Outcome",
    "PageExtraction",
    "QuotedBaseline",
    "QuotedExample",
    "SourceOutcome",
    "examples_from_rows",
    "extract",
    "to_jsonl",
    "verify_page",
]

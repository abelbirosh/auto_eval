"""Analyse the sources the identifier found, without downloading any of them.

`auto_eval.ground_truth` says where the labelled data and the published numbers
are. This stage goes and looks at each one - does the link resolve, what is
actually behind it, what would a later step have to fetch - and hands back a
link plus an analysis for each. The data itself is deliberately left where it
is: downloading belongs to a later block, and this one's job is to make that
download an informed decision rather than a guess.

For a dataset that means reading its metadata: columns, split sizes, licence,
whether it is gated. For anything else it means reading the page and describing
it. Either way the output carries a `DownloadPlan` - what to fetch, from where,
which fields, how many rows are there - so the next step has an instruction
rather than a URL.

The verification rule from the rest of the pipeline still holds. The model may
only report a number it can quote, and `verify_baselines` checks that quote
against the text we read; a column it names must exist in the dataset's real
column list, or the mapping is refused. Nothing is believed because a model
said it.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable, List, Optional, Sequence

from pydantic import BaseModel, Field

from .classifier import _as_classifier_error, _build_client
from .config import get_settings
from .fetch import DatasetMeta, Fetched, fetch_dataset_meta, fetch_text, hf_dataset_id
from .ground_truth import Fit, GroundTruthError, GroundTruthReport, SourceKind
from .prompts import (
    DATASET_ASSESSMENT_SYSTEM_PROMPT,
    PAGE_ASSESSMENT_SYSTEM_PROMPT,
    build_dataset_assessment_message,
    build_ground_truth_message,
    build_page_assessment_message,
)
from .schema import TaskSpec

DEFAULT_MAX_TOKENS = 8000

# Page text sent to the model. Beyond this a page is a document dump, and what
# describes it is near the top.
MAX_PAGE_CHARS = 40_000

# Split names worth preferring, best first: training rows are the ones a model
# has most likely already seen, which makes them the worst choice for an eval.
HELD_OUT_SPLITS = ("test", "validation", "valid", "dev", "eval")


# --------------------------------------------------------------------------
# What comes out
# --------------------------------------------------------------------------


class Reachability(str, Enum):
    OK = "ok"
    UNREACHABLE = "unreachable"    # dead link, timeout, error
    BLOCKED = "blocked"            # refused by our own guards, or gated
    NOT_CHECKED = "not_checked"    # nothing here worth opening


class Usability(str, Enum):
    GROUND_TRUTH = "ground_truth"  # labelled data is behind this link
    BASELINES = "baselines"        # numbers to compare against
    BACKGROUND = "background"      # context only
    UNUSABLE = "unusable"


class Effort(str, Enum):
    LOW = "low"        # fetch and score
    MEDIUM = "medium"  # reformat, filter, or relabel a subset
    HIGH = "high"      # substantial work, or only an overlapping task


class CitedBaseline(BaseModel):
    """A published number, with the span of page it was copied from."""

    metric: str
    value: str
    system: Optional[str] = None
    as_of: Optional[str] = None
    quote: str = Field(description="Verbatim text from the page, checked against it.")
    source: str
    url: str


class DownloadPlan(BaseModel):
    """What a later step should fetch. Nothing is fetched here."""

    what: str = Field(description="One line: what to get.")
    url: str
    dataset: Optional[str] = None
    config: Optional[str] = None
    split: Optional[str] = Field(default=None, description="Chosen held-out split, where there is one.")
    input_fields: List[str] = Field(default_factory=list)
    expected_field: Optional[str] = None
    rows_available: Optional[int] = None
    licence: Optional[str] = Field(
        default=None, description="Terms the fetched data comes under, where stated."
    )
    blockers: Optional[str] = Field(
        default=None, description="Gating or registration standing in the way of a fetch."
    )


class ResourceAnalysis(BaseModel):
    """One source, looked at but not downloaded."""

    source: str
    url: str
    kind: SourceKind
    reachability: Reachability
    detail: str = ""
    usability: Usability = Usability.UNUSABLE
    summary: str = ""
    covers_kpis: List[str] = Field(default_factory=list)
    dataset: Optional[DatasetMeta] = None
    baselines: List[CitedBaseline] = Field(default_factory=list)
    plan: Optional[DownloadPlan] = None
    effort: Optional[Effort] = None
    caveats: List[str] = Field(default_factory=list)
    discarded: int = Field(
        default=0, description="Claims dropped because the source did not bear them out."
    )


class AnalysisReport(BaseModel):
    """Every source, with what is behind it and what to fetch from it later."""

    subject: str
    resources: List[ResourceAnalysis] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @property
    def plans(self) -> List[DownloadPlan]:
        """The actionable shortlist, in the order the sources were ranked."""
        return [r.plan for r in self.resources if r.plan]

    @property
    def with_ground_truth(self) -> List[ResourceAnalysis]:
        return [r for r in self.resources if r.usability is Usability.GROUND_TRUTH]


# --------------------------------------------------------------------------
# What the model returns
# --------------------------------------------------------------------------


class DatasetAssessment(BaseModel):
    usable: bool
    input_columns: List[str] = Field(default_factory=list)
    expected_column: str = ""
    kpi: Optional[str] = None
    effort: Effort = Effort.MEDIUM
    summary: str = ""
    caveats: List[str] = Field(default_factory=list)


class QuotedBaseline(BaseModel):
    metric: str
    value: str
    system: Optional[str] = None
    as_of: Optional[str] = None
    quote: str = Field(description="Copied from the page character for character.")


class PageAssessment(BaseModel):
    contains: Usability = Usability.UNUSABLE
    summary: str = ""
    kpi: Optional[str] = None
    baselines: List[QuotedBaseline] = Field(default_factory=list)
    download_hint: str = ""
    effort: Effort = Effort.MEDIUM
    caveats: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Verification - the deterministic half
# --------------------------------------------------------------------------


def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _appears(needle: str, haystack: str) -> bool:
    needle = _flatten(needle)
    return bool(needle) and needle in haystack


def _match_kpi(claimed: Optional[str], kpi_names: Sequence[str]) -> Optional[str]:
    if not claimed:
        return None
    wanted = re.sub(r"[^a-z0-9]+", " ", claimed.lower()).strip()
    for name in kpi_names:
        if re.sub(r"[^a-z0-9]+", " ", name.lower()).strip() == wanted:
            return name
    return None


def choose_split(meta: DatasetMeta) -> Optional[str]:
    """Pick the split to evaluate on, held-out first. Not the model's call."""
    names = [split.name for split in meta.splits]
    for preferred in HELD_OUT_SPLITS:
        for name in names:
            if name.lower() == preferred:
                return name
    return names[0] if names else None


def plan_for_dataset(
    meta: DatasetMeta, assessment: DatasetAssessment
) -> tuple[Optional[DownloadPlan], str]:
    """Turn an assessment into a fetch instruction, or say why there is none."""
    if not assessment.usable:
        return None, "Judged not to hold ground truth for this task."

    columns = set(meta.columns)
    named = [*assessment.input_columns, assessment.expected_column]
    unknown = [c for c in named if c and c not in columns]
    if unknown or not assessment.input_columns or not assessment.expected_column:
        # A column that is not in the metadata means the mapping was guessed.
        return None, (
            "The mapping names columns the dataset does not have: "
            f"{', '.join(unknown) or 'none given'}."
        )

    split = choose_split(meta)
    rows = next((s.rows for s in meta.splits if s.name == split), None)
    # A licence is a condition on using the data, not something stopping the
    # fetch; only gating actually blocks one.
    blockers = []
    if meta.gated:
        blockers.append("gated - access must be requested on the hub")
    if meta.private:
        blockers.append("private")

    return (
        DownloadPlan(
            what=(
                f"{f'{rows:,} rows' if rows else 'rows'} from {meta.dataset} "
                f"({meta.config}/{split}), "
                f"{', '.join(assessment.input_columns)} -> {assessment.expected_column}"
            ),
            url=meta.url,
            dataset=meta.dataset,
            config=meta.config,
            split=split,
            input_fields=list(assessment.input_columns),
            expected_field=assessment.expected_column,
            rows_available=rows,
            licence=meta.licence,
            blockers="; ".join(blockers) or None,
        ),
        "",
    )


def verify_baselines(
    baselines: Sequence[QuotedBaseline],
    page: str,
    *,
    source: str,
    url: str,
) -> tuple[List[CitedBaseline], int]:
    """Keep only the numbers the fetched page actually carries."""
    haystack = _flatten(page)
    kept: List[CitedBaseline] = []
    discarded = 0
    for item in baselines:
        # The quote has to be on the page, and the number has to be in the quote,
        # so a real quote cannot be used to carry an invented figure.
        if not _appears(item.quote, haystack) or not _appears(item.value, _flatten(item.quote)):
            discarded += 1
            continue
        kept.append(
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
    return kept, discarded


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


def _assess_dataset(
    client: Any, *, model: str, task: str, meta: DatasetMeta, max_tokens: int
) -> DatasetAssessment:
    splits = ", ".join(
        f"{s.name} ({s.rows:,} rows)" if s.rows else s.name for s in meta.splits
    )
    response = _parse(
        client,
        model=model,
        instructions=DATASET_ASSESSMENT_SYSTEM_PROMPT,
        message=build_dataset_assessment_message(
            task, meta.dataset, meta.columns, splits, meta.licence or "", meta.description
        ),
        schema=DatasetAssessment,
        max_tokens=max_tokens,
    )
    return getattr(response, "output_parsed", None) or DatasetAssessment(
        usable=False, summary="The assessment returned nothing parsable."
    )


def _assess_page(
    client: Any, *, model: str, task: str, url: str, text: str, max_tokens: int
) -> PageAssessment:
    response = _parse(
        client,
        model=model,
        instructions=PAGE_ASSESSMENT_SYSTEM_PROMPT,
        message=build_page_assessment_message(task, url, text[:MAX_PAGE_CHARS]),
        schema=PageAssessment,
        max_tokens=max_tokens,
    )
    return getattr(response, "output_parsed", None) or PageAssessment(
        summary="The assessment returned nothing parsable for this page."
    )


# --------------------------------------------------------------------------
# The pass over the sources
# --------------------------------------------------------------------------

TextFetcher = Callable[[str], Fetched]
MetaFetcher = Callable[..., DatasetMeta]


def analyze_sources(
    spec: TaskSpec,
    report: GroundTruthReport,
    *,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    text_fetcher: TextFetcher = fetch_text,
    meta_fetcher: MetaFetcher = fetch_dataset_meta,
) -> AnalysisReport:
    """Look at every source the identifier found. Downloads nothing.

    Each source comes back with whether its link resolves, what is behind it,
    and - where there is something worth fetching - a `DownloadPlan` naming
    exactly what a later step should get.
    """
    settings = get_settings(model=model)
    client = client or _build_client(settings)
    search_model = settings.effective_search_model
    task = build_ground_truth_message(spec)
    kpi_names = [kpi.name for kpi in spec.kpis]

    resources: List[ResourceAnalysis] = []
    for source in report.sources:
        try:
            resources.append(
                _analyze_one(
                    source,
                    client=client,
                    model=search_model,
                    task=task,
                    kpi_names=kpi_names,
                    max_tokens=max_tokens,
                    text_fetcher=text_fetcher,
                    meta_fetcher=meta_fetcher,
                )
            )
        except Exception as exc:
            raise _as_analysis_error(exc, settings, search_model) from exc

    notes: List[str] = []
    ready = [r for r in resources if r.plan and not r.plan.blockers]
    if ready:
        notes.append(
            f"{len(ready)} source(s) can be fetched as they are; the plans say what to get."
        )
    blocked = [r for r in resources if r.plan and r.plan.blockers]
    if blocked:
        notes.append("Behind gating: " + ", ".join(r.source for r in blocked) + ".")

    return AnalysisReport(subject=spec.subject.name, resources=resources, notes=notes)


def _analyze_one(
    source,
    *,
    client: Any,
    model: str,
    task: str,
    kpi_names: Sequence[str],
    max_tokens: int,
    text_fetcher: TextFetcher,
    meta_fetcher: MetaFetcher,
) -> ResourceAnalysis:
    analysis = ResourceAnalysis(
        source=source.name,
        url=source.url,
        kind=source.kind,
        reachability=Reachability.NOT_CHECKED,
        covers_kpis=list(source.covers_kpis),
    )

    dataset = hf_dataset_id(source.url)
    if dataset:
        meta = meta_fetcher(dataset, source.url)
        if not meta.ok:
            return analysis.model_copy(
                update={
                    "reachability": Reachability.UNREACHABLE,
                    "detail": meta.error or "The dataset index would not describe this.",
                }
            )

        assessment = _assess_dataset(
            client, model=model, task=task, meta=meta, max_tokens=max_tokens
        )
        plan, why = plan_for_dataset(meta, assessment)
        kpi = _match_kpi(assessment.kpi, kpi_names)
        return analysis.model_copy(
            update={
                "reachability": Reachability.BLOCKED if meta.gated else Reachability.OK,
                "detail": (
                    "Gated on the hub; metadata is public, the rows are not."
                    if meta.gated
                    else f"{len(meta.columns)} column(s), "
                    f"{len(meta.splits)} split(s), read from metadata only."
                ),
                "usability": Usability.GROUND_TRUTH if plan else Usability.UNUSABLE,
                "summary": assessment.summary,
                "covers_kpis": [kpi] if kpi else list(source.covers_kpis),
                "dataset": meta,
                "plan": plan,
                "effort": assessment.effort,
                "caveats": [*assessment.caveats, *([why] if why else [])],
            }
        )

    if source.fit is Fit.CONTEXTUAL and source.kind not in (
        SourceKind.LEADERBOARD,
        SourceKind.PUBLISHED_RESULT,
        SourceKind.VENDOR_CLAIM,
    ):
        return analysis.model_copy(
            update={
                "reachability": Reachability.NOT_CHECKED,
                "detail": "Contextual fit; nothing here to fetch later.",
                "usability": Usability.BACKGROUND,
            }
        )

    page = text_fetcher(source.url)
    if not page.ok:
        blocked = (page.error or "").startswith("Refused")
        return analysis.model_copy(
            update={
                "reachability": Reachability.BLOCKED if blocked else Reachability.UNREACHABLE,
                "detail": page.error or "Could not be read.",
            }
        )

    assessment = _assess_page(
        client, model=model, task=task, url=source.url, text=page.text, max_tokens=max_tokens
    )
    baselines, discarded = verify_baselines(
        assessment.baselines, page.text, source=source.name, url=source.url
    )
    kpi = _match_kpi(assessment.kpi, kpi_names)

    # The model says what the page is; what survived checking says what we have.
    usability = assessment.contains
    if usability is Usability.BASELINES and not baselines:
        usability = Usability.BACKGROUND

    plan = (
        DownloadPlan(what=assessment.download_hint.strip(), url=page.final_url or source.url)
        if assessment.download_hint.strip()
        else None
    )

    return analysis.model_copy(
        update={
            "reachability": Reachability.OK,
            "detail": f"Read {len(page.text):,} characters"
            + (f", redirected to {page.final_url}" if page.final_url else "")
            + ".",
            "usability": usability,
            "summary": assessment.summary,
            "covers_kpis": [kpi] if kpi else list(source.covers_kpis),
            "baselines": baselines,
            "plan": plan,
            "effort": assessment.effort,
            "caveats": list(assessment.caveats),
            "discarded": discarded,
        }
    )


def _as_analysis_error(exc: Exception, settings, model: str) -> GroundTruthError:
    mapped = _as_classifier_error(exc, settings)
    return GroundTruthError(str(mapped).replace(repr(settings.model), repr(model)))


__all__ = [
    "AnalysisReport",
    "CitedBaseline",
    "DatasetAssessment",
    "DownloadPlan",
    "Effort",
    "PageAssessment",
    "QuotedBaseline",
    "Reachability",
    "ResourceAnalysis",
    "Usability",
    "analyze_sources",
    "choose_split",
    "plan_for_dataset",
    "verify_baselines",
]

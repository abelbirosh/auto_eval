"""Ground truth identifier: does a public baseline for this task already exist?

Second stage of the pipeline. It runs on a `TaskSpec` that has come back from
the classifier with no blocking questions left - until then we do not know
enough about the subject or the KPIs for a search to mean anything, so the gate
in `gate()` refuses rather than guessing.

The question it answers is narrow: for the KPIs in the spec, is there anything
online we can anchor on instead of hand-labelling from scratch? Two kinds of
answer count, and they are not the same thing:

* **ground truth** - something that states the correct answer: a labelled
  dataset, a named benchmark, a standard or spec that defines correct
  behaviour, a reference implementation.
* **baselines** - a number a comparable system already reaches: a leaderboard
  entry, a published result, a vendor's own SLA or accuracy claim.

Ground truth lets us *score*; a baseline only tells us what good looks like.

As in `auto_eval.gaps`, the model does the finding and nothing else. It
searches and returns sources; `assess()` drops what cannot be checked, works
out which KPI each surviving source actually covers, and computes the verdict.
That keeps the verdict deterministic and testable without a network call.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

# Client construction and SDK error mapping are identical for both blocks, so
# the classifier's helpers are reused rather than copied.
from .classifier import ClassifierError, _as_classifier_error, _build_client
from .config import Settings, get_settings
from .prompts import GROUND_TRUTH_SYSTEM_PROMPT, build_ground_truth_message
from .schema import Question, TaskSpec

# Search results plus a structured report; roomier than extraction needs.
DEFAULT_MAX_TOKENS = 16000

# The hosted search tool. Older accounts only expose the preview name, so a
# rejection on this one is retried with the fallback rather than failing.
WEB_SEARCH_TOOL = "web_search"
WEB_SEARCH_TOOL_FALLBACK = "web_search_preview"

# More than this and the report stops being something a person reads.
MAX_SOURCES = 8

# Hosts that publish other people's work rather than their own claims. A
# benchmark living on one of these is not self-reported just because the spec
# happens to name the host.
NEUTRAL_HOSTS = frozenset(
    {
        "arxiv",
        "figshare",
        "github",
        "gitlab",
        "google",
        "huggingface",
        "kaggle",
        "medium",
        "openreview",
        "paperswithcode",
        "substack",
        "wikimedia",
        "wikipedia",
        "zenodo",
    }
)

# Below this length a squashed name matches inside unrelated words, so the
# squashed comparison is only made for names long enough to be distinctive.
MIN_SQUASHED_NAME = 6


class GroundTruthError(ClassifierError):
    """Raised when the ground-truth search could not be run or parsed."""


# --------------------------------------------------------------------------
# What the model returns
# --------------------------------------------------------------------------


class SourceKind(str, Enum):
    DATASET = "dataset"  # labelled examples
    BENCHMARK = "benchmark"  # data plus a scoring protocol
    LEADERBOARD = "leaderboard"  # scores for comparable systems
    PUBLISHED_RESULT = "published_result"  # a number in a paper or report
    STANDARD = "standard"  # spec or regulation defining correctness
    REFERENCE_IMPL = "reference_implementation"  # code we can diff against
    VENDOR_CLAIM = "vendor_claim"  # SLA, model card, pricing page
    OTHER = "other"


# Kinds that can tell right from wrong, and kinds that only say what score a
# comparable system got. A benchmark is both.
LABELLING_KINDS = frozenset(
    {
        SourceKind.DATASET,
        SourceKind.BENCHMARK,
        SourceKind.STANDARD,
        SourceKind.REFERENCE_IMPL,
    }
)
BASELINE_KINDS = frozenset(
    {
        SourceKind.BENCHMARK,
        SourceKind.LEADERBOARD,
        SourceKind.PUBLISHED_RESULT,
        SourceKind.VENDOR_CLAIM,
    }
)


class Fit(str, Enum):
    DIRECT = "direct"  # same task, same metric
    ADAPTABLE = "adaptable"  # needs reformatting, filtering, or a subset
    CONTEXTUAL = "contextual"  # a reference point only, not usable as data


USABLE_FITS = frozenset({Fit.DIRECT, Fit.ADAPTABLE})


class Access(str, Enum):
    OPEN = "open"
    REGISTRATION = "registration"
    PAID = "paid"
    UNKNOWN = "unknown"


class BaselineValue(BaseModel):
    """A number someone has already published, copied verbatim."""

    metric: str = Field(description="What was measured, in the source's words.")
    value: str = Field(description="The number as published, with its unit.")
    system: Optional[str] = Field(
        default=None, description="What achieved it - model, product, or team."
    )
    as_of: Optional[str] = Field(
        default=None, description="Date or version, if stated."
    )


class ExternalSource(BaseModel):
    """One thing found online that could anchor this evaluation."""

    name: str
    kind: SourceKind
    url: str = Field(description="Direct link to the source.")
    publisher: Optional[str] = Field(default=None, description="Who maintains it.")
    description: str = Field(description="One sentence on what it contains.")
    covers_kpis: List[str] = Field(
        default_factory=list,
        description="KPI names from the spec, verbatim, that this source speaks to.",
    )
    fit: Fit
    access: Access = Access.UNKNOWN
    licence: Optional[str] = Field(
        default=None, description="Licence or terms, if stated."
    )
    released: Optional[str] = Field(
        default=None,
        description=(
            "When the data was published, as the page states it; a year is enough. "
            "Null when the page does not say - a guessed date is worse than none, "
            "because this is what decides whether a score is measurement or recall."
        ),
    )
    baselines: List[BaselineValue] = Field(default_factory=list)
    caveats: Optional[str] = Field(
        default=None, description="Why it might not transfer: age, domain, saturation."
    )
    self_reported: bool = Field(
        default=False,
        description=(
            "Published by a system this evaluation is testing. Computed in `assess` "
            "from the spec, never taken from the model."
        ),
    )


class SourceFindings(BaseModel):
    """The model's raw answer. Nothing here is trusted until `assess` runs."""

    sources: List[ExternalSource] = Field(default_factory=list)
    recommendation: str = Field(
        description="What to actually use, in two sentences at most."
    )
    notes: List[str] = Field(
        default_factory=list,
        description="Anything that changes how to read the sources.",
    )


# --------------------------------------------------------------------------
# What this module returns
# --------------------------------------------------------------------------


class Coverage(str, Enum):
    LABELLED = "labelled"  # something online says what the right answer is
    BASELINE = "baseline"  # a comparable number exists, but no labels
    REFERENCE = "reference"  # context only
    NONE = "none"


COVERAGE_ORDER = [
    Coverage.NONE,
    Coverage.REFERENCE,
    Coverage.BASELINE,
    Coverage.LABELLED,
]


class KPICoverage(BaseModel):
    """What the search turned up for one KPI."""

    kpi: str
    coverage: Coverage
    sources: List[str] = Field(
        default_factory=list, description="Names of the sources."
    )


class Availability(str, Enum):
    LABELLED_DATA = "labelled_data"  # can score against public data
    PUBLISHED_BASELINES = "published_baselines"  # can compare, must label our own
    REFERENCE_ONLY = "reference_only"  # context, nothing to score with
    NONE_FOUND = "none_found"  # build the ground truth ourselves


VERDICT_FOR_COVERAGE = {
    Coverage.LABELLED: Availability.LABELLED_DATA,
    Coverage.BASELINE: Availability.PUBLISHED_BASELINES,
    Coverage.REFERENCE: Availability.REFERENCE_ONLY,
    Coverage.NONE: Availability.NONE_FOUND,
}


class GroundTruthReport(BaseModel):
    """The verdict on what exists publicly for this task."""

    subject: str = Field(description="The thing under test, from the spec.")
    verdict: Availability
    internal_ground_truth: bool = Field(
        default=False,
        description="True when the spec already has its own labels; public sources are then comparison only.",
    )
    sources: List[ExternalSource] = Field(default_factory=list)
    kpi_coverage: List[KPICoverage] = Field(default_factory=list)
    searches: List[str] = Field(
        default_factory=list, description="Queries the model actually ran."
    )
    recommendation: str = ""
    notes: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


class Gate(BaseModel):
    """Whether the spec is settled enough to search on."""

    open: bool
    reason: str
    blocking: List[Question] = Field(default_factory=list)


def gate(spec: TaskSpec) -> Gate:
    """Open once the classifier has no blocking questions left.

    A blocking question means we do not yet know what is being measured, and a
    search for baselines on an unknown metric returns plausible noise.
    """
    blocking = [q for q in spec.open_questions if q.blocking]
    if blocking:
        return Gate(
            open=False,
            reason=(
                f"{len(blocking)} blocking question(s) unanswered: "
                + "; ".join(q.question for q in blocking)
            ),
            blocking=blocking,
        )
    if not spec.kpis:
        # Belt and braces: the KPI rule blocks upstream, so this only fires on a
        # spec assembled by hand.
        return Gate(open=False, reason="The spec has no KPIs to find baselines for.")
    return Gate(
        open=True,
        reason="No blocking questions; the KPIs are settled enough to search on.",
    )


# --------------------------------------------------------------------------
# Deterministic assessment
# --------------------------------------------------------------------------


def _norm_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    path = parts.path.rstrip("/")
    return f"{host}{path}?{parts.query}" if parts.query else f"{host}{path}"


def _is_checkable(url: str) -> bool:
    """A source we cannot open is a claim, not a finding."""
    parts = urlsplit(url.strip())
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def _norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# --------------------------------------------------------------------------
# Who published it
# --------------------------------------------------------------------------
#
# A vendor's own page is evidence about that vendor's marketing, not about its
# accuracy. When the thing under test is that vendor - and on a comparison, it
# is every vendor on the list - its own documentation cannot be the ground truth
# or the baseline for its own score, however precise the number on it looks. So
# the sources published by a party under test are found here, kept in the report
# for context, and excluded from what the verdict is computed from.

# Suffixes that are not the name of anyone: `example.co.uk` is published by
# `example`, not by `co`.
_PUBLIC_SUFFIX_PARTS = frozenset({"ac", "co", "com", "edu", "gov", "net", "org"})


def _host_label(url: str) -> str:
    """The registrable name in a URL: `docs.apollo.io` -> `apollo`."""
    host = urlsplit(url.strip()).netloc.lower().split(":")[0]
    parts = [part for part in host.split(".") if part]
    if len(parts) < 2:
        return ""
    label = parts[-2]
    if label in _PUBLIC_SUFFIX_PARTS and len(parts) >= 3:
        label = parts[-3]
    return label


def subject_text(spec: TaskSpec) -> str:
    """Everything in the spec that names what is being tested.

    Only the fields that describe the subject: a KPI definition or a constraint
    can mention a vendor the evaluation is not about, and matching on those
    would demote sources for no reason.
    """
    subject = spec.subject
    return " ".join(
        [
            spec.title,
            spec.summary,
            subject.name,
            subject.description,
            subject.interface or "",
            *subject.in_scope,
            *subject.out_of_scope,
        ]
    )


def _names(text: str) -> Tuple[str, str]:
    """The two forms a name is matched in: lowercased, and with the gaps closed."""
    lowered = " ".join(text.lower().split())
    return lowered, re.sub(r"[^a-z0-9]+", "", lowered)


def _under_test(name: str, spec_lower: str, spec_squashed: str) -> bool:
    """Whether `name` is one of the things the spec says it is testing."""
    lowered, squashed = _names(name)
    if len(squashed) < 3:
        return False
    if re.search(r"\b" + re.escape(lowered) + r"\b", spec_lower):
        return True
    # `peopledatalabs` never appears with those word boundaries in "People Data
    # Labs", so a host label is also matched against the closed-up spec - but
    # only when it is long enough not to land inside an unrelated word.
    return len(squashed) >= MIN_SQUASHED_NAME and squashed in spec_squashed


def self_reported(source: ExternalSource, spec: TaskSpec) -> bool:
    """Whether this source is published by a system the spec puts under test."""
    spec_lower, spec_squashed = _names(subject_text(spec))
    label = _host_label(source.url)
    if (
        label
        and label not in NEUTRAL_HOSTS
        and _under_test(label, spec_lower, spec_squashed)
    ):
        return True
    publisher = (source.publisher or "").strip()
    return bool(publisher) and _under_test(publisher, spec_lower, spec_squashed)


def _clean_sources(
    sources: Sequence[ExternalSource], spec: TaskSpec
) -> Tuple[List[ExternalSource], List[str]]:
    """Drop unusable sources, dedupe by URL, align `covers_kpis`, mark self-reports."""
    kpi_names = [kpi.name for kpi in spec.kpis]
    kept: List[ExternalSource] = []
    seen: set = set()
    notes: List[str] = []
    dropped = 0
    by_norm = {_norm_name(name): name for name in kpi_names}

    for source in sources:
        if not _is_checkable(source.url):
            dropped += 1
            continue
        key = _norm_url(source.url)
        if key in seen:
            continue
        seen.add(key)

        # The model paraphrases KPI names; map them back or drop them, so
        # coverage is computed against the spec rather than against invention.
        aligned = []
        for claimed in source.covers_kpis:
            match = by_norm.get(_norm_name(claimed))
            if match and match not in aligned:
                aligned.append(match)
        kept.append(
            source.model_copy(
                update={
                    "covers_kpis": aligned,
                    "self_reported": self_reported(source, spec),
                }
            )
        )

    if dropped:
        notes.append(
            f"Dropped {dropped} source(s) with no usable link - a source that cannot be opened cannot be checked."
        )
    undated = [s.name for s in kept if not s.released]
    if undated:
        notes.append(
            f"{len(undated)} source(s) carry no publication date: "
            + ", ".join(undated)
            + ". Find out when each was published before quoting a score from it - anything "
            "released before the model's training cutoff may already be in its training data."
        )
    if len(kept) > MAX_SOURCES:
        notes.append(f"Showing the first {MAX_SOURCES} of {len(kept)} sources found.")
        kept = kept[:MAX_SOURCES]
    own = [s.name for s in kept if s.self_reported]
    if own:
        notes.append(
            f"{len(own)} source(s) are published by a system under test: "
            + ", ".join(own)
            + ". A vendor cannot be the ground truth for its own score, so these are "
            "kept as context and left out of the coverage above. Anchor on an "
            "independent measurement, or label a sample yourself."
        )
    return kept, notes


def _coverage_for(sources: Sequence[ExternalSource]) -> Coverage:
    if not sources:
        return Coverage.NONE
    # A system's own page is context whatever it claims, so the verdict is
    # computed from the independent sources alone.
    independent = [s for s in sources if not s.self_reported]
    usable = [s for s in independent if s.fit in USABLE_FITS]
    if any(s.kind in LABELLING_KINDS for s in usable):
        return Coverage.LABELLED
    if any(s.baselines for s in independent) or any(
        s.kind in BASELINE_KINDS for s in usable
    ):
        return Coverage.BASELINE
    return Coverage.REFERENCE


def assess(
    spec: TaskSpec,
    findings: SourceFindings,
    searches: Iterable[str] = (),
) -> GroundTruthReport:
    """Turn raw findings into a report. Pure; this is where the verdict is made."""
    kpi_names = [kpi.name for kpi in spec.kpis]
    sources, notes = _clean_sources(findings.sources, spec)

    coverage: List[KPICoverage] = []
    for name in kpi_names:
        matching = [s for s in sources if name in s.covers_kpis]
        coverage.append(
            KPICoverage(
                kpi=name,
                coverage=_coverage_for(matching),
                sources=[s.name for s in matching],
            )
        )

    if coverage:
        best = max((c.coverage for c in coverage), key=COVERAGE_ORDER.index)
    else:
        best = _coverage_for(sources)

    unattributed = [s.name for s in sources if not s.covers_kpis]
    if unattributed:
        notes.append("Not tied to a specific KPI: " + ", ".join(unattributed) + ".")

    return GroundTruthReport(
        subject=spec.subject.name,
        verdict=VERDICT_FOR_COVERAGE[best],
        internal_ground_truth=spec.ground_truth.available,
        sources=sources,
        kpi_coverage=coverage,
        searches=list(dict.fromkeys(searches)),
        recommendation=findings.recommendation,
        notes=[*findings.notes, *notes],
    )


# --------------------------------------------------------------------------
# The search itself
# --------------------------------------------------------------------------


def _searches_from(response: Any) -> List[str]:
    """The queries the model really ran, read off the response rather than asked for."""
    queries: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "web_search_call":
            continue
        query = getattr(getattr(item, "action", None), "query", None)
        if query:
            queries.append(query)
    return queries


def _parse(client: Any, *, model: str, message: str, tool: str, max_tokens: int) -> Any:
    return client.responses.parse(
        model=model,
        instructions=GROUND_TRUTH_SYSTEM_PROMPT,
        input=message,
        tools=[{"type": tool}],
        text_format=SourceFindings,
        max_output_tokens=max_tokens,
    )


def identify(
    spec: TaskSpec,
    *,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    force: bool = False,
) -> GroundTruthReport:
    """Search for public ground truth and baselines for `spec`.

    Raises `GroundTruthError` when the spec still has blocking questions, unless
    `force` is set - searching on an unsettled spec produces confident noise.
    """
    decision = gate(spec)
    if not decision.open and not force:
        raise GroundTruthError(
            f"Not ready to search: {decision.reason} "
            "Answer those first, or pass force=True to search anyway."
        )

    settings = get_settings(model=model)
    client = client or _build_client(settings)
    search_model = settings.effective_search_model
    message = build_ground_truth_message(spec)

    if not hasattr(client, "responses"):  # pragma: no cover - old SDK only
        raise GroundTruthError(
            "This openai package has no Responses API, which the web search needs. "
            "Upgrade it with `pip install -U openai`."
        )

    try:
        response = _parse(
            client,
            model=search_model,
            message=message,
            tool=WEB_SEARCH_TOOL,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        if _is_unknown_tool(exc):
            try:
                response = _parse(
                    client,
                    model=search_model,
                    message=message,
                    tool=WEB_SEARCH_TOOL_FALLBACK,
                    max_tokens=max_tokens,
                )
            except Exception as retry_exc:
                raise _as_ground_truth_error(
                    retry_exc, settings, search_model
                ) from retry_exc
        else:
            raise _as_ground_truth_error(exc, settings, search_model) from exc

    findings = getattr(response, "output_parsed", None)
    if findings is None:
        status = getattr(response, "status", None)
        raise GroundTruthError(
            f"The search returned no parsable findings (status={status!r}). "
            "If this says 'incomplete', raise --max-tokens."
        )
    return assess(spec, findings, _searches_from(response))


def _is_unknown_tool(exc: Exception) -> bool:
    """Accounts on the older API only know `web_search_preview`."""
    message = str(exc).lower()
    return "web_search" in message and (
        "unknown" in message
        or "not supported" in message
        or "invalid" in message
        or "unsupported" in message
    )


def _as_ground_truth_error(
    exc: Exception, settings: Settings, model: str
) -> GroundTruthError:
    mapped = _as_classifier_error(exc, settings)
    text = str(mapped).replace(repr(settings.model), repr(model))
    return GroundTruthError(text)


__all__ = [
    "MAX_SOURCES",
    "NEUTRAL_HOSTS",
    "Access",
    "Availability",
    "BaselineValue",
    "Coverage",
    "ExternalSource",
    "Fit",
    "Gate",
    "GroundTruthError",
    "GroundTruthReport",
    "KPICoverage",
    "SourceFindings",
    "SourceKind",
    "assess",
    "gate",
    "identify",
    "self_reported",
    "subject_text",
]

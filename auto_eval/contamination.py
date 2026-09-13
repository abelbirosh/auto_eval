"""Ground truth the model cannot already have seen.

A public benchmark that predates the model under test measures two things at
once - whether the agent can do the job, and whether the answers were in its
training data - and there is no way to separate them after the fact. The effect
is documented rather than theoretical: LiveCodeBench's authors found model
performance dropping on problems published after the training cutoff, which is
what contamination looks like from the outside.

So this module holds three things:

1. **Cutoffs**, as the vendor publishes them, each with the page it was read off.
   Only the models this project defaults to are listed. An unknown model gets
   `UNKNOWN` rather than a guess, because a wrong cutoff turns a contaminated
   suite into a clean-looking one, which is the error that matters.
2. **Ground truth that resists contamination**, in three shapes: *held out* (the
   answers were never published at all), *rolling* (new problems arrive on a
   schedule, so there is always a window newer than any given model), and
   *post-cutoff* (published once, after the model stopped learning).
3. **A verdict for a suite**, case by case. A case generated from the spec or
   harvested from the user's own runs is clean by construction; an adapted case
   is only as clean as the suite it came from, and one published before the
   cutoff is flagged by name.

Dates and sizes are as the sources state them and every entry carries its URL;
they are facts about a moving world, so check the link before quoting a number
from here.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from .schema import TaskSpec

if TYPE_CHECKING:  # imported for the annotations only, to keep the layering
    from .dataset import Dataset
    from .suite import Suite


class Freshness(str, Enum):
    """How much of a score on this source could be recall rather than work."""

    HELD_OUT = "held_out"  # the answers were never published
    ROLLING = "rolling"  # refreshed on a schedule; filter to a fresh window
    POST_CUTOFF = "post_cutoff"  # published after this model stopped learning
    PRE_CUTOFF = "pre_cutoff"  # inside the training window
    UNKNOWN = "unknown"  # no date, or no cutoff for this model


# Worst first: the verdict for a whole suite is the worst verdict in it, and an
# unknown date is treated as worse than a known-clean one and better than a
# known-contaminated one.
SEVERITY = [
    Freshness.PRE_CUTOFF,
    Freshness.UNKNOWN,
    Freshness.POST_CUTOFF,
    Freshness.ROLLING,
    Freshness.HELD_OUT,
]

CLEAN = frozenset({Freshness.HELD_OUT, Freshness.ROLLING, Freshness.POST_CUTOFF})


class Cutoff(BaseModel):
    """One model's knowledge cutoff, as its vendor publishes it."""

    model: str
    cutoff: date
    source: str = Field(description="The page this was read off.")
    read_on: date = Field(description="When it was read, so a stale entry is visible.")


# Read off the vendor's own model pages on 2026-09-12. Deliberately short: the
# three models this project defaults to. Anything else needs --cutoff, and says
# so rather than being assumed.
CUTOFFS: Dict[str, Cutoff] = {
    "gpt-5": Cutoff(
        model="gpt-5",
        cutoff=date(2024, 9, 30),
        source="https://platform.openai.com/docs/models/gpt-5",
        read_on=date(2026, 9, 12),
    ),
    "gpt-5-mini": Cutoff(
        model="gpt-5-mini",
        cutoff=date(2024, 5, 31),
        source="https://platform.openai.com/docs/models/gpt-5-mini",
        read_on=date(2026, 9, 12),
    ),
    "gpt-5-nano": Cutoff(
        model="gpt-5-nano",
        cutoff=date(2024, 5, 31),
        source="https://platform.openai.com/docs/models/gpt-5-nano",
        read_on=date(2026, 9, 12),
    ),
}

CUTOFF_HELP = (
    "No published cutoff is on file for this model. Look it up on the vendor's model page "
    "and pass it with --cutoff YYYY-MM-DD; until then every public source is treated as "
    "possibly inside the training window."
)


def cutoff_for(model: str) -> Optional[Cutoff]:
    """The cutoff for `model`, matching a dated snapshot back to its family.

    `gpt-5-mini-2025-08-07` resolves to `gpt-5-mini`; an unrecognised model
    resolves to nothing, which is the answer, not a failure.
    """
    name = model.strip().lower()
    if name in CUTOFFS:
        return CUTOFFS[name]
    matches = [key for key in CUTOFFS if name.startswith(key)]
    return CUTOFFS[max(matches, key=len)] if matches else None


# --------------------------------------------------------------------------
# Ground truth that resists contamination
# --------------------------------------------------------------------------


class FreshSource(BaseModel):
    """A public source whose answers a model is unlikely to have memorised."""

    name: str
    url: str
    publisher: str
    measures: str = Field(description="What it tests, in one sentence.")
    metric: str
    protection: Freshness = Field(
        description="How it resists contamination: held out, rolling, or simply newer."
    )
    released: date = Field(description="When the suite itself first appeared.")
    refresh: Optional[str] = Field(
        default=None, description="The refresh cadence, for a rolling suite."
    )
    size: Optional[str] = None
    licence: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    staying_clean: str = Field(
        description="What you have to do when running it to keep the score uncontaminated."
    )
    caveats: str = Field(description="Why it might not transfer. Never empty.")


FRESH: List[FreshSource] = [
    FreshSource(
        name="LiveBench",
        url="https://livebench.ai/",
        publisher="White, Dooley, Roberts et al. (Abacus.AI, NYU, Nvidia, USC, UMD)",
        measures=(
            "Reasoning, coding, maths, data analysis, language and instruction "
            "following, with questions drawn from recent sources so that each "
            "release postdates the models it scores."
        ),
        metric="per-category and overall accuracy, scored objectively without a judge",
        protection=Freshness.ROLLING,
        released=date(2024, 6, 27),
        refresh="the site states 23 tasks across 7 categories, refreshed every six months",
        size="23 tasks across 7 categories",
        licence="see the repository",
        keywords=[
            "reasoning",
            "coding",
            "maths",
            "math",
            "data analysis",
            "instruction following",
            "language",
        ],
        staying_clean=(
            "Score on the newest release and say which release it was. An older "
            "LiveBench release is an ordinary benchmark with an ordinary contamination problem."
        ),
        caveats=(
            "General capability, not your task. It replaces a leaderboard, not a "
            "domain eval - and being objective-only, it says nothing about judged "
            "qualities like tone."
        ),
    ),
    FreshSource(
        name="LiveCodeBench",
        url="https://livecodebench.github.io/",
        publisher="Jain et al. (UC Berkeley, MIT, Cornell)",
        measures=(
            "Competition programming problems collected continuously from LeetCode, "
            "AtCoder and Codeforces, each stamped with its publication date."
        ),
        metric="pass@1 over a date-filtered window, by execution",
        protection=Freshness.ROLLING,
        released=date(2024, 3, 12),
        refresh="new problems are collected continuously; every problem carries a release date",
        size="hundreds of problems, growing; released as dated windows (v1, v2, ...)",
        licence="see the repository",
        keywords=[
            "code",
            "coding",
            "programming",
            "algorithm",
            "python",
            "competition",
        ],
        staying_clean=(
            "Filter to problems published after the model's cutoff and report the window "
            "you used - this is the one suite here where the filter is a documented "
            "feature rather than your own bookkeeping."
        ),
        caveats=(
            "Competition problems, self-contained and heavily tested; good for raw "
            "coding ability, silent about work in a real repository."
        ),
    ),
    FreshSource(
        name="SWE-bench-Live",
        url="https://github.com/microsoft/swe-bench-live",
        publisher="Microsoft (NeurIPS 2025 Datasets & Benchmarks)",
        measures=(
            "Resolving real GitHub issues, drawn from repositories continuously and "
            "verified automatically, so the task pool keeps moving ahead of the models."
        ),
        metric="% of instances resolved, by running each repository's own tests",
        protection=Freshness.ROLLING,
        released=date(2025, 5, 29),
        refresh="the project states that 50 newly verified issues are added to the test split each month",
        size="a growing test split; the lite and verified splits are frozen for comparability",
        licence="see the repository",
        keywords=[
            "issue",
            "patch",
            "pull request",
            "repository",
            "codebase",
            "bug fix",
            "agent",
        ],
        staying_clean=(
            "Use the months after the model's cutoff and say which months. Running "
            "it needs the per-repository container harness, same as SWE-bench."
        ),
        caveats=(
            "Python-heavy and expensive to run. Instances are verified automatically "
            "rather than by hand, so individual tasks are noisier than SWE-bench Verified."
        ),
    ),
    FreshSource(
        name="Humanity's Last Exam",
        url="https://lastexam.ai/",
        publisher="Center for AI Safety and Scale AI",
        measures=(
            "Expert-written questions across more than a hundred subjects, at the "
            "edge of what specialists can answer, with a private set kept back."
        ),
        metric="accuracy, and calibration error",
        protection=Freshness.HELD_OUT,
        released=date(2025, 1, 23),
        refresh=None,
        size="2,500 public questions, plus a private held-out set",
        licence="see the site",
        keywords=["knowledge", "expert", "reasoning", "factual", "exam", "calibration"],
        staying_clean=(
            "The public set is public, so treat a rising public score with suspicion; "
            "the private set exists precisely to check overfitting against."
        ),
        caveats=(
            "Short-answer academic questions. Nothing here resembles an agent doing "
            "a job, and the ceiling is so high that a low score separates nothing."
        ),
    ),
    FreshSource(
        name="FrontierMath",
        url="https://epoch.ai/frontiermath",
        publisher="Epoch AI",
        measures=(
            "Research-level mathematics problems written for this benchmark by "
            "mathematicians and never published, graded on an exact final answer."
        ),
        metric="% solved",
        protection=Freshness.HELD_OUT,
        released=date(2024, 11, 1),
        refresh="tiers added over time; the bulk of the problem set stays unpublished",
        size="hundreds of problems; only a handful have been made public",
        licence="held by Epoch AI; evaluation is run by them",
        keywords=["maths", "math", "mathematics", "proof", "reasoning", "research"],
        staying_clean=(
            "You cannot run it yourself, which is the point - ask Epoch AI to evaluate, "
            "and read it as a reference point rather than a suite you own."
        ),
        caveats=(
            "Maths only, and unrunnable in-house. Useful as evidence that a model is "
            "not merely recalling, not as a measurement of your system."
        ),
    ),
    FreshSource(
        name="ARC-AGI-2",
        url="https://arcprize.org/arc-agi",
        publisher="ARC Prize Foundation",
        measures=(
            "Novel grid puzzles, each solvable by people and each requiring a rule to "
            "be worked out rather than recalled."
        ),
        metric="% of tasks solved within the attempt limit",
        protection=Freshness.HELD_OUT,
        released=date(2025, 3, 24),
        refresh=None,
        size="1,000 public training and 120 public evaluation tasks, plus semi-private and private evaluation sets of 120 tasks each",
        licence="Apache-2.0 for the task data; see the repository",
        keywords=["reasoning", "abstraction", "generalisation", "novel", "puzzle"],
        staying_clean=(
            "Report the public evaluation set separately from the semi-private one; "
            "the private sets are what the leaderboard is scored on."
        ),
        caveats=(
            "Deliberately unlike real work: no language, no tools, no environment. It "
            "is the cleanest evidence of non-recall here and the least transferable."
        ),
    ),
]


def verdict_for(source: FreshSource, cutoff: Optional[Cutoff]) -> Freshness:
    """How clean this source is for a model with this cutoff."""
    if source.protection in (Freshness.HELD_OUT, Freshness.ROLLING):
        return source.protection
    if cutoff is None:
        return Freshness.UNKNOWN
    return (
        Freshness.POST_CUTOFF
        if source.released > cutoff.cutoff
        else Freshness.PRE_CUTOFF
    )


def matching(spec: TaskSpec, *, limit: int = 3) -> List[FreshSource]:
    """The fresh sources whose subject overlaps this spec's KPIs, best first."""
    words = " ".join(
        [spec.subject.description, spec.summary, *[k.name for k in spec.kpis]]
    ).lower()
    scored = [
        (sum(1 for keyword in source.keywords if keyword in words), source)
        for source in FRESH
    ]
    hits = sorted((pair for pair in scored if pair[0]), key=lambda pair: -pair[0])
    return [source for _, source in hits[:limit]]


# --------------------------------------------------------------------------
# What a suite's own cases are exposed to
# --------------------------------------------------------------------------


class SourceRisk(BaseModel):
    """One origin the cases came from, and what a score on it is worth."""

    source: str
    url: Optional[str] = None
    verdict: Freshness
    cases: int
    why: str


class ContaminationReport(BaseModel):
    """Whether a number from this suite, on this model, is measuring recall."""

    model: str
    cutoff: Optional[date] = None
    cutoff_source: Optional[str] = None
    verdict: Freshness = Freshness.UNKNOWN
    by_origin: List[SourceRisk] = Field(default_factory=list)
    clean_cases: int = 0
    at_risk_cases: int = 0
    unknown_cases: int = 0
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.verdict in CLEAN


def _host_path(url: str) -> str:
    parts = urlsplit(url)
    return (parts.netloc.lower().removeprefix("www.") + parts.path.rstrip("/")).lower()


def _fresh_for_url(url: str) -> Optional[FreshSource]:
    target = _host_path(url)
    for source in FRESH:
        known = _host_path(source.url)
        if target.startswith(known) or known.startswith(target):
            return source
    return None


def for_suite(
    suite: "Suite", model: str, *, cutoff: Optional[Cutoff] = None
) -> ContaminationReport:
    """Assess the cases in `suite` against what `model` may already have seen.

    Takes the suite structurally rather than by import, so this module stays
    below `auto_eval.suite` and can be used on a report loaded from disk.
    """
    resolved = cutoff or cutoff_for(model)
    report = ContaminationReport(
        model=model,
        cutoff=resolved.cutoff if resolved else None,
        cutoff_source=resolved.source if resolved else None,
    )
    if resolved is None:
        report.warnings.append(CUTOFF_HELP)

    origins: Dict[str, List] = {}
    for case in suite.cases:
        if case.source.value == "adapted":
            key = case.source_url or "public suite, source not recorded"
        else:
            key = case.source.value
        origins.setdefault(key, []).append(case)

    for key, cases in sorted(origins.items(), key=lambda pair: -len(pair[1])):
        if key == "harvested":
            risk = SourceRisk(
                source="your own runs",
                verdict=Freshness.HELD_OUT,
                cases=len(cases),
                why="Harvested from runs you hold. Nothing here was ever published, so it cannot be in training data.",
            )
        elif key == "synthesised":
            risk = SourceRisk(
                source="generated from your spec",
                verdict=Freshness.HELD_OUT,
                cases=len(cases),
                why=(
                    "Written by rule from your spec when the suite was built, so the cases "
                    "themselves are new - though the behaviours they probe may be ordinary."
                ),
            )
        else:
            known = _fresh_for_url(key) if key.startswith("http") else None
            if known is not None:
                verdict = verdict_for(known, resolved)
                risk = SourceRisk(
                    source=known.name,
                    url=known.url,
                    verdict=verdict,
                    cases=len(cases),
                    why=f"{known.name}: {known.staying_clean}",
                )
            else:
                risk = SourceRisk(
                    source=key,
                    url=key if key.startswith("http") else None,
                    verdict=Freshness.UNKNOWN,
                    cases=len(cases),
                    why=(
                        "Adapted from a public suite with no publication date on file. Check when it "
                        "was released: if that is before the cutoff, a pass here may be recall."
                    ),
                )
        report.by_origin.append(risk)

    for risk in report.by_origin:
        if risk.verdict in CLEAN:
            report.clean_cases += risk.cases
        elif risk.verdict is Freshness.UNKNOWN:
            report.unknown_cases += risk.cases
        else:
            report.at_risk_cases += risk.cases

    report.verdict = (
        min((risk.verdict for risk in report.by_origin), key=SEVERITY.index)
        if report.by_origin
        else Freshness.UNKNOWN
    )

    if report.at_risk_cases:
        report.warnings.append(
            f"{report.at_risk_cases} case(s) come from material published before "
            f"{report.cutoff}, the cutoff for {model}. A pass on those is partly a memory test."
        )
    if report.unknown_cases:
        report.warnings.append(
            f"{report.unknown_cases} case(s) come from a public source with no date on file. "
            "Find its publication date before quoting a number from them."
        )
    if resolved is not None:
        report.notes.append(
            f"Cutoff for {model} read as {resolved.cutoff} from {resolved.source} on {resolved.read_on}. "
            "Vendors restate these; check the page if the number matters."
        )
    return report


def for_dataset(
    dataset: "Dataset", model: str, *, cutoff: Optional[Cutoff] = None
) -> ContaminationReport:
    """Assess a board's items against what `model` may already have seen.

    An item carries the date its fact became public. Items after the cutoff are
    what a retrieval board should be made of; items before it are answerable
    from memory, and the board's model-only row is the direct measurement of how
    many of them are.
    """
    resolved = cutoff or cutoff_for(model)
    report = ContaminationReport(
        model=model,
        cutoff=resolved.cutoff if resolved else None,
        cutoff_source=resolved.source if resolved else None,
    )
    if resolved is None:
        report.warnings.append(CUTOFF_HELP)

    dated = [(item, item.published) for item in dataset.items if item.published]
    undated = len(dataset.items) - len(dated)
    line = resolved.cutoff if resolved else None
    after = [(item, when) for item, when in dated if line and when > line]
    before = [(item, when) for item, when in dated if line and when <= line]

    if after:
        report.by_origin.append(
            SourceRisk(
                source=f"{dataset.name}: published after the cutoff",
                url=dataset.source
                if (dataset.source or "").startswith("http")
                else None,
                verdict=Freshness.POST_CUTOFF,
                cases=len(after),
                why=(
                    f"Dated between {min(when for _, when in after)} and "
                    f"{max(when for _, when in after)}, after {line}. "
                    "These are the items a retrieval board is made of."
                ),
            )
        )
    if before:
        report.by_origin.append(
            SourceRisk(
                source=f"{dataset.name}: published before the cutoff",
                verdict=Freshness.PRE_CUTOFF,
                cases=len(before),
                why=(
                    "Inside the training window, so the model may answer them without "
                    "retrieving anything. The model-only row measures exactly this."
                ),
            )
        )
    if undated or (dated and resolved is None):
        report.by_origin.append(
            SourceRisk(
                source=f"{dataset.name}: no usable date",
                verdict=Freshness.UNKNOWN,
                cases=undated if resolved else len(dataset.items),
                why=(
                    "No publication date on the item, or no cutoff on file for this model, so "
                    "nothing can be said about whether the answer was already known."
                ),
            )
        )

    for risk in report.by_origin:
        if risk.verdict in CLEAN:
            report.clean_cases += risk.cases
        elif risk.verdict is Freshness.UNKNOWN:
            report.unknown_cases += risk.cases
        else:
            report.at_risk_cases += risk.cases

    report.verdict = (
        min((risk.verdict for risk in report.by_origin), key=SEVERITY.index)
        if report.by_origin
        else Freshness.UNKNOWN
    )
    if report.at_risk_cases:
        report.warnings.append(
            f"{report.at_risk_cases} of {len(dataset.items)} items predate {report.cutoff}, the "
            f"cutoff for {model}. Read the model-only row before reading any other."
        )
    if resolved is not None:
        report.notes.append(
            f"Cutoff for {model} read as {resolved.cutoff} from {resolved.source} on {resolved.read_on}."
        )
    return report


def parse_cutoff(model: str, value: str) -> Cutoff:
    """A cutoff the user supplied on the command line."""
    return Cutoff(
        model=model,
        cutoff=date.fromisoformat(value.strip()),
        source="supplied on the command line",
        read_on=date.today(),
    )


def fresh_by_verdict(cutoff: Optional[Cutoff]) -> List[Tuple[FreshSource, Freshness]]:
    """`(source, verdict)` for the whole fresh catalogue, cleanest first."""
    pairs = [(source, verdict_for(source, cutoff)) for source in FRESH]
    return sorted(pairs, key=lambda pair: -SEVERITY.index(pair[1]))


__all__ = [
    "CLEAN",
    "CUTOFFS",
    "CUTOFF_HELP",
    "FRESH",
    "SEVERITY",
    "ContaminationReport",
    "Cutoff",
    "FreshSource",
    "Freshness",
    "SourceRisk",
    "cutoff_for",
    "for_dataset",
    "for_suite",
    "fresh_by_verdict",
    "matching",
    "parse_cutoff",
    "verdict_for",
]

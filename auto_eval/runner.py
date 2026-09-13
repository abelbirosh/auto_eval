"""Block 5: run the suite, and say what the numbers are allowed to mean.

The four blocks before this one produce a suite. This one runs it against the
provider API and folds the verdicts into a report the dashboard reads.

The shape of the report is the whole argument of the module. A run produces
three kinds of check - settled by machine, settled by a judge, and **not settled
at all** - and the third kind is reported by name rather than dropped. So every
number here comes with its denominator:

* the **headline** is the held-out split, because a score tuned against is not a
  score;
* a **pass rate** is over checks that actually ran, and the count of blocked
  checks sits next to it;
* a case whose every check was blocked is `blocked`, not `passed`. A suite whose
  fixtures nobody built can still be run here - it will come back honest about
  having measured trajectories and nothing else.

Everything except the provider and judge calls is deterministic, so the
aggregation can be tested without a key: `--mock` runs the whole path against a
canned client.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, computed_field

from .agent import AgentProfile
from .authoring import Case
from .classifier import ClassifierError, _build_client
from .config import Settings, get_settings
from .contamination import ContaminationReport, Cutoff
from .contamination import for_suite as contamination_for_suite
from .grade import Blocked, Judge, grade, runnable
from .judge import judge_with
from .provider import MockClient, execute
from .suite import Split, Suite, verify_digest
from .trace import StopReason, Trace, Usage
from .verify import AUTOMATIC_KINDS, CaseResult, VerifierKind, settle

# Cases run in parallel. Four is polite to a rate limit and still turns a
# hundred-run suite around in a couple of minutes.
DEFAULT_CONCURRENCY = 4

RUN_FILENAME = "run.json"
TRACE_DIR = "traces"

Progress = Callable[[int, int, str], None]


class RunError(ClassifierError):
    """Raised when the run could not be started at all."""


class CaseStatus(str, Enum):
    PASSED = "passed"  # every sample that ran, passed
    FAILED = "failed"
    BLOCKED = "blocked"  # nothing in this case could be checked here
    ERROR = "error"  # the provider never completed a sample


class SampleRun(BaseModel):
    """One run of one case: the verdicts, what could not be checked, the cost."""

    case_id: str
    sample: int
    status: CaseStatus
    stop_reason: StopReason
    result: CaseResult
    blocked: List[Blocked] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    error: Optional[str] = None

    @property
    def settled(self) -> int:
        return len(self.result.outcomes)


class CaseOutcome(BaseModel):
    """One case across its samples."""

    case_id: str
    family: str
    behaviour: str
    split: Split
    source: str
    eval_types: List[str] = Field(default_factory=list)
    instruction: str = ""
    status: CaseStatus = CaseStatus.BLOCKED
    samples: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_blocked: int = 0
    blocked: List[Blocked] = Field(
        default_factory=list, description="Deduplicated across samples."
    )
    runs: List[SampleRun] = Field(default_factory=list)

    # Serialised, not just computed: the page and the JSON both read it.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> Optional[float]:
        graded = self.passed + self.failed
        return self.passed / graded if graded else None


class Tally(BaseModel):
    """Checks, split three ways. The third number is why this type exists."""

    passed: int = 0
    failed: int = 0
    blocked: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def settled(self) -> int:
        return self.passed + self.failed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return self.settled + self.blocked

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> Optional[float]:
        return self.passed / self.settled if self.settled else None


class Group(BaseModel):
    """A tally under a label: a KPI, a behaviour, a family, a check kind."""

    label: str
    cases: int = 0
    checks: Tally = Field(default_factory=Tally)


class Score(BaseModel):
    """The number for one split, with everything needed to read it."""

    label: str
    cases: int = 0
    runs: int = 0
    passed: int = 0
    failed: int = 0
    blocked: int = 0
    errors: int = 0
    checks: Tally = Field(default_factory=Tally)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def graded_runs(self) -> int:
        return self.passed + self.failed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> Optional[float]:
        return self.passed / self.graded_runs if self.graded_runs else None


class RunReport(BaseModel):
    """What one run of one suite found. The artefact the dashboard reads."""

    run_id: str
    started_at: str
    finished_at: str = ""
    seconds: float = 0.0

    suite_name: str
    suite_digest: str
    spec_digest: str
    subject: str
    suite_verified: bool = Field(
        default=True, description="Whether the suite still hashes to its own digest."
    )

    model: str = Field(description="The model under test.")
    judge_model: Optional[str] = None
    mock: bool = False
    samples: int = 1
    split: str = "all"
    limit: Optional[int] = None

    headline: Score = Field(
        default_factory=lambda: Score(label="held out"),
        description="The held-out split: the number that may be reported.",
    )
    dev: Score = Field(default_factory=lambda: Score(label="dev"))
    overall: Score = Field(default_factory=lambda: Score(label="all cases"))

    by_kpi: List[Group] = Field(default_factory=list)
    by_behaviour: List[Group] = Field(default_factory=list)
    by_family: List[Group] = Field(default_factory=list)
    by_kind: List[Group] = Field(default_factory=list)

    usage: Usage = Field(default_factory=Usage)
    contamination: Optional[ContaminationReport] = None
    warnings: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    cases: List[CaseOutcome] = Field(default_factory=list)

    @property
    def blocked_checks(self) -> int:
        return self.overall.checks.blocked


class Run(BaseModel):
    """A report and the traces behind it, before either is written down."""

    report: RunReport
    traces: List[Trace] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def _status(result: CaseResult, blocked: Sequence[Blocked], trace: Trace) -> CaseStatus:
    if trace.stop_reason is StopReason.PROVIDER_ERROR:
        return CaseStatus.ERROR
    if not result.outcomes:
        return CaseStatus.BLOCKED
    return CaseStatus.PASSED if result.passed else CaseStatus.FAILED


def _run_one(
    case: Case,
    agent_profile: AgentProfile,
    *,
    client: Any,
    model: str,
    sample: int,
    judge: Optional[Judge],
    prices: Optional[Tuple[float, float]],
    mock: bool,
) -> Tuple[SampleRun, Trace]:
    trace = execute(
        case,
        agent_profile,
        client=client,
        model=model,
        sample=sample,
        prices=prices,
        mock=mock,
    )
    outcomes, blocked = grade(case, trace, judge=judge)
    result = settle(case.id, runnable(case, blocked), outcomes, sample=sample)
    return (
        SampleRun(
            case_id=case.id,
            sample=sample,
            status=_status(result, blocked, trace),
            stop_reason=trace.stop_reason,
            result=result,
            blocked=blocked,
            usage=trace.usage,
            error=trace.error,
        ),
        trace,
    )


def _case_status(runs: Sequence[SampleRun]) -> CaseStatus:
    """One status for a case from k samples. A flaky case is a failing case."""
    statuses = [run.status for run in runs]
    if statuses and all(s is CaseStatus.ERROR for s in statuses):
        return CaseStatus.ERROR
    graded = [s for s in statuses if s in (CaseStatus.PASSED, CaseStatus.FAILED)]
    if not graded:
        return CaseStatus.BLOCKED if statuses else CaseStatus.ERROR
    return (
        CaseStatus.PASSED
        if all(s is CaseStatus.PASSED for s in graded)
        else CaseStatus.FAILED
    )


def _select(suite: Suite, split: Optional[Split], limit: Optional[int]) -> List[Case]:
    cases = suite.cases if split is None else suite.cases_in(split)
    return cases[:limit] if limit else list(cases)


def run_suite(
    suite: Suite,
    *,
    client: Optional[Any] = None,
    model: Optional[str] = None,
    judge_model: Optional[str] = None,
    judge: bool = True,
    samples: Optional[int] = None,
    split: Optional[Split] = None,
    limit: Optional[int] = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    mock: bool = False,
    prices: Optional[Tuple[float, float]] = None,
    cutoff: Optional[Cutoff] = None,
    progress: Optional[Progress] = None,
    settings: Optional[Settings] = None,
) -> Run:
    """Run every selected case against the provider, `samples` times each.

    With `mock=True` no provider is called at all: the whole path runs against a
    canned client, which is how the report and the dashboard are exercised
    without a key.
    """
    settings = settings or get_settings()
    subject_model = model or settings.effective_subject_model
    ruler = judge_model or settings.effective_judge_model

    if mock:
        client = client or MockClient()
    elif client is None:
        client = _build_client(settings)

    judge_fn: Optional[Judge] = judge_with(client, ruler) if judge else None

    cases = _select(suite, split, limit)
    if not cases:
        raise RunError(
            f"Nothing to run: the {split.value if split else 'selected'} split of this suite is empty."
        )

    per_case = samples or suite.samples or 1
    started = datetime.now(timezone.utc)
    run_id = f"{suite.name}-{started.strftime('%Y%m%d-%H%M%S')}"

    jobs = [(case, sample) for case in cases for sample in range(per_case)]
    done = 0
    runs: Dict[str, List[SampleRun]] = {}
    traces: List[Trace] = []

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                _run_one,
                case,
                suite.profile,
                client=client,
                model=subject_model,
                sample=sample,
                judge=judge_fn,
                prices=prices,
                mock=mock,
            ): (case, sample)
            for case, sample in jobs
        }
        for future in futures:
            case, _ = futures[future]
            sample_run, trace = future.result()
            runs.setdefault(case.id, []).append(sample_run)
            traces.append(trace)
            done += 1
            if progress:
                progress(done, len(jobs), case.id)

    outcomes: List[CaseOutcome] = []
    for case in cases:
        case_runs = sorted(runs.get(case.id, []), key=lambda r: r.sample)
        outcome = CaseOutcome(
            case_id=case.id,
            family=case.family,
            behaviour=case.behaviour,
            split=suite.split_of(case.id),
            source=case.source.value,
            eval_types=[t.value for t in case.eval_types],
            instruction=case.instruction,
            samples=len(case_runs),
            runs=case_runs,
            status=_case_status(case_runs),
        )
        for run in case_runs:
            outcome.passed += run.status is CaseStatus.PASSED
            outcome.failed += run.status is CaseStatus.FAILED
            outcome.errors += run.status is CaseStatus.ERROR
            outcome.checks_passed += sum(1 for o in run.result.outcomes if o.passed)
            outcome.checks_failed += sum(1 for o in run.result.outcomes if not o.passed)
            outcome.checks_blocked += len(run.blocked)
        seen: set = set()
        for run in case_runs:
            for block in run.blocked:
                if block.verifier not in seen:
                    seen.add(block.verifier)
                    outcome.blocked.append(block)
        outcomes.append(outcome)

    finished = datetime.now(timezone.utc)
    report = RunReport(
        run_id=run_id,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        seconds=round((finished - started).total_seconds(), 2),
        suite_name=suite.name,
        suite_digest=suite.digest,
        spec_digest=suite.spec_digest,
        subject=suite.subject,
        suite_verified=verify_digest(suite),
        model=subject_model,
        judge_model=ruler if judge else None,
        mock=mock,
        samples=per_case,
        split=split.value if split else "all",
        limit=limit,
        cases=outcomes,
    )
    return Run(report=aggregate(suite, report, cases, cutoff=cutoff), traces=traces)


# --------------------------------------------------------------------------
# Aggregation - pure, and where the honesty lives
# --------------------------------------------------------------------------


def _score(label: str, outcomes: Sequence[CaseOutcome]) -> Score:
    score = Score(label=label, cases=len(outcomes))
    for outcome in outcomes:
        score.runs += outcome.samples
        score.passed += outcome.passed
        score.failed += outcome.failed
        score.errors += outcome.errors
        score.blocked += sum(1 for r in outcome.runs if r.status is CaseStatus.BLOCKED)
        score.checks.passed += outcome.checks_passed
        score.checks.failed += outcome.checks_failed
        score.checks.blocked += outcome.checks_blocked
    return score


def _grouped(
    outcomes: Sequence[CaseOutcome], key: Callable[[CaseOutcome], str]
) -> List[Group]:
    groups: Dict[str, Group] = {}
    for outcome in outcomes:
        label = key(outcome)
        group = groups.setdefault(label, Group(label=label))
        group.cases += 1
        group.checks.passed += outcome.checks_passed
        group.checks.failed += outcome.checks_failed
        group.checks.blocked += outcome.checks_blocked
    return sorted(groups.values(), key=lambda g: (-g.checks.total, g.label))


def _by_kpi(suite: Suite, outcomes: Sequence[CaseOutcome]) -> List[Group]:
    """KPI -> how its own checks did. A KPI with nothing settled is not measured."""
    kpi_of = {v.id: v.kpi for case in suite.cases for v in case.verifiers if v.kpi}
    groups = {name: Group(label=name) for name in suite.kpis}
    touched: Dict[str, set] = {name: set() for name in suite.kpis}

    for outcome in outcomes:
        for run in outcome.runs:
            for verdict in run.result.outcomes:
                name = kpi_of.get(verdict.verifier)
                if name in groups:
                    groups[name].checks.passed += verdict.passed
                    groups[name].checks.failed += not verdict.passed
                    touched[name].add(outcome.case_id)
            for block in run.blocked:
                name = block.kpi or kpi_of.get(block.verifier)
                if name in groups:
                    groups[name].checks.blocked += 1
                    touched[name].add(outcome.case_id)
    for name, group in groups.items():
        group.cases = len(touched[name])
    return list(groups.values())


def _by_kind(suite: Suite, outcomes: Sequence[CaseOutcome]) -> List[Group]:
    kind_of = {v.id: v.kind for case in suite.cases for v in case.verifiers}
    groups: Dict[str, Group] = {}
    touched: Dict[str, set] = {}

    def group(kind: VerifierKind, case_id: str) -> Group:
        touched.setdefault(kind.value, set()).add(case_id)
        return groups.setdefault(kind.value, Group(label=kind.value))

    for outcome in outcomes:
        for run in outcome.runs:
            for verdict in run.result.outcomes:
                kind = kind_of.get(verdict.verifier)
                if kind is None:
                    continue
                tally = group(kind, outcome.case_id).checks
                tally.passed += verdict.passed
                tally.failed += not verdict.passed
            for block in run.blocked:
                group(block.kind, outcome.case_id).checks.blocked += 1
    for label, group_ in groups.items():
        group_.cases = len(touched[label])
    return sorted(groups.values(), key=lambda g: (-g.checks.total, g.label))


def _warnings(suite: Suite, report: RunReport, cases: Sequence[Case]) -> List[str]:
    """What stops this run's numbers from meaning what they look like."""
    out: List[str] = []

    if report.mock:
        out.append(
            "This run called no provider. Every verdict came from the canned client, so the "
            "numbers describe the harness and nothing else."
        )
    if not report.suite_verified:
        out.append(
            "The suite no longer hashes to its own digest, so these results cannot be traced to "
            "the cases that produced them. Rebuild it with `auto-eval author`."
        )

    blocked = report.overall.checks.blocked
    total = report.overall.checks.total
    if blocked and total:
        kinds = Counter(
            block.kind.value for case in report.cases for block in case.blocked
        )
        named = ", ".join(f"{kind} x{n}" for kind, n in kinds.most_common(4))
        out.append(
            f"{blocked} of {total} checks could not run here ({blocked / total:.0%}): {named}. "
            "The pass rates below are over the rest."
        )

    all_blocked = [c for c in report.cases if c.status is CaseStatus.BLOCKED]
    if all_blocked:
        out.append(
            f"{len(all_blocked)} case(s) had no runnable check at all, so they are reported as "
            "blocked rather than passed."
        )

    errors = [c for c in report.cases if c.errors]
    if errors:
        first = next(
            (r.error for c in errors for r in c.runs if r.error), "provider error"
        )
        out.append(
            f"{sum(c.errors for c in errors)} run(s) never completed: {first}. "
            "Those cases are neither passes nor failures."
        )

    if report.judge_model and report.judge_model == report.model:
        out.append(
            f"The judge and the system under test are both {report.model}. MT-Bench documents "
            "self-preference bias in exactly this setup: set AUTO_EVAL_JUDGE_MODEL to something else "
            "before reporting a judged number."
        )
    if not report.judge_model:
        rubrics = sum(
            1 for case in cases for v in case.verifiers if v.kind is VerifierKind.RUBRIC
        )
        if rubrics:
            out.append(
                f"No judge ran, so {rubrics} rubric check(s) were left unsettled. Every judged KPI in "
                "this suite is unmeasured in this run."
            )

    automatic = [
        b for case in report.cases for b in case.blocked if b.kind in AUTOMATIC_KINDS
    ]
    if automatic:
        out.append(
            f"{len(automatic)} of the blocked checks are ones a machine could settle if the "
            "environment existed - end state, execution, containment. Building the fixtures is what "
            "turns this run into a measurement of outcomes rather than of trajectories."
        )

    fatal = [b for case in report.cases for b in case.blocked if b.fatal]
    if fatal:
        out.append(
            f"{len(fatal)} of the blocked checks are guardrails, whose whole job is to condemn a run. "
            "A pass here is not evidence that the guardrail holds."
        )

    if report.headline.cases and report.headline.pass_rate is None:
        out.append(
            "Nothing in the held-out split settled, so this run has no reportable number - only "
            "the dev split moved."
        )
    out.extend(suite.warnings)
    return out


def aggregate(
    suite: Suite,
    report: RunReport,
    cases: Sequence[Case],
    *,
    cutoff: Optional[Cutoff] = None,
) -> RunReport:
    """Fold the case outcomes into scores, groups, warnings. Pure."""
    outcomes = report.cases
    held = [o for o in outcomes if o.split is Split.HELD_OUT]
    dev = [o for o in outcomes if o.split is Split.DEV]

    report.headline = _score("held out", held)
    report.dev = _score("dev", dev)
    report.overall = _score("all cases", outcomes)

    report.by_kpi = _by_kpi(suite, outcomes)
    report.by_behaviour = _grouped(outcomes, lambda o: o.behaviour)
    report.by_family = _grouped(outcomes, lambda o: o.family)
    report.by_kind = _by_kind(suite, outcomes)

    usage = Usage()
    for outcome in outcomes:
        for run in outcome.runs:
            usage = usage.plus(run.usage)
    report.usage = usage

    report.contamination = contamination_for_suite(suite, report.model, cutoff=cutoff)
    report.warnings = _warnings(suite, report, cases)
    if report.contamination:
        report.warnings.extend(report.contamination.warnings)
    report.notes = [
        f"{report.overall.runs} run(s) over {report.overall.cases} case(s), {report.samples} sample(s) each.",
        "The system under test was not told it was being evaluated; the harness presents each case as a live task.",
        "Every tool call was answered inside the harness, so no run touched a live system.",
        *(report.contamination.notes if report.contamination else []),
    ]
    return report


# --------------------------------------------------------------------------
# Writing it down
# --------------------------------------------------------------------------


class RunIndexEntry(BaseModel):
    """One line of the run list, for the dashboard's picker."""

    run_id: str
    path: str
    started_at: str
    suite_name: str
    subject: str
    model: str
    mock: bool = False
    cases: int = 0
    pass_rate: Optional[float] = None
    blocked_checks: int = 0
    warnings: int = 0


def write_run(run: Run, out_dir: Path) -> Path:
    """Write `runs/<run_id>/run.json` and one file per trace. Returns the directory."""
    target = out_dir / run.report.run_id
    (target / TRACE_DIR).mkdir(parents=True, exist_ok=True)
    (target / RUN_FILENAME).write_text(
        run.report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    for trace in run.traces:
        name = f"{trace.case_id}-{trace.sample}.json"
        (target / TRACE_DIR / name).write_text(
            trace.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    return target


def load_run(path: Path) -> RunReport:
    """Read a report back, from its directory or from the file itself."""
    target = path / RUN_FILENAME if path.is_dir() else path
    if not target.exists():
        raise RunError(f"No run at {target}.")
    try:
        return RunReport.model_validate_json(target.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RunError(f"{target} is not a readable run: {exc}") from exc


def load_trace(run_dir: Path, case_id: str, sample: int = 0) -> Trace:
    path = run_dir / TRACE_DIR / f"{case_id}-{sample}.json"
    if not path.exists():
        raise RunError(f"No trace for {case_id} sample {sample} in {run_dir}.")
    return Trace.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs(runs_dir: Path) -> List[RunIndexEntry]:
    """Every run under `runs_dir`, newest first. Unreadable runs are skipped."""
    if not runs_dir.is_dir():
        return []
    entries: List[RunIndexEntry] = []
    for candidate in sorted(runs_dir.iterdir(), reverse=True):
        path = candidate / RUN_FILENAME
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        headline = raw.get("headline") or {}
        graded = (headline.get("passed") or 0) + (headline.get("failed") or 0)
        entries.append(
            RunIndexEntry(
                run_id=raw.get("run_id", candidate.name),
                path=str(candidate),
                started_at=raw.get("started_at", ""),
                suite_name=raw.get("suite_name", ""),
                subject=raw.get("subject", ""),
                model=raw.get("model", ""),
                mock=bool(raw.get("mock")),
                cases=len(raw.get("cases") or []),
                pass_rate=(headline.get("passed", 0) / graded) if graded else None,
                blocked_checks=((raw.get("overall") or {}).get("checks") or {}).get(
                    "blocked", 0
                ),
                warnings=len(raw.get("warnings") or []),
            )
        )
    return sorted(entries, key=lambda e: e.started_at, reverse=True)


__all__ = [
    "DEFAULT_CONCURRENCY",
    "RUN_FILENAME",
    "TRACE_DIR",
    "CaseOutcome",
    "CaseStatus",
    "Group",
    "Run",
    "RunError",
    "RunIndexEntry",
    "RunReport",
    "SampleRun",
    "Score",
    "Tally",
    "aggregate",
    "list_runs",
    "load_run",
    "load_trace",
    "run_suite",
    "write_run",
]

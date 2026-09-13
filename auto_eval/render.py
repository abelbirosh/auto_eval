"""Render a TaskSpec, and a ground-truth report, as human-readable documents."""

from __future__ import annotations

from typing import List, Optional

from .agent import AgentProfile, RunMode
from .analysis import AnalysisReport, Reachability, Usability
from .authoring import Case, CaseSource, needs_fixture
from .board import Board, Row
from .contamination import Cutoff, Freshness, FreshSource, fresh_by_verdict
from .ground_truth import Availability, Coverage, Gate, GroundTruthReport
from .runner import CaseStatus, Group, RunReport, Score, Tally
from .saturation import LoopReport, Round
from .schema import EvidenceStatus, Readiness, TaskSpec
from .suite import Split, Suite, kpi_coverage
from .surface import CellStatus, CoverageMatrix
from .verify import automatic_share

READINESS_BLURB = {
    Readiness.READY: "enough detail to start building the eval",
    Readiness.NEEDS_INPUT: "usable, but answers below will sharpen it",
    Readiness.INSUFFICIENT: "blocked - the questions marked **blocking** must be answered first",
}

VERDICT_BLURB = {
    Availability.LABELLED_DATA: "public data with labels exists - the eval can score against it",
    Availability.PUBLISHED_BASELINES: "published numbers to compare against, but no labels - we still label our own cases",
    Availability.REFERENCE_ONLY: "context only - nothing found that can score or benchmark this",
    Availability.NONE_FOUND: "nothing usable found online - the ground truth has to be built",
}

COVERAGE_MARK = {
    Coverage.LABELLED: "labelled data",
    Coverage.BASELINE: "baseline only",
    Coverage.REFERENCE: "reference only",
    Coverage.NONE: "nothing found",
}

REACH_MARK = {
    Reachability.OK: "reachable",
    Reachability.UNREACHABLE: "could not read",
    Reachability.BLOCKED: "blocked",
    Reachability.NOT_CHECKED: "not opened",
}

USABILITY_MARK = {
    Usability.GROUND_TRUTH: "labelled data behind it",
    Usability.BASELINES: "numbers to compare against",
    Usability.BACKGROUND: "background only",
    Usability.UNUSABLE: "not usable",
}

# A description or a quote can run long; the JSON holds it in full, the
# document shows enough to recognise it.
PREVIEW_CHARS = 300

STATUS_MARK = {
    EvidenceStatus.PROVIDED: "have it",
    EvidenceStatus.MENTIONED_NOT_PROVIDED: "mentioned, not provided",
    EvidenceStatus.ABSENT: "does not exist",
}


def _or_unknown(value: Optional[str]) -> str:
    return value if value else "_not stated_"


def _bullets(items: List[str], empty: str) -> List[str]:
    if not items:
        return [f"_{empty}_"]
    return [f"- {item}" for item in items]


def render_questions(spec: TaskSpec) -> str:
    """Just the open questions - what the CLI shows when you only want the asks."""
    if not spec.open_questions:
        return "No open questions."
    lines = []
    for i, q in enumerate(spec.open_questions, 1):
        flag = " **[blocking]**" if q.blocking else ""
        lines.append(f"{i}. {q.question}{flag}")
        lines.append(f"   - _{q.why}_ (`{q.field}`)")
    return "\n".join(lines)


def render_markdown(spec: TaskSpec) -> str:
    blocking = sum(1 for q in spec.open_questions if q.blocking)
    lines: List[str] = [
        f"# {spec.title}",
        "",
        f"**Readiness:** `{spec.readiness.value}` - {READINESS_BLURB[spec.readiness]}  ",
        f"**Confidence:** subject {spec.confidence.subject:.2f} · "
        f"KPIs {spec.confidence.kpis:.2f} · "
        f"evidence {spec.confidence.evidence:.2f} · "
        f"overall {spec.confidence.overall:.2f}  ",
        f"**Open questions:** {len(spec.open_questions)} ({blocking} blocking)",
        "",
        spec.summary,
        "",
        "## 1. What we're testing",
        "",
        f"**{spec.subject.name}** — `{spec.subject.kind.value}`",
        "",
        spec.subject.description,
        "",
        f"- **Interface:** {_or_unknown(spec.subject.interface)}",
        f"- **Inputs:** {_or_unknown(spec.subject.inputs)}",
        f"- **Outputs:** {_or_unknown(spec.subject.outputs)}",
        "- **Evaluation type:** "
        + (
            ", ".join(t.value for t in spec.eval_types)
            if spec.eval_types
            else "_not stated_"
        ),
        "",
        "**In scope**",
        "",
        *_bullets(spec.subject.in_scope, "not stated"),
        "",
        "**Out of scope**",
        "",
        *_bullets(spec.subject.out_of_scope, "not stated"),
        "",
        "## 2. How success is judged",
        "",
    ]

    if not spec.kpis:
        lines += ["_No KPIs identified - see open questions._", ""]
    else:
        lines += [
            "| KPI | Type | Priority | Measured by | Unit | Target | Baseline | Source |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for kpi in spec.kpis:
            lines.append(
                f"| {kpi.name} | {kpi.kind.value} | {kpi.priority.value} | "
                f"{kpi.measurement.value} | {kpi.unit or '—'} | {kpi.target or '—'} | "
                f"{kpi.baseline or '—'} | {kpi.source.value} |"
            )
        lines += ["", "**Definitions**", ""]
        for kpi in spec.kpis:
            lines.append(f"- **{kpi.name}** ({kpi.direction.value}) — {kpi.definition}")
        lines.append("")

    lines += ["## 3. Evidence", ""]
    if not spec.evidence:
        lines += ["_Nothing supplied - see open questions._", ""]
    else:
        lines += [
            "| Kind | Reference | Location | Status |",
            "| --- | --- | --- | --- |",
        ]
        for item in spec.evidence:
            lines.append(
                f"| {item.kind.value} | {item.reference} | {item.location or '—'} | "
                f"{STATUS_MARK[item.status]} |"
            )
        lines.append("")

    gt = spec.ground_truth
    lines += [
        "**Ground truth:** "
        + (
            f"available — {_or_unknown(gt.description)}"
            + (f" (n={gt.sample_size})" if gt.sample_size else "")
            if gt.available
            else "none identified"
        ),
        "",
        "## 4. Constraints",
        "",
        *_bullets(spec.constraints, "none stated"),
        "",
        "## 5. Assumptions made by the classifier",
        "",
        *_bullets(spec.assumptions, "none"),
        "",
        "## 6. Open questions",
        "",
        render_questions(spec),
        "",
    ]
    return "\n".join(lines)


def render_ground_truth(report: GroundTruthReport) -> str:
    """The ground-truth report as a document, in the same voice as the task spec."""
    lines: List[str] = [
        f"# Ground truth for {report.subject}",
        "",
        f"**Verdict:** `{report.verdict.value}` - {VERDICT_BLURB[report.verdict]}  ",
        f"**Sources found:** {len(report.sources)}  ",
        "**We already hold our own ground truth**, so public sources are for comparison, not scoring.  "
        if report.internal_ground_truth
        else "**No ground truth of our own**, so anything usable below saves labelling work.  ",
        "",
        report.recommendation or "_No recommendation returned._",
        "",
        "## Coverage by KPI",
        "",
    ]

    if not report.kpi_coverage:
        lines += ["_No KPIs to cover._", ""]
    else:
        lines += ["| KPI | What exists | Sources |", "| --- | --- | --- |"]
        for item in report.kpi_coverage:
            lines.append(
                f"| {item.kpi} | {COVERAGE_MARK[item.coverage]} | "
                f"{', '.join(item.sources) if item.sources else '—'} |"
            )
        lines.append("")

    lines += ["## Sources", ""]
    if not report.sources:
        lines += ["_Nothing found online for this task._", ""]
    else:
        for source in report.sources:
            lines += [
                f"### {source.name}",
                "",
                f"`{source.kind.value}` · fit {source.fit.value} · access {source.access.value}"
                + (f" · published {source.released}" if source.released else "")
                + (f" · {source.licence}" if source.licence else "")
                + (f" · {source.publisher}" if source.publisher else "")
                + (" · **self-reported**" if source.self_reported else ""),
                "",
                source.description,
                "",
                f"<{source.url}>",
                "",
            ]
            if source.covers_kpis:
                lines += [f"Covers: {', '.join(source.covers_kpis)}", ""]
            if source.baselines:
                lines += [
                    "| Metric | Value | System | As of |",
                    "| --- | --- | --- | --- |",
                ]
                for value in source.baselines:
                    lines.append(
                        f"| {value.metric} | {value.value} | {value.system or '—'} | "
                        f"{value.as_of or '—'} |"
                    )
                lines.append("")
            if source.self_reported:
                lines += [
                    "Published by a system under test, so it is context and not "
                    "ground truth: nothing here can score the system that wrote it.",
                    "",
                ]
            if source.caveats:
                lines += [f"Caveat: {source.caveats}", ""]

    lines += [
        "## Notes",
        "",
        *_bullets(report.notes, "none"),
        "",
        "## Searches run",
        "",
        *_bullets(report.searches, "none recorded"),
        "",
    ]
    return "\n".join(lines)


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= PREVIEW_CHARS:
        return flat
    return flat[:PREVIEW_CHARS].rstrip() + " […]"


def render_analysis(analysis: AnalysisReport) -> str:
    """The per-source analysis as a document. Nothing here was downloaded."""
    ready = [r for r in analysis.resources if r.plan and not r.plan.blockers]
    lines: List[str] = [
        f"# Source analysis for {analysis.subject}",
        "",
        f"**Sources looked at:** {len(analysis.resources)}  ",
        f"**With labelled data behind them:** {len(analysis.with_ground_truth)}  ",
        f"**Ready to fetch as they are:** {len(ready)}",
        "",
        "Nothing below was downloaded. Each entry is the link, what is behind it, "
        "and what a later step would have to fetch.",
        "",
        "## What to fetch",
        "",
    ]

    if not analysis.plans:
        lines += ["_No source offered anything worth fetching._", ""]
    else:
        lines += [
            "| What | From | Licence | Blockers |",
            "| --- | --- | --- | --- |",
        ]
        for plan in analysis.plans:
            lines.append(
                f"| {plan.what} | <{plan.url}> | {plan.licence or '—'} | "
                f"{plan.blockers or '—'} |"
            )
        lines.append("")

    lines += ["## Source by source", ""]
    for item in analysis.resources:
        lines += [
            f"### {item.source}",
            "",
            f"`{item.kind.value}` · {REACH_MARK[item.reachability]} · "
            f"{USABILITY_MARK[item.usability]}"
            + (f" · effort {item.effort.value}" if item.effort else "")
            + (" · **self-reported**" if item.self_reported else ""),
            "",
            f"<{item.url}>",
            "",
        ]
        if item.summary:
            lines += [item.summary, ""]
        if item.detail:
            lines += [f"_{item.detail}_", ""]
        if item.covers_kpis:
            lines += [f"Covers: {', '.join(item.covers_kpis)}", ""]

        if item.dataset:
            data = item.dataset
            splits = ", ".join(
                f"{s.name} ({s.rows:,} rows)" if s.rows else s.name for s in data.splits
            )
            lines += [
                "| | |",
                "| --- | --- |",
                f"| Columns | {', '.join(data.columns) or '—'} |",
                f"| Splits | {splits or '—'} |",
                f"| Licence | {data.licence or 'not stated'} |",
                f"| Gated | {'yes' if data.gated else 'no'} |",
                f"| Downloads | {f'{data.downloads:,}' if data.downloads is not None else '—'} |",
                f"| Last modified | {data.last_modified or '—'} |",
                "",
            ]

        if item.plan:
            lines += [f"**Fetch later:** {item.plan.what}", ""]
            if item.plan.blockers:
                lines += [f"**In the way:** {item.plan.blockers}", ""]

        if item.baselines:
            lines += [
                "| Metric | Value | System | As of |",
                "| --- | --- | --- | --- |",
            ]
            for value in item.baselines:
                lines.append(
                    f"| {value.metric} | {value.value} | {value.system or '—'} | "
                    f"{value.as_of or '—'} |"
                )
            lines += ["", "**Quoted from**", ""]
            for value in item.baselines:
                lines.append(f'- {value.metric}: "{_preview(value.quote)}"')
            lines.append("")

        if item.discarded:
            lines += [
                f"_{item.discarded} claim(s) discarded: the page did not bear them out._",
                "",
            ]
        if item.caveats:
            lines += [*_bullets(item.caveats, "none"), ""]

    lines += ["## Notes", "", *_bullets(analysis.notes, "none"), ""]
    return "\n".join(lines)


RUN_MODE_BLURB = {
    RunMode.SUBPROCESS: "started as a process; the exit code and the files it leaves are the result",
    RunMode.HTTP: "started over HTTP and polled until it reaches a terminal state",
    RunMode.PYTHON: "called as a function in-process",
    RunMode.TRANSCRIPT: "not started at all - runs that already happened are scored instead",
    RunMode.UNKNOWN: "unknown - nothing in the spec says how to start one",
}

CELL_MARK = {
    CellStatus.COVERED: "",
    CellStatus.THIN: " (thin)",
    CellStatus.EMPTY: " (empty)",
}

SOURCE_BLURB = {
    CaseSource.HARVESTED: "from runs and incidents you already have - the strongest cases here",
    CaseSource.ADAPTED: "from a public suite; real tasks, but possibly in the agent's training data",
    CaseSource.SYNTHESISED: "generated from the spec; cheap, and partly a measurement of the generator",
}


def render_profile(agent_profile: AgentProfile, decision: Optional[Gate] = None) -> str:
    """What we think we can drive, and whether that is enough to author against."""
    budget = agent_profile.budget
    lines: List[str] = [
        f"# Running {agent_profile.subject}",
        "",
        f"**Run mode:** `{agent_profile.run_mode.value}` - {RUN_MODE_BLURB[agent_profile.run_mode]}  ",
        f"**Isolation:** `{agent_profile.isolation.value}`  ",
        f"**Samples per case:** {agent_profile.samples}"
        + (
            " (stochastic, so one run is a number with no error bar)"
            if agent_profile.stochastic
            else " (deterministic)"
        ),
        "",
        "| | |",
        "| --- | --- |",
        f"| Entry point | {_or_unknown(agent_profile.entrypoint)} |",
        f"| A run is over when | {agent_profile.episode_end} |",
        f"| Step ceiling | {budget.max_steps if budget.max_steps is not None else '—'} |",
        f"| Time ceiling | {f'{budget.max_seconds}s' if budget.max_seconds is not None else '—'} |",
        f"| Token ceiling | {budget.max_tokens if budget.max_tokens is not None else '—'} |",
        f"| Spend ceiling | {f'${budget.max_usd:.2f}' if budget.max_usd is not None else '—'} |",
        "",
        "## Tools",
        "",
    ]

    if not agent_profile.tools:
        lines += [
            "_None read off the request. Every trajectory check below is therefore a guess._",
            "",
        ]
    else:
        lines += [
            "| Tool | Effect | Can be stubbed | Read off |",
            "| --- | --- | --- | --- |",
        ]
        for tool in agent_profile.tools:
            lines.append(
                f"| {tool.name} | {tool.effect.value} | {'yes' if tool.stubbable else 'no'} | "
                f"{f'“{tool.evidence}”' if tool.evidence else '—'} |"
            )
        lines.append("")

    lines += ["## Assumptions", "", *_bullets(agent_profile.assumptions, "none"), ""]

    if decision is not None:
        lines += [
            "## Can a suite be authored",
            "",
            ("**Yes.** " if decision.open else "**Not yet.** ") + decision.reason,
            "",
        ]
        if decision.blocking:
            for i, question in enumerate(decision.blocking, 1):
                lines += [
                    f"{i}. {question.question}",
                    f"   - _{question.why}_ (`{question.field}`)",
                ]
            lines.append("")
    return "\n".join(lines)


def _grid(matrix: CoverageMatrix) -> List[str]:
    header = "| Behaviour | " + " | ".join(t.value for t in matrix.eval_types) + " |"
    rule = "| --- |" + " --- |" * len(matrix.eval_types)
    rows = [header, rule]
    for behaviour in matrix.behaviours:
        cells = []
        for eval_type in matrix.eval_types:
            cell = matrix.cell(behaviour, eval_type)
            if cell is None:
                cells.append("—")
                continue
            cells.append(f"{len(cell.case_ids)}/{cell.target}{CELL_MARK[cell.status]}")
        rows.append(f"| {behaviour} | " + " | ".join(cells) + " |")
    return rows


def render_suite(suite: Suite) -> str:
    """The suite as a document: what it covers, what it cannot, and what it costs."""
    held = suite.cases_in(Split.HELD_OUT)
    dev = suite.cases_in(Split.DEV)
    checks = suite.verifiers
    pending = needs_fixture(suite.cases)

    lines: List[str] = [
        f"# Eval suite for {suite.subject}",
        "",
        f"**Cases:** {len(suite.cases)} ({len(held)} held out, {len(dev)} dev)  ",
        f"**Runs per pass:** {len(suite.cases)} x {suite.samples} = {suite.total_runs}  ",
        f"**Checks:** {len(checks)}, {automatic_share(checks):.0%} of them settled by machine  ",
        f"**Digest:** `{suite.digest}` (spec `{suite.spec_digest}`)",
        "",
        "Nothing here has been run. The fixtures are specified, not built: this document "
        "says what the suite is, and what has to exist before it can be executed.",
        "",
    ]

    if suite.warnings:
        lines += ["## Read this first", ""]
        lines += [f"- {warning}" for warning in suite.warnings]
        lines.append("")

    lines += [
        "## 1. How it runs",
        "",
        f"`{suite.profile.run_mode.value}` - {RUN_MODE_BLURB[suite.profile.run_mode]}",
        "",
        f"- **Entry point:** {_or_unknown(suite.profile.entrypoint)}",
        f"- **A run is over when:** {suite.profile.episode_end}",
        f"- **Isolation:** {suite.profile.isolation.value}",
        f"- **Ceilings:** {suite.profile.budget.max_steps} steps, {suite.profile.budget.max_seconds}s",
        "",
        "## 2. Coverage",
        "",
        *_grid(suite.coverage),
        "",
    ]
    if suite.coverage.riders:
        lines += [
            "Measured on every case rather than in a cell of its own:",
            "",
            *_bullets(suite.coverage.riders, "none"),
            "",
        ]
    lines += [*_bullets(suite.coverage.notes, "no notes"), ""]

    lines += [
        "## 3. Where the cases come from",
        "",
        "| Source | Cases | What that means |",
        "| --- | --- | --- |",
    ]
    for source in CaseSource:
        found = suite.by_source(source)
        if found:
            lines.append(f"| {source.value} | {len(found)} | {SOURCE_BLURB[source]} |")
    lines.append("")

    lines += ["## 4. What the checks measure", "", "| KPI | Checks |", "| --- | --- |"]
    for name, ids in kpi_coverage(suite).items():
        lines.append(
            f"| {name} | {len(ids) if ids else 'none - this suite does not measure it'} |"
        )
    lines.append("")

    kinds: dict = {}
    for check in checks:
        kinds[check.kind.value] = kinds.get(check.kind.value, 0) + 1
    lines += [
        "By kind: "
        + ", ".join(f"{kind} ({count})" for kind, count in sorted(kinds.items()))
        + ".",
        "",
    ]

    lines += ["## 5. Fixtures still to build", ""]
    if not pending:
        lines += ["_None - every case has its starting state._", ""]
    else:
        lines += ["| Case | What has to exist |", "| --- | --- |"]
        seen: set = set()
        for case in pending:
            key = case.fixture.reference
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| `{case.id}` | {_preview(case.fixture.reference)} |")
        lines += [
            "",
            f"_{len(pending)} case(s) across {len(seen)} distinct starting states._",
            "",
        ]

    lines += [
        "## 6. The cases",
        "",
        "| Case | Split | Family | Asks for | Checks |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in suite.cases:
        lines.append(
            f"| `{case.id}` | {suite.split_of(case.id).value} | {case.family} | "
            f"{_preview(case.instruction)} | {len(case.verifiers)} |"
        )
    lines += ["", "## Notes", "", *_bullets(suite.notes, "none"), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# A run
# --------------------------------------------------------------------------

FRESHNESS_BLURB = {
    Freshness.HELD_OUT: "answers never published - a score here cannot be recall",
    Freshness.ROLLING: "refreshed on a schedule - score the window after the cutoff",
    Freshness.POST_CUTOFF: "published after this model stopped learning",
    Freshness.PRE_CUTOFF: "inside the training window - partly a memory test",
    Freshness.UNKNOWN: "publication date not on file - check before quoting a number",
}

STATUS_WORD = {
    CaseStatus.PASSED: "passed",
    CaseStatus.FAILED: "failed",
    CaseStatus.BLOCKED: "nothing runnable",
    CaseStatus.ERROR: "never ran",
}


def _rate(value: Optional[float], empty: str = "nothing settled") -> str:
    return f"{value:.0%}" if value is not None else f"_{empty}_"


def _tally(tally: Tally) -> str:
    return (
        f"{tally.passed} passed, {tally.failed} failed, {tally.blocked} blocked"
        f" ({_rate(tally.pass_rate)})"
    )


def _score_line(score: Score) -> str:
    return (
        f"| {score.label} | {score.cases} | {score.runs} | {score.passed} | {score.failed} | "
        f"{score.blocked} | {score.errors} | {_rate(score.pass_rate)} |"
    )


def _groups(groups: List[Group], first: str) -> List[str]:
    rows = [f"| {first} | Cases | Checks |", "| --- | --- | --- |"]
    for group in groups:
        rows.append(f"| {group.label} | {group.cases} | {_tally(group.checks)} |")
    return rows


def render_run(report: RunReport) -> str:
    """A run as a document. The blocked count is never far from a pass rate."""
    lines: List[str] = [
        f"# Run {report.run_id}",
        "",
        f"**Subject:** {report.subject} — suite `{report.suite_digest}`, spec `{report.spec_digest}`  ",
        f"**Model under test:** `{report.model}`"
        + (
            f" — judged by `{report.judge_model}`"
            if report.judge_model
            else " — no judge ran"
        )
        + ("  \n**Mock run: no provider was called.**  " if report.mock else "  "),
        f"**Held out:** {_rate(report.headline.pass_rate)} of {report.headline.graded_runs} graded run(s)  ",
        f"**Checks:** {_tally(report.overall.checks)}  ",
        f"**Cost:** {report.overall.runs} run(s), {report.usage.tokens:,} tokens, "
        + (
            f"${report.usage.usd:.4f}"
            if report.usage.usd is not None
            else "spend not priced"
        )
        + f", {report.seconds:.0f}s wall clock",
        "",
    ]

    if report.warnings:
        lines += ["## Read this first", ""]
        lines += [f"- {warning}" for warning in report.warnings]
        lines.append("")

    lines += [
        "## 1. The numbers",
        "",
        "A run passes a case only when every check that ran on it passed. Blocked "
        "checks are not counted either way — they are listed in full below.",
        "",
        "| Split | Cases | Runs | Passed | Failed | Blocked | Errors | Pass rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        _score_line(report.headline),
        _score_line(report.dev),
        _score_line(report.overall),
        "",
        "## 2. By KPI",
        "",
        "A KPI with nothing settled is a KPI this run did not measure, whatever the "
        "case count says.",
        "",
        *_groups(report.by_kpi, "KPI"),
        "",
        "## 3. By check kind",
        "",
        *_groups(report.by_kind, "Kind"),
        "",
        "## 4. By behaviour",
        "",
        *_groups(report.by_behaviour, "Behaviour"),
        "",
    ]

    blocked = [(case, block) for case in report.cases for block in case.blocked]
    if blocked:
        reasons: dict = {}
        for _, block in blocked:
            reasons.setdefault(block.reason, 0)
            reasons[block.reason] += 1
        lines += [
            "## 5. What could not be checked",
            "",
            "| Checks | Why not |",
            "| --- | --- |",
        ]
        for reason, count in sorted(reasons.items(), key=lambda pair: -pair[1]):
            lines.append(f"| {count} | {reason} |")
        lines.append("")

    if report.contamination:
        contamination = report.contamination
        lines += [
            "## 6. Contamination",
            "",
            f"**{contamination.verdict.value.replace('_', ' ')}** — "
            f"{FRESHNESS_BLURB[contamination.verdict]}  ",
            f"Cutoff for `{contamination.model}`: "
            + (str(contamination.cutoff) if contamination.cutoff else "_not on file_")
            + "  ",
            f"{contamination.clean_cases} case(s) the model cannot have seen, "
            f"{contamination.unknown_cases} unknown, {contamination.at_risk_cases} at risk",
            "",
            "| Origin | Cases | Verdict | Why |",
            "| --- | --- | --- | --- |",
        ]
        for origin in contamination.by_origin:
            lines.append(
                f"| {origin.source} | {origin.cases} | {origin.verdict.value} | {origin.why} |"
            )
        lines.append("")

    failures = [c for c in report.cases if c.status is not CaseStatus.PASSED]
    lines += [
        "## 7. Cases that did not pass",
        "",
        f"{len(failures)} of {len(report.cases)}. Every verdict carries the span it was read off; "
        "the full spans are in the run JSON and the stored traces.",
        "",
    ]
    for case in failures[:40]:
        lines.append(
            f"### `{case.case_id}` — {STATUS_WORD[case.status]} "
            f"({case.passed}/{case.samples} sample(s) passed)"
        )
        lines += [
            "",
            f"- **{case.behaviour}** / {case.family} / {case.split.value} / {case.source}",
            f"- {case.instruction}",
        ]
        for run in case.runs:
            for outcome in run.result.outcomes:
                if outcome.passed:
                    continue
                lines.append(
                    f"- FAILED `{outcome.verifier}` — {_preview(outcome.evidence or outcome.note or 'no evidence')}"
                )
        for block in case.blocked:
            lines.append(f"- blocked `{block.verifier}` — {block.reason}")
        lines.append("")
    if len(failures) > 40:
        lines.append(f"_{len(failures) - 40} more in the run JSON._")
        lines.append("")

    lines += ["## 8. Notes", "", *_bullets(report.notes, "none"), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# A comparison board
# --------------------------------------------------------------------------


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _usd(value: Optional[float]) -> str:
    return f"${value:,.4f}" if value is not None else "—"


def _secs(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 1000:.0f}ms" if value < 1 else f"{value:.2f}s"


def _row(row: Row) -> str:
    return (
        f"| {row.label} | {row.configuration or '—'} | {_pct(row.accuracy)} | "
        f"{_pct(row.recall_at_1)} | {_pct(row.recall_at_5)} | {_secs(row.seconds_p50)} | "
        f"{_secs(row.seconds_p95)} | {row.errors} | {_usd(row.usd)} | "
        f"{_usd(row.usd_per_1k_correct)} |"
    )


def render_board(board: Board) -> str:
    """A board as a document: the rows, then everything that qualifies them."""
    leader = board.leader()
    baseline = board.baseline

    lines: List[str] = [
        f"# {board.task or 'Comparison'}: {board.dataset}",
        "",
        f"**Board:** `{board.board_id}`  ",
        f"**Items:** {board.items} from `{board.dataset_digest}`"
        + (f" ({board.dataset_source})" if board.dataset_source else "")
        + "  ",
        f"**Held constant:** model `{board.model or 'none'}`, judge `{board.judge_model or 'none'}`  ",
        "**Model-only baseline:** "
        + (
            _pct(baseline.accuracy)
            if baseline and baseline.accuracy is not None
            else "_not measured_"
        )
        + "  ",
        "**Leader:** "
        + (
            f"{leader.label} at {_pct(leader.accuracy)}"
            if leader
            else "_nothing scored_"
        ),
        "",
    ]

    if board.warnings:
        lines += ["## Read this first", ""]
        lines += [f"- {warning}" for warning in board.warnings]
        lines.append("")

    lines += [
        "## The board",
        "",
        "Rows are alphabetical. No column decides the order.",
        "",
        "| System | Endpoint & configuration | Accuracy | AR@1 | AR@5 | p50 | p95 | Errors | Total $ | $ / 1k correct |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in board.rows:
        if row.skipped is None:
            lines.append(_row(row))
    lines.append("")

    skipped = [row for row in board.rows if row.skipped]
    if skipped:
        lines += ["### Not run", "", "| System | Why not |", "| --- | --- |"]
        lines += [f"| {row.label} | {row.skipped} |" for row in skipped]
        lines.append("")

    priced = [row for row in board.rows if row.price_note or row.docs_url]
    if priced:
        lines += [
            "### Prices, as published",
            "",
            "| System | List price | Source |",
            "| --- | --- | --- |",
        ]
        for row in priced:
            source = (
                f"<{row.price_source}>"
                if row.price_source
                else (f"<{row.docs_url}>" if row.docs_url else "—")
            )
            lines.append(f"| {row.label} | {row.price_note or '—'} | {source} |")
        lines.append("")

    lines += [
        "## How it was measured",
        "",
        *_bullets(board.methodology, "not recorded"),
        "",
    ]

    if board.contamination:
        contamination = board.contamination
        lines += [
            "## Contamination",
            "",
            f"**{contamination.verdict.value.replace('_', ' ')}** — "
            f"{FRESHNESS_BLURB[contamination.verdict]}  ",
            f"{contamination.clean_cases} item(s) the model cannot have seen, "
            f"{contamination.unknown_cases} unknown, {contamination.at_risk_cases} at risk"
            + (f", cutoff {contamination.cutoff}" if contamination.cutoff else "")
            + "",
            "",
        ]
        if contamination.by_origin:
            lines += ["| Items | Verdict | Why |", "| --- | --- | --- |"]
            for origin in contamination.by_origin:
                lines.append(
                    f"| {origin.cases} | {origin.verdict.value} | {origin.why} |"
                )
            lines.append("")

    lines += [
        "## Per item",
        "",
        "The first few, with the span each verdict was read off.",
        "",
    ]
    for row in board.rows:
        if row.skipped or not row.verdicts:
            continue
        lines += [
            f"### {row.label}",
            "",
            "| Item | Right | Rank | Evidence |",
            "| --- | --- | --- | --- |",
        ]
        for verdict in row.verdicts[:10]:
            mark = "error" if verdict.error else ("yes" if verdict.correct else "no")
            evidence = verdict.error or verdict.evidence or verdict.note or "—"
            lines.append(
                f"| {verdict.item_id} | {mark} | {verdict.rank or '—'} | {_preview(evidence)} |"
            )
        if len(row.verdicts) > 10:
            lines.append(f"| … | | | {len(row.verdicts) - 10} more in the board JSON |")
        lines.append("")
    return "\n".join(lines)


def render_fresh(cutoff: Optional[Cutoff] = None) -> str:
    """The contamination-resistant catalogue, judged against one model's cutoff."""
    lines = [
        "# Ground truth the model cannot already have seen",
        "",
    ]
    if cutoff:
        lines += [
            f"Judged against `{cutoff.model}`, cutoff **{cutoff.cutoff}** "
            f"([source]({cutoff.source}), read {cutoff.read_on}).",
            "",
        ]
    else:
        lines += [
            "No model cutoff given, so nothing is judged post-cutoff here — only the "
            "sources that are clean whatever the cutoff is.",
            "",
        ]

    for source, verdict in fresh_by_verdict(cutoff):
        lines += _fresh_entry(source, verdict)
    return "\n".join(lines)


def _fresh_entry(source: FreshSource, verdict: Freshness) -> List[str]:
    return [
        f"## {source.name} — {verdict.value.replace('_', ' ')}",
        "",
        f"{source.measures}",
        "",
        f"- **Metric:** {source.metric}",
        f"- **Released:** {source.released}"
        + (f"; {source.refresh}" if source.refresh else ""),
        f"- **Size:** {source.size or 'not stated'}",
        f"- **Licence:** {source.licence or 'not stated - check the page'}",
        f"- **Keeping it clean:** {source.staying_clean}",
        f"- **Caveat:** {source.caveats}",
        f"- {source.url}",
        "",
    ]


def render_case(case: Case) -> str:
    """One case in full - what a person reads when they want to argue with it."""
    lines = [
        f"### `{case.id}` - {case.family}",
        "",
        f"`{case.source.value}` · {', '.join(t.value for t in case.eval_types)} · covers _{case.behaviour}_",
        "",
        case.instruction,
        "",
        f"**Starting state:** {case.fixture.reference}"
        + ("" if case.fixture.materialised else " _(not built yet)_"),
        "",
    ]
    if case.stubs:
        lines += ["**Scripted tool responses**", ""]
        for stub in case.stubs:
            lines.append(
                f"- `{stub.tool}` returns {stub.behaviour.value}: {stub.payload}"
            )
        lines.append("")
    lines += ["**Checks**", ""]
    for verifier in case.verifiers:
        mark = " **[fatal]**" if verifier.fatal else ""
        lines.append(f"- `{verifier.kind.value}`{mark} — {verifier.description}")
        for line in verifier.rubric:
            lines.append(f"  - _{line}_")
    lines.append("")
    if case.notes:
        lines += [*_bullets(case.notes, "none"), ""]
    return "\n".join(lines)


def render_round(this_round: Round) -> str:
    """One round of the hardening loop: the board it ran, and what it rewrote."""
    lines: List[str] = [
        f"## Round {this_round.index}",
        "",
        f"**Board:** `{this_round.board_id}`  ",
        f"**Items:** {this_round.items} from `{this_round.dataset_digest}`  ",
        "**Model-only baseline:** " + _pct(this_round.baseline),
        "",
        "| System | Accuracy | Correct | Scored |",
        "| --- | --- | --- | --- |",
    ]
    for row in this_round.rows:
        lines.append(
            f"| {row.label} | {_pct(row.accuracy)} | {row.correct} | {row.scored} |"
        )
    lines.append("")

    ceiling = this_round.saturated_rows
    lines.append(
        "**At 100%:** "
        + (", ".join(ceiling) if ceiling else "_nothing - the board separates_")
    )
    everywhere = this_round.saturated_items
    lines.append("")
    lines.append(
        "**Items every scored system got right:** "
        + (", ".join(f"`{item}`" for item in everywhere) if everywhere else "_none_")
    )
    lines.append("")

    if this_round.rewrites:
        lines += ["### Rewritten", ""]
        for rewrite in this_round.rewrites:
            mark = "accepted" if rewrite.accepted else "kept as it was"
            lines.append(f"**`{rewrite.item_id}`** — {mark}: {rewrite.why}")
            lines.append("")
            lines.append(
                f"- was: {rewrite.was_query} → {', '.join(rewrite.was_answers) or '—'}"
            )
            if rewrite.accepted:
                lines.append(
                    f"- now: {rewrite.now_query} → {', '.join(rewrite.now_answers)}"
                )
                if rewrite.source:
                    lines.append(f"- verified against: <{rewrite.source}>")
                if rewrite.evidence:
                    lines.append(f"- evidence: _{rewrite.evidence}_")
            lines.append("")
    return "\n".join(lines)


def render_loop(report: LoopReport) -> str:
    """The loop as a document: every board it ran, in the order it ran them."""
    lines: List[str] = [
        f"# Hardening {report.dataset} against {report.cohort}",
        "",
        f"**Rounds:** {len(report.rounds)}  ",
        f"**Ceiling:** {report.threshold:.0%}  ",
        f"**Items rewritten:** {report.rewritten}  ",
        f"**Stopped because:** {report.stopped}",
        "",
    ]
    for this_round in report.rounds:
        lines.append(render_round(this_round))

    first = report.rounds[0] if report.rounds else None
    last = report.rounds[-1] if report.rounds else None
    if first is not None and last is not None and first is not last:
        lines += [
            "## What moved",
            "",
            "| System | Round 1 | " + f"Round {last.index} |",
            "| --- | --- | --- |",
        ]
        before = {row.label: row.accuracy for row in first.rows}
        for row in last.rows:
            lines.append(
                f"| {row.label} | {_pct(before.get(row.label))} | {_pct(row.accuracy)} |"
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "render_analysis",
    "render_board",
    "render_case",
    "render_fresh",
    "render_ground_truth",
    "render_loop",
    "render_markdown",
    "render_profile",
    "render_questions",
    "render_round",
    "render_run",
    "render_suite",
]

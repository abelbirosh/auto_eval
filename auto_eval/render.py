"""Render a TaskSpec, and a ground-truth report, as human-readable documents."""

from __future__ import annotations

from typing import List, Optional

from .analysis import AnalysisReport, Reachability, Usability
from .ground_truth import Availability, Coverage, GroundTruthReport
from .schema import EvidenceStatus, Readiness, TaskSpec

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
        f"- **Evaluation type:** "
        + (", ".join(t.value for t in spec.eval_types) if spec.eval_types else "_not stated_"),
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
                + (f" · {source.licence}" if source.licence else "")
                + (f" · {source.publisher}" if source.publisher else ""),
                "",
                source.description,
                "",
                f"<{source.url}>",
                "",
            ]
            if source.covers_kpis:
                lines += [f"Covers: {', '.join(source.covers_kpis)}", ""]
            if source.baselines:
                lines += ["| Metric | Value | System | As of |", "| --- | --- | --- | --- |"]
                for value in source.baselines:
                    lines.append(
                        f"| {value.metric} | {value.value} | {value.system or '—'} | "
                        f"{value.as_of or '—'} |"
                    )
                lines.append("")
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
            + (f" · effort {item.effort.value}" if item.effort else ""),
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
            lines += ["| Metric | Value | System | As of |", "| --- | --- | --- | --- |"]
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


__all__ = [
    "render_analysis",
    "render_ground_truth",
    "render_markdown",
    "render_questions",
]

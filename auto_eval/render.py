"""Render a TaskSpec, and a ground-truth report, as human-readable documents."""

from __future__ import annotations

from typing import List, Optional

from .extraction import GroundTruthSet, Outcome
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

OUTCOME_MARK = {
    Outcome.EXTRACTED: "extracted",
    Outcome.NOTHING_FOUND: "nothing in it",
    Outcome.UNREACHABLE: "could not read",
    Outcome.UNUSABLE: "not usable",
    Outcome.SKIPPED: "skipped",
}

# Ground truth can be long - a dataset row carries a whole passage. The files
# hold it in full; the document shows enough to recognise it.
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


def render_ground_truth_set(found: GroundTruthSet) -> str:
    """The extracted ground truth as a document. The data itself is in the files."""
    lines: List[str] = [
        f"# Extracted ground truth for {found.subject}",
        "",
        f"**Labelled cases:** {len(found.examples)}  ",
        f"**Published baselines:** {len(found.baselines)}  ",
        f"**Sources tried:** {len(found.outcomes)}",
        "",
    ]

    if not found.usable:
        lines += [
            "Nothing survived extraction. What each source did is below - that is the "
            "answer to what has to be labelled by hand.",
            "",
        ]

    lines += ["## Labelled cases", ""]
    if not found.examples:
        lines += ["_None extracted._", ""]
    else:
        by_source: dict = {}
        for example in found.examples:
            by_source.setdefault(example.source, []).append(example)
        for source, items in by_source.items():
            first = items[0]
            split = f", {first.split} split" if first.split else ""
            lines += [
                f"**{source}** — {len(items)} case(s), {first.origin.value}{split}",
                "",
                "| Input | Expected | KPI |",
                "| --- | --- | --- |",
            ]
            for example in items:
                lines.append(
                    f"| {_preview(example.input)} | {_preview(example.expected)} | "
                    f"{example.kpi or '—'} |"
                )
            lines.append("")

    lines += ["## Published baselines", ""]
    if not found.baselines:
        lines += ["_None found._", ""]
    else:
        lines += ["| Metric | Value | System | As of | Source |", "| --- | --- | --- | --- | --- |"]
        for baseline in found.baselines:
            lines.append(
                f"| {baseline.metric} | {baseline.value} | {baseline.system or '—'} | "
                f"{baseline.as_of or '—'} | {baseline.source} |"
            )
        lines += ["", "**Quoted from**", ""]
        for baseline in found.baselines:
            lines.append(f"- {baseline.metric}: \"{_preview(baseline.quote)}\" — <{baseline.url}>")
        lines.append("")

    lines += [
        "## What each source gave",
        "",
        "| Source | Result | Cases | Numbers | Discarded | Detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in found.outcomes:
        lines.append(
            f"| {item.source} | {OUTCOME_MARK[item.outcome]} | {item.examples} | "
            f"{item.baselines} | {item.discarded} | {item.detail} |"
        )
    lines += ["", "## Notes", "", *_bullets(found.notes, "none"), ""]
    return "\n".join(lines)


__all__ = [
    "render_ground_truth",
    "render_ground_truth_set",
    "render_markdown",
    "render_questions",
]

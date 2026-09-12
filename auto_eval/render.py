"""Render a TaskSpec as a human-readable task document."""

from __future__ import annotations

from typing import List, Optional

from .schema import EvidenceStatus, Readiness, TaskSpec

READINESS_BLURB = {
    Readiness.READY: "enough detail to start building the eval",
    Readiness.NEEDS_INPUT: "usable, but answers below will sharpen it",
    Readiness.INSUFFICIENT: "blocked - the questions marked **blocking** must be answered first",
}

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


__all__ = ["render_markdown", "render_questions"]

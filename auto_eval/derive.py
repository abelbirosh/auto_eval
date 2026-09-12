"""Work out what the spec can work out for itself.

A question the system can answer from what it already knows is a question worth
not asking. `derive` fills those in - a KPI when the request implies one, a
measurement method, which KPI is primary, what sort of evaluation this is - so
the pipeline keeps moving and the user is left only with the questions that
genuinely need them.

Nothing here invents a fact about the user's system. Every value written is
marked `inferred` and recorded in `assumptions`, and `auto_eval.gaps` still asks
about it as a normal question, so a derived spec is a starting point the user
can correct rather than a claim about what they said.
"""

from __future__ import annotations

from typing import List

from .schema import (
    KPI,
    Direction,
    EvalType,
    Measurement,
    MetricKind,
    Priority,
    Provenance,
    SubjectKind,
    TaskSpec,
)


def _kpi(
    name: str,
    definition: str,
    kind: MetricKind,
    measurement: Measurement,
    direction: Direction,
    unit: str,
) -> KPI:
    return KPI(
        name=name,
        kind=kind,
        definition=definition,
        measurement=measurement,
        direction=direction,
        unit=unit,
        priority=Priority.PRIMARY,
        source=Provenance.INFERRED,
    )


# What the user said they were worried about beats what the thing under test is:
# someone who asked for a safety evaluation wants a violation rate, whatever the
# subject happens to be. Read in this order.
BY_EVAL_TYPE = {
    EvalType.SAFETY: _kpi(
        "policy violation rate",
        "Share of outputs that breach the stated policy.",
        MetricKind.QUANTITATIVE,
        Measurement.LLM_JUDGE,
        Direction.MINIMIZE,
        "%",
    ),
    EvalType.COMPLIANCE: _kpi(
        "format adherence",
        "Share of outputs that satisfy the required format or schema.",
        MetricKind.QUANTITATIVE,
        Measurement.PROGRAMMATIC,
        Direction.MAXIMIZE,
        "%",
    ),
    EvalType.ROBUSTNESS: _kpi(
        "failure rate under messy input",
        "Share of adversarial or malformed inputs that produce a bad output.",
        MetricKind.QUANTITATIVE,
        Measurement.LLM_JUDGE,
        Direction.MINIMIZE,
        "%",
    ),
    EvalType.PERFORMANCE: _kpi(
        "p95 latency",
        "95th percentile wall-clock time for one run.",
        MetricKind.QUANTITATIVE,
        Measurement.TELEMETRY,
        Direction.MINIMIZE,
        "ms",
    ),
    EvalType.COST: _kpi(
        "cost per run",
        "Spend on one run, averaged over the sample.",
        MetricKind.QUANTITATIVE,
        Measurement.TELEMETRY,
        Direction.MINIMIZE,
        "USD",
    ),
}

# `capability` and `regression` say "does it do its job", which is a question
# about the subject, so those fall through to the table below.
EVAL_TYPE_ORDER = [
    EvalType.SAFETY,
    EvalType.COMPLIANCE,
    EvalType.ROBUSTNESS,
    EvalType.PERFORMANCE,
    EvalType.COST,
]

BY_SUBJECT_KIND = {
    SubjectKind.CLASSIFIER: _kpi(
        "accuracy",
        "Share of inputs given the correct label.",
        MetricKind.QUANTITATIVE,
        Measurement.PROGRAMMATIC,
        Direction.MAXIMIZE,
        "%",
    ),
    SubjectKind.RAG_PIPELINE: _kpi(
        "answer faithfulness",
        "Share of answers fully supported by the retrieved passages.",
        MetricKind.QUANTITATIVE,
        Measurement.LLM_JUDGE,
        Direction.MAXIMIZE,
        "%",
    ),
    SubjectKind.API_ENDPOINT: _kpi(
        "correct response rate",
        "Share of requests answered with the expected response.",
        MetricKind.QUANTITATIVE,
        Measurement.PROGRAMMATIC,
        Direction.MAXIMIZE,
        "%",
    ),
    SubjectKind.TOOL: _kpi(
        "correct response rate",
        "Share of calls that return the expected result.",
        MetricKind.QUANTITATIVE,
        Measurement.PROGRAMMATIC,
        Direction.MAXIMIZE,
        "%",
    ),
    SubjectKind.DATASET: _kpi(
        "label correctness",
        "Share of records whose label is right on inspection.",
        MetricKind.QUANTITATIVE,
        Measurement.HUMAN_REVIEW,
        Direction.MAXIMIZE,
        "%",
    ),
}

# Anything else - an agent, a prompt, a workflow, a model - is judged on whether
# the run came out the way the user wanted.
FALLBACK_KPI = _kpi(
    "task success rate",
    "Share of runs that produced the outcome the user asked for.",
    MetricKind.QUANTITATIVE,
    Measurement.LLM_JUDGE,
    Direction.MAXIMIZE,
    "%",
)

# A KPI with no stated grading method gets the one its kind implies: a number
# can be checked, a judged quality needs a judge.
MEASUREMENT_FOR_KIND = {
    MetricKind.QUANTITATIVE: Measurement.PROGRAMMATIC,
    MetricKind.QUALITATIVE: Measurement.LLM_JUDGE,
}

# With nothing said about why they are evaluating, the reading is "can it do the
# job at all".
DEFAULT_EVAL_TYPE = EvalType.CAPABILITY


def default_kpi(spec: TaskSpec) -> KPI:
    """The one KPI this request implies, when the user named none."""
    for eval_type in EVAL_TYPE_ORDER:
        if eval_type in spec.eval_types:
            return BY_EVAL_TYPE[eval_type].model_copy(deep=True)
    kind = BY_SUBJECT_KIND.get(spec.subject.kind)
    return (kind or FALLBACK_KPI).model_copy(deep=True)


def _note(assumptions: List[str], line: str) -> None:
    """Record an assumption once; `derive` is run again on a round trip."""
    if line not in assumptions:
        assumptions.append(line)


def derive(spec: TaskSpec) -> TaskSpec:
    """Return a copy of `spec` with the derivable holes filled in.

    Idempotent: running it on an already-derived spec changes nothing and adds
    no second copy of an assumption, so a spec can make the round trip through
    the API and come back without growing.
    """
    assumptions = list(spec.assumptions)
    kpis = [kpi.model_copy(deep=True) for kpi in spec.kpis]
    eval_types = list(spec.eval_types)

    if not kpis:
        proposed = default_kpi(spec)
        kpis = [proposed]
        _note(
            assumptions,
            f"No KPI was stated, so '{proposed.name}' was derived from the request "
            "and made primary. Replace it if the real bar is something else.",
        )

    for kpi in kpis:
        if kpi.measurement is Measurement.UNKNOWN:
            kpi.measurement = MEASUREMENT_FOR_KIND[kpi.kind]
            _note(
                assumptions,
                f"'{kpi.name}' had no stated grading method, so it is assumed to be "
                f"{kpi.measurement.value.replace('_', ' ')}.",
            )

    if not any(kpi.priority is Priority.PRIMARY for kpi in kpis):
        # Something the user stated outranks something we proposed; failing that,
        # the first one, which is the one the model led with.
        stated = [kpi for kpi in kpis if kpi.source is Provenance.STATED]
        chosen = (stated or kpis)[0]
        chosen.priority = Priority.PRIMARY
        _note(
            assumptions,
            f"No KPI was named as the one that decides the outcome, so "
            f"'{chosen.name}' was taken as primary.",
        )

    if not eval_types:
        eval_types = [DEFAULT_EVAL_TYPE]
        _note(
            assumptions,
            f"No kind of evaluation was stated, so this is read as a "
            f"{DEFAULT_EVAL_TYPE.value} evaluation.",
        )

    return spec.model_copy(
        update={"kpis": kpis, "eval_types": eval_types, "assumptions": assumptions}
    )


__all__ = [
    "BY_EVAL_TYPE",
    "BY_SUBJECT_KIND",
    "DEFAULT_EVAL_TYPE",
    "EVAL_TYPE_ORDER",
    "FALLBACK_KPI",
    "MEASUREMENT_FOR_KIND",
    "default_kpi",
    "derive",
]

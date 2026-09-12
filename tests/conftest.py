import pytest

from auto_eval.schema import (
    KPI,
    Confidence,
    Direction,
    EvalType,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    GroundTruth,
    Measurement,
    MetricKind,
    Priority,
    Provenance,
    Subject,
    SubjectKind,
    TaskSpec,
)


@pytest.fixture
def sparse_spec() -> TaskSpec:
    """What a one-line request looks like after extraction: almost nothing."""
    return TaskSpec(
        title="Support bot quality",
        summary="User wants their support bot evaluated.",
        subject=Subject(
            name="support bot",
            kind=SubjectKind.AGENT,
            description="Answers customer questions.",
        ),
    )


@pytest.fixture
def subjectless_spec() -> TaskSpec:
    """The one shape that still blocks: nothing names a thing to test."""
    return TaskSpec(
        title="Unclear request",
        summary="The request does not say what should be evaluated.",
        subject=Subject(name="", kind=SubjectKind.OTHER, description=""),
    )


@pytest.fixture
def full_spec() -> TaskSpec:
    """A spec with every gap filled - the only shape that should read as ready."""
    return TaskSpec(
        title="Invoice extractor accuracy",
        summary="Evaluate field-level accuracy of the invoice extraction pipeline.",
        subject=Subject(
            name="invoice extractor",
            kind=SubjectKind.WORKFLOW,
            description="Pulls structured fields out of PDF invoices.",
            interface="POST /v1/extract",
            inputs="PDF bytes",
            outputs="JSON with 8 fields",
            in_scope=["field accuracy"],
            out_of_scope=["OCR of handwriting"],
        ),
        eval_types=[EvalType.CAPABILITY],
        kpis=[
            KPI(
                name="field accuracy",
                kind=MetricKind.QUANTITATIVE,
                definition="Share of the 8 fields exactly matching the labelled value.",
                measurement=Measurement.PROGRAMMATIC,
                direction=Direction.MAXIMIZE,
                unit="%",
                target=">= 95%",
                baseline="91%",
                priority=Priority.PRIMARY,
                source=Provenance.STATED,
            )
        ],
        evidence=[
            EvidenceItem(
                kind=EvidenceKind.DOCUMENTATION,
                reference="extractor README",
                location="docs/extractor.md",
                status=EvidenceStatus.PROVIDED,
            ),
            EvidenceItem(
                kind=EvidenceKind.SUCCESSFUL_RUN,
                reference="30 accepted extractions",
                status=EvidenceStatus.PROVIDED,
            ),
            EvidenceItem(
                kind=EvidenceKind.FAILURE_CASE,
                reference="multi-page invoice misparse",
                status=EvidenceStatus.PROVIDED,
            ),
        ],
        ground_truth=GroundTruth(
            available=True, description="Hand-labelled invoices", sample_size="400"
        ),
        confidence=Confidence(subject=0.9, kpis=0.9, evidence=0.8, overall=0.87),
    )


@pytest.fixture
def agent_spec() -> TaskSpec:
    """An agent with tools, somewhere safe to run, and runs of its own to harvest."""
    return TaskSpec(
        title="Support agent",
        summary="Evaluate the support agent that issues refunds and looks up orders.",
        subject=Subject(
            name="support agent",
            kind=SubjectKind.AGENT,
            description="Answers customer email, looks orders up in the database, issues refunds.",
            interface="POST /v1/runs",
            inputs="a customer message",
            outputs="JSON with a reply and an action",
            in_scope=["refund requests", "order lookups"],
            out_of_scope=["phone calls"],
        ),
        eval_types=[EvalType.CAPABILITY, EvalType.SAFETY],
        kpis=[
            KPI(
                name="task success rate",
                kind=MetricKind.QUANTITATIVE,
                definition="Share of runs that resolved the customer's request correctly.",
                measurement=Measurement.LLM_JUDGE,
                direction=Direction.MAXIMIZE,
                unit="%",
                target=">= 90%",
                priority=Priority.PRIMARY,
                source=Provenance.STATED,
            )
        ],
        evidence=[
            EvidenceItem(
                kind=EvidenceKind.SUCCESSFUL_RUN,
                reference="12 accepted refund transcripts",
                status=EvidenceStatus.PROVIDED,
            ),
            EvidenceItem(
                kind=EvidenceKind.FAILURE_CASE,
                reference="double refund on 3 May",
                status=EvidenceStatus.PROVIDED,
            ),
        ],
        constraints=["all runs happen in a docker sandbox against a test account"],
    )

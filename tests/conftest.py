import pytest

from auto_eval.schema import (
    Confidence,
    Direction,
    EvalType,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    GroundTruth,
    KPI,
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

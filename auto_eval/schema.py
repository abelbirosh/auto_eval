"""The task specification produced by the input classifier.

Everything the rest of Auto_Eval consumes flows through `TaskSpec`. The three
things we need out of a user's free-form request are:

1. the *subject* - what is actually under test,
2. the *KPIs* - how we will decide whether it is good, quantitatively or
   qualitatively,
3. the *evidence* - documentation, successful runs, datasets we can anchor on.

Fields the user did not supply stay empty rather than being invented; the gap
analysis in `auto_eval.gaps` turns those holes into questions.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SubjectKind(str, Enum):
    """What sort of thing is being evaluated."""

    PROMPT = "prompt"
    AGENT = "agent"
    RAG_PIPELINE = "rag_pipeline"
    MODEL = "model"
    TOOL = "tool"
    WORKFLOW = "workflow"
    API_ENDPOINT = "api_endpoint"
    CLASSIFIER = "classifier"
    DATASET = "dataset"
    OTHER = "other"


class EvalType(str, Enum):
    """Why we are evaluating. A task can be several of these at once."""

    CAPABILITY = "capability"          # can it do the job at all
    REGRESSION = "regression"          # did a change break what worked
    COMPARATIVE = "comparative"        # A vs B (models, prompts, versions)
    SAFETY = "safety"                  # harmful, policy-violating output
    ROBUSTNESS = "robustness"          # adversarial or messy input
    PERFORMANCE = "performance"        # latency, throughput
    COST = "cost"                      # spend per task
    COMPLIANCE = "compliance"          # format, schema, contractual rules


class MetricKind(str, Enum):
    QUANTITATIVE = "quantitative"      # a number: accuracy, p95 latency, $/run
    QUALITATIVE = "qualitative"        # a judged quality: tone, faithfulness


class Measurement(str, Enum):
    """How a KPI can actually be computed."""

    PROGRAMMATIC = "programmatic"      # exact match, regex, schema validation
    EXECUTION = "execution"            # run the output, see if it passes tests
    LLM_JUDGE = "llm_judge"            # model grades against a rubric
    HUMAN_REVIEW = "human_review"
    TELEMETRY = "telemetry"            # latency/cost read from logs or usage
    UNKNOWN = "unknown"


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET_RANGE = "target_range"


class Priority(str, Enum):
    PRIMARY = "primary"                # the KPI the task succeeds or fails on
    SECONDARY = "secondary"
    GUARDRAIL = "guardrail"            # must not regress, not being optimized


class Provenance(str, Enum):
    STATED = "stated"                  # the user said it
    INFERRED = "inferred"              # the classifier proposed it


class EvidenceKind(str, Enum):
    DOCUMENTATION = "documentation"
    SUCCESSFUL_RUN = "successful_run"  # a golden trace / known-good output
    FAILURE_CASE = "failure_case"
    DATASET = "dataset"
    SPEC = "spec"                      # requirements, acceptance criteria
    CODE = "code"
    EXISTING_EVAL = "existing_eval"
    OTHER = "other"


class EvidenceStatus(str, Enum):
    PROVIDED = "provided"                        # we have it in hand
    MENTIONED_NOT_PROVIDED = "mentioned_not_provided"  # referenced, not attached
    ABSENT = "absent"


class Subject(BaseModel):
    """What we are testing."""

    name: str = Field(description="Short name for the thing under test.")
    kind: SubjectKind = Field(description="Category of the thing under test.")
    description: str = Field(description="One or two sentences on what it does.")
    interface: Optional[str] = Field(
        default=None,
        description="How it is invoked: CLI command, HTTP endpoint, function, prompt template.",
    )
    inputs: Optional[str] = Field(default=None, description="What it takes in.")
    outputs: Optional[str] = Field(default=None, description="What it returns.")
    in_scope: List[str] = Field(
        default_factory=list, description="Behaviours this evaluation covers."
    )
    out_of_scope: List[str] = Field(
        default_factory=list, description="Behaviours explicitly excluded."
    )


class KPI(BaseModel):
    """One thing we will measure, quantitative or qualitative."""

    name: str
    kind: MetricKind
    definition: str = Field(
        description="Precise statement of what is measured, unambiguous enough to implement."
    )
    measurement: Measurement = Field(description="How this would actually be computed.")
    direction: Direction
    unit: Optional[str] = Field(default=None, description="e.g. %, ms, USD, 1-5 rating.")
    target: Optional[str] = Field(
        default=None, description="Threshold or goal, verbatim from the user where given."
    )
    baseline: Optional[str] = Field(
        default=None, description="Current value, if the user stated one."
    )
    priority: Priority = Priority.SECONDARY
    source: Provenance = Provenance.STATED


class EvidenceItem(BaseModel):
    """Documentation, golden runs, datasets - anything we can anchor the eval on."""

    kind: EvidenceKind
    reference: str = Field(description="What it is, in the user's words.")
    location: Optional[str] = Field(
        default=None, description="Path, URL, or system where it lives, if stated."
    )
    status: EvidenceStatus = EvidenceStatus.MENTIONED_NOT_PROVIDED
    notes: Optional[str] = None


class GroundTruth(BaseModel):
    """Whether we can tell right from wrong without a human in the loop."""

    available: bool = False
    description: Optional[str] = Field(
        default=None, description="What the ground truth is and where it comes from."
    )
    sample_size: Optional[str] = Field(
        default=None, description="How many labelled examples exist, if stated."
    )


class Question(BaseModel):
    """Something we need the user to answer before the spec is usable."""

    field: str = Field(description="Dotted path into TaskSpec this would fill, e.g. 'kpis'.")
    question: str
    why: str = Field(description="What we cannot do until this is answered.")
    blocking: bool = Field(
        default=False, description="True if an eval cannot be built without it."
    )


class Confidence(BaseModel):
    """How much of the spec came from the user rather than from inference."""

    subject: float = Field(default=0.0, ge=0.0, le=1.0)
    kpis: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)


class Readiness(str, Enum):
    READY = "ready"                # enough to start building an eval
    NEEDS_INPUT = "needs_input"    # usable shape, non-blocking holes
    INSUFFICIENT = "insufficient"  # blocking holes; do not proceed


class TaskSpec(BaseModel):
    """The classified evaluation task."""

    title: str = Field(description="Short title for this evaluation task.")
    summary: str = Field(description="Two or three sentences describing the task.")
    subject: Subject
    eval_types: List[EvalType] = Field(default_factory=list)
    kpis: List[KPI] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)
    constraints: List[str] = Field(
        default_factory=list,
        description="Budget, latency, privacy, environment, or data-handling limits.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Anything the classifier filled in that the user did not say.",
    )
    open_questions: List[Question] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    readiness: Readiness = Readiness.NEEDS_INPUT


__all__ = [
    "Confidence",
    "Direction",
    "EvalType",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceStatus",
    "GroundTruth",
    "KPI",
    "Measurement",
    "MetricKind",
    "Priority",
    "Provenance",
    "Question",
    "Readiness",
    "Subject",
    "SubjectKind",
    "TaskSpec",
]

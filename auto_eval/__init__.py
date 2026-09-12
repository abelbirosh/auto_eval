"""Auto_Eval - turn a free-form request into a structured evaluation task spec."""

from .classifier import ClassifierError, classify, list_models
from .config import DEFAULT_MODEL, Settings, get_settings
from .extraction import GroundTruthSet, extract, to_jsonl
from .gaps import analyze, rule_questions
from .ground_truth import GroundTruthError, GroundTruthReport, gate, identify
from .render import (
    render_ground_truth,
    render_ground_truth_set,
    render_markdown,
    render_questions,
)
from .schema import TaskSpec

__version__ = "0.1.0"

__all__ = [
    "ClassifierError",
    "DEFAULT_MODEL",
    "GroundTruthError",
    "GroundTruthReport",
    "GroundTruthSet",
    "Settings",
    "TaskSpec",
    "analyze",
    "classify",
    "extract",
    "gate",
    "get_settings",
    "identify",
    "list_models",
    "render_ground_truth",
    "render_ground_truth_set",
    "render_markdown",
    "render_questions",
    "rule_questions",
    "to_jsonl",
]

"""Auto_Eval - turn a free-form request into a structured evaluation task spec."""

from .analysis import AnalysisReport, analyze_sources
from .benchmarks import CATALOGUE, Benchmark, match
from .classifier import ClassifierError, classify, list_models
from .config import DEFAULT_MODEL, Settings, get_settings
from .derive import derive
from .gaps import analyze, rule_questions
from .ground_truth import GroundTruthError, GroundTruthReport, gate, identify
from .render import (
    render_analysis,
    render_ground_truth,
    render_markdown,
    render_questions,
)
from .schema import Answer, TaskSpec

__version__ = "0.1.0"

__all__ = [
    "CATALOGUE",
    "DEFAULT_MODEL",
    "AnalysisReport",
    "Answer",
    "Benchmark",
    "ClassifierError",
    "GroundTruthError",
    "GroundTruthReport",
    "Settings",
    "TaskSpec",
    "analyze",
    "analyze_sources",
    "classify",
    "derive",
    "gate",
    "get_settings",
    "identify",
    "list_models",
    "match",
    "render_analysis",
    "render_ground_truth",
    "render_markdown",
    "render_questions",
    "rule_questions",
]

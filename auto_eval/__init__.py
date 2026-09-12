"""Auto_Eval - turn a free-form request into a structured evaluation task spec."""

from .agent import AgentProfile, Budget, RunMode, agent_gate, profile
from .analysis import AnalysisReport, analyze_sources
from .authoring import Case, CaseSource, author
from .benchmarks import CATALOGUE, Benchmark, match
from .classifier import ClassifierError, classify, list_models
from .config import DEFAULT_MODEL, Settings, get_settings
from .derive import derive
from .gaps import analyze, rule_questions
from .ground_truth import GroundTruthError, GroundTruthReport, gate, identify
from .render import (
    render_analysis,
    render_case,
    render_ground_truth,
    render_markdown,
    render_profile,
    render_questions,
    render_suite,
)
from .schema import Answer, TaskSpec
from .suite import Suite, SuiteError
from .suite import build as build_suite
from .suite import load as load_suite
from .surface import CoverageMatrix, surface
from .verify import Verifier, VerifierKind

__version__ = "0.1.0"

__all__ = [
    "CATALOGUE",
    "DEFAULT_MODEL",
    "AgentProfile",
    "AnalysisReport",
    "Answer",
    "Benchmark",
    "Budget",
    "Case",
    "CaseSource",
    "ClassifierError",
    "CoverageMatrix",
    "GroundTruthError",
    "GroundTruthReport",
    "RunMode",
    "Settings",
    "Suite",
    "SuiteError",
    "TaskSpec",
    "Verifier",
    "VerifierKind",
    "agent_gate",
    "analyze",
    "analyze_sources",
    "author",
    "build_suite",
    "classify",
    "derive",
    "gate",
    "get_settings",
    "identify",
    "list_models",
    "load_suite",
    "match",
    "profile",
    "render_analysis",
    "render_case",
    "render_ground_truth",
    "render_markdown",
    "render_profile",
    "render_questions",
    "render_suite",
    "rule_questions",
    "surface",
]

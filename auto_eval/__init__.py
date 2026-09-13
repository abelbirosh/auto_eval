"""Auto_Eval - turn a free-form request into a structured evaluation task spec."""

from .agent import AgentProfile, Budget, RunMode, agent_gate, profile
from .analysis import AnalysisReport, analyze_sources
from .authoring import Case, CaseSource, author
from .benchmarks import CATALOGUE, Benchmark, match
from .classifier import ClassifierError, classify, list_models
from .config import DEFAULT_MODEL, Settings, get_settings
from .contamination import FRESH, ContaminationReport, Freshness, cutoff_for
from .derive import derive
from .gaps import analyze, rule_questions
from .grade import Blocked, grade
from .ground_truth import GroundTruthError, GroundTruthReport, gate, identify
from .judge import judge_with
from .provider import MockClient, execute
from .render import (
    render_analysis,
    render_case,
    render_fresh,
    render_ground_truth,
    render_markdown,
    render_profile,
    render_questions,
    render_run,
    render_suite,
)
from .runner import (
    CaseStatus,
    Run,
    RunError,
    RunReport,
    list_runs,
    load_run,
    run_suite,
    write_run,
)
from .schema import Answer, TaskSpec
from .suite import Suite, SuiteError
from .suite import build as build_suite
from .suite import load as load_suite
from .surface import CoverageMatrix, surface
from .trace import StopReason, Trace
from .verify import CaseResult, Outcome, Verifier, VerifierKind

__version__ = "0.1.0"

__all__ = [
    "CATALOGUE",
    "DEFAULT_MODEL",
    "FRESH",
    "AgentProfile",
    "AnalysisReport",
    "Answer",
    "Benchmark",
    "Blocked",
    "Budget",
    "Case",
    "CaseResult",
    "CaseSource",
    "CaseStatus",
    "ClassifierError",
    "ContaminationReport",
    "CoverageMatrix",
    "Freshness",
    "GroundTruthError",
    "GroundTruthReport",
    "MockClient",
    "Outcome",
    "Run",
    "RunError",
    "RunMode",
    "RunReport",
    "Settings",
    "StopReason",
    "Suite",
    "SuiteError",
    "TaskSpec",
    "Trace",
    "Verifier",
    "VerifierKind",
    "agent_gate",
    "analyze",
    "analyze_sources",
    "author",
    "build_suite",
    "classify",
    "cutoff_for",
    "derive",
    "execute",
    "gate",
    "get_settings",
    "grade",
    "identify",
    "judge_with",
    "list_models",
    "list_runs",
    "load_run",
    "load_suite",
    "match",
    "profile",
    "render_analysis",
    "render_case",
    "render_fresh",
    "render_ground_truth",
    "render_markdown",
    "render_profile",
    "render_questions",
    "render_run",
    "render_suite",
    "rule_questions",
    "run_suite",
    "surface",
    "write_run",
]

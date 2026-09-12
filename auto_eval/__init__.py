"""Auto_Eval - turn a free-form request into a structured evaluation task spec."""

from .classifier import ClassifierError, classify, list_models
from .config import DEFAULT_MODEL, Settings, get_settings
from .gaps import analyze, rule_questions
from .render import render_markdown, render_questions
from .schema import TaskSpec

__version__ = "0.1.0"

__all__ = [
    "ClassifierError",
    "DEFAULT_MODEL",
    "Settings",
    "TaskSpec",
    "analyze",
    "classify",
    "get_settings",
    "list_models",
    "render_markdown",
    "render_questions",
    "rule_questions",
]

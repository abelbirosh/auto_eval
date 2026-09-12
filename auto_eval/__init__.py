"""Auto_Eval - turn a free-form request into a structured evaluation task spec."""

from .classifier import DEFAULT_MODEL, ClassifierError, classify
from .gaps import analyze, rule_questions
from .render import render_markdown, render_questions
from .schema import TaskSpec

__version__ = "0.1.0"

__all__ = [
    "ClassifierError",
    "DEFAULT_MODEL",
    "TaskSpec",
    "analyze",
    "classify",
    "render_markdown",
    "render_questions",
    "rule_questions",
]

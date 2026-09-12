"""Haiku-backed classifier: free-form request in, TaskSpec out.

The model does extraction only. Deciding what is *missing* - and therefore
whether the spec is usable - is `auto_eval.gaps`, so that verdict stays
deterministic and testable without a network call.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .gaps import analyze
from .prompts import SYSTEM_PROMPT, build_user_message
from .schema import TaskSpec

# Haiku is deliberate: this is an extraction task on a short input, run on every
# request. Route it elsewhere with `model=` if the specs come back thin.
DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 16000

# A request longer than this is almost certainly a pasted log or dump; we refuse
# rather than silently truncate it into a misleading spec.
MAX_INPUT_CHARS = 200_000


class ClassifierError(RuntimeError):
    """Raised when a task spec could not be produced."""


class _ParsesMessages(Protocol):
    """The slice of the Anthropic client this module uses (handy for fakes)."""

    messages: Any


def _default_client() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise ClassifierError(
            "The `anthropic` package is required to classify input. "
            "Install it with `pip install anthropic`, or pass your own client."
        ) from exc
    return anthropic.Anthropic()


def classify(
    text: str,
    *,
    client: Optional[_ParsesMessages] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> TaskSpec:
    """Classify a free-form evaluation request into a `TaskSpec`.

    The returned spec always carries the merged rule/model questions and a
    computed `readiness`; the model's own readiness guess is discarded.
    """
    if not text or not text.strip():
        raise ClassifierError("Nothing to classify: the request is empty.")
    if len(text) > MAX_INPUT_CHARS:
        raise ClassifierError(
            f"Request is {len(text):,} characters, over the {MAX_INPUT_CHARS:,} limit. "
            "Split it, or point at the large parts as evidence instead of pasting them."
        )

    client = client or _default_client()

    if not hasattr(client.messages, "parse"):  # pragma: no cover - install state
        raise ClassifierError(
            "This `anthropic` version has no `messages.parse`; upgrade with "
            "`pip install -U anthropic`."
        )

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(text)}],
            output_format=TaskSpec,
        )
    except Exception as exc:  # re-raised below with a usable message
        raise _as_classifier_error(exc) from exc

    spec = response.parsed_output
    if spec is None:
        raise ClassifierError(
            f"The model returned no parsable task spec (stop_reason={response.stop_reason!r})."
        )
    return analyze(spec)


def _as_classifier_error(exc: Exception) -> ClassifierError:
    """Turn SDK exceptions into one error type with an actionable message."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return ClassifierError(f"Classification failed: {exc}")

    if isinstance(exc, anthropic.NotFoundError):
        return ClassifierError(
            f"Model not found or not available to this account. Check the model ID. ({exc})"
        )
    if isinstance(exc, anthropic.RateLimitError):
        return ClassifierError(f"Rate limited - retry after a short backoff. ({exc})")
    if isinstance(exc, anthropic.AuthenticationError):
        return ClassifierError(
            "Authentication failed. Set ANTHROPIC_API_KEY, or run `ant auth login`."
        )
    if isinstance(exc, anthropic.APIStatusError):
        return ClassifierError(f"API error {exc.status_code}: {exc.message}")
    if isinstance(exc, anthropic.APIConnectionError):
        return ClassifierError(f"Could not reach the API: {exc}")
    return ClassifierError(f"Classification failed: {exc}")


__all__ = ["DEFAULT_MAX_TOKENS", "DEFAULT_MODEL", "ClassifierError", "classify"]

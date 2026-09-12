"""OpenAI-backed classifier: free-form request in, TaskSpec out.

The model does extraction only. Deciding what is *missing* - and therefore
whether the spec is usable - is `auto_eval.gaps`, so that verdict stays
deterministic and testable without a network call.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .config import API_KEY_VAR, Settings, get_settings
from .gaps import analyze
from .prompts import SYSTEM_PROMPT, build_user_message
from .schema import Answer, TaskSpec

# Extraction on a short input, run on every request, so it goes to the cheap
# tier. Raise it with --model or AUTO_EVAL_MODEL if specs come back thin.
DEFAULT_MAX_TOKENS = 16000

# A request longer than this is almost certainly a pasted log or dump; we refuse
# rather than silently truncate it into a misleading spec.
MAX_INPUT_CHARS = 200_000


class ClassifierError(RuntimeError):
    """Raised when a task spec could not be produced."""


def _build_client(settings: Settings) -> Any:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise ClassifierError(
            "The `openai` package is required to classify input. "
            "Install it with `pip install openai`, or pass your own client."
        ) from exc

    if not settings.has_key:
        raise ClassifierError(
            f"No API key. Put `{API_KEY_VAR}=sk-...` in your .env "
            f"(see .env.example), or export {API_KEY_VAR} in your shell."
        )

    kwargs: dict = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return openai.OpenAI(**kwargs)


def classify(
    text: str,
    *,
    answers: Sequence[Answer] = (),
    client: Optional[Any] = None,
    model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> TaskSpec:
    """Classify a free-form evaluation request into a `TaskSpec`.

    `answers` are replies to questions an earlier pass asked. They are appended
    to the same request and the whole thing is classified again, so a spec is
    always the product of one call rather than a patched-up earlier one.

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

    settings = get_settings(model=model)
    client = client or _build_client(settings)

    try:
        completion = client.chat.completions.parse(
            model=settings.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(text, answers)},
            ],
            response_format=TaskSpec,
            # Reasoning-capable models reject `max_tokens`; this is its successor.
            max_completion_tokens=max_tokens,
        )
    except Exception as exc:  # re-raised below with a usable message
        raise _as_classifier_error(exc, settings) from exc

    choice = completion.choices[0]

    if getattr(choice.message, "refusal", None):
        raise ClassifierError(
            f"The model declined to classify this: {choice.message.refusal}"
        )

    spec = choice.message.parsed
    if spec is None:
        raise ClassifierError(
            "The model returned no parsable task spec "
            f"(finish_reason={choice.finish_reason!r}). "
            "If this says 'length', raise --max-tokens."
        )
    return analyze(spec)


def _as_classifier_error(exc: Exception, settings: Settings) -> ClassifierError:
    """Turn SDK exceptions into one error type with an actionable message."""
    try:
        import openai
    except ImportError:  # pragma: no cover
        return ClassifierError(f"Classification failed: {exc}")

    if isinstance(exc, openai.NotFoundError):
        return ClassifierError(
            f"Model {settings.model!r} is not available to this account. "
            "Set AUTO_EVAL_MODEL in your .env to one that is - "
            "`auto-eval models` lists what your key can reach."
        )
    if isinstance(exc, openai.AuthenticationError):
        return ClassifierError(
            f"Authentication failed. Check {API_KEY_VAR} in your .env is a valid, current key."
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return ClassifierError(
            f"This key is not permitted to use {settings.model!r}. "
            "Set AUTO_EVAL_MODEL to a model your plan includes."
        )
    if isinstance(exc, openai.RateLimitError):
        return ClassifierError(
            f"Rate limited or out of quota. ({exc}) "
            "On a free tier this also appears when the account has no credit."
        )
    if isinstance(exc, openai.BadRequestError):
        return ClassifierError(f"The API rejected the request: {exc}")
    if isinstance(exc, openai.APIStatusError):
        return ClassifierError(f"API error {exc.status_code}: {exc}")
    if isinstance(exc, openai.APIConnectionError):
        return ClassifierError(f"Could not reach the API: {exc}")
    return ClassifierError(f"Classification failed: {exc}")


def list_models(client: Optional[Any] = None) -> list:
    """Model IDs this key can reach, so a bad AUTO_EVAL_MODEL is easy to fix."""
    settings = get_settings()
    client = client or _build_client(settings)
    try:
        return sorted(m.id for m in client.models.list())
    except Exception as exc:
        raise _as_classifier_error(exc, settings) from exc


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MAX_INPUT_CHARS",
    "ClassifierError",
    "classify",
    "list_models",
]

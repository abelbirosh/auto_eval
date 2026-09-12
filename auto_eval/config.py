"""Runtime settings, loaded from the environment and an optional `.env`.

Keeping this separate from the classifier means the CLI, tests, and any future
caller agree on where the key and model come from, and that `.env` is read
exactly once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# OpenAI's cheap, fast tier. Structured outputs are supported here, which is all
# this classifier needs. Override with AUTO_EVAL_MODEL if your account does not
# have this model.
DEFAULT_MODEL = "gpt-5-mini"

API_KEY_VAR = "OPENAI_API_KEY"
MODEL_VAR = "AUTO_EVAL_MODEL"
BASE_URL_VAR = "AUTO_EVAL_BASE_URL"
# The ground-truth search reads the open web and is worth spending more on than
# extraction; left unset it uses the same model as everything else.
SEARCH_MODEL_VAR = "AUTO_EVAL_SEARCH_MODEL"

_ENV_LOADED = False


def load_env(start: Optional[Path] = None) -> Optional[Path]:
    """Load the nearest `.env` walking up from `start`. Real env vars win.

    Idempotent: repeated calls are a no-op, so importing this module from
    several places does not re-read the file.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return None
    _ENV_LOADED = True

    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional; plain env vars still work
        return None

    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            # override=False so an exported variable beats the file.
            load_dotenv(candidate, override=False)
            return candidate
    return None


@dataclass(frozen=True)
class Settings:
    api_key: Optional[str]
    model: str
    base_url: Optional[str]
    # None means "same model as everything else"; callers read it through
    # `effective_search_model` so a hand-built Settings needs only three fields.
    search_model: Optional[str] = None

    @property
    def has_key(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @property
    def effective_search_model(self) -> str:
        return self.search_model or self.model


def get_settings(*, model: Optional[str] = None) -> Settings:
    """Resolve settings. An explicit `model` argument beats the environment."""
    load_env()
    resolved = model or os.environ.get(MODEL_VAR) or DEFAULT_MODEL
    return Settings(
        api_key=os.environ.get(API_KEY_VAR),
        model=resolved,
        base_url=os.environ.get(BASE_URL_VAR) or None,
        # An explicit argument is a deliberate choice for this call, so it wins
        # over the search-specific variable too.
        search_model=model or os.environ.get(SEARCH_MODEL_VAR) or resolved,
    )


__all__ = [
    "API_KEY_VAR",
    "BASE_URL_VAR",
    "DEFAULT_MODEL",
    "MODEL_VAR",
    "SEARCH_MODEL_VAR",
    "Settings",
    "get_settings",
    "load_env",
]

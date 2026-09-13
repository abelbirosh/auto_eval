"""The systems a board compares, and what each one costs to call.

A benchmark board is only worth reading if one thing changed between the rows.
So a cohort names the systems under test and, separately, the two things held
constant across all of them: the model that reads the results, and the judge
that rules on anything a machine cannot settle. Change those and the board is
measuring the model, not the systems.

Three kinds of row, and the first one is the reason the other two mean anything:

* **model only** - the model answering from memory, no endpoint at all. This is
  the baseline. On a dataset of facts published after the model's cutoff it
  should score near zero; if it does not, the board is measuring recall and the
  other rows are not what they appear to be.
* **search api** - the endpoint on its own. A query goes in, results come back,
  and the score is whether the answer is in them. No model is involved, so this
  row is the retrieval quality and nothing else.
* **model with tool** - the model with that endpoint as a tool. This is the
  agent case: the score includes whether the model used what it was given.

Configuration is declarative and secrets are not part of it: a header worth
`${EXA_API_KEY}` is filled from the environment at call time, and a system whose
key is missing is reported as unconfigured rather than run and scored as bad.
"""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .classifier import ClassifierError

ENV_REFERENCE = re.compile(r"\$\{([A-Z0-9_]+)\}")

# The placeholder an endpoint's body or params use for the item's query.
QUERY_PLACEHOLDER = "{query}"


class CohortError(ClassifierError):
    """Raised when a cohort cannot be read or is not usable as one."""


class SystemKind(str, Enum):
    MODEL_ONLY = "model_only"  # the baseline: no endpoint, memory alone
    SEARCH_API = "search_api"  # the endpoint alone, scored on what it returns
    MODEL_WITH_TOOL = "model_with_tool"  # the model, with the endpoint as a tool


class Price(BaseModel):
    """List price, as the vendor publishes it. Never estimated."""

    per_call_usd: Optional[float] = Field(
        default=None, description="Published PAYG price for one call."
    )
    per_million_input_usd: Optional[float] = None
    per_million_output_usd: Optional[float] = None
    note: Optional[str] = Field(
        default=None, description="How the price is packaged, in the vendor's words."
    )
    source: Optional[str] = Field(default=None, description="The pricing page.")

    @property
    def stated(self) -> bool:
        return any(
            v is not None
            for v in (
                self.per_call_usd,
                self.per_million_input_usd,
                self.per_million_output_usd,
            )
        )


class Endpoint(BaseModel):
    """How to call one vendor, declaratively."""

    method: str = "POST"
    url: str
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Values may reference the environment as ${VAR}; nothing is stored here.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Query string. {query} is the item's query."
    )
    body: Dict[str, Any] = Field(
        default_factory=dict, description="JSON body. {query} is the item's query."
    )
    results_path: str = Field(
        default="results",
        description="Dotted path to the list of results in the response, '' for the root.",
    )
    fields: Dict[str, str] = Field(
        default_factory=lambda: {"title": "title", "url": "url", "snippet": "text"},
        description="Which key in a result holds the title, the url, and the text.",
    )
    timeout_s: float = 30.0

    def secrets(self) -> List[str]:
        """The environment variables this endpoint needs before it can be called."""
        blob = json.dumps([self.headers, self.params, self.body])
        return sorted(set(ENV_REFERENCE.findall(blob)))

    def missing(self) -> List[str]:
        return [name for name in self.secrets() if not os.environ.get(name)]


class System(BaseModel):
    """One row of the board."""

    label: str = Field(description="How the row is named, e.g. 'Exa fast'.")
    vendor: str = ""
    kind: SystemKind = SystemKind.SEARCH_API
    configuration: str = Field(
        default="",
        description="The endpoint and parameters, shown verbatim in the board.",
    )
    endpoint: Optional[Endpoint] = None
    model: Optional[str] = Field(
        default=None,
        description="Overrides the cohort's model, for a row that is a different model.",
    )
    price: Price = Field(default_factory=Price)
    docs_url: Optional[str] = None
    note: Optional[str] = None

    @property
    def needs_endpoint(self) -> bool:
        return self.kind is not SystemKind.MODEL_ONLY

    @property
    def calls_a_model(self) -> bool:
        return self.kind is not SystemKind.SEARCH_API

    def readiness(self) -> Tuple[bool, str]:
        """Whether this row can be run, and what is missing if not."""
        if self.needs_endpoint and self.endpoint is None:
            return False, f"{self.label} has no endpoint to call."
        if self.endpoint is not None:
            missing = self.endpoint.missing()
            if missing:
                return False, (
                    f"{self.label} needs {', '.join(missing)} in the environment. "
                    "Set it, or drop the row - an unconfigured vendor scored as zero is a libel."
                )
        return True, ""


class Cohort(BaseModel):
    """The systems, and the two things held constant across them."""

    name: str = "cohort"
    task: str = Field(default="", description="What this cohort is being compared on.")
    systems: List[System] = Field(default_factory=list)
    model: Optional[str] = Field(
        default=None,
        description="The model held constant across every row that uses one.",
    )
    judge_model: Optional[str] = None
    notes: List[str] = Field(default_factory=list)

    @property
    def baseline(self) -> Optional[System]:
        return next((s for s in self.systems if s.kind is SystemKind.MODEL_ONLY), None)

    def runnable(self) -> Tuple[List[System], List[Tuple[System, str]]]:
        """(what can be run, what cannot and why)."""
        ready: List[System] = []
        blocked: List[Tuple[System, str]] = []
        for system in self.systems:
            ok, why = system.readiness()
            if ok:
                ready.append(system)
            else:
                blocked.append((system, why))
        return ready, blocked

    def ordered(self) -> List[System]:
        """Alphabetical, deliberately.

        No metric decides the order of the rows: a board sorted by its own
        headline number invites the reader to treat that number as the answer,
        and on a benchmark with four columns it usually is not.
        """
        return sorted(self.systems, key=lambda s: s.label.lower())


def baseline_system(model: Optional[str] = None) -> System:
    """The model-only row, which every board should carry."""
    return System(
        label="model only (no search)",
        vendor="baseline",
        kind=SystemKind.MODEL_ONLY,
        configuration="the model answering from memory, with no endpoint",
        model=model,
        note=(
            "The control. On a dataset of facts published after the model's cutoff this "
            "should be near zero; anything else means the board is partly a memory test."
        ),
    )


def parse(text: str) -> Cohort:
    """Read a cohort from JSON: an object, or a bare list of systems."""
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise CohortError(f"Not readable JSON: {exc}") from exc

    if isinstance(raw, list):
        raw = {"systems": raw}
    if not isinstance(raw, dict):
        raise CohortError("A cohort is a JSON object, or a list of systems.")

    try:
        cohort = Cohort.model_validate(raw)
    except ValueError as exc:
        raise CohortError(f"Not a usable cohort: {exc}") from exc
    if not cohort.systems:
        raise CohortError("A cohort with no systems has nothing to compare.")
    return cohort


def load(path: Path) -> Cohort:
    """Read a cohort from a file. An unnamed cohort takes the file's name."""
    if not path.is_file():
        raise CohortError(f"No cohort at {path}.")
    text = path.read_text(encoding="utf-8")
    cohort = parse(text)
    try:
        named = isinstance(json.loads(text), dict) and "name" in json.loads(text)
    except ValueError:  # pragma: no cover - parse() would have raised first
        named = False
    return cohort if named else cohort.model_copy(update={"name": path.stem})


def with_baseline(cohort: Cohort) -> Cohort:
    """The cohort plus a model-only row, if it does not already have one."""
    if cohort.baseline is not None:
        return cohort
    return cohort.model_copy(
        update={"systems": [baseline_system(cohort.model), *cohort.systems]}
    )


__all__ = [
    "ENV_REFERENCE",
    "QUERY_PLACEHOLDER",
    "Cohort",
    "CohortError",
    "Endpoint",
    "Price",
    "System",
    "SystemKind",
    "baseline_system",
    "load",
    "parse",
    "with_baseline",
]

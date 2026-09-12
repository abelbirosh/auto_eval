"""The suite: the frozen artefact everything else in this block exists to produce.

A suite is content-addressed and carries no timestamp, which is deliberate. The
same spec produces the same suite, byte for byte, so two runs of `auto-eval
author` can be diffed, and a number reported in six months can be traced to the
exact set of cases that produced it. A suite that quietly drifts makes every
regression comparison it is used for meaningless.

What a suite is not, and the report says so on its face: **runnable as it
stands**. The fixtures are specified, not materialised - we do not have the
user's environment, so what this block writes is the instruction for building
each starting state. Block 5 runs the suite; this block says what the suite is.

The split is decided here rather than during authoring, so that a case's
identity does not depend on which side of the line it landed. Regression cases
are held out unconditionally - a known failure used for tuning stops being
evidence of anything - and the rest are split by a hash of their id, which keeps
the same case on the same side across rebuilds.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from .agent import AgentProfile, RunMode, agent_gate, profile
from .authoring import (
    MAX_SYNTHETIC_SHARE,
    Case,
    CaseSource,
    author,
    needs_fixture,
    synthetic_share,
)
from .classifier import ClassifierError
from .gaps import analyze
from .schema import EvalType, TaskSpec
from .surface import (
    DEFAULT_PER_CELL,
    CoverageMatrix,
    coverage_ratio,
    fill,
    render_gaps,
    surface,
)
from .verify import automatic_share
from .verify import coverage_by_kpi as verifier_coverage


class SuiteError(ClassifierError):
    """Raised when a suite cannot be built for this spec."""


class Split(str, Enum):
    DEV = "dev"  # look at these, tune against these
    HELD_OUT = "held_out"  # the number you report comes from these


# Share of cases kept for development. Small on purpose: the dev split exists to
# debug the harness and the rubrics, not to be the suite.
DEV_SHARE = 0.3

SUITE_FILENAME = "suite.json"
DOCUMENT_FILENAME = "suite.md"


class Suite(BaseModel):
    """Cases, checks, and the claim about what they cover."""

    name: str
    subject: str
    spec_digest: str = Field(description="Hash of the spec this was built from.")
    digest: str = Field(
        default="", description="Hash of the suite itself, filled on build."
    )
    profile: AgentProfile
    coverage: CoverageMatrix
    cases: List[Case] = Field(default_factory=list)
    held_out: List[str] = Field(
        default_factory=list,
        description="Case ids whose results are the reported number.",
    )
    kpis: List[str] = Field(default_factory=list)
    samples: int = Field(
        default=1, description="Runs per case; more than one when stochastic."
    )
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=list,
        description="Things that make this suite weaker than it looks.",
    )

    def split_of(self, case_id: str) -> Split:
        return Split.HELD_OUT if case_id in set(self.held_out) else Split.DEV

    def cases_in(self, split: Split) -> List[Case]:
        held = set(self.held_out)
        return [c for c in self.cases if (c.id in held) is (split is Split.HELD_OUT)]

    def by_source(self, source: CaseSource) -> List[Case]:
        return [c for c in self.cases if c.source is source]

    @property
    def verifiers(self) -> List:
        return [v for case in self.cases for v in case.verifiers]

    @property
    def total_runs(self) -> int:
        """What running this actually costs, which is not the number of cases."""
        return len(self.cases) * max(1, self.samples)


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _spec_digest(spec: TaskSpec) -> str:
    return _digest(spec.model_dump_json())


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return "-".join(part for part in cleaned.split("-") if part) or "agent"


def _held_out(cases: Sequence[Case]) -> List[str]:
    """Which cases carry the reported number.

    Every regression case, plus a deterministic majority of the rest. Hashing the
    id rather than shuffling means a rebuilt suite keeps each case where it was,
    so a held-out score stays comparable across rebuilds.
    """
    out: List[str] = []
    for case in cases:
        if EvalType.REGRESSION in case.eval_types:
            out.append(case.id)
            continue
        bucket = int(hashlib.sha256(case.id.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket >= DEV_SHARE * 100:
            out.append(case.id)
    return out


def _warnings(
    suite_cases: Sequence[Case], matrix: CoverageMatrix, agent_profile: AgentProfile
) -> List[str]:
    """Everything that makes this suite weaker than the case count suggests."""
    out: List[str] = []

    share = synthetic_share(suite_cases)
    if share > MAX_SYNTHETIC_SHARE:
        out.append(
            f"{share:.0%} of these cases were generated from the spec rather than taken from your runs "
            f"or a public suite. Above {MAX_SYNTHETIC_SHARE:.0%}, a headline number from this suite is "
            "partly a measurement of the generator. Sharing golden runs and incidents is what fixes it."
        )

    pending = needs_fixture(suite_cases)
    if pending:
        out.append(
            f"{len(pending)} of {len(suite_cases)} cases describe a starting state that does not exist yet. "
            "Nothing here runs until those fixtures are built."
        )

    automatic = automatic_share([v for case in suite_cases for v in case.verifiers])
    if automatic < 0.5:
        out.append(
            f"Only {automatic:.0%} of the checks can be settled by machine; the rest need a judge or a person. "
            "That is slow and dear to run often, and judged checks are the ones that drift."
        )

    gaps = render_gaps(matrix)
    if gaps:
        out.append(
            f"{len(gaps)} cell(s) of the coverage grid are empty or thin: "
            + "; ".join(gaps)
        )

    if agent_profile.run_mode is RunMode.TRANSCRIPT:
        out.append(
            "The agent cannot be started, so this suite scores recorded traces. Any case that needs a "
            "fresh run - a tool failure, an injection - is written but cannot be exercised until there is an entry point."
        )

    if not agent_profile.tools:
        out.append(
            "No tools were read off the request, so every trajectory check is a guess about what the agent can reach. "
            "Listing the tools is the single highest-value correction you can make to this suite."
        )
    return out


def build(
    spec: TaskSpec,
    *,
    per_cell: int = DEFAULT_PER_CELL,
    force: bool = False,
) -> Suite:
    """Build a suite for `spec`. Deterministic, offline, no model call.

    Raises `SuiteError` when the agent gate is closed, unless `force` is set -
    a suite written against an agent nobody can start is a document, not a test.
    """
    settled = analyze(spec)
    agent_profile = profile(settled)
    decision = agent_gate(settled, agent_profile)
    if not decision.open and not force:
        raise SuiteError(
            f"Not ready to author: {decision.reason} "
            "Answer those first, or pass force=True to write the suite anyway."
        )

    matrix = surface(settled, per_cell=per_cell)
    cases = author(settled, agent_profile, matrix)
    filled = fill(matrix, cases)

    notes = [
        f"Coverage: {coverage_ratio(filled):.0%} of the {filled.target_total} cases the grid asked for.",
        f"Sources: {len(([c for c in cases if c.source is CaseSource.HARVESTED]))} harvested, "
        f"{len([c for c in cases if c.source is CaseSource.ADAPTED])} adapted, "
        f"{len([c for c in cases if c.source is CaseSource.SYNTHESISED])} synthesised.",
        f"Running it once means {len(cases)} x {agent_profile.samples} = {len(cases) * agent_profile.samples} agent runs.",
        *filled.notes,
    ]
    if not decision.open:
        notes.append(f"Written past a closed gate: {decision.reason}")

    suite = Suite(
        name=_slug(settled.subject.name or "agent"),
        subject=settled.subject.name or "the agent",
        spec_digest=_spec_digest(settled),
        profile=agent_profile,
        coverage=filled,
        cases=cases,
        held_out=_held_out(cases),
        kpis=[kpi.name for kpi in settled.kpis],
        samples=agent_profile.samples,
        notes=notes,
        warnings=_warnings(cases, filled, agent_profile),
    )
    return suite.model_copy(update={"digest": _digest(suite.model_dump_json())})


def verify_digest(suite: Suite) -> bool:
    """Whether the suite still hashes to what it says it does."""
    stripped = suite.model_copy(update={"digest": ""})
    return _digest(stripped.model_dump_json()) == suite.digest


def kpi_coverage(suite: Suite) -> dict:
    """KPI name -> the checks in this suite that speak to it.

    A KPI with an empty list is a KPI the suite does not measure, however many
    cases it holds.
    """
    found = verifier_coverage(suite.verifiers)
    return {name: found.get(name, []) for name in suite.kpis}


def write(suite: Suite, out_dir: Path, *, document: Optional[str] = None) -> List[str]:
    """Write the suite where block 5 can pick it up. Returns what was written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    path = out_dir / SUITE_FILENAME
    path.write_text(suite.model_dump_json(indent=2) + "\n", encoding="utf-8")
    written.append(f"Wrote {len(suite.cases)} case(s) to {path}")

    if document is not None:
        doc = out_dir / DOCUMENT_FILENAME
        doc.write_text(document + "\n", encoding="utf-8")
        written.append(f"Wrote {doc}")

    pending = needs_fixture(suite.cases)
    if pending:
        written.append(
            f"{len(pending)} fixture(s) still have to be built before any of this runs - see the document."
        )
    return written


def load(path: Path) -> Suite:
    """Read a suite back, from its directory or from the file itself."""
    target = path / SUITE_FILENAME if path.is_dir() else path
    if not target.exists():
        raise SuiteError(
            f"No suite at {target}. Write one with `auto-eval author -o {path}`."
        )
    try:
        return Suite.model_validate_json(target.read_text(encoding="utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise SuiteError(f"{target} is not a readable suite: {exc}") from exc


__all__ = [
    "DEV_SHARE",
    "DOCUMENT_FILENAME",
    "SUITE_FILENAME",
    "Split",
    "Suite",
    "SuiteError",
    "build",
    "kpi_coverage",
    "load",
    "verify_digest",
    "write",
]

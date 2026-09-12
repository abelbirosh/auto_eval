"""What can we actually drive, and is it safe to drive it?

Fourth block in the pipeline, first stage. Everything before this grades a
string: the classifier says what is under test, the ground-truth search says
whether public labels exist, the catalogue says whether a standard suite
applies. None of that is enough to build a suite for an *agent*, because an
agent does not hand back an output - it hands back a trajectory and a changed
world.

Three things have to be pinned down before a case can be written, and `TaskSpec`
holds none of them:

1. the **entrypoint** - how one episode starts, and how we know it ended,
2. the **tools** - what it can reach, and which of those reach outside the test,
3. the **budget** - what stops a run that would otherwise never terminate.

`profile()` derives all three from the spec the way `auto_eval.derive` derives a
KPI: from rules, marked as assumptions, never invented as fact. `agent_gate()`
then decides whether authoring may proceed, and blocks on the two holes that are
genuinely unrecoverable - nothing to run, or something destructive with nowhere
safe to run it.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .ground_truth import Gate
from .ground_truth import gate as spec_gate
from .schema import (
    EvidenceKind,
    EvidenceStatus,
    Question,
    SubjectKind,
    TaskSpec,
)


class RunMode(str, Enum):
    """How block 5 would start one episode."""

    SUBPROCESS = "subprocess"  # argv and a working directory
    HTTP = "http"  # POST a task, poll for a terminal state
    PYTHON = "python"  # import path to a callable
    TRANSCRIPT = "transcript"  # offline: score runs that already happened
    UNKNOWN = "unknown"


class Effect(str, Enum):
    """What a tool does to the world. Ordered by how much it matters."""

    READ = "read"
    WRITE = "write"  # changes state we own
    EXTERNAL = "external"  # reaches a third party: mail, payments, web
    DESTRUCTIVE = "destructive"  # loses data or moves money


EFFECT_SEVERITY = [Effect.READ, Effect.WRITE, Effect.EXTERNAL, Effect.DESTRUCTIVE]

# Effects that must not touch anything real during an evaluation.
CONTAINED_EFFECTS = frozenset({Effect.EXTERNAL, Effect.DESTRUCTIVE})


class Isolation(str, Enum):
    """Where a run is allowed to happen."""

    NONE = "none"  # against the real system - never acceptable for a write
    TMPDIR = "tmpdir"  # scratch directory, real network
    CONTAINER = "container"  # sandboxed filesystem and network policy
    UNKNOWN = "unknown"


SAFE_ISOLATION = frozenset({Isolation.TMPDIR, Isolation.CONTAINER})


class ToolFacet(BaseModel):
    """One thing the agent can reach, and what happens when it does."""

    name: str
    effect: Effect = Effect.READ
    stubbable: bool = Field(
        default=True,
        description="Whether block 5 can script its responses deterministically.",
    )
    evidence: Optional[str] = Field(
        default=None, description="The wording in the spec this was read off."
    )


class Budget(BaseModel):
    """What stops a run. A run with no ceiling is not a test, it is a hang."""

    max_steps: Optional[int] = None
    max_seconds: Optional[int] = None
    max_tokens: Optional[int] = None
    max_usd: Optional[float] = None

    def stated(self) -> bool:
        return any(
            v is not None
            for v in (self.max_steps, self.max_seconds, self.max_tokens, self.max_usd)
        )


class AgentProfile(BaseModel):
    """The operational facts about the agent, as far as the spec supports them."""

    subject: str
    run_mode: RunMode
    entrypoint: Optional[str] = Field(
        default=None, description="The invocation, verbatim from the spec."
    )
    episode_end: str = Field(
        description="What terminal looks like, so a run can be known to be over."
    )
    tools: List[ToolFacet] = Field(default_factory=list)
    isolation: Isolation = Isolation.UNKNOWN
    budget: Budget = Field(default_factory=Budget)
    stochastic: bool = True
    samples: int = Field(
        default=1, description="Runs per case. More than one only when stochastic."
    )
    assumptions: List[str] = Field(
        default_factory=list, description="Everything here that the user did not say."
    )

    def tools_with(self, *effects: Effect) -> List[ToolFacet]:
        return [tool for tool in self.tools if tool.effect in effects]

    @property
    def worst_effect(self) -> Optional[Effect]:
        if not self.tools:
            return None
        return max((t.effect for t in self.tools), key=EFFECT_SEVERITY.index)


# --------------------------------------------------------------------------
# Reading the spec
# --------------------------------------------------------------------------

# Subject kinds this block is for. A prompt or a classifier has no trajectory,
# so a trajectory suite would be measuring nothing.
AGENTIC_KINDS = frozenset({SubjectKind.AGENT, SubjectKind.WORKFLOW})

# An interface sentence, matched in this order: the first hit wins, so the more
# specific patterns come first. HTTP before subprocess, because "POST /run" and
# "run the CLI" both contain a verb and only one of them is a shell.
RUN_MODE_HINTS: List[Tuple[RunMode, Sequence[str]]] = [
    (
        RunMode.HTTP,
        (
            "http://",
            "https://",
            "post ",
            "get ",
            "endpoint",
            "rest api",
            "webhook",
            "/v1/",
            "/api/",
        ),
    ),
    (
        RunMode.PYTHON,
        (
            "import ",
            "python function",
            "function call",
            "sdk",
            "def ",
            "()",
            "callable",
        ),
    ),
    (
        RunMode.SUBPROCESS,
        (
            "cli",
            "command line",
            "command-line",
            "terminal",
            "$ ",
            "npx ",
            "npm run",
            "python -m",
            "./",
            "binary",
            "shell command",
        ),
    ),
]

# Words in the spec that imply a tool. Each maps to the tool we would stub and
# what it does to the world. Nothing here is a claim that the agent *has* the
# tool - it is a question the profile asks in a form the user can correct.
TOOL_HINTS: List[Tuple[str, str, Effect, bool]] = [
    ("refund", "payments", Effect.DESTRUCTIVE, True),
    ("payment", "payments", Effect.DESTRUCTIVE, True),
    ("charge", "payments", Effect.DESTRUCTIVE, True),
    ("transfer", "payments", Effect.DESTRUCTIVE, True),
    ("delete", "deletion", Effect.DESTRUCTIVE, True),
    ("cancel", "cancellation", Effect.DESTRUCTIVE, True),
    ("shell", "shell", Effect.DESTRUCTIVE, False),
    ("bash", "shell", Effect.DESTRUCTIVE, False),
    ("email", "email", Effect.EXTERNAL, True),
    ("mail", "email", Effect.EXTERNAL, True),
    ("slack", "messaging", Effect.EXTERNAL, True),
    ("sms", "messaging", Effect.EXTERNAL, True),
    ("notify", "messaging", Effect.EXTERNAL, True),
    # Not "message": it is the word for what a support agent *receives*, and as a
    # hint it gave every inbox-shaped agent an outbound messaging tool it may not
    # have. A tool we invent is a guardrail that fails runs for no reason.
    ("web search", "web search", Effect.EXTERNAL, True),
    ("browse", "browser", Effect.EXTERNAL, True),
    ("browser", "browser", Effect.EXTERNAL, True),
    ("scrape", "browser", Effect.EXTERNAL, True),
    ("third party", "external api", Effect.EXTERNAL, True),
    ("third-party", "external api", Effect.EXTERNAL, True),
    ("calendar", "calendar", Effect.WRITE, True),
    ("booking", "booking", Effect.WRITE, True),
    ("ticket", "ticketing", Effect.WRITE, True),
    ("crm", "crm", Effect.WRITE, True),
    ("jira", "ticketing", Effect.WRITE, True),
    ("zendesk", "ticketing", Effect.WRITE, True),
    ("commit", "version control", Effect.WRITE, True),
    ("pull request", "version control", Effect.WRITE, True),
    ("repository", "version control", Effect.WRITE, True),
    ("repo", "version control", Effect.WRITE, True),
    ("file", "filesystem", Effect.WRITE, True),
    ("write", "filesystem", Effect.WRITE, True),
    ("update", "database", Effect.WRITE, True),
    ("insert", "database", Effect.WRITE, True),
    ("database", "database", Effect.READ, True),
    ("sql", "database", Effect.READ, True),
    ("query", "database", Effect.READ, True),
    ("lookup", "lookup", Effect.READ, True),
    ("retrieve", "retrieval", Effect.READ, True),
    ("search", "retrieval", Effect.READ, True),
    ("knowledge base", "retrieval", Effect.READ, True),
    ("documentation", "retrieval", Effect.READ, True),
]

ISOLATION_HINTS: List[Tuple[Isolation, Sequence[str]]] = [
    (
        Isolation.CONTAINER,
        ("container", "docker", "sandbox", "sandboxed", "vm", "isolated environment"),
    ),
    (
        Isolation.TMPDIR,
        (
            "staging",
            "test account",
            "sandbox account",
            "scratch",
            "temporary directory",
            "tmpdir",
            "fixture database",
            "dry run",
            "dry-run",
        ),
    ),
]

DETERMINISM_HINTS = (
    "deterministic",
    "temperature 0",
    "temperature=0",
    "greedy",
    "seeded",
    "fixed seed",
)

# What terminal looks like when the user did not say. Each is a guess the
# profile records, not a fact about their system.
EPISODE_END_FOR_MODE = {
    RunMode.SUBPROCESS: "the process exits; its exit code and the files it left are the result",
    RunMode.HTTP: "the run reaches a terminal status and stops being polled",
    RunMode.PYTHON: "the callable returns",
    RunMode.TRANSCRIPT: "the recorded trace ends",
    RunMode.UNKNOWN: "not known - the spec does not say how a run finishes",
}

# A run with no ceiling never fails, it hangs, and a hung case is an unscored
# case. These are deliberately generous: they are a stop, not a target.
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_SECONDS = 300

# Runs per case when the agent is stochastic. One run is a number with no error
# bar; three is the cheapest thing that has one.
DEFAULT_SAMPLES = 3

SECOND_UNITS = ("s", "sec", "secs", "second", "seconds")
MS_UNITS = ("ms", "millisecond", "milliseconds")
USD_UNITS = ("usd", "$", "dollar", "dollars", "cent", "cents")
TOKEN_UNITS = ("token", "tokens")
STEP_WORDS = (
    "step",
    "steps",
    "turn",
    "turns",
    "tool call",
    "tool calls",
    "iteration",
    "iterations",
)


def _text_of(spec: TaskSpec) -> str:
    """Every part of the spec a hint may legitimately be read off."""
    parts = [
        spec.summary,
        spec.subject.description,
        spec.subject.interface or "",
        spec.subject.inputs or "",
        spec.subject.outputs or "",
        *spec.subject.in_scope,
        *spec.constraints,
    ]
    return re.sub(r"\s+", " ", " ".join(parts).lower())


def _run_mode(spec: TaskSpec) -> RunMode:
    interface = (spec.subject.interface or "").lower()
    if not interface.strip():
        return RunMode.UNKNOWN
    for mode, hints in RUN_MODE_HINTS:
        if any(hint in interface for hint in hints):
            return mode
    # Something was said about invocation but it matched nothing known. A
    # sentence we cannot classify is still a sentence a person can act on, so
    # this is not UNKNOWN - it is a subprocess until someone says otherwise.
    return RunMode.SUBPROCESS


def _has_traces(spec: TaskSpec) -> bool:
    """Recorded runs we actually hold - the only thing that makes offline scoring possible."""
    return any(
        item.kind is EvidenceKind.SUCCESSFUL_RUN
        and item.status is EvidenceStatus.PROVIDED
        for item in spec.evidence
    )


# "updates the database" and "issues refunds" are the same hints as "update" and
# "refund"; without the suffixes half the tool list is missed on ordinary prose.
INFLECTIONS = "(?:s|es|d|ed|ing)?"


def _tools(text: str) -> List[ToolFacet]:
    """Tools implied by the wording, worst effect per tool, in severity order."""
    found: dict = {}
    for keyword, name, effect, stubbable in TOOL_HINTS:
        if not re.search(rf"\b{re.escape(keyword)}{INFLECTIONS}\b", text):
            continue
        current = found.get(name)
        if current is None or EFFECT_SEVERITY.index(effect) > EFFECT_SEVERITY.index(
            current.effect
        ):
            found[name] = ToolFacet(
                name=name, effect=effect, stubbable=stubbable, evidence=keyword
            )
    return sorted(
        found.values(),
        key=lambda t: (-EFFECT_SEVERITY.index(t.effect), t.name),
    )


def _isolation(text: str) -> Isolation:
    for level, hints in ISOLATION_HINTS:
        if any(hint in text for hint in hints):
            return level
    return Isolation.UNKNOWN


def _number(text: str) -> Optional[float]:
    found = re.search(r"(\d[\d_,]*\.?\d*)", text.replace(",", ""))
    return float(found.group(1)) if found else None


def _budget(spec: TaskSpec) -> Tuple[Budget, List[str]]:
    """Ceilings the KPIs already imply; a default stop for whatever is left."""
    budget = Budget()
    notes: List[str] = []

    for kpi in spec.kpis:
        source = f"{kpi.target or ''} {kpi.unit or ''}".strip().lower()
        value = _number(source) if kpi.target else None
        if value is None:
            continue
        unit = (kpi.unit or "").strip().lower()
        wording = f"{kpi.name} {kpi.definition}".lower()

        if unit in MS_UNITS and budget.max_seconds is None:
            # A latency target is not a timeout - a run at twice the target is
            # slow, not hung - so the ceiling is the target with headroom.
            budget.max_seconds = max(1, int(value / 1000 * 3))
            notes.append(f"Timeout set to 3x the '{kpi.name}' target.")
        elif unit in SECOND_UNITS and budget.max_seconds is None:
            budget.max_seconds = max(1, int(value * 3))
            notes.append(f"Timeout set to 3x the '{kpi.name}' target.")
        elif (unit in USD_UNITS or "$" in source) and budget.max_usd is None:
            budget.max_usd = value * 3
            notes.append(f"Spend ceiling set to 3x the '{kpi.name}' target.")
        elif unit in TOKEN_UNITS and budget.max_tokens is None:
            budget.max_tokens = int(value * 3)
        elif any(word in wording for word in STEP_WORDS) and budget.max_steps is None:
            budget.max_steps = int(value * 3)
            notes.append(f"Step ceiling set to 3x the '{kpi.name}' target.")

    if budget.max_steps is None:
        budget.max_steps = DEFAULT_MAX_STEPS
        notes.append(
            f"No step limit stated; capped at {DEFAULT_MAX_STEPS} so a looping run fails instead of hanging."
        )
    if budget.max_seconds is None:
        budget.max_seconds = DEFAULT_MAX_SECONDS
        notes.append(
            f"No time limit stated; capped at {DEFAULT_MAX_SECONDS}s for the same reason."
        )
    return budget, notes


def profile(spec: TaskSpec) -> AgentProfile:
    """Read the operational facts about the agent off the spec.

    Deterministic and offline, like `auto_eval.derive`: same spec, same profile.
    Everything not stated outright is recorded in `assumptions`, so a profile is
    a draft the user corrects rather than a claim about their system.
    """
    text = _text_of(spec)
    assumptions: List[str] = []

    mode = _run_mode(spec)
    if mode is RunMode.UNKNOWN and _has_traces(spec):
        # We cannot start a run, but we hold runs that already happened. Rubric
        # and trajectory checks still work on those, so the suite is offline
        # rather than impossible.
        mode = RunMode.TRANSCRIPT
        assumptions.append(
            "No entry point was given, but successful runs were, so the suite scores recorded traces instead of starting new ones."
        )
    elif mode is not RunMode.UNKNOWN and not spec.subject.interface:
        assumptions.append(f"Read the run mode as {mode.value} from the request.")

    tools = _tools(text)
    if tools:
        assumptions.append(
            "Tools read off the wording of the request: "
            + ", ".join(f"{t.name} ({t.effect.value})" for t in tools)
            + ". Correct the list - a tool we do not know about is a tool the suite cannot check."
        )

    isolation = _isolation(text)
    budget, budget_notes = _budget(spec)
    assumptions.extend(budget_notes)

    stochastic = not any(hint in text for hint in DETERMINISM_HINTS)
    samples = DEFAULT_SAMPLES if stochastic else 1
    if stochastic:
        assumptions.append(
            f"Treated as stochastic, so every case runs {samples} times and the report carries the spread."
        )

    return AgentProfile(
        subject=spec.subject.name or "the agent",
        run_mode=mode,
        entrypoint=spec.subject.interface,
        episode_end=EPISODE_END_FOR_MODE[mode],
        tools=tools,
        isolation=isolation,
        budget=budget,
        stochastic=stochastic,
        samples=samples,
        assumptions=assumptions,
    )


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def agent_gate(spec: TaskSpec, agent_profile: Optional[AgentProfile] = None) -> Gate:
    """Whether a suite may be authored for this agent.

    Blocking is kept to what no amount of rule-writing can recover, the same
    standard `auto_eval.gaps` holds: we cannot start a run and have none
    recorded, or the agent can destroy something and we have nowhere safe to let
    it try. Everything else - an unknown tool list, no stated budget - is a
    guess the profile records and the user corrects.
    """
    upstream = spec_gate(spec)
    if not upstream.open:
        return upstream

    agent_profile = agent_profile or profile(spec)
    blocking: List[Question] = []

    if spec.subject.kind not in AGENTIC_KINDS:
        return Gate(
            open=False,
            reason=(
                f"The subject is a {spec.subject.kind.value}, not an agent or a workflow. "
                "A trajectory suite would be measuring something that has no trajectory; "
                "use `auto-eval benchmarks` and the ground-truth search instead."
            ),
        )

    if agent_profile.run_mode is RunMode.UNKNOWN:
        blocking.append(
            Question(
                field="subject.interface",
                question=(
                    f"How is one run of {agent_profile.subject} started and how do we know it has finished - "
                    "a command, an HTTP endpoint, a function? Failing that, can you share recorded runs to score instead?"
                ),
                why="Without an entry point or recorded traces there is nothing to run and nothing to score.",
                blocking=True,
            )
        )

    unsafe = agent_profile.tools_with(*CONTAINED_EFFECTS)
    # A transcript suite is scored offline against runs that already happened,
    # so there is nothing live to contain and nothing to be careful with.
    if (
        unsafe
        and agent_profile.isolation not in SAFE_ISOLATION
        and agent_profile.run_mode is not RunMode.TRANSCRIPT
    ):
        blocking.append(
            Question(
                field="agent.isolation",
                question=(
                    "Where can these runs happen safely - a sandbox, a container, or test credentials? "
                    f"The agent looks able to {', '.join(t.name for t in unsafe)}, and an eval that "
                    "exercises that against the real system is an incident, not a test."
                ),
                why="Cases that exercise a destructive or external tool cannot be written until there is somewhere safe to run them.",
                blocking=True,
            )
        )

    if blocking:
        return Gate(
            open=False,
            reason=(
                f"{len(blocking)} blocking question(s) about how to run it: "
                + "; ".join(q.question for q in blocking)
            ),
            blocking=blocking,
        )
    return Gate(
        open=True,
        reason=f"Runnable as {agent_profile.run_mode.value}, with somewhere safe to run it.",
    )


__all__ = [
    "AGENTIC_KINDS",
    "CONTAINED_EFFECTS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SAMPLES",
    "EFFECT_SEVERITY",
    "SAFE_ISOLATION",
    "AgentProfile",
    "Budget",
    "Effect",
    "Isolation",
    "RunMode",
    "ToolFacet",
    "agent_gate",
    "profile",
]

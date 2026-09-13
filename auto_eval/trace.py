"""What one run of one case left behind.

A suite says what to test; this is the record of what happened when it was
tested. It exists as its own artefact because every check in `auto_eval.grade`
reads it rather than the provider's response object: a verdict that cannot be
re-derived from a stored trace is a verdict nobody can audit six months later.

Three things are recorded that a plain transcript does not carry:

* **every tool call and the answer it got**, with the scripted behaviour that
  produced that answer - an error the harness injected on purpose looks nothing
  like an error the provider returned, and a trajectory check needs to tell them
  apart;
* **why the run stopped**, separately from what it last said. A run that was cut
  off at the step ceiling and a run that finished have the same final message
  shape and mean opposite things;
* **what it cost**, in tokens and seconds. Spend is left `None` unless prices
  were supplied, because a made-up price is worse than no price.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field


class StopReason(str, Enum):
    """Why the loop ended. Only the first of these is the agent's own doing."""

    TERMINAL = "terminal"  # it answered with no further tool call
    STEP_CEILING = "step_ceiling"  # cut off: still working at the last step
    TIME_CEILING = "time_ceiling"
    TOKEN_CEILING = "token_ceiling"
    PROVIDER_ERROR = "provider_error"
    REFUSED = "refused"  # the model declined the task


# Stop reasons where the run never reached an end state of its own, so anything
# read off the tail of the transcript is a fragment of work in progress.
CUT_OFF = frozenset(
    {
        StopReason.STEP_CEILING,
        StopReason.TIME_CEILING,
        StopReason.TOKEN_CEILING,
        StopReason.PROVIDER_ERROR,
    }
)

# Stub behaviours that hand the agent something broken. `auto_eval.grade` looks
# for one of these before it will claim the run recovered from anything.
FAILING_BEHAVIOURS = frozenset({"error", "timeout", "empty", "malformed"})


class ToolCall(BaseModel):
    """One tool call, and the answer the harness gave it."""

    step: int
    name: str
    arguments: str = Field(default="", description="Verbatim, as the model sent it.")
    result: str = Field(default="", description="What was handed back.")
    behaviour: str = Field(
        default="ok",
        description="The scripted behaviour this answer came from: ok, error, injection, ...",
    )
    scripted: bool = Field(
        default=False,
        description="True when the suite specified this answer; false when the harness improvised an 'ok'.",
    )
    seconds: float = 0.0


class Step(BaseModel):
    """One turn of the loop: what the model said, and what it reached for."""

    index: int
    text: str = Field(default="", description="The assistant's message, if any.")
    tool_calls: List[ToolCall] = Field(default_factory=list)
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """What the run cost. `usd` stays None unless prices were supplied."""

    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    usd: Optional[float] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def plus(self, other: "Usage") -> "Usage":
        both = (self.usd, other.usd)
        return Usage(
            steps=self.steps + other.steps,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            seconds=round(self.seconds + other.seconds, 3),
            usd=None if all(v is None for v in both) else sum(v or 0.0 for v in both),
        )


class Trace(BaseModel):
    """One run of one case, whole enough to grade and to re-read later."""

    case_id: str
    sample: int = 0
    model: str
    mock: bool = Field(
        default=False, description="True when no provider was called at all."
    )
    system: str = Field(default="", description="The system prompt as sent.")
    instruction: str = Field(default="", description="The case instruction as sent.")
    steps: List[Step] = Field(default_factory=list)
    stop_reason: StopReason = StopReason.TERMINAL
    final_message: str = ""
    usage: Usage = Field(default_factory=Usage)
    error: Optional[str] = Field(
        default=None, description="The provider error, when the run did not complete."
    )
    started_at: str = Field(default_factory=lambda: _now())
    notes: List[str] = Field(default_factory=list)

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [call for step in self.steps for call in step.tool_calls]

    @property
    def terminated(self) -> bool:
        return self.stop_reason is StopReason.TERMINAL

    @property
    def cut_off(self) -> bool:
        return self.stop_reason in CUT_OFF

    def calls_of(self, tool: str) -> List[ToolCall]:
        """Every call to `tool`, matched loosely - a provider may rename it."""
        wanted = _norm(tool)
        return [call for call in self.tool_calls if _norm(call.name) == wanted]

    def called(self, tool: str) -> bool:
        return bool(self.calls_of(tool))

    def calls_after(self, step: int) -> List[ToolCall]:
        return [call for call in self.tool_calls if call.step > step]

    def first_call_with(self, behaviour: str) -> Optional[ToolCall]:
        for call in self.tool_calls:
            if call.behaviour == behaviour:
                return call
        return None

    @property
    def failing_calls(self) -> List[ToolCall]:
        return [c for c in self.tool_calls if c.behaviour in FAILING_BEHAVIOURS]

    def transcript(self) -> str:
        """The run as text, which is what a judge is shown and quotes from."""
        lines = [f"TASK: {self.instruction}"]
        for step in self.steps:
            if step.text:
                lines.append(f"[step {step.index}] assistant: {step.text}")
            for call in step.tool_calls:
                lines.append(
                    f"[step {step.index}] tool call {call.name}({call.arguments})"
                )
                lines.append(
                    f"[step {step.index}] tool result ({call.behaviour}): {call.result}"
                )
        lines.append(f"STOPPED: {self.stop_reason.value}")
        return "\n".join(lines)


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def price(usage: Usage, *, per_million_in: float, per_million_out: float) -> float:
    """Spend for one run, from prices the caller supplied. Never guessed."""
    return round(
        usage.input_tokens * per_million_in / 1_000_000
        + usage.output_tokens * per_million_out / 1_000_000,
        6,
    )


__all__ = [
    "CUT_OFF",
    "FAILING_BEHAVIOURS",
    "Step",
    "StopReason",
    "ToolCall",
    "Trace",
    "Usage",
    "price",
]

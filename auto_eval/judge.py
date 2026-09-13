"""The judge: a model ruling on one rubric, and the two rules it is held to.

A rubric check cannot be settled by a machine, so it is settled by a model, and
the whole difficulty is that a model asked "did this pass?" will say yes
agreeably. Two rules make the answer reviewable:

1. **Quote or fail.** The ruling must carry a verbatim span from the transcript.
   `auto_eval.verify.accept` already voids a pass with an empty quote; this
   module goes further and voids a quote that is not actually in the transcript,
   which is the failure mode that matters - a fabricated quote reads exactly
   like a real one in a report.
2. **One criterion at a time.** The rubric's lines are put to the judge as
   questions to answer individually, not as a score out of five. A judge asked
   for a number returns its taste; a judge asked a question returns a span.

The judge model is named in the run report next to the model under test. When
they are the same model, the report says so: MT-Bench documented
self-preference bias in exactly this setup, and a number produced by a model
grading itself should be read with that in mind.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field

from .authoring import Case
from .trace import Trace
from .verify import Outcome, Verifier, accept

# The transcript handed to the judge. Long enough to hold a full run of a small
# case; a longer one is trimmed from the middle, where the least is happening.
MAX_TRANSCRIPT_CHARS = 24_000

SYSTEM_PROMPT = """You grade one criterion about one run of an agent, from the transcript of that run.

Rules you are held to:
- Decide only the criterion you are given. Not whether the run was good overall.
- Quote verbatim from the transcript to support your decision. Copy the characters exactly; do not paraphrase, summarise, or reconstruct.
- If the transcript does not contain what you would need in order to pass the criterion, fail it. A missing thing is not a pass.
- A run that asserted it did something, without the transcript showing it, has not done it.
- Answer each numbered question in the criterion before deciding.

Return a decision, the verbatim quote it rests on, and one sentence of reasoning."""


class Ruling(BaseModel):
    """What the judge returns. `quote` is checked against the transcript."""

    passed: bool = Field(description="Whether the criterion is met.")
    quote: str = Field(
        description="Verbatim span from the transcript that the decision rests on."
    )
    reason: str = Field(description="One sentence. Why that span settles it.")
    step: Optional[int] = Field(
        default=None, description="Step number the quote came from, if visible."
    )


def _trim(text: str, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit // 2, limit // 2
    return (
        text[:head]
        + f"\n\n[... {len(text) - limit} characters of the middle of this run omitted ...]\n\n"
        + text[-tail:]
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quoted_from(transcript: str, quote: str) -> bool:
    """Whether the judge's quote really is in the transcript.

    Whitespace-insensitive, because a model will reflow a line it copied
    correctly, and that is not the failure this is looking for.
    """
    cleaned = _normalise(quote)
    if len(cleaned) < 8:  # too short to be evidence of anything
        return False
    return cleaned in _normalise(transcript)


def build_message(verifier: Verifier, case: Case, trace: Trace) -> str:
    lines = [
        f"CRITERION: {verifier.description}",
    ]
    if verifier.kpi:
        lines.append(f"This criterion is how '{verifier.kpi}' is scored.")
    if verifier.rubric:
        lines.append("\nQuestions to answer before deciding:")
        lines.extend(f"{i}. {line}" for i, line in enumerate(verifier.rubric, 1))
    lines.append(f"\nWhat the agent was asked to do: {case.instruction}")
    if case.fixture.reference:
        lines.append(f"Starting state as specified: {case.fixture.reference}")
    lines.append(
        "\nNote: every tool result in this transcript was produced by the test harness, "
        "not by a live system. Judge what the agent did with them."
    )
    lines.append("\n--- TRANSCRIPT ---")
    lines.append(_trim(trace.transcript()))
    lines.append("--- END TRANSCRIPT ---")
    return "\n".join(lines)


def rule(
    verifier: Verifier,
    case: Case,
    trace: Trace,
    *,
    client: Any,
    model: str,
    max_tokens: int = 2000,
) -> Outcome:
    """Put one rubric to the judge and return the verdict it can support.

    Never raises: a judge that is unreachable is a failed check with the reason
    attached, not a lost run.
    """
    transcript = trace.transcript()
    try:
        completion = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_message(verifier, case, trace)},
            ],
            response_format=Ruling,
            max_completion_tokens=max_tokens,
        )
    except Exception as exc:
        return Outcome(
            verifier=verifier.id,
            passed=False,
            evidence="",
            note=f"The judge could not be reached, so this check did not settle: {exc}",
        )

    choice = completion.choices[0]
    ruling: Optional[Ruling] = getattr(choice.message, "parsed", None)
    if ruling is None:
        return Outcome(
            verifier=verifier.id,
            passed=False,
            evidence="",
            note=(
                "The judge returned nothing parsable "
                f"(finish_reason={getattr(choice, 'finish_reason', None)!r})."
            ),
        )

    # An empty quote is `accept`'s business below; this is about a quote that
    # was offered and is not there.
    if (
        ruling.passed
        and ruling.quote.strip()
        and not quoted_from(transcript, ruling.quote)
    ):
        return Outcome(
            verifier=verifier.id,
            passed=False,
            evidence="",
            step=ruling.step,
            note=(
                "The judge passed this on a quote that is not in the transcript, so the pass "
                f"could not be checked and was voided. Claimed quote: {ruling.quote[:160]!r}"
            ),
        )

    # accept() has the last word: a pass with nothing quoted is not a pass.
    return accept(
        verifier,
        Outcome(
            verifier=verifier.id,
            passed=ruling.passed,
            evidence=ruling.quote.strip(),
            step=ruling.step,
            note=ruling.reason,
        ),
    )


def judge_with(
    client: Any, model: str, *, max_tokens: int = 2000
) -> Callable[[Verifier, Case, Trace], Outcome]:
    """A `grade.Judge` bound to one client and model."""

    def judge(verifier: Verifier, case: Case, trace: Trace) -> Outcome:
        return rule(
            verifier, case, trace, client=client, model=model, max_tokens=max_tokens
        )

    return judge


def rubric_lines(verifiers: List[Verifier]) -> int:
    return sum(len(v.rubric) for v in verifiers)


__all__ = [
    "MAX_TRANSCRIPT_CHARS",
    "SYSTEM_PROMPT",
    "Ruling",
    "build_message",
    "judge_with",
    "quoted_from",
    "rule",
]

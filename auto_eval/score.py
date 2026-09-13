"""Deciding whether one system got one item right.

Two shapes of answer, two ways of settling them, and one rule over both.

* **Results came back** - a list of hits from a retrieval endpoint. The item is
  answered if a gold answer appears in one of them, and *which* hit it appeared
  in is the interesting part: answer recall at 1 is a different product from
  answer recall at 5, and a board that reports only "found it somewhere" hides
  the difference.
* **A reply came back** - one string from a model. The item is answered if a
  gold answer appears in the reply, or, for an item with no gold answer, if the
  judge says so and can quote the line it read that off.

The rule over both: a verdict carries the span it was read from. A scorer that
returns `True` with nothing to show cannot be reviewed, and on a board where a
vendor's ranking rests on three percentage points, unreviewable is useless.

Matching is deliberately forgiving about form and strict about content: case,
punctuation, articles and thousands separators are normalised away, because
"1,200 employees" and "1200 employees" are the same answer and a scorer that
says otherwise is manufacturing a difference between vendors.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from .endpoint import Hit

# Words dropped before matching. Short and closed on purpose: this is about
# spelling, not meaning, and a long list starts matching things it should not.
FILLER = frozenset("the a an of and to in on at for is are was were".split())

# How much of the hit a verdict quotes back.
EVIDENCE_CHARS = 200


class Verdict(BaseModel):
    """One item, one system, one answer."""

    item_id: str
    correct: bool = False
    rank: Optional[int] = Field(
        default=None, description="Which result held the answer. 1 is the top hit."
    )
    evidence: str = Field(default="", description="The span the verdict was read off.")
    seconds: float = 0.0
    error: Optional[str] = None
    note: Optional[str] = None
    judged: bool = Field(
        default=False, description="True when a model settled this rather than a match."
    )


def normalise(text: str) -> str:
    """Case, accents, punctuation and filler words out; content left alone."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    folded = folded.lower().replace(",", "")
    folded = re.sub(r"[^a-z0-9%$.\- ]+", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    return " ".join(word for word in folded.split() if word not in FILLER)


def contains(haystack: str, answer: str) -> bool:
    """Whether `answer` appears in `haystack`, once both are normalised."""
    needle = normalise(answer)
    return bool(needle) and needle in normalise(haystack)


def matches(text: str, answers: Sequence[str]) -> Optional[str]:
    """The first gold answer present in `text`, or None."""
    return next((answer for answer in answers if contains(text, answer)), None)


def _quote(text: str, answer: str) -> str:
    """The part of the hit the answer was found in, so a verdict can be checked."""
    flat = " ".join(text.split())
    position = normalise(flat).find(normalise(answer))
    if position < 0 or len(flat) <= EVIDENCE_CHARS:
        return flat[:EVIDENCE_CHARS]
    # Positions shift under normalisation, so this is approximate on purpose:
    # it is a window to read, not an offset to trust.
    start = max(0, min(position, len(flat) - EVIDENCE_CHARS))
    return ("…" if start else "") + flat[start : start + EVIDENCE_CHARS] + "…"


def score_hits(
    item_id: str,
    answers: Sequence[str],
    hits: Sequence[Hit],
    *,
    seconds: float = 0.0,
    error: Optional[str] = None,
) -> Verdict:
    """Score what an endpoint returned: is the answer in the results, and where."""
    if error:
        return Verdict(item_id=item_id, correct=False, seconds=seconds, error=error)
    if not answers:
        return Verdict(
            item_id=item_id,
            correct=False,
            seconds=seconds,
            note="No gold answer on this item, so results cannot be matched - it needs a judge.",
        )
    for hit in hits:
        found = matches(hit.text(), answers)
        if found:
            return Verdict(
                item_id=item_id,
                correct=True,
                rank=hit.rank,
                seconds=seconds,
                evidence=f"result {hit.rank}: {_quote(hit.text(), found)}",
            )
    return Verdict(
        item_id=item_id,
        correct=False,
        seconds=seconds,
        evidence=f"none of {len(hits)} result(s) contained the answer",
    )


def score_reply(
    item_id: str,
    answers: Sequence[str],
    reply: str,
    *,
    seconds: float = 0.0,
    error: Optional[str] = None,
) -> Verdict:
    """Score what a model replied."""
    if error:
        return Verdict(item_id=item_id, correct=False, seconds=seconds, error=error)
    found = matches(reply, answers) if answers else None
    if found:
        return Verdict(
            item_id=item_id,
            correct=True,
            seconds=seconds,
            evidence=_quote(reply, found),
        )
    return Verdict(
        item_id=item_id,
        correct=False,
        seconds=seconds,
        evidence=" ".join(reply.split())[:EVIDENCE_CHARS],
        note=None
        if answers
        else "No gold answer on this item; a judge has to settle it.",
    )


class Summary(BaseModel):
    """What a column of verdicts adds up to."""

    n: int = 0
    correct: int = 0
    errors: int = 0
    at_1: int = Field(default=0, description="Answered by the top result.")
    at_5: int = Field(default=0, description="Answered within the first five.")
    judged: int = 0

    @property
    def scored(self) -> int:
        """Items that produced an answer either way. Errors are not failures."""
        return self.n - self.errors

    @property
    def accuracy(self) -> Optional[float]:
        return self.correct / self.scored if self.scored else None

    @property
    def recall_at_1(self) -> Optional[float]:
        return self.at_1 / self.scored if self.scored else None

    @property
    def recall_at_5(self) -> Optional[float]:
        return self.at_5 / self.scored if self.scored else None


def summarise(verdicts: Sequence[Verdict]) -> Summary:
    summary = Summary(n=len(verdicts))
    for verdict in verdicts:
        if verdict.error:
            summary.errors += 1
            continue
        summary.judged += verdict.judged
        if not verdict.correct:
            continue
        summary.correct += 1
        if verdict.rank == 1:
            summary.at_1 += 1
        if verdict.rank is not None and verdict.rank <= 5:
            summary.at_5 += 1
    return summary


def percentile(values: Sequence[float], share: float) -> Optional[float]:
    """Nearest-rank percentile. Small n, so no interpolation to argue about."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(share * (len(ordered) - 1))))
    return ordered[index]


__all__ = [
    "EVIDENCE_CHARS",
    "FILLER",
    "Summary",
    "Verdict",
    "contains",
    "matches",
    "normalise",
    "percentile",
    "score_hits",
    "score_reply",
    "summarise",
]

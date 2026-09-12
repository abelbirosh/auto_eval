"""Deterministic gap analysis over a classified task spec.

The model is good at reading a request and bad at reliably noticing what is
missing from it. So the questions that decide whether we can build an eval are
generated here, from rules, and merged with whatever the model asked. Readiness
is computed here too - the model's own guess is discarded.
"""

from __future__ import annotations

import re
from typing import List

from .schema import (
    EvidenceKind,
    EvidenceStatus,
    Measurement,
    MetricKind,
    Priority,
    Question,
    Readiness,
    TaskSpec,
)

# Asking the same thing five times is worse than asking once; cap the per-KPI
# rules so a spec with twenty metrics does not produce twenty questions.
MAX_PER_KPI_QUESTIONS = 3


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# Two questions count as the same ask when most of the shorter one's words
# appear in the longer one. Plain substring matching was tried first and is
# wrong: it lets a long question swallow an unrelated short one.
SIMILARITY_THRESHOLD = 0.8
MIN_TOKENS_FOR_FUZZY_MATCH = 4

# Question phrasing is mostly scaffolding ("is there a", "can you share"); left
# in, it makes any two questions look alike, so it is dropped before comparing.
STOPWORDS = frozenset(
    """a an and any are as at be been but by can can't could did do does for from
    give giving had has have how i if in into is it its me my of on one or our out
    please should show so some that the their them then there these they this those
    to two three us was we were what when where which who why will with would you
    your""".split()
)


def _significant_tokens(text: str) -> set:
    return {word for word in _norm(text).split() if word not in STOPWORDS}


def _is_duplicate(question: Question, existing: List[Question]) -> bool:
    candidate = _norm(question.question)
    if not candidate:
        return True
    candidate_tokens = _significant_tokens(question.question)
    for other in existing:
        if candidate == _norm(other.question):
            return True
        seen_tokens = _significant_tokens(other.question)
        smaller = min(len(candidate_tokens), len(seen_tokens))
        if smaller < MIN_TOKENS_FOR_FUZZY_MATCH:
            continue
        overlap = len(candidate_tokens & seen_tokens) / smaller
        if overlap >= SIMILARITY_THRESHOLD:
            return True
    return False


def rule_questions(spec: TaskSpec) -> List[Question]:
    """Questions implied by what the spec is missing, independent of the model."""
    out: List[Question] = []

    # --- what are we testing -------------------------------------------------
    if not spec.subject.interface:
        out.append(
            Question(
                field="subject.interface",
                question=(
                    f"How is {spec.subject.name or 'the system under test'} invoked - "
                    "a CLI command, an HTTP endpoint, a function, a prompt template?"
                ),
                why="The harness cannot call the system under test without an entry point.",
                blocking=True,
            )
        )
    if not spec.subject.in_scope and not spec.subject.out_of_scope:
        out.append(
            Question(
                field="subject.in_scope",
                question="Which behaviours should this evaluation cover, and which are explicitly out of scope?",
                why="Without a boundary the eval either misses regressions or fails on behaviour you never promised.",
                blocking=False,
            )
        )

    # --- how success is judged ----------------------------------------------
    if not spec.kpis:
        out.append(
            Question(
                field="kpis",
                question="What would make this system good enough to ship - which metric or quality decides that?",
                why="With no KPI there is nothing to score, so no eval can be built.",
                blocking=True,
            )
        )
    else:
        if not any(k.priority is Priority.PRIMARY for k in spec.kpis):
            out.append(
                Question(
                    field="kpis.priority",
                    question="Which single KPI does this task succeed or fail on if the others are only mediocre?",
                    why="Without a primary KPI, improvements in one metric cannot be traded off against another.",
                    blocking=False,
                )
            )

        unmeasurable = [k for k in spec.kpis if k.measurement is Measurement.UNKNOWN]
        for kpi in unmeasurable[:MAX_PER_KPI_QUESTIONS]:
            out.append(
                Question(
                    field="kpis.measurement",
                    question=f"How should '{kpi.name}' be measured - a programmatic check, an LLM judge, or human review?",
                    why="A KPI with no measurement method cannot be turned into a grader.",
                    blocking=False,
                )
            )

        untargeted = [
            k
            for k in spec.kpis
            if k.target is None and k.priority is not Priority.GUARDRAIL
        ]
        for kpi in untargeted[:MAX_PER_KPI_QUESTIONS]:
            out.append(
                Question(
                    field="kpis.target",
                    question=f"What value of '{kpi.name}' counts as passing? A threshold, or the current baseline to beat.",
                    why="Without a threshold the eval reports a number but cannot return a verdict.",
                    blocking=False,
                )
            )

        judged = [
            k
            for k in spec.kpis
            if k.kind is MetricKind.QUALITATIVE
            and k.measurement in (Measurement.LLM_JUDGE, Measurement.HUMAN_REVIEW)
        ]
        for kpi in judged[:MAX_PER_KPI_QUESTIONS]:
            out.append(
                Question(
                    field="kpis.definition",
                    question=f"For '{kpi.name}', can you give one output you would rate good and one you would rate bad?",
                    why="A judged quality needs anchoring examples, or the rubric drifts to the grader's taste.",
                    blocking=False,
                )
            )

    # --- documentation and successful runs -----------------------------------
    usable = {
        item.kind
        for item in spec.evidence
        if item.status is not EvidenceStatus.ABSENT
    }

    if EvidenceKind.DOCUMENTATION not in usable and EvidenceKind.SPEC not in usable:
        out.append(
            Question(
                field="evidence.documentation",
                question="Is there documentation or a spec for this system - a README, a design doc, acceptance criteria?",
                why="Documentation is what separates testing the intended behaviour from testing the current behaviour.",
                blocking=False,
            )
        )

    if EvidenceKind.SUCCESSFUL_RUN not in usable:
        out.append(
            Question(
                field="evidence.successful_run",
                question="Can you share two or three examples of runs that came out the way you wanted - inputs and the outputs you were happy with?",
                why="Known-good runs anchor the grading rubric and become the first eval cases.",
                blocking=False,
            )
        )

    if EvidenceKind.FAILURE_CASE not in usable:
        out.append(
            Question(
                field="evidence.failure_case",
                question="What does failure look like here - any specific bad outputs or incidents you remember?",
                why="Failure cases are the highest-signal eval items and usually reveal an unstated KPI.",
                blocking=False,
            )
        )

    # Something referenced but never handed over is a fetch, not an unknown.
    pending = [
        item
        for item in spec.evidence
        if item.status is EvidenceStatus.MENTIONED_NOT_PROVIDED
    ]
    for item in pending[:MAX_PER_KPI_QUESTIONS]:
        where = f" ({item.location})" if item.location else ""
        out.append(
            Question(
                field="evidence.pending",
                question=f"Can you share the contents of {item.reference}{where}?",
                why="It is referenced in the request but was not included, so it cannot be used yet.",
                blocking=False,
            )
        )

    # --- ground truth --------------------------------------------------------
    if not spec.ground_truth.available and any(
        k.measurement is Measurement.PROGRAMMATIC for k in spec.kpis
    ):
        out.append(
            Question(
                field="ground_truth",
                question="For the programmatic checks, where does the expected answer come from - labelled data, a reference implementation, or hand-written cases?",
                why="A programmatic grader needs something to compare against.",
                blocking=True,
            )
        )

    return out


def merge_questions(model_questions: List[Question], rules: List[Question]) -> List[Question]:
    """Rule questions win; model questions survive when they ask something new."""
    merged: List[Question] = []
    for question in rules:
        if not _is_duplicate(question, merged):
            merged.append(question)
    for question in model_questions:
        if not _is_duplicate(question, merged):
            merged.append(question)
    merged.sort(key=lambda q: not q.blocking)  # blocking first, order preserved within
    return merged


def readiness_for(questions: List[Question]) -> Readiness:
    if any(q.blocking for q in questions):
        return Readiness.INSUFFICIENT
    if questions:
        return Readiness.NEEDS_INPUT
    return Readiness.READY


def analyze(spec: TaskSpec) -> TaskSpec:
    """Return a copy of `spec` with merged questions and a computed readiness."""
    questions = merge_questions(spec.open_questions, rule_questions(spec))
    return spec.model_copy(
        update={
            "open_questions": questions,
            "readiness": readiness_for(questions),
        }
    )


__all__ = ["analyze", "merge_questions", "readiness_for", "rule_questions"]

"""A small catalogue of real, public benchmarks, and a matcher onto a spec.

`auto_eval.ground_truth` asks a model to search the web, which is the right way
to find something specific to a task and the wrong way to answer "is there a
standard suite for this?". The model can misremember a name, a URL, or a number.
This module is the deterministic half: ten benchmarks that actually exist, each
one a suite with a scoring protocol rather than a loose dataset, matched onto a
`TaskSpec` by rules.

Ten is a sample, not a survey. They were picked to cover the shapes a request
usually takes - knowledge, code, maths, agentic work, truthfulness, retrieval,
RAG faithfulness, open-ended judged quality, classification, safety - so that a
classified spec usually finds at least one thing to anchor on before anyone
searches the web.

What is deliberately not here:

- **Numbers.** No leaderboard scores, no "GPT-4 gets X". Those go stale in weeks
  and a wrong one is worse than none, so `baselines` is left empty and
  `auto_eval.analysis` fills it by opening the page and quoting it.
- **Latency and cost.** There is no public benchmark worth citing: both are
  properties of your deployment, not of a model, so measure them yourself.

Sizes and licences are as the source publishes them and can change; every entry
carries the URL, and the analysis stage checks it rather than trusting this
file.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from .ground_truth import USABLE_FITS, Access, ExternalSource, Fit, SourceKind
from .schema import EvalType, SubjectKind, TaskSpec


class Benchmark(BaseModel):
    """One public benchmark suite: data, a task, and a way to score it."""

    name: str
    url: str
    publisher: str
    measures: str = Field(description="What it actually tests, in one sentence.")
    metric: str = Field(description="The score it reports, in its own words.")
    size: Optional[str] = Field(
        default=None, description="How much data, as the source states it."
    )
    licence: Optional[str] = Field(
        default=None, description="Left null where the source does not state one."
    )
    access: Access = Access.OPEN
    subjects: List[SubjectKind] = Field(
        default_factory=list, description="Subject kinds this suits."
    )
    eval_types: List[EvalType] = Field(
        default_factory=list, description="Reasons for evaluating that it serves."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Words in a KPI that mean this benchmark speaks to it.",
    )
    caveats: str = Field(description="Why it might not transfer. Never empty.")


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------

CATALOGUE: List[Benchmark] = [
    Benchmark(
        name="MMLU",
        url="https://github.com/hendrycks/test",
        publisher="Hendrycks et al.",
        measures=(
            "Multiple-choice knowledge and reasoning across 57 subjects, from "
            "elementary maths to professional law."
        ),
        metric="accuracy (4-way multiple choice)",
        size="57 subjects, about 16,000 multiple-choice questions",
        licence="MIT",
        subjects=[SubjectKind.MODEL, SubjectKind.PROMPT],
        eval_types=[EvalType.CAPABILITY, EvalType.COMPARATIVE],
        keywords=["knowledge", "reasoning", "factual", "exam", "subject matter"],
        caveats=(
            "Saturated at the top of the range and widely present in training "
            "data, so a high score says little about a frontier model. Multiple "
            "choice, so it does not test generation at all."
        ),
    ),
    Benchmark(
        name="HumanEval",
        url="https://github.com/openai/human-eval",
        publisher="OpenAI",
        measures=(
            "Writing a Python function from a docstring, scored by running the "
            "problem's own unit tests."
        ),
        metric="pass@k (execution)",
        size="164 hand-written problems",
        licence="MIT",
        subjects=[SubjectKind.MODEL, SubjectKind.PROMPT, SubjectKind.TOOL],
        eval_types=[EvalType.CAPABILITY, EvalType.COMPARATIVE],
        keywords=["code", "compile", "unit test", "python", "snippet", "programming"],
        caveats=(
            "Small, Python-only, and self-contained - it says nothing about work "
            "in a real repository. Effectively saturated for frontier models."
        ),
    ),
    Benchmark(
        name="GSM8K",
        url="https://github.com/openai/grade-school-math",
        publisher="OpenAI",
        measures=(
            "Grade-school maths word problems needing two to eight reasoning "
            "steps, scored on the final answer."
        ),
        metric="exact-match accuracy on the final answer",
        size="8.5K problems, split 7.5K train / 1K test",
        licence="MIT",
        subjects=[SubjectKind.MODEL, SubjectKind.PROMPT],
        eval_types=[EvalType.CAPABILITY, EvalType.COMPARATIVE],
        keywords=["reasoning", "arithmetic", "maths", "math", "word problem"],
        caveats=(
            "Saturated, and contamination is documented - freshly written "
            "equivalents have shown models scoring lower than on the original set."
        ),
    ),
    Benchmark(
        name="SWE-bench",
        url="https://www.swebench.com/",
        publisher="Princeton NLP",
        measures=(
            "Resolving real GitHub issues in Python repositories, scored by "
            "whether the repository's own tests pass afterwards."
        ),
        metric="% of instances resolved",
        size="2,294 task instances; the human-validated Verified subset is 500",
        licence="MIT",
        subjects=[SubjectKind.AGENT, SubjectKind.WORKFLOW, SubjectKind.TOOL],
        eval_types=[EvalType.CAPABILITY, EvalType.COMPARATIVE],
        keywords=[
            "issue",
            "patch",
            "pull request",
            "repository",
            "codebase",
            "bug fix",
        ],
        caveats=(
            "Python only, and running it needs a per-repository container "
            "harness, so it is a day of setup rather than an afternoon. Use the "
            "Verified subset unless you have a reason not to."
        ),
    ),
    Benchmark(
        name="TruthfulQA",
        url="https://github.com/sylinrl/TruthfulQA",
        publisher="Lin et al.",
        measures=(
            "Whether a model repeats common human misconceptions on questions "
            "written to provoke exactly that."
        ),
        metric="% true, and % true and informative",
        size="817 questions across 38 categories",
        licence="Apache-2.0",
        subjects=[SubjectKind.MODEL, SubjectKind.PROMPT, SubjectKind.AGENT],
        eval_types=[EvalType.SAFETY, EvalType.ROBUSTNESS, EvalType.CAPABILITY],
        keywords=[
            "truthful",
            "hallucination",
            "hallucinate",
            "factual",
            "misinformation",
            "made up",
        ],
        caveats=(
            "Adversarially written against an early GPT-3, and it measures "
            "imitative falsehood specifically - not hallucination in your own "
            "domain, where the model has no popular misconception to copy."
        ),
    ),
    Benchmark(
        name="BEIR",
        url="https://github.com/beir-cellar/beir",
        publisher="UKP Lab / Thakur et al.",
        measures=(
            "Zero-shot retrieval quality across 18 heterogeneous IR datasets - "
            "the retrieval half of a RAG pipeline."
        ),
        metric="nDCG@10",
        size="18 datasets; several need a separate licence or request",
        licence="Apache-2.0 for the toolkit; per-dataset terms vary",
        subjects=[SubjectKind.RAG_PIPELINE, SubjectKind.WORKFLOW, SubjectKind.TOOL],
        eval_types=[EvalType.CAPABILITY, EvalType.COMPARATIVE],
        keywords=[
            "retrieval",
            "retrieve",
            "ranking",
            "relevance",
            "relevant",
            "search",
            "ndcg",
        ],
        caveats=(
            "Scores the retriever, not the answer the pipeline finally gives. "
            "Check each dataset's terms separately - the toolkit's licence is "
            "not the data's."
        ),
    ),
    Benchmark(
        name="RAGTruth",
        url="https://github.com/ParticleMedia/RAGTruth",
        publisher="Niu et al. (ACL 2024)",
        measures=(
            "Whether a RAG answer is supported by its retrieved context, with "
            "hallucinations annotated at word level."
        ),
        metric="hallucination detection precision / recall / F1",
        size="~18,000 annotated responses over QA, data-to-text and summarisation",
        subjects=[SubjectKind.RAG_PIPELINE, SubjectKind.WORKFLOW, SubjectKind.AGENT],
        eval_types=[EvalType.CAPABILITY, EvalType.SAFETY],
        keywords=[
            "faithful",
            "faithfulness",
            "grounded",
            "grounding",
            "hallucination",
            "hallucinate",
            "supported",
            "citation",
            "attribution",
        ],
        caveats=(
            "The annotations are on responses from a fixed set of older models, "
            "so it trains and checks a hallucination detector better than it "
            "scores your pipeline directly."
        ),
    ),
    Benchmark(
        name="MT-Bench",
        url="https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge",
        publisher="LMSYS",
        measures=(
            "Open-ended multi-turn instruction following over 8 categories, "
            "graded 1-10 by a strong model acting as judge."
        ),
        metric="mean judge score out of 10",
        size="80 two-turn questions",
        licence="Apache-2.0",
        subjects=[SubjectKind.PROMPT, SubjectKind.AGENT, SubjectKind.MODEL],
        eval_types=[EvalType.CAPABILITY, EvalType.COMPARATIVE],
        keywords=[
            "helpfulness",
            "helpful",
            "tone",
            "instruction following",
            "coherence",
            "judge",
            "rubric",
            "conversation",
            "multi-turn",
        ],
        caveats=(
            "80 questions is too few to separate close systems. Its real value "
            "here is as the reference point for LLM-as-judge itself: the paper "
            "reports judge-human agreement at around the level of two humans "
            "agreeing, and documents position, verbosity and self-preference "
            "bias you will inherit if you build a judge."
        ),
    ),
    Benchmark(
        name="SuperGLUE",
        url="https://super.gluebenchmark.com/",
        publisher="NYU / Wang et al.",
        measures=(
            "Eight classification and inference tasks - entailment, coreference, "
            "question answering - with a held-out test set and a public "
            "leaderboard."
        ),
        metric="per-task accuracy / F1 and an aggregate score",
        size="8 tasks, each a few hundred to several thousand examples",
        licence="per-task; the tasks carry their own terms",
        subjects=[SubjectKind.CLASSIFIER, SubjectKind.MODEL, SubjectKind.PROMPT],
        eval_types=[EvalType.CAPABILITY, EvalType.COMPARATIVE, EvalType.REGRESSION],
        keywords=[
            "classification",
            "classify",
            "label",
            "entailment",
            "sentiment",
        ],
        caveats=(
            "Long since beaten by models above the human baseline, so it is a "
            "sanity check and a source of task format, not a discriminating "
            "score."
        ),
    ),
    Benchmark(
        name="RealToxicityPrompts",
        url="https://allenai.org/data/real-toxicity-prompts",
        publisher="Allen Institute for AI",
        measures=(
            "How often ordinary, non-adversarial prompts lead a model into toxic "
            "continuations."
        ),
        metric="expected maximum toxicity, and probability of toxicity over k samples",
        size="100,000 naturally occurring prompts, scored with Perspective API",
        subjects=[SubjectKind.MODEL, SubjectKind.PROMPT, SubjectKind.AGENT],
        eval_types=[EvalType.SAFETY, EvalType.ROBUSTNESS],
        keywords=[
            "toxicity",
            "toxic",
            "harmful",
            "safety",
            "policy violation",
            "offensive",
            "abuse",
        ],
        caveats=(
            "Scores come from Perspective API, whose model changes over time, so "
            "numbers are only comparable when scored in the same window. English "
            "only, and it tests unprompted toxicity rather than jailbreak "
            "resistance."
        ),
    ),
]


# --------------------------------------------------------------------------
# Matching a spec onto the catalogue
# --------------------------------------------------------------------------

# A metric name is not a subject. Nearly every benchmark reports "accuracy", so
# a KPI called "policy adherence accuracy" matched MMLU, GSM8K and SuperGLUE
# equally - four confident recommendations, none of them about billing policy.
# These words are therefore never keywords on an entry; what a benchmark is
# *about* is what makes it match. A test enforces it.
GENERIC_METRIC_WORDS = frozenset(
    {
        "accuracy",
        "correct",
        "correctness",
        "precision",
        "recall",
        "f1",
        "score",
        "rate",
        "pass rate",
        "error rate",
        "quality",
        "exact",
        "exact match",
        "performance",
    }
)

# Weights are arbitrary but ordered on purpose: the thing under test matters
# more than the reason for testing, and a KPI the benchmark actually reports
# matters most of all.
SUBJECT_WEIGHT = 3
EVAL_TYPE_WEIGHT = 2
KPI_WEIGHT = 4


class BenchmarkMatch(BaseModel):
    """One catalogue entry, and why it came back for this spec."""

    benchmark: Benchmark
    score: int
    fit: Fit
    covers_kpis: List[str] = Field(
        default_factory=list, description="KPI names from the spec, verbatim."
    )
    reasons: List[str] = Field(
        default_factory=list, description="Why this matched, in plain words."
    )


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _mentions(haystack: str, keyword: str) -> bool:
    """Whole-word containment, so 'math' does not match 'aftermath'."""
    return re.search(rf"\b{re.escape(keyword)}\b", haystack) is not None


def _kpi_hits(benchmark: Benchmark, spec: TaskSpec) -> List[str]:
    """KPI names whose wording overlaps what this benchmark reports."""
    hits = []
    for kpi in spec.kpis:
        text = _norm(f"{kpi.name} {kpi.definition}")
        if any(_mentions(text, _norm(word).strip()) for word in benchmark.keywords):
            hits.append(kpi.name)
    return hits


def _fit_for(subject_hit: bool, kpi_hits: Sequence[str]) -> Fit:
    """Same subject and a KPI it reports is as close as a public suite gets.

    Reporting nothing you measure caps it at `contextual` however well the rest
    lines up: SWE-bench and a billing agent are both "an agent, tested for
    capability", and it is still not data you can score a billing agent with.
    """
    if kpi_hits:
        return Fit.DIRECT if subject_hit else Fit.ADAPTABLE
    return Fit.CONTEXTUAL


def match(spec: TaskSpec, *, limit: int = 5) -> List[BenchmarkMatch]:
    """Catalogue entries worth anchoring this spec on, best first.

    Deterministic and offline: same spec, same answer, no API call. An entry
    that matches nothing is left out entirely - a benchmark returned "because
    it is famous" is noise the user has to read and dismiss - and so is one
    that matches only on the kind of thing under test.

    Entries that report one of the spec's KPIs come back on their own. Only
    when none does are the `contextual` near-misses returned instead, so an
    empty-handed answer still points somewhere.
    """
    matches: List[BenchmarkMatch] = []

    for benchmark in CATALOGUE:
        score = 0
        reasons: List[str] = []

        subject_hit = spec.subject.kind in benchmark.subjects
        if subject_hit:
            score += SUBJECT_WEIGHT
            reasons.append(f"built for a {spec.subject.kind.value.replace('_', ' ')}")

        shared_types = [e for e in spec.eval_types if e in benchmark.eval_types]
        if shared_types:
            score += EVAL_TYPE_WEIGHT * len(shared_types)
            reasons.append(
                "covers " + ", ".join(e.value for e in shared_types) + " evaluation"
            )

        kpi_hits = _kpi_hits(benchmark, spec)
        if kpi_hits:
            score += KPI_WEIGHT * len(kpi_hits)
            reasons.append("reports something close to " + ", ".join(kpi_hits))

        # Matching on the subject alone is not enough: "it is also a model" is
        # how GSM8K ends up recommended for a toxicity evaluation. Something
        # about *why* they are evaluating, or *what* they measure, has to hit.
        if score and (shared_types or kpi_hits):
            matches.append(
                BenchmarkMatch(
                    benchmark=benchmark,
                    score=score,
                    fit=_fit_for(subject_hit, kpi_hits),
                    covers_kpis=kpi_hits,
                    reasons=reasons,
                )
            )

    # Sorted by score; `sorted` is stable, so ties keep catalogue order rather
    # than shuffling between runs.
    matches.sort(key=lambda m: m.score, reverse=True)

    # Once something reports what you actually measure, the "it is also an
    # agent" entries are padding. They only earn their place when nothing
    # usable came back at all, where the nearest shape is better than silence.
    usable = [m for m in matches if m.fit in USABLE_FITS]
    return (usable or matches)[:limit]


def as_sources(matches: Sequence[BenchmarkMatch]) -> List[ExternalSource]:
    """Turn matches into the source type the ground-truth report already reads.

    That way a curated hit renders, and is opened and checked by
    `auto_eval.analysis`, exactly like one the model found by searching.
    `baselines` stays empty on purpose - see this module's docstring.
    """
    return [
        ExternalSource(
            name=m.benchmark.name,
            kind=SourceKind.BENCHMARK,
            url=m.benchmark.url,
            publisher=m.benchmark.publisher,
            description=m.benchmark.measures,
            covers_kpis=list(m.covers_kpis),
            fit=m.fit,
            access=m.benchmark.access,
            licence=m.benchmark.licence,
            caveats=m.benchmark.caveats,
        )
        for m in matches
    ]


def by_name(name: str) -> Optional[Benchmark]:
    """Look one up case-insensitively, for a CLI argument or a test."""
    wanted = name.strip().lower()
    return next((b for b in CATALOGUE if b.name.lower() == wanted), None)


__all__ = [
    "CATALOGUE",
    "EVAL_TYPE_WEIGHT",
    "GENERIC_METRIC_WORDS",
    "KPI_WEIGHT",
    "SUBJECT_WEIGHT",
    "Benchmark",
    "BenchmarkMatch",
    "as_sources",
    "by_name",
    "match",
]

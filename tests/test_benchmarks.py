import pytest

from auto_eval.benchmarks import (
    CATALOGUE,
    GENERIC_METRIC_WORDS,
    as_sources,
    by_name,
    match,
)
from auto_eval.gaps import analyze
from auto_eval.ground_truth import USABLE_FITS, Fit, SourceKind
from auto_eval.schema import (
    KPI,
    Direction,
    EvalType,
    Measurement,
    MetricKind,
    Subject,
    SubjectKind,
    TaskSpec,
)


def spec_for(kind, *, eval_types=(), kpis=()):
    """A spec of a given shape, run through `analyze` like a real one would be."""
    return analyze(
        TaskSpec(
            title="Test",
            summary="A spec to match against.",
            subject=Subject(name="thing", kind=kind, description="Does something."),
            eval_types=list(eval_types),
            kpis=[
                KPI(
                    name=name,
                    kind=MetricKind.QUANTITATIVE,
                    definition=definition,
                    measurement=Measurement.PROGRAMMATIC,
                    direction=Direction.MAXIMIZE,
                )
                for name, definition in kpis
            ],
        )
    )


def names(matches):
    return [m.benchmark.name for m in matches]


# --- the catalogue itself --------------------------------------------------


def test_the_catalogue_holds_ten_benchmarks():
    assert len(CATALOGUE) == 10
    assert len({b.name for b in CATALOGUE}) == 10


def test_every_entry_carries_a_link_and_a_caveat():
    """A benchmark with no stated caveat reads as a recommendation. None is."""
    for entry in CATALOGUE:
        assert entry.url.startswith("https://"), entry.name
        assert entry.caveats.strip(), entry.name
        assert entry.metric.strip(), entry.name
        assert entry.subjects, entry.name


def test_no_entry_keys_off_a_generic_metric_word():
    """Nearly everything reports "accuracy"; it cannot be what makes a match."""
    offenders = [
        (entry.name, keyword)
        for entry in CATALOGUE
        for keyword in entry.keywords
        if keyword.lower() in GENERIC_METRIC_WORDS
    ]
    assert offenders == []


def test_no_entry_carries_a_baseline_number():
    """Scores go stale; `auto_eval.analysis` quotes them from the live page."""
    assert all(not entry.model_dump().get("baselines") for entry in CATALOGUE)


def test_lookup_by_name_is_case_insensitive():
    assert by_name("humaneval").name == "HumanEval"
    assert by_name("  SWE-bench ").name == "SWE-bench"
    assert by_name("not a benchmark") is None


# --- matching --------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, eval_types, kpis, expected",
    [
        (
            SubjectKind.RAG_PIPELINE,
            [EvalType.CAPABILITY],
            [("faithfulness", "Answers are grounded in the retrieved passages.")],
            "RAGTruth",
        ),
        (
            SubjectKind.RAG_PIPELINE,
            [EvalType.CAPABILITY],
            [("retrieval relevance", "Share of retrieved passages that are relevant.")],
            "BEIR",
        ),
        (
            SubjectKind.AGENT,
            [EvalType.CAPABILITY],
            [
                (
                    "issue resolution",
                    "Share of GitHub issues resolved in the repository.",
                )
            ],
            "SWE-bench",
        ),
        (
            SubjectKind.CLASSIFIER,
            [EvalType.CAPABILITY],
            [("label accuracy", "Share of tickets given the right label.")],
            "SuperGLUE",
        ),
        (
            SubjectKind.MODEL,
            [EvalType.SAFETY],
            [("toxicity rate", "Share of completions that are toxic.")],
            "RealToxicityPrompts",
        ),
        (
            SubjectKind.MODEL,
            [EvalType.CAPABILITY],
            [("hallucination rate", "Share of answers that are made up.")],
            "TruthfulQA",
        ),
    ],
)
def test_a_spec_finds_the_benchmark_written_for_it(kind, eval_types, kpis, expected):
    matches = match(spec_for(kind, eval_types=eval_types, kpis=kpis))
    assert names(matches)[0] == expected
    assert matches[0].fit is Fit.DIRECT
    assert matches[0].covers_kpis == [kpis[0][0]]


def test_a_direct_hit_crowds_out_the_near_misses():
    matches = match(
        spec_for(
            SubjectKind.RAG_PIPELINE,
            eval_types=[EvalType.CAPABILITY],
            kpis=[("faithfulness", "Answers grounded in the retrieved context.")],
        )
    )
    assert names(matches) == ["RAGTruth"]


def test_an_in_house_metric_gets_honest_near_misses_not_a_false_direct():
    """Nothing public scores a billing agent, and saying so beats inventing it."""
    matches = match(
        spec_for(
            SubjectKind.AGENT,
            eval_types=[EvalType.CAPABILITY],
            kpis=[
                (
                    "refund policy adherence",
                    "Share of replies stating the refund window correctly.",
                )
            ],
        )
    )
    assert matches
    assert all(m.fit is Fit.CONTEXTUAL for m in matches)
    assert all(m.covers_kpis == [] for m in matches)


def test_matching_only_the_subject_is_not_a_match():
    """ "It is also a model" is how GSM8K ends up under a toxicity evaluation.

    MMLU, GSM8K and HumanEval all list `model` as a subject they suit, and all
    three are about capability - so a safety evaluation shares the subject with
    them and nothing else. They must not come back.
    """
    matches = match(spec_for(SubjectKind.MODEL, eval_types=[EvalType.SAFETY]))

    assert names(matches)
    assert {"GSM8K", "MMLU", "HumanEval"}.isdisjoint(names(matches))


def test_the_derived_generic_kpi_does_not_conjure_swe_bench():
    """`derive` names a bare agent KPI "task success rate" - not SWE-bench's."""
    matches = match(spec_for(SubjectKind.AGENT, eval_types=[EvalType.CAPABILITY]))
    assert all(m.fit is Fit.CONTEXTUAL for m in matches)


def test_limit_is_honoured_and_order_is_stable():
    spec = spec_for(SubjectKind.MODEL, eval_types=[EvalType.CAPABILITY])
    assert len(match(spec, limit=2)) <= 2
    assert names(match(spec)) == names(match(spec))


def test_matching_needs_no_network_or_key(monkeypatch):
    """It is a lookup table; an absent API key must not change the answer."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert match(spec_for(SubjectKind.CLASSIFIER, eval_types=[EvalType.CAPABILITY]))


# --- handing matches to the rest of the pipeline ---------------------------


def test_matches_become_sources_the_report_already_understands():
    matches = match(
        spec_for(
            SubjectKind.CLASSIFIER,
            eval_types=[EvalType.CAPABILITY],
            kpis=[("label accuracy", "Share of tickets given the right label.")],
        )
    )
    sources = as_sources(matches)

    assert [s.name for s in sources] == names(matches)
    assert all(s.kind is SourceKind.BENCHMARK for s in sources)
    assert sources[0].covers_kpis == ["label accuracy"]
    assert sources[0].fit in USABLE_FITS
    assert sources[0].caveats
    # Numbers are the analysis stage's job, quoted from the live page.
    assert sources[0].baselines == []

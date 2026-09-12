from auto_eval.derive import default_kpi, derive
from auto_eval.gaps import analyze
from auto_eval.schema import (
    KPI,
    Direction,
    EvalType,
    Measurement,
    MetricKind,
    Priority,
    Provenance,
    Readiness,
    SubjectKind,
)


def test_a_spec_with_no_kpi_gets_one_rather_than_a_blocking_question(sparse_spec):
    derived = derive(sparse_spec)

    assert [k.name for k in derived.kpis] == ["task success rate"]
    assert derived.kpis[0].source is Provenance.INFERRED
    assert derived.kpis[0].priority is Priority.PRIMARY
    assert any("task success rate" in line for line in derived.assumptions)


def test_the_derived_kpi_follows_the_subject(sparse_spec):
    sparse_spec.subject.kind = SubjectKind.CLASSIFIER
    assert default_kpi(sparse_spec).name == "accuracy"

    sparse_spec.subject.kind = SubjectKind.RAG_PIPELINE
    assert default_kpi(sparse_spec).name == "answer faithfulness"


def test_a_stated_eval_type_outranks_the_subject_kind(sparse_spec):
    sparse_spec.subject.kind = SubjectKind.CLASSIFIER
    sparse_spec.eval_types = [EvalType.SAFETY]
    assert default_kpi(sparse_spec).name == "policy violation rate"


def test_capability_falls_through_to_the_subject(sparse_spec):
    """`capability` asks whether it does its job, which the subject defines."""
    sparse_spec.subject.kind = SubjectKind.CLASSIFIER
    sparse_spec.eval_types = [EvalType.CAPABILITY]
    assert default_kpi(sparse_spec).name == "accuracy"


def test_an_unmeasurable_kpi_gets_the_method_its_kind_implies(full_spec):
    full_spec.kpis[0].measurement = Measurement.UNKNOWN
    assert derive(full_spec).kpis[0].measurement is Measurement.PROGRAMMATIC

    full_spec.kpis[0].kind = MetricKind.QUALITATIVE
    full_spec.kpis[0].measurement = Measurement.UNKNOWN
    assert derive(full_spec).kpis[0].measurement is Measurement.LLM_JUDGE


def test_a_stated_kpi_is_preferred_over_an_inferred_one_for_primary(full_spec):
    full_spec.kpis[0].priority = Priority.SECONDARY
    full_spec.kpis.insert(
        0,
        KPI(
            name="tone",
            kind=MetricKind.QUALITATIVE,
            definition="Reads like a person.",
            measurement=Measurement.LLM_JUDGE,
            direction=Direction.MAXIMIZE,
            source=Provenance.INFERRED,
        ),
    )

    derived = derive(full_spec)
    primary = [k.name for k in derived.kpis if k.priority is Priority.PRIMARY]
    assert primary == ["field accuracy"]


def test_missing_eval_type_reads_as_capability(sparse_spec):
    assert derive(sparse_spec).eval_types == [EvalType.CAPABILITY]


def test_nothing_is_touched_when_the_user_said_it(full_spec):
    before = full_spec.model_copy(deep=True)
    after = derive(full_spec)

    assert after.kpis == before.kpis
    assert after.eval_types == before.eval_types
    assert after.assumptions == before.assumptions


def test_derive_is_idempotent(sparse_spec):
    """A spec makes the round trip through the API and comes back unchanged."""
    once = derive(sparse_spec)
    twice = derive(once)

    assert twice.kpis == once.kpis
    assert twice.assumptions == once.assumptions


def test_a_derived_spec_is_not_blocked(sparse_spec):
    spec = analyze(sparse_spec)

    assert spec.readiness is Readiness.NEEDS_INPUT
    assert not any(q.blocking for q in spec.open_questions)
    # What it filled in is still asked about, just without stopping anything.
    assert "kpis.target" in {q.field for q in spec.open_questions}


def test_the_model_cannot_declare_its_own_question_blocking(sparse_spec):
    from auto_eval.schema import Question

    sparse_spec.open_questions = [
        Question(
            field="evidence.dataset",
            question="Can you send me the six months of transcripts?",
            why="I would like them.",
            blocking=True,
        )
    ]

    spec = analyze(sparse_spec)
    asked = [q for q in spec.open_questions if "transcripts" in q.question]
    assert asked and asked[0].blocking is False
    assert spec.readiness is Readiness.NEEDS_INPUT

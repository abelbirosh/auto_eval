from auto_eval.gaps import analyze, merge_questions, readiness_for, rule_questions
from auto_eval.schema import (
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    Measurement,
    MetricKind,
    Priority,
    Question,
    Readiness,
)


def fields(questions):
    return {q.field for q in questions}


def test_only_missing_kpis_blocks(sparse_spec):
    """Everything else is worth asking about but does not stop a spec."""
    questions = rule_questions(sparse_spec)
    blocking = {q.field for q in questions if q.blocking}
    assert blocking == {"kpis"}


def test_a_missing_entry_point_is_asked_about_but_does_not_block(sparse_spec):
    interface = [
        q for q in rule_questions(sparse_spec) if q.field == "subject.interface"
    ]
    assert len(interface) == 1
    assert interface[0].blocking is False


def test_sparse_spec_always_asks_for_docs_and_successful_runs(sparse_spec):
    asked = fields(rule_questions(sparse_spec))
    assert "evidence.documentation" in asked
    assert "evidence.successful_run" in asked
    assert "evidence.failure_case" in asked


def test_full_spec_has_no_gaps(full_spec):
    assert rule_questions(full_spec) == []
    assert analyze(full_spec).readiness is Readiness.READY


def test_mentioned_evidence_becomes_a_fetch_request(full_spec):
    full_spec.evidence.append(
        EvidenceItem(
            kind=EvidenceKind.EXISTING_EVAL,
            reference="the old eval suite",
            location="evals/legacy",
            status=EvidenceStatus.MENTIONED_NOT_PROVIDED,
        )
    )
    questions = rule_questions(full_spec)
    assert [q.field for q in questions] == ["evidence.pending"]
    assert "evals/legacy" in questions[0].question


def test_absent_evidence_does_not_count_as_available(full_spec):
    for item in full_spec.evidence:
        item.status = EvidenceStatus.ABSENT
    asked = fields(rule_questions(full_spec))
    assert {"evidence.documentation", "evidence.successful_run"} <= asked


def test_qualitative_judged_kpi_asks_for_anchoring_examples(full_spec):
    kpi = full_spec.kpis[0]
    kpi.kind = MetricKind.QUALITATIVE
    kpi.measurement = Measurement.LLM_JUDGE
    assert "kpis.definition" in fields(rule_questions(full_spec))


def test_untargeted_kpi_asks_for_a_threshold_but_guardrails_are_exempt(full_spec):
    full_spec.kpis[0].target = None
    assert "kpis.target" in fields(rule_questions(full_spec))

    full_spec.kpis[0].priority = Priority.GUARDRAIL
    questions = rule_questions(full_spec)
    assert "kpis.target" not in fields(questions)
    assert "kpis.priority" in fields(questions)  # nothing is primary any more


def test_programmatic_kpi_without_ground_truth_asks_but_does_not_block(full_spec):
    full_spec.ground_truth.available = False
    questions = rule_questions(full_spec)
    assert [q.field for q in questions] == ["ground_truth"]
    assert questions[0].blocking is False


def test_a_spec_with_kpis_but_gaps_reads_needs_input_not_insufficient(full_spec):
    full_spec.ground_truth.available = False
    full_spec.subject.interface = None
    assert analyze(full_spec).readiness is Readiness.NEEDS_INPUT


def test_per_kpi_questions_are_capped(full_spec):
    template = full_spec.kpis[0]
    full_spec.kpis = [
        template.model_copy(update={"name": f"metric {i}", "target": None})
        for i in range(10)
    ]
    targets = [q for q in rule_questions(full_spec) if q.field == "kpis.target"]
    assert len(targets) == 3


def test_merge_drops_model_questions_that_restate_a_rule():
    rule = Question(
        field="kpis", question="What metric decides success?", why="x", blocking=True
    )
    echo = Question(field="kpis", question="what metric decides success", why="y")
    fresh = Question(
        field="constraints", question="What is the latency budget?", why="z"
    )

    merged = merge_questions([echo, fresh], [rule])

    assert [q.question for q in merged] == [rule.question, fresh.question]


def test_merge_puts_blocking_questions_first():
    a = Question(field="a", question="non blocking one", why="")
    b = Question(field="b", question="blocking one", why="", blocking=True)
    assert [q.field for q in merge_questions([], [a, b])] == ["b", "a"]


def test_readiness_levels():
    blocking = Question(field="a", question="q", why="", blocking=True)
    soft = Question(field="b", question="r", why="")
    assert readiness_for([blocking, soft]) is Readiness.INSUFFICIENT
    assert readiness_for([soft]) is Readiness.NEEDS_INPUT
    assert readiness_for([]) is Readiness.READY


def test_analyze_overrides_model_supplied_readiness(sparse_spec):
    sparse_spec.readiness = Readiness.READY
    assert analyze(sparse_spec).readiness is Readiness.INSUFFICIENT


def test_merge_drops_a_reworded_echo_of_a_rule_question():
    rule = Question(
        field="evidence.successful_run",
        question="Can you share two or three examples of runs that came out the way you wanted?",
        why="",
    )
    echo = Question(
        field="evidence.successful_run",
        question="Can you share examples of runs that came out the way you wanted?",
        why="",
    )
    assert len(merge_questions([echo], [rule])) == 1


def test_merge_keeps_short_questions_that_merely_share_words():
    rule = Question(field="a", question="Is there a documented latency budget?", why="")
    other = Question(field="b", question="Is there a budget?", why="")
    assert len(merge_questions([other], [rule])) == 2

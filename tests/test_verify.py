from auto_eval.agent import profile
from auto_eval.schema import KPI, Direction, Measurement, MetricKind
from auto_eval.verify import (
    AUTOMATIC_KINDS,
    Outcome,
    Reads,
    Verifier,
    VerifierKind,
    accept,
    automatic_share,
    coverage_by_kpi,
    guardrails,
    settle,
    verifiers_for_kpi,
)


def kpi_with(measurement, name="task success rate"):
    return KPI(
        name=name,
        kind=MetricKind.QUANTITATIVE,
        definition="Share of runs that resolved the request correctly.",
        measurement=measurement,
        direction=Direction.MAXIMIZE,
        unit="%",
    )


def kinds(verifiers):
    return [v.kind for v in verifiers]


# --- the measurement mapping ----------------------------------------------


def test_every_measurement_maps_to_a_check(agent_spec):
    """A KPI with no check is a KPI the suite silently does not measure."""
    agent_profile = profile(agent_spec)
    for measurement in Measurement:
        built = verifiers_for_kpi(kpi_with(measurement), agent_profile)
        assert built, measurement


def test_the_mapping_is_what_the_spec_asked_for(agent_spec):
    agent_profile = profile(agent_spec)
    expected = {
        Measurement.PROGRAMMATIC: VerifierKind.END_STATE,
        Measurement.EXECUTION: VerifierKind.EXECUTION,
        Measurement.TELEMETRY: VerifierKind.TRACE_BUDGET,
        Measurement.LLM_JUDGE: VerifierKind.RUBRIC,
        Measurement.HUMAN_REVIEW: VerifierKind.HUMAN,
        Measurement.UNKNOWN: VerifierKind.RUBRIC,
    }
    for measurement, kind in expected.items():
        assert kinds(verifiers_for_kpi(kpi_with(measurement), agent_profile)) == [kind]


def test_a_judged_check_reads_the_trace_as_well_as_the_answer(agent_spec):
    """The whole point: an output-only grader cannot see what it did on the way."""
    built = verifiers_for_kpi(kpi_with(Measurement.LLM_JUDGE), profile(agent_spec))[0]
    assert built.reads is Reads.BOTH
    assert len(built.rubric) >= 2
    assert any("quote" in line.lower() for line in built.rubric)


def test_verifier_ids_are_namespaced_so_two_cases_do_not_collide(agent_spec):
    agent_profile = profile(agent_spec)
    first = verifiers_for_kpi(
        kpi_with(Measurement.LLM_JUDGE), agent_profile, prefix="a."
    )
    second = verifiers_for_kpi(
        kpi_with(Measurement.LLM_JUDGE), agent_profile, prefix="b."
    )
    assert first[0].id != second[0].id


# --- guardrails ------------------------------------------------------------


def test_every_case_checks_that_the_run_stopped_and_stayed_in_budget(agent_spec):
    built = kinds(guardrails(profile(agent_spec)))
    assert VerifierKind.TERMINATED in built
    assert VerifierKind.TRACE_BUDGET in built


def test_a_destructive_tool_is_forbidden_by_default_and_fatally_so(agent_spec):
    built = guardrails(profile(agent_spec))
    forbidden = [v for v in built if v.kind is VerifierKind.TOOL_FORBIDDEN]
    assert [v.tool for v in forbidden] == ["payments"]
    assert all(v.fatal for v in forbidden)


def test_a_case_that_needs_the_tool_does_not_also_forbid_it(agent_spec):
    """Otherwise no case about issuing refunds could ever pass."""
    built = guardrails(profile(agent_spec), allow=["payments"])
    assert not [v for v in built if v.kind is VerifierKind.TOOL_FORBIDDEN]


def test_escaping_the_sandbox_is_fatal(agent_spec):
    escape = [
        v for v in guardrails(profile(agent_spec)) if v.kind is VerifierKind.NO_ESCAPE
    ]
    assert len(escape) == 1
    assert escape[0].fatal is True


def test_a_read_only_agent_gets_no_containment_check(agent_spec):
    agent_spec.subject.description = "Looks orders up in the database."
    agent_spec.summary = "Evaluate a read-only lookup agent."
    agent_spec.subject.in_scope = ["order lookups"]
    built = kinds(guardrails(profile(agent_spec)))
    assert VerifierKind.NO_ESCAPE not in built


# --- evidence --------------------------------------------------------------


def test_a_pass_with_nothing_to_show_for_it_is_a_failure():
    verifier = Verifier(id="v", kind=VerifierKind.RUBRIC, description="judged")
    settled = accept(verifier, Outcome(verifier="v", passed=True, evidence="  "))
    assert settled.passed is False
    assert "evidence" in (settled.note or "")


def test_a_pass_that_quotes_something_stands():
    verifier = Verifier(id="v", kind=VerifierKind.RUBRIC, description="judged")
    outcome = Outcome(
        verifier="v", passed=True, evidence="step 4: refund_tool returned ok"
    )
    assert accept(verifier, outcome).passed is True


def test_a_failure_needs_no_evidence_to_stay_a_failure():
    verifier = Verifier(id="v", kind=VerifierKind.RUBRIC, description="judged")
    assert accept(verifier, Outcome(verifier="v", passed=False)).passed is False


# --- settling a run --------------------------------------------------------


def make(kind=VerifierKind.END_STATE, vid="v", fatal=False):
    return Verifier(id=vid, kind=kind, description="x", fatal=fatal)


def test_a_check_that_never_ran_counts_as_failed():
    result = settle("c1", [make()], [])
    assert result.passed is False
    assert result.outcomes[0].note == "Never ran; counted as a failure."


def test_a_case_passes_only_when_everything_passed():
    verifiers = [make(vid="a"), make(vid="b")]
    outcomes = [
        Outcome(verifier="a", passed=True, evidence="diff"),
        Outcome(verifier="b", passed=True, evidence="diff"),
    ]
    assert settle("c1", verifiers, outcomes).passed is True

    outcomes[1] = Outcome(verifier="b", passed=False, evidence="diff")
    assert settle("c1", verifiers, outcomes).passed is False


def test_a_guardrail_failure_condemns_the_run():
    verifiers = [make(vid="ok"), make(kind=VerifierKind.NO_ESCAPE, vid="escape")]
    outcomes = [
        Outcome(verifier="ok", passed=True, evidence="diff"),
        Outcome(verifier="escape", passed=False, evidence="wrote /etc/hosts"),
    ]
    result = settle("c1", verifiers, outcomes)
    assert result.passed is False
    assert any("guardrail" in note for note in result.notes)


def test_extra_verdicts_for_checks_the_case_does_not_have_are_ignored():
    result = settle(
        "c1",
        [make(vid="a")],
        [
            Outcome(verifier="a", passed=True, evidence="diff"),
            Outcome(verifier="ghost", passed=True, evidence="diff"),
        ],
    )
    assert [o.verifier for o in result.outcomes] == ["a"]


# --- reporting -------------------------------------------------------------


def test_automatic_share_counts_what_a_machine_can_settle():
    assert automatic_share([make(), make(kind=VerifierKind.RUBRIC, vid="r")]) == 0.5
    assert automatic_share([]) == 0.0


def test_the_judge_families_are_not_counted_as_automatic():
    assert VerifierKind.RUBRIC not in AUTOMATIC_KINDS
    assert VerifierKind.HUMAN not in AUTOMATIC_KINDS


def test_coverage_by_kpi_ignores_guardrails(agent_spec):
    built = [
        *guardrails(profile(agent_spec)),
        *verifiers_for_kpi(kpi_with(Measurement.LLM_JUDGE), profile(agent_spec)),
    ]
    assert list(coverage_by_kpi(built)) == ["task success rate"]

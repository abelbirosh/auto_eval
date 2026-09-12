from auto_eval.agent import profile
from auto_eval.authoring import (
    MAX_SYNTHETIC_SHARE,
    CaseSource,
    FixtureKind,
    StubBehaviour,
    adapted,
    author,
    harvested,
    nearest_behaviour,
    needs_fixture,
    synthetic_share,
)
from auto_eval.gaps import analyze
from auto_eval.schema import (
    EvalType,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    SubjectKind,
)
from auto_eval.surface import surface
from auto_eval.verify import VerifierKind


def build(spec, per_cell=5):
    settled = analyze(spec)
    agent_profile = profile(settled)
    matrix = surface(settled, per_cell=per_cell)
    return settled, agent_profile, matrix, author(settled, agent_profile, matrix)


def families(cases):
    return {case.family for case in cases}


def kinds_in(case):
    return {v.kind for v in case.verifiers}


# --- harvested -------------------------------------------------------------


def test_the_users_own_runs_become_cases(agent_spec):
    cases = harvested(analyze(agent_spec), profile(agent_spec), ["refund requests"])
    assert families(cases) == {"harvested.golden_run", "harvested.known_failure"}
    assert all(case.source is CaseSource.HARVESTED for case in cases)


def test_a_known_failure_is_a_regression_case(agent_spec):
    cases = harvested(analyze(agent_spec), profile(agent_spec), ["refund requests"])
    failure = next(c for c in cases if c.family == "harvested.known_failure")
    assert failure.eval_types == [EvalType.REGRESSION]
    assert VerifierKind.END_STATE in kinds_in(failure)


def test_a_golden_run_is_scored_against_the_state_it_reached(agent_spec):
    cases = harvested(analyze(agent_spec), profile(agent_spec), ["refund requests"])
    golden = next(c for c in cases if c.family == "harvested.golden_run")
    assert golden.fixture.kind is FixtureKind.RECORDED_TRACE
    assert golden.fixture.materialised is True


def test_evidence_we_do_not_have_makes_a_case_that_says_so(agent_spec):
    agent_spec.evidence = [
        EvidenceItem(
            kind=EvidenceKind.SUCCESSFUL_RUN,
            reference="transcripts in the ticket system",
            status=EvidenceStatus.MENTIONED_NOT_PROVIDED,
        )
    ]
    case = harvested(analyze(agent_spec), profile(agent_spec), ["refund requests"])[0]
    assert case.fixture.materialised is False
    assert any("placeholder" in note for note in case.notes)


def test_absent_evidence_produces_nothing(agent_spec):
    for item in agent_spec.evidence:
        item.status = EvidenceStatus.ABSENT
    assert (
        harvested(analyze(agent_spec), profile(agent_spec), ["refund requests"]) == []
    )


def test_documentation_is_not_a_case(agent_spec):
    agent_spec.evidence = [
        EvidenceItem(
            kind=EvidenceKind.DOCUMENTATION,
            reference="the runbook",
            status=EvidenceStatus.PROVIDED,
        )
    ]
    assert (
        harvested(analyze(agent_spec), profile(agent_spec), ["refund requests"]) == []
    )


def test_a_harvested_case_lands_on_the_behaviour_it_mentions(agent_spec):
    cases = harvested(
        analyze(agent_spec), profile(agent_spec), ["order lookups", "refund requests"]
    )
    assert {c.behaviour for c in cases} == {"refund requests"}


def test_nearest_behaviour_falls_back_rather_than_dropping_a_case():
    assert nearest_behaviour("something unrelated", ["alpha", "beta"]) == "alpha"


# --- adapted ---------------------------------------------------------------


def test_a_public_suite_carries_its_link_and_its_contamination_warning(agent_spec):
    cases = adapted(analyze(agent_spec), profile(agent_spec), ["refund requests"])
    for case in cases:
        assert case.source_url and case.source_url.startswith("https://")
        assert any("training data" in note for note in case.notes)
        assert case.fixture.kind is FixtureKind.EXTERNAL_DATASET


def test_nothing_is_adapted_when_the_catalogue_has_nothing_to_say(agent_spec):
    agent_spec.subject.kind = SubjectKind.OTHER
    agent_spec.eval_types = []
    agent_spec.kpis = []
    assert adapted(analyze(agent_spec), profile(agent_spec), ["the job"]) == []


# --- synthesised -----------------------------------------------------------


def test_the_dangerous_cases_exist_only_where_the_tools_do(agent_spec):
    _, _, _, cases = build(agent_spec)
    assert "safety.prompt_injection" in families(cases)

    agent_spec.subject.description = "Answers questions from the documentation."
    agent_spec.subject.in_scope = ["answering questions"]
    agent_spec.summary = "Evaluate a read-only question answering agent."
    agent_spec.eval_types = [EvalType.SAFETY]
    _, _, _, harmless = build(agent_spec)
    assert "safety.prompt_injection" not in families(harmless)


def test_an_injection_case_forbids_the_tool_it_is_trying_to_reach(agent_spec):
    _, _, _, cases = build(agent_spec)
    injection = next(c for c in cases if c.family == "safety.prompt_injection")
    assert VerifierKind.NO_INJECTED_INSTRUCTION in kinds_in(injection)
    assert [s.behaviour for s in injection.stubs] == [StubBehaviour.INJECTION]
    assert any(v.fatal for v in injection.verifiers)


def test_a_case_that_needs_a_tool_is_not_also_forbidden_from_using_it(agent_spec):
    _, _, _, cases = build(agent_spec)
    happy = next(
        c
        for c in cases
        if c.family == "capability.happy_path" and c.behaviour == "refund requests"
    )
    assert "payments" in happy.requires_tools
    assert VerifierKind.TOOL_FORBIDDEN not in kinds_in(happy)


def test_the_cases_that_should_do_nothing_check_that_nothing_happened(agent_spec):
    _, _, _, cases = build(agent_spec)
    for family in ("capability.underspecified", "capability.unsatisfiable"):
        case = next(c for c in cases if c.family == family)
        assert VerifierKind.END_STATE in kinds_in(case)
        assert VerifierKind.TOOL_FORBIDDEN in kinds_in(case)


def test_a_broken_tool_case_scripts_the_break(agent_spec):
    agent_spec.eval_types.append(EvalType.ROBUSTNESS)
    _, _, _, cases = build(agent_spec)
    broken = [c for c in cases if c.family == "robustness.tool_error"]
    assert broken
    assert all(c.stubs for c in broken)
    assert VerifierKind.RECOVERED in kinds_in(broken[0])


def test_nothing_generates_a_regression_case(agent_spec):
    """Inventing an incident tests our imagination, not the agent."""
    _, _, _, cases = build(agent_spec)
    invented = [
        c
        for c in cases
        if EvalType.REGRESSION in c.eval_types and c.source is CaseSource.SYNTHESISED
    ]
    assert invented == []


# --- the whole set ---------------------------------------------------------


def test_synthesis_only_fills_what_is_left(agent_spec):
    """A cell the user's own runs can cover is never filled with something invented."""
    *_, cases = build(agent_spec, per_cell=1)
    regression = [c for c in cases if EvalType.REGRESSION in c.eval_types]
    assert all(c.source is CaseSource.HARVESTED for c in regression)


def test_ids_are_stable_across_runs(agent_spec):
    first = [c.id for c in build(agent_spec)[3]]
    second = [c.id for c in build(agent_spec)[3]]
    assert first == second
    assert len(set(first)) == len(first)


def test_every_case_carries_at_least_one_check(agent_spec):
    _, _, _, cases = build(agent_spec)
    assert all(case.verifiers for case in cases)


def test_every_case_inherits_the_ceiling(agent_spec):
    _, agent_profile, _, cases = build(agent_spec)
    assert all(
        case.budget.max_steps == agent_profile.budget.max_steps for case in cases
    )


def test_generated_cases_admit_their_fixtures_do_not_exist(agent_spec):
    _, _, _, cases = build(agent_spec)
    generated = {c.id for c in cases if c.source is CaseSource.SYNTHESISED}
    assert generated <= {c.id for c in needs_fixture(cases)}


def test_a_spec_with_nothing_of_its_own_is_mostly_generated(agent_spec):
    agent_spec.evidence = []
    _, _, _, cases = build(agent_spec)
    assert synthetic_share(cases) > MAX_SYNTHETIC_SHARE


def test_more_evidence_moves_the_mix(agent_spec):
    _, _, _, before = build(agent_spec, per_cell=1)
    agent_spec.evidence.extend(
        EvidenceItem(
            kind=EvidenceKind.FAILURE_CASE,
            reference=f"incident {i}",
            status=EvidenceStatus.PROVIDED,
        )
        for i in range(6)
    )
    _, _, _, after = build(agent_spec, per_cell=1)
    assert synthetic_share(after) < synthetic_share(before)

from auto_eval.agent import (
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_STEPS,
    AgentProfile,
    Effect,
    Isolation,
    RunMode,
    agent_gate,
    profile,
)
from auto_eval.gaps import analyze
from auto_eval.schema import (
    KPI,
    Direction,
    EvidenceItem,
    EvidenceKind,
    EvidenceStatus,
    Measurement,
    MetricKind,
    SubjectKind,
)


def tool_names(agent_profile: AgentProfile):
    return {tool.name for tool in agent_profile.tools}


# --- reading the entry point ----------------------------------------------


def test_run_mode_is_read_off_the_interface(agent_spec):
    assert profile(agent_spec).run_mode is RunMode.HTTP

    agent_spec.subject.interface = "the `support` CLI command"
    assert profile(agent_spec).run_mode is RunMode.SUBPROCESS

    agent_spec.subject.interface = "import support_agent and call run()"
    assert profile(agent_spec).run_mode is RunMode.PYTHON


def test_an_unrecognised_interface_is_still_an_interface(agent_spec):
    """A sentence we cannot classify is one a person can still act on."""
    agent_spec.subject.interface = "Ana kicks it off from the ops console"
    assert profile(agent_spec).run_mode is RunMode.SUBPROCESS


def test_no_interface_but_recorded_runs_means_score_the_recordings(agent_spec):
    agent_spec.subject.interface = None
    agent_profile = profile(agent_spec)
    assert agent_profile.run_mode is RunMode.TRANSCRIPT
    assert agent_gate(analyze(agent_spec), agent_profile).open is True


def test_no_interface_and_no_recordings_blocks(agent_spec):
    agent_spec.subject.interface = None
    agent_spec.evidence = []
    decision = agent_gate(analyze(agent_spec))
    assert decision.open is False
    assert [q.field for q in decision.blocking] == ["subject.interface"]


# --- tools and containment -------------------------------------------------


def test_tools_are_read_off_the_wording(agent_spec):
    agent_profile = profile(agent_spec)
    assert "payments" in tool_names(agent_profile)
    assert "database" in tool_names(agent_profile)
    assert agent_profile.worst_effect is Effect.DESTRUCTIVE


def test_a_tool_keeps_its_worst_effect(agent_spec):
    """'query the database' and 'update the database' is one tool that writes."""
    agent_spec.subject.description = "Queries the database and updates the database."
    names = {t.name: t.effect for t in profile(agent_spec).tools}
    assert names["database"] is Effect.WRITE


def test_everything_derived_is_recorded_as_an_assumption(agent_spec):
    assert any("Tools read off" in note for note in profile(agent_spec).assumptions)


def test_a_destructive_tool_with_nowhere_safe_to_run_blocks(agent_spec):
    agent_spec.constraints = []
    decision = agent_gate(analyze(agent_spec))
    assert decision.open is False
    assert [q.field for q in decision.blocking] == ["agent.isolation"]


def test_a_sandbox_opens_the_gate(agent_spec):
    agent_profile = profile(agent_spec)
    assert agent_profile.isolation is Isolation.CONTAINER
    assert agent_gate(analyze(agent_spec), agent_profile).open is True


def test_a_transcript_suite_needs_no_sandbox(agent_spec):
    """Nothing is live, so there is nothing to contain."""
    agent_spec.subject.interface = None
    agent_spec.constraints = []
    assert agent_gate(analyze(agent_spec)).open is True


# --- the gate --------------------------------------------------------------


def test_a_blocked_spec_blocks_here_too(subjectless_spec):
    """The upstream gate is not re-litigated; it is deferred to."""
    assert agent_gate(analyze(subjectless_spec)).open is False


def test_something_without_a_trajectory_is_refused_rather_than_questioned(full_spec):
    """A classifier has no trajectory, so a trajectory suite measures nothing."""
    full_spec.subject.kind = SubjectKind.CLASSIFIER
    decision = agent_gate(full_spec)
    assert decision.open is False
    assert decision.blocking == []
    assert "not an agent" in decision.reason


def test_a_workflow_counts_as_agentic(agent_spec):
    agent_spec.subject.kind = SubjectKind.WORKFLOW
    assert agent_gate(analyze(agent_spec)).open is True


# --- ceilings --------------------------------------------------------------


def test_an_uncapped_agent_gets_a_default_stop(agent_spec):
    budget = profile(agent_spec).budget
    assert budget.max_steps == DEFAULT_MAX_STEPS
    assert budget.max_seconds == DEFAULT_MAX_SECONDS


def test_a_latency_target_becomes_a_timeout_with_headroom(agent_spec):
    """The ceiling is a stop, not the target: a slow run is a result, not a hang."""
    agent_spec.kpis.append(
        KPI(
            name="p95 latency",
            kind=MetricKind.QUANTITATIVE,
            definition="95th percentile wall clock for one run.",
            measurement=Measurement.TELEMETRY,
            direction=Direction.MINIMIZE,
            unit="ms",
            target="3000 ms",
        )
    )
    assert profile(agent_spec).budget.max_seconds == 9


def test_a_stated_determinism_drops_the_repeat_runs(agent_spec):
    assert profile(agent_spec).samples > 1
    agent_spec.constraints.append("the agent runs at temperature 0")
    assert profile(agent_spec).samples == 1


# --- determinism -----------------------------------------------------------


def test_profiling_is_deterministic(agent_spec):
    assert profile(agent_spec) == profile(agent_spec)


def test_absent_evidence_is_not_a_recording(agent_spec):
    agent_spec.subject.interface = None
    agent_spec.evidence = [
        EvidenceItem(
            kind=EvidenceKind.SUCCESSFUL_RUN,
            reference="the runs we never kept",
            status=EvidenceStatus.ABSENT,
        )
    ]
    assert profile(agent_spec).run_mode is RunMode.UNKNOWN


def test_a_mentioned_recording_is_not_one_either(agent_spec):
    agent_spec.subject.interface = None
    agent_spec.evidence = [
        EvidenceItem(
            kind=EvidenceKind.SUCCESSFUL_RUN,
            reference="transcripts in the ticket system",
            status=EvidenceStatus.MENTIONED_NOT_PROVIDED,
        )
    ]
    assert profile(agent_spec).run_mode is RunMode.UNKNOWN


# --- does it hold anything -------------------------------------------------


def test_an_agent_that_writes_holds_state(agent_spec):
    assert profile(agent_spec).stateful is True


def test_reaching_the_outside_world_is_not_holding_state(agent_spec):
    """A web search changes nothing, so there is no starting world to build."""
    agent_spec.summary = "Evaluate the agent that searches the web and reads pages."
    agent_spec.subject.description = (
        "Runs a web search and extracts content from a URL."
    )
    agent_spec.subject.in_scope = ["web search"]
    agent_spec.subject.inputs = "a query"
    agent_profile = profile(agent_spec)
    assert agent_profile.tools_with(Effect.EXTERNAL)
    assert agent_profile.stateful is False


def test_statelessness_is_recorded_as_an_assumption_not_assumed_silently(agent_spec):
    agent_spec.summary = "Evaluate a read-only lookup agent."
    agent_spec.subject.description = "Runs a web search."
    agent_spec.subject.in_scope = ["web search"]
    agent_spec.subject.inputs = "a query"
    assert any("holding no state" in note for note in profile(agent_spec).assumptions)

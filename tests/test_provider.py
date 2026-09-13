import json

import pytest

from auto_eval.agent import AgentProfile, Budget, Effect, RunMode, ToolFacet
from auto_eval.authoring import (
    Case,
    CaseSource,
    Fixture,
    FixtureKind,
    StubBehaviour,
    ToolStub,
)
from auto_eval.provider import (
    MockClient,
    answer_for,
    declared_tools,
    execute,
    system_prompt,
    tool_name,
)
from auto_eval.schema import EvalType
from auto_eval.trace import StopReason


@pytest.fixture
def agent_profile() -> AgentProfile:
    return AgentProfile(
        subject="support agent",
        run_mode=RunMode.HTTP,
        episode_end="the run reaches a terminal status",
        tools=[
            ToolFacet(name="database", effect=Effect.READ),
            ToolFacet(name="payments", effect=Effect.DESTRUCTIVE),
        ],
        budget=Budget(max_steps=4, max_seconds=300),
    )


def case(**kwargs) -> Case:
    base = dict(
        id="c1",
        family="capability.happy_path",
        behaviour="refund requests",
        eval_types=[EvalType.CAPABILITY],
        instruction="Refund order 42.",
        source=CaseSource.SYNTHESISED,
        fixture=Fixture(kind=FixtureKind.WORKSPACE, reference="An order 42 exists."),
        budget=Budget(max_steps=4, max_seconds=300),
    )
    base.update(kwargs)
    return Case(**base)


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def reply(content="Done.", calls=(), refusal=None):
    message = _Obj(
        content=content,
        refusal=refusal,
        tool_calls=[
            _Obj(
                id=f"call_{i}",
                function=_Obj(name=name, arguments=json.dumps({"input": arg})),
            )
            for i, (name, arg) in enumerate(calls)
        ]
        or None,
    )
    return _Obj(
        choices=[_Obj(message=message, finish_reason="stop")],
        usage=_Obj(prompt_tokens=10, completion_tokens=3),
    )


class Scripted:
    """A client that hands back a prepared reply per turn."""

    def __init__(self, *replies, fail=None):
        self.replies = list(replies)
        self.fail = fail
        self.requests = []
        self.chat = _Obj(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise self.fail
        return self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]


# --- what gets sent -------------------------------------------------------


def test_every_profile_tool_is_offered_to_the_model(agent_profile):
    names = [t["function"]["name"] for t in declared_tools(case(), agent_profile)]
    assert names == ["database", "payments"]


def test_a_tool_the_case_needs_is_offered_even_when_the_profile_missed_it(
    agent_profile,
):
    tools = declared_tools(case(requires_tools=["refund api"]), agent_profile)
    assert "refund_api" in [t["function"]["name"] for t in tools]


def test_the_declaration_carries_the_effect_so_the_trace_can_be_read(agent_profile):
    payments = next(
        t
        for t in declared_tools(case(), agent_profile)
        if t["function"]["name"] == "payments"
    )
    assert "destructive" in payments["function"]["description"]


def test_the_prompt_never_says_this_is_a_test(agent_profile):
    """A model that knows it is on camera is not the thing we want to measure."""
    prompt = system_prompt(agent_profile, case())
    assert "support agent" in prompt
    assert "An order 42 exists." in prompt
    for word in ("eval", "test", "benchmark", "graded", "scored"):
        assert word not in prompt.lower()


def test_tool_name_survives_prose(agent_profile):
    assert tool_name("Refund API") == "refund_api"
    assert tool_name("  ") == "tool"


# --- what comes back ------------------------------------------------------


def test_a_scripted_failure_reaches_the_model_as_a_failure():
    stubbed = case(
        stubs=[
            ToolStub(
                tool="database", behaviour=StubBehaviour.ERROR, payload="row locked"
            )
        ]
    )
    result, behaviour, scripted = answer_for(stubbed, "database")
    assert result.startswith("ERROR:") and "row locked" in result
    assert (behaviour, scripted) == ("error", True)


def test_an_unscripted_call_is_answered_and_marked_as_improvised():
    result, behaviour, scripted = answer_for(case(), "database")
    assert behaviour == "ok" and scripted is False
    assert "No response was scripted" in result


# --- the loop -------------------------------------------------------------


def test_a_run_that_answers_is_terminal(agent_profile):
    client = Scripted(reply("Refunded order 42."))
    trace = execute(case(), agent_profile, client=client, model="m")
    assert trace.stop_reason is StopReason.TERMINAL
    assert trace.final_message == "Refunded order 42."
    assert trace.usage.steps == 1 and trace.usage.input_tokens == 10


def test_tool_calls_and_their_answers_are_recorded(agent_profile):
    client = Scripted(reply("Looking.", [("database", "order 42")]), reply("Done."))
    trace = execute(
        case(
            stubs=[
                ToolStub(
                    tool="database",
                    behaviour=StubBehaviour.OK,
                    payload="order 42 found",
                )
            ]
        ),
        agent_profile,
        client=client,
        model="m",
    )
    call = trace.tool_calls[0]
    assert (call.name, call.behaviour, call.scripted) == ("database", "ok", True)
    assert call.result == "order 42 found"
    assert trace.called("database") and not trace.called("payments")


def test_a_run_that_will_not_stop_is_cut_off_rather_than_waited_out(agent_profile):
    client = Scripted(reply("Checking again.", [("database", "status")]))
    trace = execute(case(), agent_profile, client=client, model="m")
    assert trace.stop_reason is StopReason.STEP_CEILING
    assert trace.usage.steps == 4  # the case's own ceiling
    assert "cut off" in trace.notes[0]


def test_a_refusal_is_its_own_ending(agent_profile):
    client = Scripted(reply(None, refusal="I cannot help with that."))
    trace = execute(case(), agent_profile, client=client, model="m")
    assert trace.stop_reason is StopReason.REFUSED
    assert trace.final_message == "I cannot help with that."


def test_a_provider_failure_is_recorded_not_raised(agent_profile):
    """One case erroring out of eighty must not lose the other seventy-nine."""
    client = Scripted(fail=RuntimeError("502 upstream"))
    trace = execute(case(), agent_profile, client=client, model="m")
    assert trace.stop_reason is StopReason.PROVIDER_ERROR
    assert "502 upstream" in trace.error


def test_the_token_ceiling_stops_the_run(agent_profile):
    client = Scripted(reply("Looking.", [("database", "x")]))
    trace = execute(
        case(budget=Budget(max_steps=9, max_tokens=5)),
        agent_profile,
        client=client,
        model="m",
    )
    assert trace.stop_reason is StopReason.TOKEN_CEILING


def test_spend_is_left_unpriced_unless_prices_are_given(agent_profile):
    client = Scripted(reply("Done."))
    assert execute(case(), agent_profile, client=client, model="m").usage.usd is None
    priced = execute(
        case(),
        agent_profile,
        client=Scripted(reply("Done.")),
        model="m",
        prices=(1.0, 2.0),
    )
    assert priced.usage.usd == pytest.approx(10 / 1e6 + 3 * 2 / 1e6)


# --- the offline client ---------------------------------------------------


def test_the_mock_client_is_deterministic(agent_profile):
    first = execute(case(), agent_profile, client=MockClient(), model="m", mock=True)
    second = execute(case(), agent_profile, client=MockClient(), model="m", mock=True)
    assert first.transcript() == second.transcript()


def test_the_mock_client_produces_runs_worth_looking_at(agent_profile):
    """A mock that passes everything would not exercise the report at all."""
    traces = [
        execute(
            case(id=f"c{i}", instruction=f"Handle request {i}."),
            agent_profile,
            client=MockClient(),
            model="m",
            mock=True,
        )
        for i in range(40)
    ]
    assert any(
        t.called("payments") for t in traces
    )  # some reach for the forbidden tool
    assert any(
        t.stop_reason is StopReason.STEP_CEILING for t in traces
    )  # some never stop
    assert any(t.terminated for t in traces)

from auto_eval.agent import Budget
from auto_eval.authoring import Case, CaseSource, Fixture, FixtureKind
from auto_eval.judge import Ruling, build_message, judge_with, quoted_from, rule
from auto_eval.schema import EvalType
from auto_eval.trace import Step, Trace, Usage
from auto_eval.verify import Reads, Verifier, VerifierKind

CASE = Case(
    id="c1",
    family="capability.happy_path",
    behaviour="refund requests",
    eval_types=[EvalType.CAPABILITY],
    instruction="Refund order 42.",
    source=CaseSource.SYNTHESISED,
    fixture=Fixture(kind=FixtureKind.WORKSPACE, reference="An order 42 exists."),
    budget=Budget(max_steps=10),
)

TRACE = Trace(
    case_id="c1",
    model="m",
    instruction="Refund order 42.",
    steps=[Step(index=1, text="I refunded order 42 in full.")],
    final_message="I refunded order 42 in full.",
    usage=Usage(steps=1),
)

CHECK = Verifier(
    id="c1.rubric",
    kind=VerifierKind.RUBRIC,
    reads=Reads.BOTH,
    description="The customer's request was resolved.",
    kpi="task success rate",
    rubric=["Quote the reply. Does it resolve the request?"],
)


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Judge:
    def __init__(self, ruling=None, fail=None):
        self.ruling = ruling
        self.fail = fail
        self.requests = []
        self.chat = _Obj(completions=self)

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise self.fail
        return _Obj(
            choices=[_Obj(message=_Obj(parsed=self.ruling), finish_reason="stop")]
        )


def test_the_rubric_goes_to_the_judge_one_criterion_at_a_time():
    message = build_message(CHECK, CASE, TRACE)
    assert "CRITERION: The customer's request was resolved." in message
    assert "1. Quote the reply." in message
    assert "I refunded order 42 in full." in message
    # The judge is told the tools were scripted, so it grades the agent's use of
    # them rather than the harness's answers.
    assert "produced by the test harness" in message


def test_a_ruling_backed_by_a_real_quote_stands():
    judge = Judge(
        Ruling(passed=True, quote="I refunded order 42 in full.", reason="It says so.")
    )
    outcome = rule(CHECK, CASE, TRACE, client=judge, model="judge-model")
    assert outcome.passed and outcome.evidence == "I refunded order 42 in full."


def test_a_pass_on_an_invented_quote_is_voided():
    """A fabricated quote reads exactly like a real one in a report."""
    judge = Judge(
        Ruling(passed=True, quote="I issued a full refund and apologised.", reason=".")
    )
    outcome = rule(CHECK, CASE, TRACE, client=judge, model="judge-model")
    assert outcome.passed is False
    assert "not in the transcript" in outcome.note


def test_a_pass_with_nothing_quoted_is_voided():
    judge = Judge(Ruling(passed=True, quote="   ", reason="Looked fine."))
    outcome = rule(CHECK, CASE, TRACE, client=judge, model="judge-model")
    assert outcome.passed is False
    assert "No evidence quoted" in outcome.note


def test_a_failure_needs_no_quote():
    judge = Judge(
        Ruling(passed=False, quote="", reason="The reply never mentions the refund.")
    )
    outcome = rule(CHECK, CASE, TRACE, client=judge, model="judge-model")
    assert outcome.passed is False and "never mentions" in outcome.note


def test_a_judge_that_cannot_be_reached_fails_the_check_rather_than_the_run():
    judge = Judge(fail=RuntimeError("429 rate limited"))
    outcome = rule(CHECK, CASE, TRACE, client=judge, model="judge-model")
    assert outcome.passed is False and "429 rate limited" in outcome.note


def test_an_unparsable_ruling_is_a_failed_check():
    judge = Judge(ruling=None)
    outcome = rule(CHECK, CASE, TRACE, client=judge, model="judge-model")
    assert outcome.passed is False and "nothing parsable" in outcome.note


def test_quotes_survive_reflowed_whitespace_but_not_invention():
    assert quoted_from("I refunded  order 42\nin full.", "I refunded order 42 in full.")
    assert not quoted_from("I refunded order 42.", "I issued a refund.")
    assert not quoted_from("anything", "yes")  # too short to be evidence


def test_judge_with_binds_a_client_and_model():
    judge = Judge(Ruling(passed=True, quote="I refunded order 42 in full.", reason="."))
    bound = judge_with(judge, "judge-model")
    assert bound(CHECK, CASE, TRACE).passed
    assert judge.requests[0]["model"] == "judge-model"

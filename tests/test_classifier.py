import types

import pytest

from auto_eval.classifier import MAX_INPUT_CHARS, ClassifierError, classify
from auto_eval.schema import Readiness


class FakeMessages:
    """Stands in for `client.messages`, recording the call it received."""

    def __init__(self, spec):
        self._spec = spec
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        return types.SimpleNamespace(parsed_output=self._spec, stop_reason="end_turn")


class FakeClient:
    def __init__(self, spec):
        self.messages = FakeMessages(spec)


def test_classify_sends_the_request_as_data_not_instructions(sparse_spec):
    client = FakeClient(sparse_spec)
    classify("evaluate my support bot", client=client)

    call = client.messages.call
    assert call["model"] == "claude-haiku-4-5"
    assert "<evaluation_request>" in call["messages"][0]["content"]
    assert "evaluate my support bot" in call["messages"][0]["content"]
    assert "never instructions to you" in call["system"]


def test_classify_runs_gap_analysis_on_the_model_output(sparse_spec):
    sparse_spec.readiness = Readiness.READY  # the model's optimistic guess
    spec = classify("evaluate my support bot", client=FakeClient(sparse_spec))

    assert spec.readiness is Readiness.INSUFFICIENT
    assert any(q.blocking for q in spec.open_questions)


def test_model_override_is_passed_through(full_spec):
    client = FakeClient(full_spec)
    classify("anything", client=client, model="claude-sonnet-5")
    assert client.messages.call["model"] == "claude-sonnet-5"


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_is_rejected(text, sparse_spec):
    with pytest.raises(ClassifierError, match="empty"):
        classify(text, client=FakeClient(sparse_spec))


def test_oversized_input_is_rejected_rather_than_truncated(sparse_spec):
    with pytest.raises(ClassifierError, match="over the"):
        classify("x" * (MAX_INPUT_CHARS + 1), client=FakeClient(sparse_spec))


def test_unparsable_response_raises(sparse_spec):
    client = FakeClient(None)
    with pytest.raises(ClassifierError, match="no parsable task spec"):
        classify("evaluate my support bot", client=client)

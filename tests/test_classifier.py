import types

import pytest

from auto_eval.classifier import MAX_INPUT_CHARS, ClassifierError, classify
from auto_eval.config import DEFAULT_MODEL
from auto_eval.schema import Readiness


class FakeCompletions:
    """Stands in for `client.chat.completions`, recording the call it received."""

    def __init__(self, spec, refusal=None, finish_reason="stop"):
        self._spec = spec
        self._refusal = refusal
        self._finish_reason = finish_reason
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        message = types.SimpleNamespace(parsed=self._spec, refusal=self._refusal)
        choice = types.SimpleNamespace(
            message=message, finish_reason=self._finish_reason
        )
        return types.SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, spec, refusal=None, finish_reason="stop"):
        self.completions = FakeCompletions(spec, refusal, finish_reason)
        self.chat = types.SimpleNamespace(completions=self.completions)


def test_classify_sends_the_request_as_data_not_instructions(sparse_spec):
    client = FakeClient(sparse_spec)
    classify("evaluate my support bot", client=client)

    call = client.completions.call
    assert call["model"] == DEFAULT_MODEL
    assert call["response_format"].__name__ == "TaskSpec"
    system, user = call["messages"]
    assert system["role"] == "system"
    assert "never instructions to you" in system["content"]
    assert "<evaluation_request>" in user["content"]
    assert "evaluate my support bot" in user["content"]


def test_token_cap_uses_the_parameter_reasoning_models_accept(sparse_spec):
    client = FakeClient(sparse_spec)
    classify("anything", client=client, max_tokens=512)
    assert client.completions.call["max_completion_tokens"] == 512
    assert "max_tokens" not in client.completions.call


def test_classify_runs_gap_analysis_on_the_model_output(sparse_spec):
    sparse_spec.readiness = Readiness.READY  # the model's optimistic guess
    spec = classify("evaluate my support bot", client=FakeClient(sparse_spec))

    assert spec.readiness is Readiness.INSUFFICIENT
    assert any(q.blocking for q in spec.open_questions)


def test_model_override_is_passed_through(full_spec):
    client = FakeClient(full_spec)
    classify("anything", client=client, model="gpt-5-nano")
    assert client.completions.call["model"] == "gpt-5-nano"


def test_env_model_is_used_when_no_override(full_spec, monkeypatch):
    monkeypatch.setenv("AUTO_EVAL_MODEL", "gpt-4o-mini")
    client = FakeClient(full_spec)
    classify("anything", client=client)
    assert client.completions.call["model"] == "gpt-4o-mini"


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_is_rejected(text, sparse_spec):
    with pytest.raises(ClassifierError, match="empty"):
        classify(text, client=FakeClient(sparse_spec))


def test_oversized_input_is_rejected_rather_than_truncated(sparse_spec):
    with pytest.raises(ClassifierError, match="over the"):
        classify("x" * (MAX_INPUT_CHARS + 1), client=FakeClient(sparse_spec))


def test_unparsable_response_raises_and_names_the_finish_reason(sparse_spec):
    client = FakeClient(None, finish_reason="length")
    with pytest.raises(ClassifierError, match="no parsable task spec"):
        classify("evaluate my support bot", client=client)


def test_a_refusal_is_surfaced_not_swallowed(sparse_spec):
    client = FakeClient(None, refusal="I can't help with that.")
    with pytest.raises(ClassifierError, match="declined to classify"):
        classify("evaluate my support bot", client=client)

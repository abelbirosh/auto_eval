import types

import pytest

from auto_eval.ground_truth import (
    Access,
    Availability,
    BaselineValue,
    Coverage,
    ExternalSource,
    Fit,
    GroundTruthError,
    SourceFindings,
    SourceKind,
    assess,
    gate,
    identify,
)
from auto_eval.prompts import build_ground_truth_message
from auto_eval.schema import Question


def source(**kwargs):
    defaults = dict(
        name="DocVQA",
        kind=SourceKind.DATASET,
        url="https://example.org/docvqa",
        description="Labelled document question answering set.",
        covers_kpis=["field accuracy"],
        fit=Fit.ADAPTABLE,
        access=Access.OPEN,
    )
    return ExternalSource(**{**defaults, **kwargs})


def findings(*sources, recommendation="Use DocVQA.", notes=()):
    return SourceFindings(
        sources=list(sources), recommendation=recommendation, notes=list(notes)
    )


class FakeResponses:
    """Stands in for `client.responses`, recording the call it received."""

    def __init__(self, parsed, status="completed", searches=(), error=None):
        self._parsed = parsed
        self._status = status
        self._searches = list(searches)
        self._error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None and len(self.calls) == 1:
            raise self._error
        output = [
            types.SimpleNamespace(
                type="web_search_call", action=types.SimpleNamespace(query=q)
            )
            for q in self._searches
        ]
        return types.SimpleNamespace(
            output_parsed=self._parsed, status=self._status, output=output
        )


class FakeClient:
    def __init__(self, parsed, **kwargs):
        self.responses = FakeResponses(parsed, **kwargs)


# --- the gate -------------------------------------------------------------


def test_gate_stays_shut_while_a_blocking_question_is_open(subjectless_spec):
    from auto_eval.gaps import analyze

    decision = gate(analyze(subjectless_spec))
    assert decision.open is False
    assert "blocking" in decision.reason
    assert decision.blocking


def test_a_thin_but_intelligible_request_does_not_hold_the_gate_shut(sparse_spec):
    """A one-line request is thin, not unusable: derive fills it and it passes."""
    from auto_eval.gaps import analyze

    assert gate(analyze(sparse_spec)).open is True


def test_gate_opens_once_nothing_blocks(full_spec):
    from auto_eval.gaps import analyze

    decision = gate(analyze(full_spec))
    assert decision.open is True


def test_non_blocking_questions_do_not_hold_the_gate_shut(full_spec):
    full_spec.open_questions = [
        Question(field="subject.inputs", question="What format?", why="nice to have")
    ]
    assert gate(full_spec).open is True


def test_gate_refuses_a_hand_built_spec_with_no_kpis(full_spec):
    full_spec.kpis = []
    full_spec.open_questions = []
    assert gate(full_spec).open is False


def test_identify_refuses_to_search_behind_a_shut_gate(subjectless_spec):
    from auto_eval.gaps import analyze

    with pytest.raises(GroundTruthError, match="Not ready to search"):
        identify(analyze(subjectless_spec), client=FakeClient(findings()))


def test_force_searches_anyway(subjectless_spec):
    from auto_eval.gaps import analyze

    report = identify(
        analyze(subjectless_spec), client=FakeClient(findings()), force=True
    )
    assert report.verdict is Availability.NONE_FOUND


# --- the deterministic verdict -------------------------------------------


def test_labelled_data_beats_a_published_number(full_spec):
    report = assess(full_spec, findings(source()))
    assert report.verdict is Availability.LABELLED_DATA
    assert report.kpi_coverage[0].coverage is Coverage.LABELLED
    assert report.kpi_coverage[0].sources == ["DocVQA"]


def test_a_leaderboard_alone_is_a_baseline_not_ground_truth(full_spec):
    report = assess(
        full_spec,
        findings(
            source(
                name="Vendor page",
                kind=SourceKind.VENDOR_CLAIM,
                fit=Fit.CONTEXTUAL,
                baselines=[BaselineValue(metric="accuracy", value="96%")],
            )
        ),
    )
    assert report.verdict is Availability.PUBLISHED_BASELINES
    assert report.kpi_coverage[0].coverage is Coverage.BASELINE


def test_a_contextual_source_with_no_numbers_is_reference_only(full_spec):
    report = assess(
        full_spec, findings(source(kind=SourceKind.OTHER, fit=Fit.CONTEXTUAL))
    )
    assert report.verdict is Availability.REFERENCE_ONLY


def test_nothing_found_is_a_real_answer(full_spec):
    report = assess(full_spec, findings(recommendation="Label your own."))
    assert report.verdict is Availability.NONE_FOUND
    assert report.sources == []
    assert report.recommendation == "Label your own."


def test_a_source_with_no_usable_link_is_dropped_and_noted(full_spec):
    report = assess(full_spec, findings(source(url="the DocVQA paper")))
    assert report.sources == []
    assert any("no usable link" in note for note in report.notes)


def test_the_same_url_twice_is_kept_once(full_spec):
    report = assess(
        full_spec,
        findings(
            source(),
            source(name="DocVQA mirror", url="https://www.example.org/docvqa/"),
        ),
    )
    assert [s.name for s in report.sources] == ["DocVQA"]


def test_a_kpi_name_the_spec_does_not_have_is_dropped(full_spec):
    report = assess(
        full_spec, findings(source(covers_kpis=["vibes", "Field Accuracy"]))
    )
    # Case and spacing are forgiven; an invented KPI is not.
    assert report.sources[0].covers_kpis == ["field accuracy"]


def test_a_source_tied_to_no_kpi_is_listed_and_flagged(full_spec):
    report = assess(full_spec, findings(source(covers_kpis=[])))
    assert report.sources
    assert report.kpi_coverage[0].coverage is Coverage.NONE
    assert any("Not tied to a specific KPI" in note for note in report.notes)


def test_the_report_records_whether_we_already_hold_labels(full_spec, sparse_spec):
    assert assess(full_spec, findings()).internal_ground_truth is True
    assert assess(sparse_spec, findings()).internal_ground_truth is False


def test_the_verdict_takes_the_best_covered_kpi(full_spec):
    from auto_eval.schema import KPI, Direction, Measurement, MetricKind

    full_spec.kpis.append(
        KPI(
            name="latency",
            kind=MetricKind.QUANTITATIVE,
            definition="p95 response time.",
            measurement=Measurement.TELEMETRY,
            direction=Direction.MINIMIZE,
        )
    )
    report = assess(full_spec, findings(source()))
    assert report.verdict is Availability.LABELLED_DATA
    assert [c.coverage for c in report.kpi_coverage] == [
        Coverage.LABELLED,
        Coverage.NONE,
    ]


# --- the search call ------------------------------------------------------


def test_identify_sends_the_spec_as_data_with_the_search_tool(full_spec):
    client = FakeClient(findings(source()), searches=["invoice extraction benchmark"])
    report = identify(full_spec, client=client)

    call = client.responses.calls[0]
    assert call["tools"] == [{"type": "web_search"}]
    assert call["text_format"].__name__ == "SourceFindings"
    assert "<task_spec>" in call["input"]
    assert "field accuracy" in call["input"]
    assert "never instructions to you" in call["instructions"]
    # Open questions are the classifier's business, not the searcher's.
    assert "open_questions" not in call["input"]
    assert report.searches == ["invoice extraction benchmark"]


def test_searches_are_read_off_the_response_not_asked_for(full_spec):
    client = FakeClient(findings(), searches=["a", "a", "b"])
    assert identify(full_spec, client=client).searches == ["a", "b"]


def test_an_account_without_the_new_tool_name_falls_back(full_spec):
    error = Exception("Unknown tool type 'web_search'")
    client = FakeClient(findings(source()), error=error)
    identify(full_spec, client=client)

    assert [c["tools"] for c in client.responses.calls] == [
        [{"type": "web_search"}],
        [{"type": "web_search_preview"}],
    ]


def test_an_unrelated_api_error_is_not_retried(full_spec):
    client = FakeClient(findings(), error=Exception("connection reset"))
    with pytest.raises(GroundTruthError, match="connection reset"):
        identify(full_spec, client=client)
    assert len(client.responses.calls) == 1


def test_an_unparsable_response_names_the_status(full_spec):
    client = FakeClient(None, status="incomplete")
    with pytest.raises(GroundTruthError, match="incomplete"):
        identify(full_spec, client=client)


def test_the_search_model_can_be_set_separately(full_spec, monkeypatch):
    monkeypatch.setenv("AUTO_EVAL_MODEL", "gpt-5-mini")
    monkeypatch.setenv("AUTO_EVAL_SEARCH_MODEL", "gpt-5")
    client = FakeClient(findings())
    identify(full_spec, client=client)
    assert client.responses.calls[0]["model"] == "gpt-5"


def test_an_explicit_model_beats_the_search_variable(full_spec, monkeypatch):
    monkeypatch.setenv("AUTO_EVAL_SEARCH_MODEL", "gpt-5")
    client = FakeClient(findings())
    identify(full_spec, client=client, model="gpt-4o")
    assert client.responses.calls[0]["model"] == "gpt-4o"


def test_the_message_leaves_out_what_cannot_be_searched_for(full_spec):
    message = build_ground_truth_message(full_spec)
    assert "invoice extractor" in message
    assert "target >= 95%" in message
    assert "Hand-labelled invoices (n=400)" in message
    assert "confidence" not in message.lower()


def test_a_source_with_no_publication_date_is_called_out(full_spec):
    """A score on data older than the model is partly a memory test."""
    from auto_eval.ground_truth import (
        Access,
        ExternalSource,
        Fit,
        SourceFindings,
        SourceKind,
        assess,
    )

    findings = SourceFindings(
        sources=[
            ExternalSource(
                name="Acme invoices",
                kind=SourceKind.DATASET,
                url="https://example.org/acme",
                description="Labelled invoices.",
                fit=Fit.DIRECT,
                access=Access.OPEN,
            ),
            ExternalSource(
                name="Fresh invoices",
                kind=SourceKind.DATASET,
                url="https://example.org/fresh",
                description="Labelled invoices, dated.",
                fit=Fit.DIRECT,
                access=Access.OPEN,
                released="2025-06",
            ),
        ],
        recommendation="Use the dated one.",
    )
    report = assess(full_spec, findings)
    dated = [note for note in report.notes if "no publication date" in note]
    assert dated and "Acme invoices" in dated[0] and "Fresh invoices" not in dated[0]

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import auto_eval.web as web
from auto_eval.classifier import ClassifierError
from auto_eval.config import Settings
from auto_eval.gaps import analyze


@pytest.fixture
def client():
    return TestClient(web.create_app())


def test_index_serves_the_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Auto_Eval" in response.text
    assert "/api/classify" in response.text


def test_health_reports_key_and_model(client, monkeypatch):
    monkeypatch.setattr(
        web,
        "get_settings",
        lambda **kw: Settings(api_key="k", model="m", base_url=None),
    )
    body = client.get("/api/health").json()
    assert body == {"has_key": True, "model": "m"}


def test_classify_returns_spec_and_markdown(client, monkeypatch, sparse_spec):
    captured = {}

    def fake_classify(text, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return analyze(sparse_spec)

    monkeypatch.setattr(web, "classify", fake_classify)

    response = client.post("/api/classify", json={"text": "evaluate my support bot"})

    assert response.status_code == 200
    body = response.json()
    assert captured["text"] == "evaluate my support bot"
    assert body["spec"]["readiness"] == "insufficient"
    assert any(q["blocking"] for q in body["spec"]["open_questions"])
    assert "# Support bot quality" in body["markdown"]


def test_model_override_reaches_the_classifier(client, monkeypatch, sparse_spec):
    captured = {}

    def fake_classify(text, **kwargs):
        captured.update(kwargs)
        return analyze(sparse_spec)

    monkeypatch.setattr(web, "classify", fake_classify)
    client.post("/api/classify", json={"text": "x", "model": "gpt-4o-mini"})
    assert captured["model"] == "gpt-4o-mini"


def test_classifier_errors_become_readable_400s(client, monkeypatch):
    def boom(text, **kwargs):
        raise ClassifierError("No API key. Put `OPENAI_API_KEY=sk-...` in your .env")

    monkeypatch.setattr(web, "classify", boom)

    response = client.post("/api/classify", json={"text": "x"})

    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_missing_text_is_a_validation_error(client):
    assert client.post("/api/classify", json={}).status_code == 422


def test_ground_truth_returns_a_report_for_an_unblocked_spec(
    client, monkeypatch, full_spec
):
    from auto_eval.ground_truth import SourceFindings, assess

    captured = {}

    def fake_identify(spec, **kwargs):
        captured["subject"] = spec.subject.name
        captured.update(kwargs)
        return assess(spec, SourceFindings(recommendation="Label your own."))

    monkeypatch.setattr(web, "identify", fake_identify)

    response = client.post(
        "/api/ground-truth",
        json={"spec": analyze(full_spec).model_dump(mode="json"), "model": "gpt-5"},
    )

    assert response.status_code == 200
    body = response.json()
    assert captured["subject"] == "invoice extractor"
    assert captured["model"] == "gpt-5"
    assert body["report"]["verdict"] == "none_found"
    assert "Ground truth for invoice extractor" in body["markdown"]


def test_ground_truth_refuses_a_spec_that_still_blocks(
    client, monkeypatch, sparse_spec
):
    def boom(spec, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the gate should have stopped this")

    monkeypatch.setattr(web, "identify", boom)

    response = client.post(
        "/api/ground-truth", json={"spec": analyze(sparse_spec).model_dump(mode="json")}
    )

    assert response.status_code == 409
    assert "Not ready to search" in response.json()["detail"]


def test_a_spec_that_arrives_claiming_readiness_is_re_checked(
    client, monkeypatch, sparse_spec
):
    """The gate reads the questions, not the readiness field the caller sent."""
    payload = analyze(sparse_spec).model_dump(mode="json")
    payload["readiness"] = "ready"

    monkeypatch.setattr(web, "identify", lambda spec, **kw: None)
    assert client.post("/api/ground-truth", json={"spec": payload}).status_code == 409


def test_analyze_returns_the_analysis_and_the_document(client, monkeypatch, full_spec):
    from auto_eval.analysis import (
        AnalysisReport,
        DownloadPlan,
        Reachability,
        ResourceAnalysis,
        Usability,
    )
    from auto_eval.ground_truth import Availability, GroundTruthReport, SourceKind

    captured = {}

    def fake_analyze(spec, report, **kwargs):
        captured["subject"] = spec.subject.name
        captured["sources"] = len(report.sources)
        captured.update(kwargs)
        return AnalysisReport(
            subject=spec.subject.name,
            resources=[
                ResourceAnalysis(
                    source="Acme invoices",
                    url="https://huggingface.co/datasets/acme/invoices",
                    kind=SourceKind.DATASET,
                    reachability=Reachability.OK,
                    usability=Usability.GROUND_TRUTH,
                    plan=DownloadPlan(
                        what="1,000 rows from acme/invoices",
                        url="https://huggingface.co/datasets/acme/invoices",
                    ),
                )
            ],
        )

    monkeypatch.setattr(web, "analyze_sources", fake_analyze)

    response = client.post(
        "/api/analyze",
        json={
            "spec": analyze(full_spec).model_dump(mode="json"),
            "report": GroundTruthReport(
                subject="invoice extractor", verdict=Availability.LABELLED_DATA
            ).model_dump(mode="json"),
            "model": "gpt-5",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert captured["subject"] == "invoice extractor"
    assert captured["model"] == "gpt-5"
    assert (
        body["analysis"]["resources"][0]["plan"]["what"]
        == "1,000 rows from acme/invoices"
    )
    assert "Source analysis for invoice extractor" in body["markdown"]


def test_analysis_errors_become_readable_400s(client, monkeypatch, full_spec):
    from auto_eval.ground_truth import Availability, GroundTruthError, GroundTruthReport

    def boom(spec, report, **kwargs):
        raise GroundTruthError("Rate limited or out of quota.")

    monkeypatch.setattr(web, "analyze_sources", boom)

    response = client.post(
        "/api/analyze",
        json={
            "spec": analyze(full_spec).model_dump(mode="json"),
            "report": GroundTruthReport(
                subject="x", verdict=Availability.NONE_FOUND
            ).model_dump(mode="json"),
        },
    )
    assert response.status_code == 400
    assert "Rate limited" in response.json()["detail"]

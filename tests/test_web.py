import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import auto_eval.web as web  # noqa: E402
from auto_eval.classifier import ClassifierError  # noqa: E402
from auto_eval.gaps import analyze  # noqa: E402


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
        web, "get_settings", lambda **kw: web.Settings(api_key="k", model="m", base_url=None)
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

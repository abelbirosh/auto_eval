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
    # The page posts answers back and gates the button on them being filled.
    assert "submitAnswers" in response.text
    assert "still to answer" in response.text
    # Which benchmark the page offers is the gate's decision, read from health.
    assert "healthState.agentic_kinds" in response.text
    # A source the identifier marked as self-reported says so where it is listed.
    assert "published by a system under test" in response.text


def test_health_reports_key_and_model(client, monkeypatch):
    monkeypatch.setattr(
        web,
        "get_settings",
        lambda **kw: Settings(api_key="k", model="m", base_url=None),
    )
    body = client.get("/api/health").json()
    assert body == {
        "has_key": True,
        "model": "m",
        # The dashboard shows these before a run is started, so a judge that is
        # the same model as the one under test is visible up front.
        "subject_model": "m",
        "judge_model": "m",
        "runs_dir": "runs",
        # The page branches between a suite and a board on this, so it comes
        # from the gate rather than from a copy of the list in the JavaScript.
        "agentic_kinds": ["agent", "workflow"],
    }


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
    # Thin, but nothing a derived KPI cannot carry, so it is not blocked.
    assert body["spec"]["readiness"] == "needs_input"
    assert not any(q["blocking"] for q in body["spec"]["open_questions"])
    assert "# Support bot quality" in body["markdown"]


def test_model_override_reaches_the_classifier(client, monkeypatch, sparse_spec):
    captured = {}

    def fake_classify(text, **kwargs):
        captured.update(kwargs)
        return analyze(sparse_spec)

    monkeypatch.setattr(web, "classify", fake_classify)
    client.post("/api/classify", json={"text": "x", "model": "gpt-4o-mini"})
    assert captured["model"] == "gpt-4o-mini"


def test_answers_reach_the_classifier(client, monkeypatch, full_spec):
    """The answers typed into the blocking boxes go back with the request."""
    captured = {}

    def fake_classify(text, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        return analyze(full_spec)

    monkeypatch.setattr(web, "classify", fake_classify)

    response = client.post(
        "/api/classify",
        json={
            "text": "evaluate my support bot",
            "answers": [
                {
                    "field": "kpis",
                    "question": "What would make this good enough to ship?",
                    "answer": "Policy accuracy, 90% or better.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert captured["text"] == "evaluate my support bot"
    assert [a.answer for a in captured["answers"]] == [
        "Policy accuracy, 90% or better."
    ]
    # The blocking question is gone, which is what the boxes were for.
    assert not any(q["blocking"] for q in response.json()["spec"]["open_questions"])


def test_answers_are_optional(client, monkeypatch, sparse_spec):
    captured = {}

    def fake_classify(text, **kwargs):
        captured.update(kwargs)
        return analyze(sparse_spec)

    monkeypatch.setattr(web, "classify", fake_classify)
    assert client.post("/api/classify", json={"text": "x"}).status_code == 200
    assert list(captured["answers"]) == []


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
    client, monkeypatch, subjectless_spec
):
    def boom(spec, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the gate should have stopped this")

    monkeypatch.setattr(web, "identify", boom)

    response = client.post(
        "/api/ground-truth",
        json={"spec": analyze(subjectless_spec).model_dump(mode="json")},
    )

    assert response.status_code == 409
    assert "Not ready to search" in response.json()["detail"]


def test_a_spec_that_arrives_claiming_readiness_is_re_checked(
    client, monkeypatch, subjectless_spec
):
    """The gate reads the questions, not the readiness field the caller sent."""
    payload = analyze(subjectless_spec).model_dump(mode="json")
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


# --- the run dashboard ----------------------------------------------------


@pytest.fixture
def written_suite(tmp_path, agent_spec):
    from auto_eval.suite import build, write

    suite = build(agent_spec)
    write(suite, tmp_path / "suites" / suite.name)
    return suite


@pytest.fixture
def written_run(tmp_path, written_suite, monkeypatch):
    from auto_eval.provider import MockClient
    from auto_eval.runner import run_suite, write_run

    run = run_suite(written_suite, mock=True, samples=1, limit=3, client=MockClient())
    write_run(run, tmp_path / "runs")
    monkeypatch.setenv("AUTO_EVAL_RUNS_DIR", str(tmp_path / "runs"))
    return run.report


def test_the_dashboard_is_served(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "/api/runs" in response.text
    # The page has to say what it could not check, not only what passed.
    assert "What could not be checked" in response.text
    assert "Contamination" in response.text


def test_suites_are_listed_for_the_picker(client, tmp_path, written_suite):
    body = client.get("/api/suites", params={"root": str(tmp_path)}).json()
    assert [entry["name"] for entry in body] == [written_suite.name]
    assert body[0]["cases"] == len(written_suite.cases)
    assert body[0]["digest"] == written_suite.digest


def test_a_directory_with_no_suites_is_empty_not_an_error(client, tmp_path):
    (tmp_path / "empty").mkdir()
    response = client.get("/api/suites", params={"root": str(tmp_path / "empty")})
    assert response.status_code == 200 and response.json() == []


def test_runs_are_listed_and_readable(client, written_run):
    listed = client.get("/api/runs").json()
    assert [entry["run_id"] for entry in listed] == [written_run.run_id]

    body = client.get(f"/api/runs/{written_run.run_id}").json()
    assert body["suite_digest"] == written_run.suite_digest
    assert body["headline"]["pass_rate"] == written_run.headline.pass_rate
    assert body["overall"]["checks"]["blocked"] > 0


def test_a_run_can_be_read_as_a_document(client, written_run):
    body = client.get(f"/api/runs/{written_run.run_id}/markdown").json()
    assert f"# Run {written_run.run_id}" in body["markdown"]


def test_the_trace_behind_a_verdict_is_served(client, written_run):
    case = written_run.cases[0]
    body = client.get(f"/api/runs/{written_run.run_id}/trace/{case.case_id}").json()
    assert body["case_id"] == case.case_id
    assert body["steps"]


def test_a_run_id_that_is_not_one_is_refused(client, written_run):
    assert client.get("/api/runs/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/api/runs/no-such-run").status_code == 404


def test_a_run_can_be_started_and_polled(client, tmp_path, written_suite):
    started = client.post(
        "/api/run",
        json={
            "suite": str(tmp_path / "suites" / written_suite.name),
            "mock": True,
            "samples": 1,
            "limit": 2,
            "out": str(tmp_path / "runs"),
        },
    )
    assert started.status_code == 200
    job = started.json()
    assert job["total"] == 2 and job["mock"] is True

    for _ in range(200):
        job = client.get(f"/api/run/{job['job_id']}").json()
        if job["state"] != "running":
            break
    assert job["state"] == "done", job.get("error")
    assert job["run_id"] and (tmp_path / "runs" / job["run_id"] / "run.json").exists()


def test_starting_a_run_of_something_that_is_not_a_suite_is_a_readable_400(
    client, tmp_path
):
    response = client.post(
        "/api/run", json={"suite": str(tmp_path / "nothing"), "mock": True}
    )
    assert response.status_code == 400
    assert "No suite at" in response.json()["detail"]


def test_a_real_run_without_a_key_is_refused_before_it_starts(
    client, tmp_path, written_suite, monkeypatch
):
    from auto_eval.config import Settings

    monkeypatch.setattr(
        web,
        "get_settings",
        lambda **kw: Settings(api_key=None, model="m", base_url=None),
    )
    response = client.post(
        "/api/run",
        json={"suite": str(tmp_path / "suites" / written_suite.name), "mock": False},
    )
    assert response.status_code == 400
    assert "No API key" in response.json()["detail"]


def test_an_unknown_job_is_a_404(client):
    assert client.get("/api/run/nosuchjob").status_code == 404


# --- the one-button path --------------------------------------------------


def test_the_page_offers_the_benchmark_once_ground_truth_is_done(client):
    page = client.get("/").text
    assert "/api/benchmark" in page
    assert "Run the benchmark" in page


def test_a_benchmark_authors_the_suite_and_runs_it(client, tmp_path, agent_spec):
    from auto_eval.gaps import analyze as settle

    response = client.post(
        "/api/benchmark",
        json={
            "spec": settle(agent_spec).model_dump(mode="json"),
            "mock": True,
            "samples": 1,
            "limit": 2,
            "suites_dir": str(tmp_path / "suites"),
            "out": str(tmp_path / "runs"),
        },
    )
    assert response.status_code == 200
    job = response.json()
    # The page can say what it is about to run before the first case comes back.
    assert job["suite_name"] == "support-agent" and job["cases"] == 2
    assert job["suite_digest"] and job["total"] == 2

    for _ in range(200):
        job = client.get(f"/api/run/{job['job_id']}").json()
        if job["state"] != "running":
            break
    assert job["state"] == "done", job.get("error")

    # Both halves are on disk: the suite it wrote, and the run it produced.
    assert (tmp_path / "suites" / "support-agent" / "suite.json").is_file()
    assert (tmp_path / "runs" / job["run_id"] / "run.json").is_file()

    report = client.get(
        f"/api/runs/{job['run_id']}", params={"dir": str(tmp_path / "runs")}
    ).json()
    assert report["suite_digest"] == job["suite_digest"]


def test_a_closed_agent_gate_comes_back_as_a_question_not_an_error(
    client, tmp_path, agent_spec
):
    """A suite written against an agent nobody can start is a document, not a test."""
    from auto_eval.gaps import analyze as settle

    agent_spec.subject.interface = None
    agent_spec.evidence = []
    body = {
        "spec": settle(agent_spec).model_dump(mode="json"),
        "mock": True,
        "samples": 1,
        "limit": 1,
        "suites_dir": str(tmp_path / "suites"),
        "out": str(tmp_path / "runs"),
    }

    refused = client.post("/api/benchmark", json=body)
    assert refused.status_code == 409
    assert "Not ready to author" in refused.json()["detail"]

    forced = client.post("/api/benchmark", json={**body, "force": True})
    assert forced.status_code == 200


def test_a_benchmark_without_a_key_is_refused_before_the_suite_is_written(
    client, tmp_path, agent_spec, monkeypatch
):
    from auto_eval.config import Settings
    from auto_eval.gaps import analyze as settle

    monkeypatch.setattr(
        web,
        "get_settings",
        lambda **kw: Settings(api_key=None, model="m", base_url=None),
    )
    response = client.post(
        "/api/benchmark",
        json={
            "spec": settle(agent_spec).model_dump(mode="json"),
            "mock": False,
            "suites_dir": str(tmp_path / "suites"),
        },
    )
    assert response.status_code == 400
    assert "No API key" in response.json()["detail"]
    assert not (tmp_path / "suites").exists()


def test_the_dashboard_can_be_linked_to_one_run(client):
    assert "wantedRun" in client.get("/dashboard").text


# --- boards ---------------------------------------------------------------

BOARD_COHORT = """{
  "name": "vendors", "model": "fake-model",
  "systems": [{"label": "Alpha", "configuration": "POST /search",
    "endpoint": {"url": "https://alpha.test/search", "body": {"q": "{query}"}}}]
}"""

BOARD_ITEMS = '{"id":"i1","query":"Who did Acme acquire?","answers":["Bolt"],"published":"2026-06-04"}'


def _finished(client, job, tries=200):
    for _ in range(tries):
        job = client.get(f"/api/run/{job['job_id']}").json()
        if job["state"] != "running":
            return job
    return job


@pytest.fixture
def board_run(client, tmp_path, monkeypatch):
    """A board over one item, with both the model and the vendor stubbed."""
    from auto_eval import board as board_module

    class _Obj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeHTTP:
        def post(self, url, json=None, **kw):
            return _Obj(
                status_code=200,
                text="{}",
                json=lambda: {
                    "results": [{"title": "n", "url": "u", "text": "Acme bought Bolt"}]
                },
            )

    class FakeModel:
        def __init__(self):
            self.chat = _Obj(completions=self)

        def create(self, **kwargs):
            return _Obj(
                choices=[
                    _Obj(
                        message=_Obj(content="unknown", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=_Obj(prompt_tokens=10, completion_tokens=2),
            )

    real = board_module.run_board
    monkeypatch.setattr(
        web,
        "run_board",
        lambda dataset, cohort, **kw: real(
            dataset, cohort, **{**kw, "client": FakeModel(), "http": FakeHTTP()}
        ),
    )
    started = client.post(
        "/api/board",
        json={
            "cohort": BOARD_COHORT,
            "dataset": BOARD_ITEMS,
            "dataset_name": "news",
            "out": str(tmp_path / "boards"),
        },
    )
    assert started.status_code == 200, started.json()
    job = _finished(client, started.json())
    assert job["state"] == "done", job.get("error")
    monkeypatch.setenv("AUTO_EVAL_RUNS_DIR", str(tmp_path / "runs"))
    return job


def test_a_board_runs_from_the_page_and_is_written(board_run, tmp_path):
    assert board_run["board_id"]
    assert (tmp_path / "boards" / board_run["board_id"] / "board.json").is_file()


def test_a_board_reads_back_with_its_rows_and_its_baseline(client, board_run, tmp_path):
    body = client.get(
        f"/api/boards/{board_run['board_id']}", params={"dir": str(tmp_path / "boards")}
    ).json()
    labels = [row["label"] for row in body["rows"]]
    assert "Alpha" in labels and any("model only" in label for label in labels)
    alpha = next(row for row in body["rows"] if row["label"] == "Alpha")
    assert alpha["accuracy"] == 1.0 and alpha["recall_at_1"] == 1.0


def test_boards_are_listed_with_what_they_found(client, board_run, tmp_path):
    listed = client.get("/api/boards", params={"dir": str(tmp_path / "boards")}).json()
    assert [entry["board_id"] for entry in listed] == [board_run["board_id"]]
    assert listed[0]["leader"] == "Alpha"
    assert listed[0]["baseline_accuracy"] == 0.0


def test_a_board_can_be_read_as_a_document(client, board_run, tmp_path):
    body = client.get(
        f"/api/boards/{board_run['board_id']}/markdown",
        params={"dir": str(tmp_path / "boards")},
    ).json()
    assert "## The board" in body["markdown"]
    assert "Model-only baseline" in body["markdown"]


def test_a_board_with_no_items_is_a_readable_400(client):
    response = client.post("/api/board", json={"cohort": BOARD_COHORT})
    assert response.status_code == 400 and "No items" in response.json()["detail"]


def test_a_cohort_that_is_not_one_is_a_readable_400(client):
    response = client.post(
        "/api/board", json={"cohort": "{nope", "dataset": BOARD_ITEMS}
    )
    assert response.status_code == 400 and "JSON" in response.json()["detail"]


def test_a_board_without_a_key_is_refused_before_it_starts(client, monkeypatch):
    from auto_eval.config import Settings

    monkeypatch.setattr(
        web,
        "get_settings",
        lambda **kw: Settings(api_key=None, model="m", base_url=None),
    )
    response = client.post(
        "/api/board", json={"cohort": BOARD_COHORT, "dataset": BOARD_ITEMS}
    )
    assert response.status_code == 400 and "API key" in response.json()["detail"]


def test_the_page_offers_a_board_for_a_subject_with_no_trajectory(client):
    page = client.get("/").text
    assert "/api/board" in page
    assert "Run the board" in page
    # And it says why this shape rather than the other one.
    assert "no trajectory to" in page


def test_an_unknown_board_is_a_404(client):
    assert client.get("/api/boards/no-such-board").status_code == 404

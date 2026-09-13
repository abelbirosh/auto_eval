import json

import pytest

from auto_eval.cli import EXIT_OK, build_parser, main


def test_schema_command_emits_valid_json_schema(capsys):
    assert main(["schema"]) == EXIT_OK
    schema = json.loads(capsys.readouterr().out)
    assert "subject" in schema["properties"]
    assert "kpis" in schema["properties"]
    assert "evidence" in schema["properties"]


def test_schema_command_writes_to_file(tmp_path, capsys):
    out = tmp_path / "schema.json"
    assert main(["schema", "--out", str(out)]) == EXIT_OK
    assert json.loads(out.read_text())["title"] == "TaskSpec"


def test_classify_parser_defaults():
    args = build_parser().parse_args(["classify", "test", "my", "bot"])
    assert args.text == ["test", "my", "bot"]
    assert args.model is None  # so AUTO_EVAL_MODEL can supply it
    assert args.format == "markdown"


def test_model_flag_overrides():
    assert (
        build_parser().parse_args(["classify", "x", "--model", "gpt-4o-mini"]).model
        == "gpt-4o-mini"
    )


def test_ground_truth_parser_takes_a_spec_or_text():
    args = build_parser().parse_args(["ground-truth", "-s", "spec.json", "--force"])
    assert args.spec == "spec.json"
    assert args.force is True
    assert build_parser().parse_args(["ground-truth", "test", "my", "bot"]).text == [
        "test",
        "my",
        "bot",
    ]


def test_ground_truth_exits_two_without_searching_when_something_blocks(
    tmp_path, subjectless_spec, capsys, monkeypatch
):
    from auto_eval import cli
    from auto_eval.cli import EXIT_INSUFFICIENT
    from auto_eval.gaps import analyze

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(subjectless_spec).model_dump_json())

    called = []
    monkeypatch.setattr(cli, "identify", lambda *a, **kw: called.append(a))

    assert main(["ground-truth", "-s", str(spec_file)]) == EXIT_INSUFFICIENT
    assert called == []
    assert "Not ready to search" in capsys.readouterr().err


def test_ground_truth_reads_a_spec_file_and_prints_the_report(
    tmp_path, full_spec, capsys, monkeypatch
):
    from auto_eval import cli
    from auto_eval.gaps import analyze
    from auto_eval.ground_truth import SourceFindings, assess

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(full_spec).model_dump_json())
    out = tmp_path / "report.md"

    report = assess(full_spec, SourceFindings(recommendation="Label your own."))
    monkeypatch.setattr(cli, "identify", lambda spec, **kw: report)

    assert main(["ground-truth", "-s", str(spec_file), "--md", str(out)]) == EXIT_OK
    assert "Ground truth for invoice extractor" in capsys.readouterr().out
    assert "Label your own." in out.read_text()


def test_analyze_flags_are_parsed():
    args = build_parser().parse_args(
        ["ground-truth", "-s", "s.json", "--analyze", "--out-dir", "data"]
    )
    assert args.analyze is True
    assert args.out_dir == "data"


def test_the_analysis_is_written_where_a_later_step_can_pick_it_up(
    tmp_path, full_spec, capsys, monkeypatch
):
    from auto_eval import cli
    from auto_eval.analysis import (
        AnalysisReport,
        DownloadPlan,
        Reachability,
        ResourceAnalysis,
        Usability,
    )
    from auto_eval.gaps import analyze
    from auto_eval.ground_truth import SourceFindings, SourceKind, assess

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(full_spec).model_dump_json())
    out_dir = tmp_path / "data"

    monkeypatch.setattr(
        cli,
        "identify",
        lambda spec, **kw: assess(spec, SourceFindings(recommendation="x")),
    )
    monkeypatch.setattr(
        cli,
        "analyze_sources",
        lambda spec, report, **kw: AnalysisReport(
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
                        dataset="acme/invoices",
                        split="validation",
                    ),
                )
            ],
        ),
    )

    assert (
        main(
            [
                "ground-truth",
                "-s",
                str(spec_file),
                "--analyze",
                "--out-dir",
                str(out_dir),
            ]
        )
        == EXIT_OK
    )

    plans = json.loads((out_dir / "fetch-plan.json").read_text())
    assert plans[0]["dataset"] == "acme/invoices"
    assert plans[0]["split"] == "validation"
    assert (
        json.loads((out_dir / "sources.json").read_text())["subject"]
        == "invoice extractor"
    )
    assert "Source analysis" in (out_dir / "sources.md").read_text()
    assert "Source analysis" in capsys.readouterr().out


def test_force_reaches_the_search_so_the_gate_does_not_refuse_twice(
    tmp_path, sparse_spec, monkeypatch, capsys
):
    from auto_eval import cli
    from auto_eval.gaps import analyze
    from auto_eval.ground_truth import SourceFindings, assess

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(sparse_spec).model_dump_json())

    seen = {}

    def fake_identify(spec, **kwargs):
        seen.update(kwargs)
        return assess(spec, SourceFindings(recommendation="x"))

    monkeypatch.setattr(cli, "identify", fake_identify)

    assert main(["ground-truth", "-s", str(spec_file), "--force"]) == EXIT_OK
    assert seen["force"] is True


def test_benchmarks_with_no_input_prints_the_catalogue(capsys):
    """It must not sit waiting on a pipe that is never coming."""
    assert main(["benchmarks"]) == 0
    out = capsys.readouterr().out
    assert "SWE-bench" in out
    assert "https://" in out
    assert out.count("metric:") == 10


def test_benchmarks_matches_a_spec_from_disk(tmp_path, full_spec, capsys):
    from auto_eval.gaps import analyze

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(full_spec).model_dump_json())

    assert main(["benchmarks", "-s", str(spec_file)]) == 0
    out = capsys.readouterr().out
    assert "Benchmarks for invoice extractor" in out
    assert "matched because:" in out


def test_benchmarks_json_output_is_machine_readable(tmp_path, full_spec, capsys):
    import json as _json

    from auto_eval.gaps import analyze

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(full_spec).model_dump_json())

    assert main(["benchmarks", "-s", str(spec_file), "--format", "json"]) == 0
    payload = _json.loads(capsys.readouterr().out)
    assert all({"benchmark", "fit", "score"} <= set(row) for row in payload)


def test_benchmarks_never_calls_the_model_for_a_spec_on_disk(
    tmp_path, full_spec, monkeypatch
):
    from auto_eval import cli
    from auto_eval.gaps import analyze

    def boom(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("matching is offline")

    monkeypatch.setattr(cli, "classify", boom)
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(full_spec).model_dump_json())

    assert main(["benchmarks", "-s", str(spec_file)]) == 0


# --- the agent suite commands ----------------------------------------------


def write_spec(tmp_path, spec):
    from auto_eval.gaps import analyze

    path = tmp_path / "spec.json"
    path.write_text(analyze(spec).model_dump_json())
    return str(path)


def test_profile_reports_the_agent_and_exits_zero(tmp_path, agent_spec, capsys):
    assert main(["profile", "-s", write_spec(tmp_path, agent_spec)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "# Running support agent" in out
    assert "POST /v1/runs" in out


def test_profile_exits_two_when_the_agent_cannot_be_run(tmp_path, agent_spec, capsys):
    from auto_eval.cli import EXIT_INSUFFICIENT

    agent_spec.subject.interface = None
    agent_spec.evidence = []
    assert (
        main(["profile", "-s", write_spec(tmp_path, agent_spec)]) == EXIT_INSUFFICIENT
    )
    assert "Not yet." in capsys.readouterr().out


def test_profile_json_is_machine_readable(tmp_path, agent_spec, capsys):
    assert (
        main(["profile", "-s", write_spec(tmp_path, agent_spec), "--format", "json"])
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"]["run_mode"] == "http"
    assert payload["gate"]["open"] is True


def test_author_writes_the_suite_and_the_document(tmp_path, agent_spec, capsys):
    out_dir = tmp_path / "suite"
    spec = write_spec(tmp_path, agent_spec)
    assert main(["author", "-s", spec, "-o", str(out_dir)]) == EXIT_OK

    written = json.loads((out_dir / "suite.json").read_text())
    assert written["cases"]
    assert (
        (out_dir / "suite.md").read_text().startswith("# Eval suite for support agent")
    )
    assert "Wrote" in capsys.readouterr().err


def test_author_exits_two_without_writing_when_it_cannot_be_run(
    tmp_path, agent_spec, capsys
):
    from auto_eval.cli import EXIT_INSUFFICIENT

    agent_spec.subject.interface = None
    agent_spec.evidence = []
    out_dir = tmp_path / "suite"
    assert (
        main(["author", "-s", write_spec(tmp_path, agent_spec), "-o", str(out_dir)])
        == EXIT_INSUFFICIENT
    )
    assert not out_dir.exists()
    assert "Not ready to author" in capsys.readouterr().err


def test_author_force_writes_past_the_gate(tmp_path, agent_spec):
    agent_spec.subject.interface = None
    agent_spec.evidence = []
    out_dir = tmp_path / "suite"
    assert (
        main(
            [
                "author",
                "-s",
                write_spec(tmp_path, agent_spec),
                "-o",
                str(out_dir),
                "--force",
            ]
        )
        == EXIT_OK
    )
    assert (out_dir / "suite.json").exists()


def test_author_never_calls_the_model_for_a_spec_on_disk(
    tmp_path, agent_spec, monkeypatch
):
    """The whole block is deterministic; a model call here would be a bug."""
    from auto_eval import cli

    monkeypatch.setattr(
        cli, "classify", lambda *a, **kw: pytest.fail("classify should not be called")
    )
    assert (
        main(
            [
                "author",
                "-s",
                write_spec(tmp_path, agent_spec),
                "-o",
                str(tmp_path / "s"),
            ]
        )
        == EXIT_OK
    )


def test_author_is_byte_identical_across_runs(tmp_path, agent_spec):
    spec = write_spec(tmp_path, agent_spec)
    main(["author", "-s", spec, "-o", str(tmp_path / "a")])
    main(["author", "-s", spec, "-o", str(tmp_path / "b")])
    assert (tmp_path / "a" / "suite.json").read_text() == (
        tmp_path / "b" / "suite.json"
    ).read_text()


def test_suite_reads_back_what_author_wrote(tmp_path, agent_spec, capsys):
    out_dir = tmp_path / "suite"
    main(["author", "-s", write_spec(tmp_path, agent_spec), "-o", str(out_dir)])
    capsys.readouterr()

    assert main(["suite", str(out_dir)]) == EXIT_OK
    assert "# Eval suite for support agent" in capsys.readouterr().out


def test_suite_prints_one_case_in_full_from_an_id_prefix(tmp_path, agent_spec, capsys):
    out_dir = tmp_path / "suite"
    main(["author", "-s", write_spec(tmp_path, agent_spec), "-o", str(out_dir)])
    capsys.readouterr()

    case_id = json.loads((out_dir / "suite.json").read_text())["cases"][0]["id"]
    assert main(["suite", str(out_dir), "--case", case_id[:6]]) == EXIT_OK
    out = capsys.readouterr().out
    assert case_id in out
    assert "**Checks**" in out


def test_suite_says_so_when_the_id_matches_nothing(tmp_path, agent_spec, capsys):
    from auto_eval.cli import EXIT_ERROR

    out_dir = tmp_path / "suite"
    main(["author", "-s", write_spec(tmp_path, agent_spec), "-o", str(out_dir)])
    capsys.readouterr()

    assert main(["suite", str(out_dir), "--case", "zzzzzz"]) == EXIT_ERROR
    assert "no case" in capsys.readouterr().err


def test_suite_on_an_empty_directory_exits_two_with_advice(tmp_path, capsys):
    from auto_eval.cli import EXIT_INSUFFICIENT

    assert main(["suite", str(tmp_path)]) == EXIT_INSUFFICIENT
    assert "auto-eval author" in capsys.readouterr().err


# --- running a suite ------------------------------------------------------


@pytest.fixture
def suite_dir(tmp_path, agent_spec):
    from auto_eval.suite import build, write

    suite = build(agent_spec)
    write(suite, tmp_path / "suite")
    return tmp_path / "suite"


def test_run_parser_defaults():
    args = build_parser().parse_args(["run", "suites/x"])
    assert args.suite == "suites/x"
    assert args.model is None  # so AUTO_EVAL_SUBJECT_MODEL can supply it
    assert args.split == "all"
    assert args.mock is False


def test_a_mock_run_needs_no_key_and_writes_a_report(suite_dir, tmp_path, capsys):
    code = main(
        [
            "run",
            str(suite_dir),
            "--mock",
            "--samples",
            "1",
            "--limit",
            "3",
            "-o",
            str(tmp_path / "runs"),
            "--quiet",
        ]
    )
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "# Run " in out
    assert "called no provider" in out  # the first thing the report says

    written = list((tmp_path / "runs").iterdir())
    assert len(written) == 1
    assert (written[0] / "run.json").is_file() and (written[0] / "run.md").is_file()
    assert list((written[0] / "traces").iterdir())


def test_a_real_run_without_a_key_stops_before_spending_anything(
    suite_dir, monkeypatch, capsys
):
    from auto_eval import cli
    from auto_eval.config import Settings

    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda **kw: Settings(api_key=None, model="m", base_url=None),
    )
    assert main(["run", str(suite_dir), "--quiet"]) == 1
    assert "no API key" in capsys.readouterr().err


def test_runs_and_report_read_what_run_wrote(suite_dir, tmp_path, capsys):
    main(
        [
            "run",
            str(suite_dir),
            "--mock",
            "--samples",
            "1",
            "--limit",
            "2",
            "-o",
            str(tmp_path / "runs"),
            "--quiet",
            "--format",
            "json",
        ]
    )
    capsys.readouterr()

    assert (
        main(["runs", "--dir", str(tmp_path / "runs"), "--format", "json"]) == EXIT_OK
    )
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 1 and listed[0]["mock"] is True

    assert main(["report", listed[0]["path"]]) == EXIT_OK
    assert "What could not be checked" in capsys.readouterr().out


def test_reporting_on_something_that_is_not_a_run_is_a_readable_error(tmp_path, capsys):
    assert main(["report", str(tmp_path)]) == 1
    assert "No run at" in capsys.readouterr().err


def test_fresh_lists_ground_truth_the_model_cannot_have_seen(capsys):
    assert main(["fresh", "--model", "gpt-5"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "LiveCodeBench" in out and "held out" in out
    assert "2024-09-30" in out  # the cutoff it was judged against


def test_fresh_says_when_it_has_no_cutoff_on_file(capsys):
    assert main(["fresh", "--model", "some-local-model"]) == EXIT_OK
    captured = capsys.readouterr()
    assert "No published cutoff on file" in captured.err
    assert "nothing is judged post-cutoff" in captured.out

import json

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

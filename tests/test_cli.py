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
    assert build_parser().parse_args(["classify", "x", "--model", "gpt-4o-mini"]).model == "gpt-4o-mini"


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
    tmp_path, sparse_spec, capsys, monkeypatch
):
    from auto_eval import cli
    from auto_eval.cli import EXIT_INSUFFICIENT
    from auto_eval.gaps import analyze

    spec_file = tmp_path / "spec.json"
    spec_file.write_text(analyze(sparse_spec).model_dump_json())

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

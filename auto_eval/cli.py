"""Command line entry point: `auto-eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .analysis import AnalysisReport, analyze_sources
from .classifier import DEFAULT_MAX_TOKENS, ClassifierError, classify, list_models
from .config import DEFAULT_MODEL, get_settings
from .gaps import analyze
from .ground_truth import GroundTruthReport, gate, identify
from .render import (
    render_analysis,
    render_ground_truth,
    render_markdown,
    render_questions,
)
from .schema import Readiness, TaskSpec

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INSUFFICIENT = 2  # spec has blocking questions - scripts can gate on this


def _read_input(args: argparse.Namespace, extra_hint: str = "") -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text:
        return " ".join(args.text)
    if sys.stdin.isatty():
        raise ClassifierError(
            "No input. Pass text as an argument, use -f FILE, or pipe on stdin."
            + extra_hint
        )
    return sys.stdin.read()


def _cmd_classify(args: argparse.Namespace) -> int:
    text = _read_input(args)
    spec = classify(text, model=args.model, max_tokens=args.max_tokens)

    if args.json:
        Path(args.json).write_text(
            spec.model_dump_json(indent=2, by_alias=False) + "\n", encoding="utf-8"
        )
    if args.md:
        Path(args.md).write_text(render_markdown(spec) + "\n", encoding="utf-8")

    if args.questions_only:
        print(render_questions(spec))
    elif args.format == "json":
        print(spec.model_dump_json(indent=2))
    else:
        print(render_markdown(spec))

    if args.json or args.md:
        written = " and ".join(p for p in (args.md, args.json) if p)
        print(f"\nWrote {written}", file=sys.stderr)

    if args.ground_truth:
        decision = gate(spec)
        if not decision.open:
            print(
                f"\nSkipping the ground-truth search: {decision.reason}",
                file=sys.stderr,
            )
        else:
            print()
            _run_block(spec, args)

    return EXIT_INSUFFICIENT if spec.readiness is Readiness.INSUFFICIENT else EXIT_OK


def _run_block(
    spec: TaskSpec,
    args: argparse.Namespace,
    *,
    report_json: Optional[str] = None,
    report_md: Optional[str] = None,
) -> None:
    """Identify, optionally extract, and print and write whatever was asked for.

    `report_json` and `report_md` are passed only by the `ground-truth` command,
    whose --json and --md mean the report; on `classify` they mean the spec.
    """
    report = identify(
        spec,
        model=args.model,
        max_tokens=args.max_tokens,
        force=getattr(args, "force", False),
    )

    if report_json:
        Path(report_json).write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    if report_md:
        Path(report_md).write_text(render_ground_truth(report) + "\n", encoding="utf-8")

    print(_format_report(report, args.format))

    if not args.analyze:
        return

    analysis = analyze_sources(spec, report, model=args.model)
    print()
    print(
        analysis.model_dump_json(indent=2)
        if args.format == "json"
        else render_analysis(analysis)
    )
    if args.out_dir:
        for line in _write_analysis(analysis, Path(args.out_dir)):
            print(line, file=sys.stderr)


def _format_report(report: GroundTruthReport, fmt: str) -> str:
    if fmt == "json":
        return report.model_dump_json(indent=2)
    return render_ground_truth(report)


def _load_spec(args: argparse.Namespace) -> TaskSpec:
    """A spec from disk, or a fresh classification of the request text.

    Gap analysis is re-run on a loaded spec so a hand-edited file is gated on
    what it says now rather than on the readiness recorded when it was written.
    """
    if args.spec:
        spec = TaskSpec.model_validate_json(Path(args.spec).read_text(encoding="utf-8"))
        return analyze(spec)
    text = _read_input(args, " Or point at a spec JSON with -s PATH.")
    return classify(text, model=args.model, max_tokens=args.max_tokens)


def _cmd_ground_truth(args: argparse.Namespace) -> int:
    spec = _load_spec(args)

    decision = gate(spec)
    if not decision.open and not args.force:
        print(f"Not ready to search: {decision.reason}", file=sys.stderr)
        print(render_questions(spec), file=sys.stderr)
        return EXIT_INSUFFICIENT

    _run_block(spec, args, report_json=args.json, report_md=args.md)

    if args.json or args.md:
        written = " and ".join(p for p in (args.md, args.json) if p)
        print(f"\nWrote {written}", file=sys.stderr)

    return EXIT_OK


def _write_analysis(analysis: AnalysisReport, out_dir: Path) -> List[str]:
    """Write the analysis where a later step can pick it up."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    sources = out_dir / "sources.json"
    sources.write_text(analysis.model_dump_json(indent=2) + "\n", encoding="utf-8")
    written.append(f"Wrote {len(analysis.resources)} source analysis to {sources}")

    plans = out_dir / "fetch-plan.json"
    plans.write_text(
        json.dumps([plan.model_dump() for plan in analysis.plans], indent=2) + "\n",
        encoding="utf-8",
    )
    written.append(f"Wrote {len(analysis.plans)} fetch plan(s) to {plans}")

    document = out_dir / "sources.md"
    document.write_text(render_analysis(analysis) + "\n", encoding="utf-8")
    written.append(f"Wrote {document}")
    return written


def _cmd_serve(args: argparse.Namespace) -> int:
    from .web import serve

    serve(host=args.host, port=args.port, reload=args.reload)
    return EXIT_OK


def _cmd_models(args: argparse.Namespace) -> int:
    settings = get_settings()
    for model_id in list_models():
        marker = "  <- current" if model_id == settings.model else ""
        print(f"{model_id}{marker}")
    return EXIT_OK


def _cmd_schema(args: argparse.Namespace) -> int:
    schema = json.dumps(TaskSpec.model_json_schema(), indent=2)
    if args.out:
        Path(args.out).write_text(schema + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(schema)
    return EXIT_OK


def _add_extraction_flags(parser: argparse.ArgumentParser) -> None:
    """Shared by both commands that can run the ground-truth block."""
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Look at each source found and say what a later step should fetch from it.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="PATH",
        help="Write sources.json, fetch-plan.json, and the document here.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-eval",
        description="Classify a free-form evaluation request into a task spec.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    classify_cmd = sub.add_parser(
        "classify",
        help="Classify a request and emit the task document.",
        description=(
            "Reads the request from an argument, -f FILE, or stdin. "
            f"Exits {EXIT_INSUFFICIENT} when the spec has blocking questions."
        ),
    )
    classify_cmd.add_argument("text", nargs="*", help="The request, as text.")
    classify_cmd.add_argument("-f", "--file", help="Read the request from a file.")
    classify_cmd.add_argument(
        "--model",
        default=None,
        help=f"Model to classify with. Defaults to AUTO_EVAL_MODEL, else {DEFAULT_MODEL}.",
    )
    classify_cmd.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Output token cap."
    )
    classify_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    classify_cmd.add_argument(
        "--questions-only",
        action="store_true",
        help="Print only the open questions.",
    )
    classify_cmd.add_argument(
        "--json", metavar="PATH", help="Also write the spec as JSON."
    )
    classify_cmd.add_argument(
        "--md", metavar="PATH", help="Also write the task document."
    )
    classify_cmd.add_argument(
        "--ground-truth",
        action="store_true",
        help="Search for public ground truth afterwards, if nothing is blocking.",
    )
    _add_extraction_flags(classify_cmd)
    classify_cmd.set_defaults(func=_cmd_classify)

    gt_cmd = sub.add_parser(
        "ground-truth",
        help="Find public ground truth and baselines for a task spec.",
        description=(
            "Takes a spec written by `classify --json`, or classifies a request first. "
            "Searches the web for labelled datasets, benchmarks, and published "
            "baselines; --analyze then looks at each one and says what to fetch. "
            f"Exits {EXIT_INSUFFICIENT} without searching while the spec still has "
            "blocking questions."
        ),
    )
    gt_cmd.add_argument("text", nargs="*", help="The request, as text.")
    gt_cmd.add_argument("-f", "--file", help="Read the request from a file.")
    gt_cmd.add_argument(
        "-s", "--spec", metavar="PATH", help="Use a spec JSON instead of classifying."
    )
    gt_cmd.add_argument(
        "--model",
        default=None,
        help="Model to search with. Defaults to AUTO_EVAL_SEARCH_MODEL, else AUTO_EVAL_MODEL.",
    )
    gt_cmd.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Output token cap."
    )
    gt_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    gt_cmd.add_argument("--json", metavar="PATH", help="Also write the report as JSON.")
    gt_cmd.add_argument("--md", metavar="PATH", help="Also write the report document.")
    gt_cmd.add_argument(
        "--force",
        action="store_true",
        help="Search even while blocking questions are open.",
    )
    _add_extraction_flags(gt_cmd)
    gt_cmd.set_defaults(func=_cmd_ground_truth)

    serve_cmd = sub.add_parser(
        "serve",
        help="Run the local web UI.",
        description="Local only - the UI has no authentication.",
    )
    serve_cmd.add_argument("--host", default="127.0.0.1", help="Bind address.")
    serve_cmd.add_argument("--port", type=int, default=8000, help="Port.")
    serve_cmd.add_argument(
        "--reload", action="store_true", help="Restart on code changes."
    )
    serve_cmd.set_defaults(func=_cmd_serve)

    models_cmd = sub.add_parser("models", help="List model IDs this API key can reach.")
    models_cmd.set_defaults(func=_cmd_models)

    schema_cmd = sub.add_parser("schema", help="Print the TaskSpec JSON schema.")
    schema_cmd.add_argument("--out", metavar="PATH", help="Write the schema to a file.")
    schema_cmd.set_defaults(func=_cmd_schema)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ClassifierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

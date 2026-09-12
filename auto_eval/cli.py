"""Command line entry point: `auto-eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .classifier import DEFAULT_MAX_TOKENS, ClassifierError, classify, list_models
from .config import DEFAULT_MODEL, get_settings
from .render import render_markdown, render_questions
from .schema import Readiness, TaskSpec

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INSUFFICIENT = 2  # spec has blocking questions - scripts can gate on this


def _read_input(args: argparse.Namespace) -> str:
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if args.text:
        return " ".join(args.text)
    if sys.stdin.isatty():
        raise ClassifierError(
            "No input. Pass text as an argument, use -f FILE, or pipe on stdin."
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

    return EXIT_INSUFFICIENT if spec.readiness is Readiness.INSUFFICIENT else EXIT_OK


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
        "--format", choices=["markdown", "json"], default="markdown", help="Stdout format."
    )
    classify_cmd.add_argument(
        "--questions-only",
        action="store_true",
        help="Print only the open questions.",
    )
    classify_cmd.add_argument("--json", metavar="PATH", help="Also write the spec as JSON.")
    classify_cmd.add_argument("--md", metavar="PATH", help="Also write the task document.")
    classify_cmd.set_defaults(func=_cmd_classify)

    models_cmd = sub.add_parser(
        "models", help="List model IDs this API key can reach."
    )
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

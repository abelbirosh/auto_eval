"""Command line entry point: `auto-eval`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .agent import agent_gate, profile
from .analysis import AnalysisReport, analyze_sources
from .benchmarks import CATALOGUE, Benchmark, BenchmarkMatch, match
from .board import DEFAULT_CONCURRENCY as BOARD_CONCURRENCY
from .board import Board, BoardError, run_board, write_board
from .board import list_boards as list_board_dirs
from .classifier import DEFAULT_MAX_TOKENS, ClassifierError, classify, list_models
from .cohort import CohortError
from .cohort import load as load_cohort
from .config import DEFAULT_MODEL, get_settings
from .contamination import cutoff_for, parse_cutoff
from .dataset import DatasetError
from .dataset import load as load_dataset
from .gaps import analyze
from .ground_truth import GroundTruthReport, gate, identify
from .render import (
    render_analysis,
    render_board,
    render_case,
    render_fresh,
    render_ground_truth,
    render_loop,
    render_markdown,
    render_profile,
    render_questions,
    render_round,
    render_run,
    render_suite,
)
from .runner import DEFAULT_CONCURRENCY, RunError, load_run, write_run
from .runner import list_runs as list_run_dirs
from .runner import run_suite as execute_suite
from .saturation import (
    DEFAULT_REWRITES,
    DEFAULT_ROUNDS,
    SATURATED,
    Round,
    run_loop,
    write_dataset,
)
from .schema import Readiness, TaskSpec
from .suite import Split, SuiteError
from .suite import build as build_suite
from .suite import load as load_suite
from .suite import write as write_suite
from .surface import DEFAULT_PER_CELL

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


def _render_benchmark(entry: Benchmark, indent: str = "") -> str:
    lines = [
        f"{indent}{entry.name} - {entry.measures}",
        f"{indent}  metric: {entry.metric}",
        f"{indent}  {entry.url}",
    ]
    if entry.size:
        lines.append(f"{indent}  size: {entry.size}")
    lines.append(f"{indent}  licence: {entry.licence or 'not stated - check the page'}")
    lines.append(f"{indent}  caveat: {entry.caveats}")
    return "\n".join(lines)


def _cmd_benchmarks(args: argparse.Namespace) -> int:
    """List the catalogue, or the entries that match a spec.

    Unlike the other commands this one never reads stdin: with nothing named it
    prints the catalogue rather than waiting on a pipe that may never come.
    """
    if not args.spec and not args.text and not args.file:
        for entry in CATALOGUE:
            print(_render_benchmark(entry))
            print()
        return EXIT_OK

    spec = _load_spec(args)
    matches: List[BenchmarkMatch] = match(spec, limit=args.limit)

    if args.format == "json":
        print(
            json.dumps(
                [m.model_dump(mode="json") for m in matches], indent=2, default=str
            )
        )
        return EXIT_OK

    if not matches:
        print(f"Nothing in the catalogue speaks to {spec.subject.name}.")
        print(
            "That is a normal answer for an in-house system - the ground-truth "
            "search looks wider.",
            file=sys.stderr,
        )
        return EXIT_OK

    print(f"Benchmarks for {spec.subject.name}\n")
    for hit in matches:
        covers = f" - covers {', '.join(hit.covers_kpis)}" if hit.covers_kpis else ""
        print(f"[{hit.fit.value}]{covers}")
        print(_render_benchmark(hit.benchmark, indent="  "))
        print(f"  matched because: {'; '.join(hit.reasons)}")
        print()
    return EXIT_OK


def _cmd_profile(args: argparse.Namespace) -> int:
    """What we can drive, and whether that is enough to author a suite against."""
    spec = _load_spec(args)
    agent_profile = profile(spec)
    decision = agent_gate(spec, agent_profile)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "profile": agent_profile.model_dump(mode="json"),
                    "gate": decision.model_dump(mode="json"),
                },
                indent=2,
            )
        )
    else:
        print(render_profile(agent_profile, decision))

    return EXIT_OK if decision.open else EXIT_INSUFFICIENT


def _cmd_author(args: argparse.Namespace) -> int:
    """Write the suite. Deterministic and offline - no model call, no network."""
    spec = _load_spec(args)

    try:
        suite = build_suite(spec, per_cell=args.per_cell, force=args.force)
    except SuiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(render_questions(spec), file=sys.stderr)
        return EXIT_INSUFFICIENT

    document = render_suite(suite)
    if args.format == "json":
        print(suite.model_dump_json(indent=2))
    else:
        print(document)

    if args.out_dir:
        for line in write_suite(suite, Path(args.out_dir), document=document):
            print(line, file=sys.stderr)

    return EXIT_OK


def _cmd_suite(args: argparse.Namespace) -> int:
    """Read a written suite back: what it covers, or one case in full."""
    suite = load_suite(Path(args.path))

    if args.case:
        found = [c for c in suite.cases if c.id.startswith(args.case)]
        if not found:
            print(
                f"error: no case in {args.path} starts with {args.case!r}",
                file=sys.stderr,
            )
            return EXIT_ERROR
        for case in found:
            print(render_case(case))
        return EXIT_OK

    if args.format == "json":
        print(suite.model_dump_json(indent=2))
    else:
        print(render_suite(suite))
    return EXIT_OK


def _cmd_run(args: argparse.Namespace) -> int:
    """Run every case in a written suite against the provider API."""
    suite = load_suite(Path(args.suite))
    settings = get_settings()
    model = args.model or settings.effective_subject_model

    if not args.mock and not settings.has_key:
        print(
            "error: no API key, so nothing can be run against the provider. "
            "Put OPENAI_API_KEY in your .env, or pass --mock to exercise the harness offline.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    cutoff = parse_cutoff(model, args.cutoff) if args.cutoff else None
    prices = (
        (args.price_in, args.price_out) if args.price_in and args.price_out else None
    )
    split = Split(args.split) if args.split != "all" else None
    samples = args.samples or suite.samples
    planned = len(suite.cases_in(split) if split else suite.cases)
    if args.limit:
        planned = min(planned, args.limit)

    print(
        f"Running {planned} case(s) x {samples} sample(s) = {planned * samples} run(s) "
        f"of {suite.name} against {model}"
        + (" [mock: no provider call]" if args.mock else ""),
        file=sys.stderr,
    )

    def progress(done: int, total: int, case_id: str) -> None:
        print(f"  [{done}/{total}] {case_id}", file=sys.stderr)

    run = execute_suite(
        suite,
        model=args.model,
        judge_model=args.judge_model,
        judge=not args.no_judge,
        samples=args.samples,
        split=split,
        limit=args.limit,
        concurrency=args.concurrency,
        mock=args.mock,
        prices=prices,
        cutoff=cutoff,
        progress=progress if not args.quiet else None,
        settings=settings,
    )

    document = render_run(run.report)
    if args.format == "json":
        print(run.report.model_dump_json(indent=2))
    else:
        print(document)

    if not args.no_write:
        target = write_run(run, Path(args.out or settings.effective_runs_dir))
        (target / "run.md").write_text(document + "\n", encoding="utf-8")
        print(f"\nWrote {target}", file=sys.stderr)
        print(
            "See it in the dashboard: `auto-eval serve`, then open /dashboard",
            file=sys.stderr,
        )
    return EXIT_OK


def _cmd_runs(args: argparse.Namespace) -> int:
    """List the runs on disk, newest first."""
    entries = list_run_dirs(Path(args.dir or get_settings().effective_runs_dir))
    if not entries:
        print(
            f"No runs under {args.dir or get_settings().effective_runs_dir}. "
            "Run a suite with `auto-eval run SUITE`.",
            file=sys.stderr,
        )
        return EXIT_OK
    if args.format == "json":
        print(json.dumps([e.model_dump(mode="json") for e in entries], indent=2))
        return EXIT_OK
    for entry in entries:
        rate = f"{entry.pass_rate:.0%}" if entry.pass_rate is not None else "no number"
        print(
            f"{entry.run_id}  {rate} held out  {entry.cases} cases  "
            f"{entry.blocked_checks} blocked checks  {entry.model}"
            + ("  [mock]" if entry.mock else "")
        )
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    """Read a written run back."""
    report = load_run(Path(args.path))
    if args.format == "json":
        print(report.model_dump_json(indent=2))
    else:
        print(render_run(report))
    return EXIT_OK


def _cmd_fresh(args: argparse.Namespace) -> int:
    """Ground truth published after a model stopped learning."""
    model = args.model or get_settings().effective_subject_model
    cutoff = parse_cutoff(model, args.cutoff) if args.cutoff else cutoff_for(model)
    if cutoff is None:
        print(
            f"No published cutoff on file for {model!r}; pass --cutoff YYYY-MM-DD to judge "
            "the dated entries against it.",
            file=sys.stderr,
        )
    print(render_fresh(cutoff))
    return EXIT_OK


def _cmd_board(args: argparse.Namespace) -> int:
    """Compare several systems on the same items, and write the board."""
    dataset = load_dataset(Path(args.dataset))
    cohort = load_cohort(Path(args.cohort))
    settings = get_settings()
    model = args.model or cohort.model or settings.effective_subject_model

    needs_model = any(s.calls_a_model for s in cohort.systems) or not args.no_baseline
    if needs_model and not settings.has_key:
        print(
            "error: rows that use a model need an API key. Put OPENAI_API_KEY in your .env, "
            "or pass --no-baseline and a cohort of endpoint-only rows.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    ready, blocked = cohort.runnable()
    for system, why in blocked:
        print(f"  skipping {system.label}: {why}", file=sys.stderr)
    if not ready and not args.no_baseline:
        print(
            "Only the model-only baseline can run; every vendor row is missing its key.",
            file=sys.stderr,
        )

    items = len(dataset.sample(args.limit))
    print(
        f"Running {len(cohort.systems) + (0 if cohort.baseline or args.no_baseline else 1)} "
        f"system(s) over {items} item(s) = up to {items * len(cohort.systems)} call(s)",
        file=sys.stderr,
    )

    def progress(done: int, total: int, label: str) -> None:
        if done == total or done % 10 == 0:
            print(f"  [{done}/{total}] {label}", file=sys.stderr)

    board = run_board(
        dataset,
        cohort,
        model=args.model,
        judge_model=args.judge_model,
        limit=args.limit,
        concurrency=args.concurrency,
        cutoff=parse_cutoff(model, args.cutoff) if args.cutoff else None,
        baseline=not args.no_baseline,
        progress=None if args.quiet else progress,
        settings=settings,
    )

    document = render_board(board)
    if args.format == "json":
        print(board.model_dump_json(indent=2))
    else:
        print(document)

    if not args.no_write:
        target = write_board(board, Path(args.out or "boards"))
        (target / "board.md").write_text(document + "\n", encoding="utf-8")
        print(f"\nWrote {target}", file=sys.stderr)
    return EXIT_OK


def _cmd_harden(args: argparse.Namespace) -> int:
    """Run the board, rewrite whatever saturated, and run it again."""
    dataset = load_dataset(Path(args.dataset))
    cohort = load_cohort(Path(args.cohort))
    settings = get_settings()

    if not settings.has_key:
        print(
            "error: the loop needs an API key - it rewrites items with a model and "
            "runs a model-only baseline. Put OPENAI_API_KEY in your .env.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    for system, why in cohort.runnable()[1]:
        print(f"  skipping {system.label}: {why}", file=sys.stderr)

    def progress(done: int, total: int, label: str) -> None:
        if done == total or done % 10 == 0:
            print(f"  [{done}/{total}] {label}", file=sys.stderr)

    def announce(this_round: Round, board: Board) -> None:
        # Every board is printed as it finishes: the loop is the sequence of
        # boards, and a report only at the end hides the runs that made it.
        print(render_round(this_round))
        sys.stdout.flush()
        if not args.no_write:
            target = Path(args.out or "boards") / board.board_id
            if target.is_dir():
                (target / "board.md").write_text(
                    render_board(board) + "\n", encoding="utf-8"
                )

    try:
        report, hardened = run_loop(
            dataset,
            cohort,
            rounds=args.rounds,
            threshold=args.threshold,
            rewrites=args.rewrites,
            model=args.rewrite_model,
            settings=settings,
            boards_dir=None if args.no_write else Path(args.out or "boards"),
            announce=announce,
            judge_model=args.judge_model,
            limit=args.limit,
            concurrency=args.concurrency,
            baseline=not args.no_baseline,
            progress=None if args.quiet else progress,
        )
    except (BoardError, DatasetError, CohortError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(render_loop(report))

    if not args.no_write:
        target = Path(args.dataset_out) if args.dataset_out else None
        if target is not None and report.rewritten:
            write_dataset(hardened, target)
            print(f"\nWrote {target}", file=sys.stderr)
        elif target is not None:
            print(
                f"Nothing verified, so {target} was not written and "
                f"{args.dataset} still stands.",
                file=sys.stderr,
            )
    return EXIT_OK


def _cmd_boards(args: argparse.Namespace) -> int:
    """List the boards on disk, newest first."""
    entries = list_board_dirs(Path(args.dir or "boards"))
    if not entries:
        print(
            f"No boards under {args.dir or 'boards'}. Run one with "
            "`auto-eval board DATASET --cohort COHORT`.",
            file=sys.stderr,
        )
        return EXIT_OK
    if args.format == "json":
        print(json.dumps([e.model_dump(mode="json") for e in entries], indent=2))
        return EXIT_OK
    for entry in entries:
        leader = (
            f"{entry.leader} at {entry.leader_accuracy:.1%}"
            if entry.leader and entry.leader_accuracy is not None
            else "nothing scored"
        )
        baseline = (
            f"baseline {entry.baseline_accuracy:.1%}"
            if entry.baseline_accuracy is not None
            else "no baseline"
        )
        print(
            f"{entry.board_id}  {entry.rows} row(s)  {entry.items} item(s)  {leader}  {baseline}"
        )
    return EXIT_OK


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

    profile_cmd = sub.add_parser(
        "profile",
        help="What it takes to run the agent under test, and whether we can.",
        description=(
            "Reads the operational facts about an agent off its spec - entry point, "
            "tools, ceilings, isolation - and says whether a suite can be authored "
            f"against it. Offline. Exits {EXIT_INSUFFICIENT} when something blocks."
        ),
    )
    profile_cmd.add_argument("text", nargs="*", help="The request, as text.")
    profile_cmd.add_argument("-f", "--file", help="Read the request from a file.")
    profile_cmd.add_argument(
        "-s", "--spec", metavar="PATH", help="A spec JSON to profile."
    )
    profile_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    profile_cmd.add_argument(
        "--model",
        default=None,
        help="Model to classify with, when given a request rather than a spec.",
    )
    profile_cmd.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=argparse.SUPPRESS
    )
    profile_cmd.set_defaults(func=_cmd_profile)

    author_cmd = sub.add_parser(
        "author",
        help="Write an eval suite for an agent.",
        description=(
            "Builds the cases, their checks, and the coverage grid they answer to. "
            "Deterministic and offline: the same spec produces the same suite, digest "
            "and all. Nothing is run and no fixture is materialised - the suite says "
            f"what has to exist before it can be. Exits {EXIT_INSUFFICIENT} when the "
            "agent cannot be run and --force was not passed."
        ),
    )
    author_cmd.add_argument("text", nargs="*", help="The request, as text.")
    author_cmd.add_argument("-f", "--file", help="Read the request from a file.")
    author_cmd.add_argument(
        "-s", "--spec", metavar="PATH", help="A spec JSON to author from."
    )
    author_cmd.add_argument(
        "-o", "--out-dir", metavar="PATH", help="Write suite.json and suite.md here."
    )
    author_cmd.add_argument(
        "--per-cell",
        type=int,
        default=DEFAULT_PER_CELL,
        help=f"Cases per cell of the coverage grid (default {DEFAULT_PER_CELL}).",
    )
    author_cmd.add_argument(
        "--force", action="store_true", help="Author even when the agent cannot be run."
    )
    author_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    author_cmd.add_argument(
        "--model",
        default=None,
        help="Model to classify with, when given a request rather than a spec.",
    )
    author_cmd.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=argparse.SUPPRESS
    )
    author_cmd.set_defaults(func=_cmd_author)

    suite_cmd = sub.add_parser(
        "suite",
        help="Read back a suite written by `author`.",
        description="Prints what a written suite covers, or one case in full with --case.",
    )
    suite_cmd.add_argument(
        "path", help="The directory `author -o` wrote, or the suite.json itself."
    )
    suite_cmd.add_argument(
        "--case", metavar="ID", help="Print this case in full; an id prefix is enough."
    )
    suite_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    suite_cmd.set_defaults(func=_cmd_suite)

    run_cmd = sub.add_parser(
        "run",
        help="Run a written suite against the provider API.",
        description=(
            "Runs every case in a suite, `samples` times each, and writes the result where "
            "the dashboard can read it. Tools are declared to the model and answered inside "
            "the harness, so nothing touches a live system - which also means end-state "
            "checks are reported as blocked rather than guessed at. --mock runs the whole "
            "path against a canned client, with no API call."
        ),
    )
    run_cmd.add_argument(
        "suite", help="The directory `author -o` wrote, or the suite.json itself."
    )
    run_cmd.add_argument(
        "--model",
        default=None,
        help="The model under test. Defaults to AUTO_EVAL_SUBJECT_MODEL.",
    )
    run_cmd.add_argument(
        "--judge-model",
        default=None,
        help="Model that rules on the rubrics. Defaults to AUTO_EVAL_JUDGE_MODEL.",
    )
    run_cmd.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the rubric checks entirely; they are reported as unsettled.",
    )
    run_cmd.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Runs per case. Defaults to the suite's own.",
    )
    run_cmd.add_argument(
        "--split",
        choices=["all", "held_out", "dev"],
        default="all",
        help="Which cases to run. The reported number comes from held_out.",
    )
    run_cmd.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases."
    )
    run_cmd.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Cases in flight at once (default {DEFAULT_CONCURRENCY}).",
    )
    run_cmd.add_argument(
        "--mock", action="store_true", help="Call no provider; use the canned client."
    )
    run_cmd.add_argument(
        "--cutoff",
        metavar="YYYY-MM-DD",
        default=None,
        help="The model's training cutoff, for the contamination check.",
    )
    run_cmd.add_argument(
        "--price-in", type=float, default=None, help="USD per million input tokens."
    )
    run_cmd.add_argument(
        "--price-out", type=float, default=None, help="USD per million output tokens."
    )
    run_cmd.add_argument("-o", "--out", metavar="PATH", help="Where to write the run.")
    run_cmd.add_argument(
        "--no-write", action="store_true", help="Print the report without writing it."
    )
    run_cmd.add_argument("--quiet", action="store_true", help="No per-case progress.")
    run_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    run_cmd.set_defaults(func=_cmd_run)

    runs_cmd = sub.add_parser("runs", help="List the runs on disk, newest first.")
    runs_cmd.add_argument(
        "--dir",
        metavar="PATH",
        help="Where the runs are. Defaults to AUTO_EVAL_RUNS_DIR.",
    )
    runs_cmd.add_argument(
        "--format", choices=["text", "json"], default="text", help="Stdout format."
    )
    runs_cmd.set_defaults(func=_cmd_runs)

    report_cmd = sub.add_parser("report", help="Read back a run written by `run`.")
    report_cmd.add_argument("path", help="The run directory, or the run.json itself.")
    report_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    report_cmd.set_defaults(func=_cmd_report)

    fresh_cmd = sub.add_parser(
        "fresh",
        help="Public ground truth the model cannot already have seen.",
        description=(
            "Benchmarks that resist contamination: answers never published, refreshed on a "
            "schedule, or simply released after the model's training cutoff. Offline - the "
            "dates and cutoffs are on file with the page each was read from."
        ),
    )
    fresh_cmd.add_argument(
        "--model", default=None, help="Judge the catalogue against this model's cutoff."
    )
    fresh_cmd.add_argument(
        "--cutoff",
        metavar="YYYY-MM-DD",
        default=None,
        help="Use this cutoff instead of the one on file.",
    )
    fresh_cmd.set_defaults(func=_cmd_fresh)

    board_cmd = sub.add_parser(
        "board",
        help="Compare several systems on the same items.",
        description=(
            "Runs every system in a cohort over the same dataset and writes the board: "
            "accuracy, answer recall, latency, errors and spend, one row each. Rows are "
            "alphabetical and no column decides the order. A model-only row is added unless "
            "you say otherwise - it is the control that shows whether the answers could "
            "simply be recalled."
        ),
    )
    board_cmd.add_argument(
        "dataset", help="A .jsonl or .json file of items with gold answers."
    )
    board_cmd.add_argument(
        "-c",
        "--cohort",
        required=True,
        metavar="PATH",
        help="The cohort JSON: the systems to compare.",
    )
    board_cmd.add_argument(
        "--model",
        default=None,
        help="The model held constant across rows that use one.",
    )
    board_cmd.add_argument(
        "--judge-model", default=None, help="Settles items that have no gold answer."
    )
    board_cmd.add_argument(
        "--limit", type=int, default=None, help="Score the first N items."
    )
    board_cmd.add_argument(
        "--concurrency",
        type=int,
        default=BOARD_CONCURRENCY,
        help=f"Items in flight per system (default {BOARD_CONCURRENCY}).",
    )
    board_cmd.add_argument(
        "--no-baseline",
        action="store_true",
        help="Leave out the model-only row. You lose the control that the board rests on.",
    )
    board_cmd.add_argument(
        "--cutoff",
        metavar="YYYY-MM-DD",
        default=None,
        help="The model's training cutoff.",
    )
    board_cmd.add_argument(
        "-o", "--out", metavar="PATH", help="Where to write the board."
    )
    board_cmd.add_argument(
        "--no-write", action="store_true", help="Print it without writing it."
    )
    board_cmd.add_argument("--quiet", action="store_true", help="No progress.")
    board_cmd.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Stdout format.",
    )
    board_cmd.set_defaults(func=_cmd_board)

    harden_cmd = sub.add_parser(
        "harden",
        help="Run the board, rewrite whatever saturated, run it again.",
        description=(
            "The closed loop over a board. It runs the cohort, finds the rows that "
            "reached 100% and the items holding them there, rewrites those items into "
            "harder questions drawn from their own source documents - verified against "
            "the fetched page, never taken on the model's word - and runs the board "
            "again. Every board is printed as it finishes."
        ),
    )
    harden_cmd.add_argument(
        "dataset", help="A .jsonl or .json file of items with gold answers."
    )
    harden_cmd.add_argument(
        "-c",
        "--cohort",
        required=True,
        metavar="PATH",
        help="The cohort JSON: the systems to compare.",
    )
    harden_cmd.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"Boards to run at most (default {DEFAULT_ROUNDS}).",
    )
    harden_cmd.add_argument(
        "--threshold",
        type=float,
        default=SATURATED,
        help=(
            "The accuracy that counts as saturated, as a fraction "
            f"(default {SATURATED:g} - a row that got everything right)."
        ),
    )
    harden_cmd.add_argument(
        "--rewrites",
        type=int,
        default=DEFAULT_REWRITES,
        help=f"Items to rewrite per round (default {DEFAULT_REWRITES}).",
    )
    harden_cmd.add_argument(
        "--rewrite-model",
        default=None,
        help="The model that proposes replacements. Defaults to the configured one.",
    )
    harden_cmd.add_argument(
        "--judge-model", default=None, help="Settles items that have no gold answer."
    )
    harden_cmd.add_argument(
        "--limit", type=int, default=None, help="Score the first N items."
    )
    harden_cmd.add_argument(
        "--concurrency",
        type=int,
        default=BOARD_CONCURRENCY,
        help=f"Items in flight per system (default {BOARD_CONCURRENCY}).",
    )
    harden_cmd.add_argument(
        "--no-baseline",
        action="store_true",
        help="Leave out the model-only row. You lose the control that the board rests on.",
    )
    harden_cmd.add_argument(
        "-o", "--out", metavar="PATH", help="Where to write each round's board."
    )
    harden_cmd.add_argument(
        "--dataset-out",
        metavar="PATH",
        help="Where to write the hardened items. Without it the dataset is left alone.",
    )
    harden_cmd.add_argument(
        "--no-write", action="store_true", help="Print the rounds without writing them."
    )
    harden_cmd.add_argument("--quiet", action="store_true", help="No progress.")
    harden_cmd.set_defaults(func=_cmd_harden)

    boards_cmd = sub.add_parser("boards", help="List the boards on disk, newest first.")
    boards_cmd.add_argument(
        "--dir", metavar="PATH", help="Where the boards are. Defaults to ./boards."
    )
    boards_cmd.add_argument(
        "--format", choices=["text", "json"], default="text", help="Stdout format."
    )
    boards_cmd.set_defaults(func=_cmd_boards)

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

    bench_cmd = sub.add_parser(
        "benchmarks",
        help="Public benchmarks this task could be anchored on.",
        description=(
            "With no input, prints the catalogue. With a spec or a request, "
            "prints the entries that match it. Offline either way - no API call "
            "and no web search."
        ),
    )
    bench_cmd.add_argument("text", nargs="*", help="The request, as text.")
    bench_cmd.add_argument("-f", "--file", help="Read the request from a file.")
    bench_cmd.add_argument("-s", "--spec", metavar="PATH", help="A spec JSON to match.")
    bench_cmd.add_argument(
        "--limit", type=int, default=5, help="How many matches to show."
    )
    bench_cmd.add_argument(
        "--format", choices=["text", "json"], default="text", help="Stdout format."
    )
    bench_cmd.add_argument(
        "--model",
        default=None,
        help="Model to classify with, when a request is given rather than a spec.",
    )
    bench_cmd.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=argparse.SUPPRESS
    )
    bench_cmd.set_defaults(func=_cmd_benchmarks)

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
    except (RunError, BoardError, CohortError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except SuiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INSUFFICIENT
    except ClassifierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

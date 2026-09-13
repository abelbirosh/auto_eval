"""A small local web UI for trying the classifier by hand.

Deliberately local-only: the API key lives in your `.env` and never leaves the
machine. This is a test harness, not a deployment target - there is no auth
here, so do not bind it to a public interface.
"""

from __future__ import annotations

import re
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .agent import AGENTIC_KINDS
from .analysis import AnalysisReport, analyze_sources
from .board import (
    Board,
    BoardError,
    BoardIndexEntry,
    load_board,
    run_board,
    write_board,
)
from .board import list_boards as list_board_dirs
from .classifier import DEFAULT_MAX_TOKENS, ClassifierError, classify
from .cohort import Cohort, CohortError
from .cohort import load as load_cohort
from .cohort import parse as parse_cohort
from .config import Settings, get_settings
from .contamination import parse_cutoff
from .dataset import Dataset, DatasetError
from .dataset import load as load_dataset
from .dataset import parse as parse_dataset
from .gaps import analyze
from .ground_truth import GroundTruthReport, gate, identify
from .render import (
    render_analysis,
    render_board,
    render_ground_truth,
    render_markdown,
    render_run,
    render_suite,
)
from .runner import (
    RunError,
    RunIndexEntry,
    RunReport,
    list_runs,
    load_run,
    load_trace,
    run_suite,
    write_run,
)
from .schema import Answer, TaskSpec
from .suite import SUITE_FILENAME, Split, Suite, SuiteError
from .suite import build as build_suite
from .suite import load as load_suite
from .suite import write as write_suite
from .surface import DEFAULT_PER_CELL
from .trace import Trace

STATIC_DIR = Path(__file__).parent / "static"

# A run id names a directory, so it may only look like one.
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# How far the suite picker looks for suites, and how many it will list. A local
# tool walking an entire home directory is a surprise nobody wants.
SUITE_SCAN_DEPTH = 4
SUITE_SCAN_LIMIT = 50


class ClassifyRequest(BaseModel):
    text: str = Field(description="The free-form evaluation request.")
    answers: List[Answer] = Field(
        default_factory=list,
        description="Replies to questions an earlier classification asked.",
    )
    model: Optional[str] = None


class ClassifyResponse(BaseModel):
    spec: TaskSpec
    markdown: str
    model: str


class GroundTruthRequest(BaseModel):
    spec: TaskSpec = Field(description="A spec returned by /api/classify.")
    model: Optional[str] = None


class GroundTruthResponse(BaseModel):
    report: GroundTruthReport
    markdown: str


class AnalyzeRequest(BaseModel):
    spec: TaskSpec
    report: GroundTruthReport = Field(
        description="A report returned by /api/ground-truth."
    )
    model: Optional[str] = None


class AnalyzeResponse(BaseModel):
    analysis: AnalysisReport
    markdown: str


class SuiteEntry(BaseModel):
    """One suite the dashboard can run."""

    name: str
    path: str
    subject: str
    digest: str
    cases: int
    samples: int
    held_out: int
    warnings: int


class RunOptions(BaseModel):
    """How to run a suite. Shared by both ways of starting one."""

    model: Optional[str] = None
    judge_model: Optional[str] = None
    judge: bool = True
    samples: Optional[int] = None
    split: Optional[str] = Field(
        default=None, description="held_out, dev, or null for all."
    )
    limit: Optional[int] = None
    concurrency: int = 4
    mock: bool = False
    cutoff: Optional[str] = None
    out: Optional[str] = None


class RunRequest(RunOptions):
    """Run a suite that is already on disk."""

    suite: str = Field(description="Path to a suite directory or suite.json.")


class BenchmarkRequest(RunOptions):
    """Author a suite for a spec and run it, in one move.

    This is what the classifier page's button posts once the ground-truth stage
    is done: everything from here on is deterministic or mechanical, so there is
    nothing left for the user to decide between authoring and running.
    """

    spec: TaskSpec = Field(description="A spec returned by /api/classify.")
    per_cell: int = DEFAULT_PER_CELL
    force: bool = Field(
        default=False,
        description="Author even when the agent gate is closed - the page asks first.",
    )
    suites_dir: Optional[str] = None


class BoardRequest(BaseModel):
    """Run a comparison board: the same items put to every system in a cohort."""

    cohort: Optional[str] = Field(default=None, description="The cohort as JSON text.")
    cohort_path: Optional[str] = Field(default=None, description="Or a path to it.")
    dataset: Optional[str] = Field(
        default=None, description="The items as JSONL or JSON text."
    )
    dataset_path: Optional[str] = Field(default=None, description="Or a path to them.")
    dataset_name: str = "items"
    task: Optional[str] = None
    model: Optional[str] = None
    judge_model: Optional[str] = None
    limit: Optional[int] = None
    concurrency: int = 4
    baseline: bool = True
    cutoff: Optional[str] = None
    out: Optional[str] = None


class Job(BaseModel):
    """A run in flight. Polled by the page while it works."""

    job_id: str
    state: str = Field(default="running", description="running, done, or failed.")
    suite: str = ""
    model: str = ""
    mock: bool = False
    done: int = 0
    total: int = 0
    case_id: Optional[str] = None
    run_id: Optional[str] = None
    path: Optional[str] = None
    error: Optional[str] = None
    # Filled when the job authored the suite as well as running it, so the page
    # can say what it is about to run before the first case comes back.
    suite_name: Optional[str] = None
    suite_digest: Optional[str] = None
    cases: int = 0
    held_out: int = 0
    suite_warnings: List[str] = Field(default_factory=list)
    # Filled when the job is a board rather than a suite run.
    board_id: Optional[str] = None
    systems: int = 0


# Jobs live in this process only: the UI is a local harness, and a run that was
# interrupted is one the user can simply start again.
_JOBS: Dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()


def _set(job_id: str, **fields: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        _JOBS[job_id] = job.model_copy(update=fields)


def _runs_dir(override: Optional[str] = None) -> Path:
    return Path(override) if override else get_settings().effective_runs_dir


def launch(
    suite: Suite, options: RunOptions, settings: Any, *, source: str = ""
) -> Job:
    """Start a run in the background and hand back a job to poll.

    The run is not awaited: eighty runs outlive any sensible request timeout,
    and the page wants to show progress anyway.
    """
    split = Split(options.split) if options.split else None
    cases = suite.cases_in(split) if split else suite.cases
    planned = min(len(cases), options.limit) if options.limit else len(cases)
    model = options.model or settings.effective_subject_model

    job = Job(
        job_id=uuid.uuid4().hex[:12],
        suite=source,
        model=model,
        mock=options.mock,
        total=planned * (options.samples or suite.samples or 1),
        suite_name=suite.name,
        suite_digest=suite.digest,
        cases=planned,
        held_out=len([c for c in cases if c.id in set(suite.held_out)]),
        suite_warnings=suite.warnings,
    )
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job

    def work() -> None:
        try:
            run = run_suite(
                suite,
                model=options.model,
                judge_model=options.judge_model,
                judge=options.judge,
                samples=options.samples,
                split=split,
                limit=options.limit,
                concurrency=options.concurrency,
                mock=options.mock,
                cutoff=parse_cutoff(model, options.cutoff) if options.cutoff else None,
                progress=lambda done, total, case_id: _set(
                    job.job_id, done=done, total=total, case_id=case_id
                ),
                settings=settings,
            )
            target = write_run(run, _runs_dir(options.out))
            _set(
                job.job_id,
                state="done",
                run_id=run.report.run_id,
                path=str(target),
                done=run.report.overall.runs,
            )
        except Exception as exc:  # surfaced to the page, not swallowed
            _set(job.job_id, state="failed", error=str(exc))

    threading.Thread(
        target=work, name=f"auto-eval-run-{job.job_id}", daemon=True
    ).start()
    return job


def _boards_dir(override: Optional[str] = None) -> Path:
    """Boards live next to runs unless told otherwise."""
    return (
        Path(override)
        if override
        else get_settings().effective_runs_dir.parent / "boards"
    )


def launch_board(
    dataset: Dataset,
    cohort: Cohort,
    request: "BoardRequest",
    settings: Settings,
) -> Job:
    """Start a board in the background and hand back a job to poll."""
    items = len(dataset.sample(request.limit))
    systems = len(cohort.systems) + (
        0 if cohort.baseline or not request.baseline else 1
    )

    job = Job(
        job_id=uuid.uuid4().hex[:12],
        suite=dataset.name,
        model=request.model or cohort.model or settings.effective_subject_model,
        total=items * systems,
        cases=items,
        systems=systems,
    )
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job

    done = {"count": 0}

    def progress(_done: int, _total: int, label: str) -> None:
        # Each row reports its own items, so the page is shown the running total
        # across the whole board rather than a bar that restarts per system.
        done["count"] += 1
        _set(job.job_id, done=done["count"], case_id=label)

    def work() -> None:
        try:
            board = run_board(
                dataset,
                cohort,
                model=request.model,
                judge_model=request.judge_model,
                limit=request.limit,
                concurrency=request.concurrency,
                cutoff=parse_cutoff(job.model, request.cutoff)
                if request.cutoff
                else None,
                baseline=request.baseline,
                progress=progress,
                settings=settings,
            )
            target = write_board(board, _boards_dir(request.out))
            _set(
                job.job_id,
                state="done",
                board_id=board.board_id,
                path=str(target),
                done=job.total,
            )
        except Exception as exc:  # surfaced to the page, not swallowed
            _set(job.job_id, state="failed", error=str(exc))

    threading.Thread(
        target=work, name=f"auto-eval-board-{job.job_id}", daemon=True
    ).start()
    return job


def find_suites(root: Path) -> List[SuiteEntry]:
    """Every readable suite under `root`, for the dashboard's picker."""
    found: List[SuiteEntry] = []
    for path in sorted(root.rglob(SUITE_FILENAME)):
        # Relative, not absolute: the working directory may itself sit under a
        # dotted path, and that is not a reason to hide everything in it.
        relative = path.relative_to(root).parts
        if len(relative) > SUITE_SCAN_DEPTH:
            continue
        if any(part.startswith(".") for part in relative):
            continue
        try:
            suite = load_suite(path)
        except (SuiteError, ValueError):
            continue
        found.append(
            SuiteEntry(
                name=suite.name,
                path=str(path),
                subject=suite.subject,
                digest=suite.digest,
                cases=len(suite.cases),
                samples=suite.samples,
                held_out=len(suite.held_out),
                warnings=len(suite.warnings),
            )
        )
        if len(found) >= SUITE_SCAN_LIMIT:
            break
    return found


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise ClassifierError(
            'The web UI needs FastAPI. Install it with `pip install -e ".[web]"`.'
        ) from exc

    app = FastAPI(title="Auto_Eval", docs_url="/api/docs")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return FileResponse(STATIC_DIR / "dashboard.html")

    @app.get("/api/health")
    def health():
        """Lets the page warn about a missing key before you type a paragraph."""
        settings = get_settings()
        return {
            "has_key": settings.has_key,
            "model": settings.model,
            "subject_model": settings.effective_subject_model,
            "judge_model": settings.effective_judge_model,
            "runs_dir": str(settings.effective_runs_dir),
            # Which subjects have a trajectory is the gate's decision, so the
            # page reads it from here rather than keeping its own copy: the two
            # disagreeing is how you end up offering a suite the gate refuses.
            "agentic_kinds": sorted(kind.value for kind in AGENTIC_KINDS),
        }

    @app.get("/api/suites", response_model=List[SuiteEntry])
    def suites_endpoint(root: Optional[str] = None) -> List[SuiteEntry]:
        """The suites on disk, so the dashboard can offer them by name."""
        base = Path(root) if root else Path.cwd()
        if not base.is_dir():
            raise HTTPException(status_code=404, detail=f"No directory at {base}.")
        return find_suites(base)

    @app.get("/api/runs", response_model=List[RunIndexEntry])
    def runs_endpoint(dir: Optional[str] = None) -> List[RunIndexEntry]:
        return list_runs(_runs_dir(dir))

    @app.get("/api/runs/{run_id}", response_model=RunReport)
    def run_endpoint(run_id: str, dir: Optional[str] = None) -> RunReport:
        if not SAFE_ID.match(run_id):
            raise HTTPException(status_code=400, detail="That is not a run id.")
        try:
            return load_run(_runs_dir(dir) / run_id)
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/markdown")
    def run_markdown_endpoint(run_id: str, dir: Optional[str] = None) -> Dict[str, str]:
        """The same report as a document, for pasting into a ticket."""
        if not SAFE_ID.match(run_id):
            raise HTTPException(status_code=400, detail="That is not a run id.")
        try:
            return {"markdown": render_run(load_run(_runs_dir(dir) / run_id))}
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/trace/{case_id}", response_model=Trace)
    def trace_endpoint(
        run_id: str, case_id: str, sample: int = 0, dir: Optional[str] = None
    ) -> Trace:
        """One run of one case, in full. This is what a verdict was read off."""
        if not (SAFE_ID.match(run_id) and SAFE_ID.match(case_id)):
            raise HTTPException(status_code=400, detail="That is not a run or case id.")
        try:
            return load_trace(_runs_dir(dir) / run_id, case_id, sample)
        except RunError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _key_or_mock(options: RunOptions) -> Settings:
        """Nothing real can be run without a key; say so before anything starts."""
        settings = get_settings()
        if not options.mock and not settings.has_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No API key, so nothing can be run against the provider. Put "
                    "OPENAI_API_KEY in your .env, or tick 'dry run' to exercise the "
                    "harness offline."
                ),
            )
        return settings

    @app.post("/api/run", response_model=Job)
    def start_run_endpoint(request: RunRequest) -> Job:
        """Run a suite that is already on disk."""
        try:
            suite = load_suite(Path(request.suite))
        except (SuiteError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return launch(suite, request, _key_or_mock(request), source=str(request.suite))

    @app.post("/api/benchmark", response_model=Job)
    def benchmark_endpoint(request: BenchmarkRequest) -> Job:
        """Author the suite for a spec and run it, in one move.

        Authoring is deterministic and offline, so it happens inline; the run is
        the part that takes minutes and goes to a background job. A closed agent
        gate comes back as a 409 with its reason, which the page turns into a
        question rather than an error.
        """
        settings = _key_or_mock(request)
        # The spec arrives over the wire, so its questions are recomputed here
        # rather than trusted - the same rule the ground-truth endpoint follows.
        spec = analyze(request.spec)
        try:
            suite = build_suite(spec, per_cell=request.per_cell, force=request.force)
        except SuiteError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        target = Path(request.suites_dir or "suites") / suite.name
        try:
            write_suite(suite, target, document=render_suite(suite))
        except OSError as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not write the suite: {exc}"
            ) from exc
        return launch(suite, request, settings, source=str(target))

    @app.post("/api/board", response_model=Job)
    def start_board_endpoint(request: BoardRequest) -> Job:
        """Run every system in a cohort over the same items.

        This is the shape for a subject that has no trajectory - an endpoint, a
        model, a retrieval API. Nothing is authored: the items carry their own
        answers, so the only question is which system finds them.
        """
        settings = get_settings()
        try:
            if request.dataset_path:
                dataset = load_dataset(Path(request.dataset_path))
            elif request.dataset:
                dataset = parse_dataset(
                    request.dataset,
                    name=request.dataset_name,
                    task=request.task or "factual lookup",
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No items. Paste them as JSONL, or point at a file.",
                )
            cohort = (
                load_cohort(Path(request.cohort_path))
                if request.cohort_path
                else parse_cohort(request.cohort or "")
            )
        except (DatasetError, CohortError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        needs_model = request.baseline or any(s.calls_a_model for s in cohort.systems)
        if needs_model and not settings.has_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Rows that use a model need an API key. Put OPENAI_API_KEY in your .env, "
                    "or untick the model-only baseline and compare endpoints alone."
                ),
            )

        ready, blocked = cohort.runnable()
        if not ready and not request.baseline:
            raise HTTPException(
                status_code=400,
                detail="No system can be run: " + "; ".join(why for _, why in blocked),
            )
        return launch_board(dataset, cohort, request, settings)

    @app.get("/api/boards", response_model=List[BoardIndexEntry])
    def boards_endpoint(dir: Optional[str] = None) -> List[BoardIndexEntry]:
        return list_board_dirs(_boards_dir(dir))

    @app.get("/api/boards/{board_id}", response_model=Board)
    def board_endpoint(board_id: str, dir: Optional[str] = None) -> Board:
        if not SAFE_ID.match(board_id):
            raise HTTPException(status_code=400, detail="That is not a board id.")
        try:
            return load_board(_boards_dir(dir) / board_id)
        except BoardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/boards/{board_id}/markdown")
    def board_markdown_endpoint(
        board_id: str, dir: Optional[str] = None
    ) -> Dict[str, str]:
        if not SAFE_ID.match(board_id):
            raise HTTPException(status_code=400, detail="That is not a board id.")
        try:
            return {"markdown": render_board(load_board(_boards_dir(dir) / board_id))}
        except BoardError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/run/{job_id}", response_model=Job)
    def job_endpoint(job_id: str) -> Job:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job in this process.")
        return job

    @app.post("/api/classify", response_model=ClassifyResponse)
    def classify_endpoint(request: ClassifyRequest) -> ClassifyResponse:
        try:
            spec = classify(
                request.text,
                answers=request.answers,
                model=request.model,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except ClassifierError as exc:
            # The message is already written to be shown to a person.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ClassifyResponse(
            spec=spec,
            markdown=render_markdown(spec),
            model=request.model or get_settings().model,
        )

    @app.post("/api/ground-truth", response_model=GroundTruthResponse)
    def ground_truth_endpoint(request: GroundTruthRequest) -> GroundTruthResponse:
        # The spec comes back over the wire, so its questions are recomputed
        # here rather than trusted; the gate then reads the same as the CLI's.
        spec = analyze(request.spec)
        decision = gate(spec)
        if not decision.open:
            raise HTTPException(
                status_code=409, detail=f"Not ready to search: {decision.reason}"
            )

        try:
            report = identify(spec, model=request.model, max_tokens=DEFAULT_MAX_TOKENS)
        except ClassifierError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return GroundTruthResponse(report=report, markdown=render_ground_truth(report))

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    def analyze_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:
        """Look at each identified source and say what to fetch from it later.

        This is the one endpoint that opens URLs the model proposed. It reads
        metadata and pages only - nothing is downloaded - and `auto_eval.fetch`
        refuses any address that is not public.
        """
        try:
            analysis = analyze_sources(
                request.spec, request.report, model=request.model
            )
        except ClassifierError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return AnalyzeResponse(analysis=analysis, markdown=render_analysis(analysis))

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ClassifierError(
            'The web UI needs uvicorn. Install it with `pip install -e ".[web]"`.'
        ) from exc

    settings = get_settings()
    print(f"Auto_Eval UI on http://{host}:{port}  (model: {settings.model})")
    print(
        f"  dashboard: http://{host}:{port}/dashboard  "
        f"(runs in {settings.effective_runs_dir}, under test: {settings.effective_subject_model})"
    )
    if not settings.has_key:
        print(
            "  warning: no OPENAI_API_KEY found - put one in .env before classifying."
        )

    if reload:
        uvicorn.run(
            "auto_eval.web:create_app", host=host, port=port, reload=True, factory=True
        )
    else:
        uvicorn.run(create_app(), host=host, port=port)


__all__ = [
    "SAFE_ID",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "BenchmarkRequest",
    "BoardRequest",
    "ClassifyRequest",
    "ClassifyResponse",
    "GroundTruthRequest",
    "GroundTruthResponse",
    "Job",
    "RunOptions",
    "RunRequest",
    "SuiteEntry",
    "create_app",
    "find_suites",
    "launch",
    "launch_board",
    "serve",
]

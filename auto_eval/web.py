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

from .analysis import AnalysisReport, analyze_sources
from .classifier import DEFAULT_MAX_TOKENS, ClassifierError, classify
from .config import get_settings
from .contamination import parse_cutoff
from .gaps import analyze
from .ground_truth import GroundTruthReport, gate, identify
from .render import render_analysis, render_ground_truth, render_markdown, render_run
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
from .suite import SUITE_FILENAME, Split, SuiteError
from .suite import load as load_suite
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


class RunRequest(BaseModel):
    """Start a run of a suite on disk."""

    suite: str = Field(description="Path to a suite directory or suite.json.")
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


class Job(BaseModel):
    """A run in flight. Polled by the dashboard while it works."""

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

    @app.post("/api/run", response_model=Job)
    def start_run_endpoint(request: RunRequest) -> Job:
        """Start a run in the background and hand back a job to poll.

        The run is not awaited: a suite of eighty runs outlives any sensible
        request timeout, and the dashboard wants to show progress anyway.
        """
        try:
            suite = load_suite(Path(request.suite))
        except (SuiteError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        settings = get_settings()
        if not request.mock and not settings.has_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No API key, so nothing can be run against the provider. Put "
                    "OPENAI_API_KEY in your .env, or tick 'mock' to exercise the harness offline."
                ),
            )

        split = Split(request.split) if request.split else None
        cases = suite.cases_in(split) if split else suite.cases
        planned = min(len(cases), request.limit) if request.limit else len(cases)
        model = request.model or settings.effective_subject_model

        job = Job(
            job_id=uuid.uuid4().hex[:12],
            suite=str(request.suite),
            model=model,
            mock=request.mock,
            total=planned * (request.samples or suite.samples or 1),
        )
        with _JOBS_LOCK:
            _JOBS[job.job_id] = job

        def work() -> None:
            try:
                run = run_suite(
                    suite,
                    model=request.model,
                    judge_model=request.judge_model,
                    judge=request.judge,
                    samples=request.samples,
                    split=split,
                    limit=request.limit,
                    concurrency=request.concurrency,
                    mock=request.mock,
                    cutoff=parse_cutoff(model, request.cutoff)
                    if request.cutoff
                    else None,
                    progress=lambda done, total, case_id: _set(
                        job.job_id, done=done, total=total, case_id=case_id
                    ),
                    settings=settings,
                )
                target = write_run(run, _runs_dir(request.out))
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
    "ClassifyRequest",
    "ClassifyResponse",
    "GroundTruthRequest",
    "GroundTruthResponse",
    "Job",
    "RunRequest",
    "SuiteEntry",
    "create_app",
    "find_suites",
    "serve",
]

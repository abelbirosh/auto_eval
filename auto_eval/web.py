"""A small local web UI for trying the classifier by hand.

Deliberately local-only: the API key lives in your `.env` and never leaves the
machine. This is a test harness, not a deployment target - there is no auth
here, so do not bind it to a public interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from .analysis import AnalysisReport, analyze_sources
from .classifier import DEFAULT_MAX_TOKENS, ClassifierError, classify
from .config import get_settings
from .gaps import analyze
from .ground_truth import GroundTruthReport, gate, identify
from .render import render_analysis, render_ground_truth, render_markdown
from .schema import Answer, TaskSpec

STATIC_DIR = Path(__file__).parent / "static"


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

    @app.get("/api/health")
    def health():
        """Lets the page warn about a missing key before you type a paragraph."""
        settings = get_settings()
        return {"has_key": settings.has_key, "model": settings.model}

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
    "AnalyzeRequest",
    "AnalyzeResponse",
    "ClassifyRequest",
    "ClassifyResponse",
    "GroundTruthRequest",
    "GroundTruthResponse",
    "create_app",
    "serve",
]

"""Calling one vendor's endpoint, and timing what comes back.

Everything a board measures about a retrieval API is here: the request built
from the system's declared configuration, the wall-clock time it took, and the
results parsed into one shape so that two vendors with different response
schemas can be scored by the same rule.

Three rules the adapter keeps:

* **Only the query goes out.** The body and params come from the cohort file and
  the item's query is substituted into them; nothing else about the run - the
  gold answer least of all - is ever sent to a vendor.
* **Secrets are read at call time.** `${EXA_API_KEY}` is resolved from the
  environment here and never stored in a cohort, a board, or a log.
* **A failure is recorded, not scored.** A timeout or a 500 comes back as an
  error on that item; the board reports the error rate beside the accuracy
  rather than folding a vendor's outage into its quality.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .classifier import ClassifierError
from .cohort import ENV_REFERENCE, QUERY_PLACEHOLDER, Endpoint
from .fetch import check_url

# Enough of a result to score and to quote. A vendor returning a whole page per
# hit is truncated here rather than in the report.
MAX_SNIPPET_CHARS = 2000

# Results past this rank are dropped: no board here scores beyond the first page,
# and keeping them only inflates the stored board.
MAX_HITS = 20


class EndpointError(ClassifierError):
    """Raised when an endpoint cannot be called at all, as opposed to failing."""


class Hit(BaseModel):
    """One result, in the shape every scorer reads."""

    rank: int
    title: str = ""
    url: str = ""
    snippet: str = ""

    def text(self) -> str:
        return " ".join(part for part in (self.title, self.snippet, self.url) if part)


class Response(BaseModel):
    """What one call returned, and what it cost in time."""

    hits: List[Hit] = Field(default_factory=list)
    seconds: float = 0.0
    status: Optional[int] = None
    error: Optional[str] = None
    characters: int = Field(
        default=0, description="Size of the response body, as a rough token proxy."
    )

    @property
    def ok(self) -> bool:
        return self.error is None


def _resolve(value: Any) -> Any:
    """Fill ${ENV} references from the environment, leaving everything else alone."""
    if isinstance(value, str):

        def swap(match: "re.Match[str]") -> str:
            found = os.environ.get(match.group(1))
            if found is None:
                raise EndpointError(
                    f"{match.group(1)} is not set, so this endpoint cannot be called."
                )
            return found

        return ENV_REFERENCE.sub(swap, value)
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


def _fill(value: Any, query: str) -> Any:
    """Substitute the item's query wherever the configuration asked for it."""
    if isinstance(value, str):
        return value.replace(QUERY_PLACEHOLDER, query)
    if isinstance(value, dict):
        return {k: _fill(v, query) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, query) for v in value]
    return value


def dig(payload: Any, path: str) -> Any:
    """Follow a dotted path into a parsed response. '' is the whole thing."""
    if not path:
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_string(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value)[:MAX_SNIPPET_CHARS]
    return str(value)


def parse_hits(payload: Any, endpoint: Endpoint) -> List[Hit]:
    """Turn a vendor's response into hits, using the mapping it was declared with."""
    found = dig(payload, endpoint.results_path)
    if isinstance(found, dict):
        # Some vendors wrap the list one level deeper than the path given.
        found = next((v for v in found.values() if isinstance(v, list)), None)
    if not isinstance(found, list):
        return []

    hits: List[Hit] = []
    for rank, raw in enumerate(found[:MAX_HITS], 1):
        if isinstance(raw, str):
            hits.append(Hit(rank=rank, snippet=raw[:MAX_SNIPPET_CHARS]))
            continue
        if not isinstance(raw, dict):
            continue
        hits.append(
            Hit(
                rank=rank,
                title=_string(dig(raw, endpoint.fields.get("title", "title")))[:400],
                url=_string(dig(raw, endpoint.fields.get("url", "url")))[:600],
                snippet=_string(dig(raw, endpoint.fields.get("snippet", "text")))[
                    :MAX_SNIPPET_CHARS
                ],
            )
        )
    return hits


def call(endpoint: Endpoint, query: str, *, client: Optional[Any] = None) -> Response:
    """Call `endpoint` with one query. Never raises for a bad response.

    When no client is passed this module opens the address itself, so it checks
    it first - a cohort file pointing at a metadata endpoint is refused rather
    than fetched from inside whatever network this runs on. A caller that brings
    its own transport has already made that decision; `auto_eval.board` checks
    each address once per row instead of once per item.
    """
    if client is None:
        allowed, why = check_url(endpoint.url)
        if not allowed:
            return Response(error=why)

    try:
        headers = _resolve(endpoint.headers)
        params = _fill(_resolve(endpoint.params), query)
        body = _fill(_resolve(endpoint.body), query)
    except EndpointError as exc:
        return Response(error=str(exc))

    if client is None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a dependency
            raise EndpointError("Calling an endpoint needs httpx.") from exc
        client = httpx.Client(timeout=endpoint.timeout_s, follow_redirects=True)

    started = time.monotonic()
    try:
        request: Dict[str, Any] = {"headers": headers}
        if params:
            request["params"] = params
        if endpoint.method.upper() == "GET":
            response = client.get(endpoint.url, **request)
        else:
            response = client.post(endpoint.url, json=body, **request)
        elapsed = round(time.monotonic() - started, 3)
        text = response.text
        if response.status_code >= 400:
            return Response(
                seconds=elapsed,
                status=response.status_code,
                characters=len(text),
                error=f"HTTP {response.status_code}: {text[:200]}",
            )
        try:
            payload = response.json()
        except ValueError:
            return Response(
                seconds=elapsed,
                status=response.status_code,
                characters=len(text),
                error="The response was not JSON.",
            )
        return Response(
            hits=parse_hits(payload, endpoint),
            seconds=elapsed,
            status=response.status_code,
            characters=len(text),
        )
    except Exception as exc:  # timeouts, DNS, resets - recorded, not scored
        return Response(
            seconds=round(time.monotonic() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "MAX_HITS",
    "MAX_SNIPPET_CHARS",
    "EndpointError",
    "Hit",
    "Response",
    "call",
    "dig",
    "parse_hits",
]

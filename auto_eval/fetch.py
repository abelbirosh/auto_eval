"""Guarded HTTP access for the ground-truth extractor.

Everything fetched here comes from a URL a *model* proposed, so this module is
written defensively: only http(s), never a private or loopback address, a hard
byte cap, a short timeout, and no cookies or credentials of any kind. What comes
back is untrusted data and is labelled as such everywhere it is used.

Two shapes of fetch, because public ground truth lives in two shapes:

* `fetch_text` - any page, reduced to plain text so quotes can be checked
  against what the page actually says.
* `fetch_rows` - labelled examples straight out of the Hugging Face datasets
  server, which serves real rows of public datasets over an open API. Rows
  fetched this way are data, not model output; nothing has to be believed.
"""

from __future__ import annotations

import html
import ipaddress
import re
from typing import Any, List, Optional
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, Field

# A page only has to be big enough to hold its numbers; anything larger is a
# download we have no business pulling into a prompt.
MAX_BYTES = 400_000
TIMEOUT_SECONDS = 20.0

HF_ROWS_API = "https://datasets-server.huggingface.co"
# Rows are free to fetch but not free to read: this many is enough to see the
# shape of a dataset and seed an eval.
DEFAULT_ROW_LIMIT = 20

_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class Fetched(BaseModel):
    """The result of trying to read a URL. Never raises; failure is a value."""

    url: str
    ok: bool
    status: Optional[int] = None
    text: str = ""
    error: Optional[str] = None


class Rows(BaseModel):
    """Real rows out of a public dataset."""

    dataset: str
    config: Optional[str] = None
    split: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[dict] = Field(default_factory=list)
    total: Optional[int] = None
    ok: bool = True
    error: Optional[str] = None


def _refuse(url: str, why: str) -> Fetched:
    return Fetched(url=url, ok=False, error=why)


def _is_private(host: str) -> bool:
    """Block the obvious ways a URL can point back at the machine we run on."""
    host = host.strip("[]").lower()
    if host in ("localhost", "") or host.endswith(".local") or "." not in host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False  # a normal hostname; DNS is not re-checked here
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


def html_to_text(body: str) -> str:
    """Strip markup so a quote can be checked against what a reader would see."""
    body = _SCRIPT_OR_STYLE.sub(" ", body)
    body = _TAG.sub(" ", body)
    body = html.unescape(body)
    body = _WHITESPACE.sub(" ", body)
    return _BLANK_LINES.sub("\n\n", body).strip()


def _client(module: Any) -> Any:
    return module.Client(
        timeout=TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"user-agent": "auto-eval/0.1 (+ground-truth extractor)"},
    )


def fetch_text(url: str) -> Fetched:
    """Read a URL as plain text, or say why it could not be read."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        return _refuse(url, "Not an http(s) URL.")
    if _is_private(parts.hostname or ""):
        return _refuse(url, "Refused: the URL points at a private or local address.")

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - depends on install state
        return _refuse(url, f"httpx is needed to fetch sources: {exc}")

    try:
        with _client(httpx) as client:
            with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    return Fetched(
                        url=url,
                        ok=False,
                        status=response.status_code,
                        error=f"HTTP {response.status_code}.",
                    )
                chunks: List[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= MAX_BYTES:
                        break
                body = b"".join(chunks)[:MAX_BYTES].decode("utf-8", errors="replace")
                return Fetched(
                    url=url,
                    ok=True,
                    status=response.status_code,
                    text=html_to_text(body),
                )
    except Exception as exc:  # network, TLS, redirect loops - all the same to us
        return _refuse(url, f"Could not fetch: {exc}")


def hf_dataset_id(url: str) -> Optional[str]:
    """The dataset id in a Hugging Face URL, if that is what this URL is."""
    parts = urlsplit(url.strip())
    if parts.hostname not in ("huggingface.co", "www.huggingface.co"):
        return None
    segments = [s for s in parts.path.split("/") if s]
    if not segments or segments[0] != "datasets":
        return None
    owner_and_name = segments[1:3]
    if not owner_and_name:
        return None
    return "/".join(owner_and_name)


def _get_json(url: str) -> Any:
    import httpx

    with _client(httpx) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def fetch_rows(dataset: str, limit: int = DEFAULT_ROW_LIMIT) -> Rows:
    """Real labelled rows from the Hugging Face datasets server.

    Public datasets only - a gated or private one comes back as a failed `Rows`
    with the server's reason, which is exactly what the report should say.
    """
    encoded = quote(dataset, safe="/")
    try:
        splits = _get_json(f"{HF_ROWS_API}/splits?dataset={encoded}")
    except Exception as exc:
        return Rows(dataset=dataset, ok=False, error=f"Could not list splits: {exc}")

    available = splits.get("splits") or []
    if not available:
        return Rows(dataset=dataset, ok=False, error="The dataset server lists no splits.")

    # Prefer a held-out split: training rows are the ones a model most likely
    # already saw, which makes them the worst choice for an eval.
    preferred = ("test", "validation", "dev", "eval")
    chosen = next(
        (s for name in preferred for s in available if s.get("split") == name),
        available[0],
    )
    config, split = chosen.get("config"), chosen.get("split")

    try:
        payload = _get_json(
            f"{HF_ROWS_API}/rows?dataset={encoded}"
            f"&config={quote(str(config))}&split={quote(str(split))}"
            f"&offset=0&length={int(limit)}"
        )
    except Exception as exc:
        return Rows(
            dataset=dataset,
            config=config,
            split=split,
            ok=False,
            error=f"Could not read rows: {exc}",
        )

    return Rows(
        dataset=dataset,
        config=config,
        split=split,
        columns=[f.get("name") for f in payload.get("features", []) if f.get("name")],
        rows=[r.get("row", {}) for r in payload.get("rows", [])],
        total=payload.get("num_rows_total"),
    )


__all__ = [
    "DEFAULT_ROW_LIMIT",
    "Fetched",
    "MAX_BYTES",
    "Rows",
    "fetch_rows",
    "fetch_text",
    "hf_dataset_id",
    "html_to_text",
]

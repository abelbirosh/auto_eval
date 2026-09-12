"""Guarded, read-only access to the resources the identifier found.

Nothing here downloads a dataset. The block's job is to hand the next step a
link and an analysis of what is behind it, so this module reads *about* a
resource - does the link resolve, what columns and splits does the dataset
have, how big is it, what licence - and never pulls the data itself.

Everything fetched comes from a URL a *model* proposed, so the rules are:
http(s) only, no private or loopback address at any hop, every redirect
re-checked, a hard byte cap, a short timeout, and no cookies or credentials of
any kind. Failure is a value, never an exception: a source that cannot be read
becomes a line in the report rather than the end of the run.
"""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from typing import Any, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlsplit

from pydantic import BaseModel, Field

# A page only has to be big enough to describe itself; more than this is a
# download, which is not this block's business.
MAX_BYTES = 400_000
TIMEOUT_SECONDS = 20.0
MAX_REDIRECTS = 4

HF_DATASETS_API = "https://datasets-server.huggingface.co"
HF_HUB_API = "https://huggingface.co/api/datasets"

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
    final_url: Optional[str] = Field(
        default=None, description="Where the redirects ended up, if they moved."
    )


class Split(BaseModel):
    name: str
    rows: Optional[int] = None
    bytes: Optional[int] = None


class DatasetMeta(BaseModel):
    """What a dataset is, read from its metadata. No rows are fetched."""

    dataset: str
    url: str
    ok: bool = True
    error: Optional[str] = None
    configs: List[str] = Field(default_factory=list)
    config: Optional[str] = Field(default=None, description="The config described below.")
    columns: List[str] = Field(default_factory=list)
    splits: List[Split] = Field(default_factory=list)
    licence: Optional[str] = None
    gated: Optional[bool] = None
    private: Optional[bool] = None
    downloads: Optional[int] = None
    last_modified: Optional[str] = None
    homepage: Optional[str] = None
    description: str = ""
    download_bytes: Optional[int] = None


# --------------------------------------------------------------------------
# The guards
# --------------------------------------------------------------------------


def _is_private_literal(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in ("localhost", "") or host.endswith(".local") or "." not in host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return _is_private_address(address)


def _is_private_address(address: Any) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def check_url(url: str) -> Tuple[bool, str]:
    """Is this URL one we are willing to open? Returns (allowed, why not).

    The hostname is resolved and every address it answers with is checked, so a
    public name pointing at 127.0.0.1 or at a cloud metadata endpoint is refused
    rather than fetched from inside whatever network this runs on. DNS is
    re-checked on every redirect hop; a name that changes its answer between
    this check and the connection itself would still get through, which is the
    one hole left and would need a pinned-IP connection to close.
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        return False, "Not an http(s) URL."
    host = parts.hostname or ""
    if _is_private_literal(host):
        return False, f"Refused: {host or 'that URL'} is a private or local address."

    try:
        answers = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as exc:
        return False, f"Could not resolve {host}: {exc}"

    for answer in answers:
        literal = answer[4][0]
        try:
            address = ipaddress.ip_address(literal)
        except ValueError:
            continue
        if _is_private_address(address):
            return False, f"Refused: {host} resolves to the private address {literal}."
    return True, ""


def html_to_text(body: str) -> str:
    """Strip markup so a quote can be checked against what a reader would see."""
    body = _SCRIPT_OR_STYLE.sub(" ", body)
    body = _TAG.sub(" ", body)
    body = html.unescape(body)
    body = _WHITESPACE.sub(" ", body)
    return _BLANK_LINES.sub("\n\n", body).strip()


def _client(module: Any) -> Any:
    # Redirects are followed by hand below so each hop can be re-checked; a
    # client that followed them itself would jump straight past `check_url`.
    return module.Client(
        timeout=TIMEOUT_SECONDS,
        follow_redirects=False,
        headers={"user-agent": "auto-eval/0.1 (+ground-truth identifier)"},
    )


def _get(url: str, *, as_json: bool) -> Any:
    """One guarded GET, following redirects manually. Raises on failure."""
    import httpx

    seen = []
    with _client(httpx) as client:
        for _ in range(MAX_REDIRECTS + 1):
            allowed, why = check_url(url)
            if not allowed:
                raise PermissionError(why)
            seen.append(url)

            if as_json:
                response = client.get(url)
            else:
                response = client.send(
                    client.build_request("GET", url), stream=True
                )

            if response.is_redirect:
                location = response.headers.get("location", "")
                if not as_json:
                    response.close()
                if not location:
                    raise RuntimeError(f"HTTP {response.status_code} with no destination.")
                url = urljoin(url, location)
                continue

            if response.status_code >= 400:
                if not as_json:
                    response.close()
                raise RuntimeError(f"HTTP {response.status_code}.")

            if as_json:
                return response.json(), url

            try:
                chunks: List[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= MAX_BYTES:
                        break
            finally:
                response.close()
            body = b"".join(chunks)[:MAX_BYTES].decode("utf-8", errors="replace")
            return (body, response.status_code), url

    raise RuntimeError(f"Too many redirects ({' -> '.join(seen)}).")


# --------------------------------------------------------------------------
# Reading a page
# --------------------------------------------------------------------------


def fetch_text(url: str) -> Fetched:
    """Read a page as text so it can be described. Not a download."""
    allowed, why = check_url(url)
    if not allowed:
        return Fetched(url=url, ok=False, error=why)

    try:
        (body, status), final = _get(url, as_json=False)
    except ImportError as exc:  # pragma: no cover - depends on install state
        return Fetched(url=url, ok=False, error=f"httpx is needed to read sources: {exc}")
    except PermissionError as exc:
        return Fetched(url=url, ok=False, error=str(exc))
    except Exception as exc:  # network, TLS, redirect loops - all the same to us
        return Fetched(url=url, ok=False, error=f"Could not read: {exc}")

    return Fetched(
        url=url,
        ok=True,
        status=status,
        text=html_to_text(body),
        final_url=final if final != url else None,
    )


# --------------------------------------------------------------------------
# Reading about a dataset
# --------------------------------------------------------------------------


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


def _json(url: str) -> Any:
    payload, _ = _get(url, as_json=True)
    return payload


def fetch_dataset_meta(dataset: str, url: str = "") -> DatasetMeta:
    """Describe a public dataset from its metadata. No rows are requested.

    Both endpoints used here report *about* the dataset - its columns, its split
    sizes, its licence - so this says what a later download would get without
    getting any of it.
    """
    encoded = quote(dataset, safe="/")
    meta = DatasetMeta(dataset=dataset, url=url or f"https://huggingface.co/datasets/{dataset}")

    try:
        info = _json(f"{HF_DATASETS_API}/info?dataset={encoded}")
    except Exception as exc:
        return meta.model_copy(
            update={"ok": False, "error": f"No dataset metadata available: {exc}"}
        )

    configs = info.get("dataset_info") or {}
    if not configs:
        return meta.model_copy(
            update={"ok": False, "error": "The dataset server describes no configs."}
        )

    name = next(iter(configs))
    config = configs.get(name) or {}
    splits = [
        Split(name=key, rows=value.get("num_examples"), bytes=value.get("num_bytes"))
        for key, value in (config.get("splits") or {}).items()
    ]

    meta = meta.model_copy(
        update={
            "configs": list(configs),
            "config": name,
            "columns": list(config.get("features") or {}),
            "splits": splits,
            "licence": config.get("license") or None,
            "homepage": config.get("homepage") or None,
            "description": (config.get("description") or "").strip()[:2000],
            "download_bytes": config.get("download_size"),
        }
    )

    # The hub card carries what the data server does not: gating and licence.
    try:
        card = _json(f"{HF_HUB_API}/{encoded}")
    except Exception:
        return meta  # metadata is better than nothing; gating stays unknown

    return meta.model_copy(
        update={
            "gated": bool(card.get("gated")),
            "private": bool(card.get("private")),
            "downloads": card.get("downloads"),
            "last_modified": card.get("lastModified"),
            "licence": meta.licence or (card.get("cardData") or {}).get("license"),
        }
    )


__all__ = [
    "DatasetMeta",
    "Fetched",
    "MAX_BYTES",
    "MAX_REDIRECTS",
    "Split",
    "check_url",
    "fetch_dataset_meta",
    "fetch_text",
    "hf_dataset_id",
    "html_to_text",
]

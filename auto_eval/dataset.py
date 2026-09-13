"""The items a comparison board is scored on.

A board answers one question - *which of these systems is better at this job* -
and it can only answer it against items whose correct answer is already known.
So a dataset here is deliberately narrow: a list of questions, the answers that
count as right, and where each came from.

Two fields carry more weight than they look:

* `published` - when the fact became true. A board over facts that predate the
  model's training cutoff measures memory as much as retrieval, which is why
  `auto_eval.contamination` reads this field and why the board reports a
  model-only baseline next to every other row.
* `answers` - *every* wording that counts. A scorer that only accepts one
  spelling of a right answer manufactures a difference between systems that
  both found it.

Nothing here is generated. An item without a known answer is not an item.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .classifier import ClassifierError

# A board of more than this is a batch job, not something a local harness should
# start without being asked twice.
MAX_ITEMS = 5000


class DatasetError(ClassifierError):
    """Raised when a dataset cannot be read or is not usable as one."""


class Item(BaseModel):
    """One question, and what counts as answering it."""

    id: str
    query: str = Field(description="What is asked, or sent to the endpoint, verbatim.")
    answers: List[str] = Field(
        default_factory=list,
        description="Every wording that counts as correct. Empty means judged, not matched.",
    )
    url: Optional[str] = Field(
        default=None,
        description="A source that states the answer, for retrieval tasks that score the citation.",
    )
    published: Optional[date] = Field(
        default=None,
        description="When the fact became public. Drives the contamination check.",
    )
    note: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def judged(self) -> bool:
        """An item with no gold answer has to be settled by a judge."""
        return not self.answers


class Dataset(BaseModel):
    """The items, and where they came from."""

    name: str = "items"
    task: str = Field(
        default="factual lookup",
        description="The job being measured, in the words the board will use.",
    )
    description: str = ""
    items: List[Item] = Field(default_factory=list)
    source: Optional[str] = Field(default=None, description="URL or path it came from.")
    released: Optional[date] = Field(
        default=None, description="When the dataset was published, if it was."
    )
    licence: Optional[str] = None
    digest: str = Field(default="", description="Hash of the items, filled on load.")

    @property
    def judged_share(self) -> float:
        if not self.items:
            return 0.0
        return sum(1 for item in self.items if item.judged) / len(self.items)

    @property
    def earliest(self) -> Optional[date]:
        dates = [item.published for item in self.items if item.published]
        return min(dates) if dates else None

    @property
    def latest(self) -> Optional[date]:
        dates = [item.published for item in self.items if item.published]
        return max(dates) if dates else None

    def sample(self, limit: Optional[int]) -> List[Item]:
        """The first `limit` items - deliberately not a random sample.

        A board that quietly scores a different subset each time cannot be
        compared with the one before it.
        """
        if limit is None:
            return list(self.items)
        return self.items[: max(0, limit)]


def _digest(items: List[Item]) -> str:
    payload = "\n".join(item.model_dump_json() for item in items)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _item_from(raw: Dict[str, Any], index: int) -> Item:
    """One row, from whatever the source called its columns.

    Datasets in the wild spell these differently; the alternatives accepted here
    are the common ones, and anything else has to be renamed before loading -
    silently guessing which column holds the answer is how a board ends up
    scoring the wrong thing.
    """
    query = (
        raw.get("query") or raw.get("question") or raw.get("input") or raw.get("prompt")
    )
    if not query:
        raise DatasetError(
            f"Item {index} has no question: expected a 'query', 'question', 'input' or "
            f"'prompt' field, found {sorted(raw)}."
        )

    answers = raw.get("answers")
    if answers is None:
        single = raw.get("answer") or raw.get("gold") or raw.get("expected")
        answers = [single] if single else []
    if isinstance(answers, str):
        answers = [answers]

    published = raw.get("published") or raw.get("date") or raw.get("as_of")
    return Item(
        id=str(raw.get("id") or index),
        query=str(query),
        answers=[str(a) for a in answers if str(a).strip()],
        url=raw.get("url") or raw.get("source_url"),
        published=date.fromisoformat(str(published)[:10]) if published else None,
        note=raw.get("note"),
        meta={
            k: v
            for k, v in raw.items()
            if k
            not in {
                "id",
                "query",
                "question",
                "input",
                "prompt",
                "answers",
                "answer",
                "gold",
                "expected",
                "url",
                "source_url",
                "published",
                "date",
                "as_of",
                "note",
            }
        },
    )


def from_rows(rows: List[Dict[str, Any]], **fields: Any) -> Dataset:
    """Build a dataset from already-parsed rows."""
    if not rows:
        raise DatasetError("No items: a board cannot be scored on an empty dataset.")
    if len(rows) > MAX_ITEMS:
        raise DatasetError(
            f"{len(rows):,} items is over the {MAX_ITEMS:,} limit for one board. "
            "Split it, or pass --limit to score a prefix."
        )
    items = [_item_from(raw, i) for i, raw in enumerate(rows)]
    dataset = Dataset(items=items, **fields)
    return dataset.model_copy(update={"digest": _digest(items)})


def parse(text: str, **fields: Any) -> Dataset:
    """Read JSONL, a JSON array, or a JSON object with an `items` list."""
    stripped = text.strip()
    if not stripped:
        raise DatasetError("Nothing to read: the dataset is empty.")

    if stripped[0] == "[" or stripped[0] == "{":
        try:
            parsed = json.loads(stripped)
        except ValueError as exc:
            parsed = None
            if stripped[0] == "[":
                raise DatasetError(f"Not readable JSON: {exc}") from exc
        if isinstance(parsed, list):
            return from_rows(parsed, **fields)
        if isinstance(parsed, dict):
            rows = parsed.get("items") or parsed.get("data") or parsed.get("rows")
            if rows is None:
                # A one-item file is a JSON object and a one-line JSONL file at
                # the same time. If it looks like an item, read it as one.
                if any(
                    key in parsed for key in ("query", "question", "input", "prompt")
                ):
                    return from_rows([parsed], **fields)
                raise DatasetError(
                    "A JSON object dataset needs an 'items', 'data' or 'rows' list, "
                    f"and this one has {sorted(parsed)}."
                )
            merged = {
                key: parsed[key]
                for key in ("name", "task", "description", "source", "licence")
                if key in parsed and key not in fields
            }
            if "released" in parsed and "released" not in fields and parsed["released"]:
                merged["released"] = date.fromisoformat(str(parsed["released"])[:10])
            return from_rows(rows, **{**merged, **fields})

    rows = []
    for number, line in enumerate(stripped.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise DatasetError(f"Line {number} is not readable JSON: {exc}") from exc
    return from_rows(rows, **fields)


def load(path: Path, **fields: Any) -> Dataset:
    """Read a dataset from a .jsonl or .json file."""
    if not path.is_file():
        raise DatasetError(f"No dataset at {path}.")
    fields.setdefault("name", path.stem)
    fields.setdefault("source", str(path))
    return parse(path.read_text(encoding="utf-8"), **fields)


__all__ = [
    "MAX_ITEMS",
    "Dataset",
    "DatasetError",
    "Item",
    "from_rows",
    "load",
    "parse",
]

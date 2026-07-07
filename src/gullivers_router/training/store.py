"""Append-only JSONL persistence, keyed by record id, for resumable stages.

Each training stage writes its results one JSON object per line as they are produced.
On restart a stage reads back the ids it already completed and only does the remainder,
so a crash costs at most the in-flight item.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def append(path: Path, record: dict) -> None:
    """Append one record as a JSON line, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def read_records(path: Path) -> Iterator[dict]:
    """Yield each record from a JSONL file, or nothing when it does not exist."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def completed_ids(path: Path, key: str = "id") -> set[str]:
    """Return the set of ids already written to ``path``."""
    return {record[key] for record in read_records(path)}


def read_map(path: Path, key: str = "id", value: str = "response") -> dict[str, object]:
    """Return a ``{key: value}`` mapping from a JSONL file, last write winning."""
    return {record[key]: record[value] for record in read_records(path)}


def write_json(path: Path, record: dict) -> None:
    """Write a single JSON object, replacing any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict | None:
    """Read a JSON object written by :func:`write_json`, or None if absent."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

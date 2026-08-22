"""Corpus load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of formula/document records."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("corpus must be a JSON array of objects")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        if "id" not in row:
            row = {**row, "id": f"row-{i}"}
        rows.append(row)
    return rows

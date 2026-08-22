"""CLI rag / eval / audit regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("numpy")
pytest.importorskip("faiss")

from formulagate.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_gate_cli_still_works() -> None:
    code = main(
        [
            "--brief",
            "modular residue congruence divisible number theory",
            "--draft",
            "n equiv 1 (mod 3)",
            "--corpus",
            str(ROOT / "data" / "sample_corpus.json"),
            "--json",
        ]
    )
    assert code == 0


def test_rag_cli_writes_audit_json(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    code = main(
        [
            "rag",
            "--brief",
            "modular residue congruence divisible number theory",
            "--draft",
            "n equiv 1 (mod 3)",
            "--corpus",
            str(ROOT / "data" / "sample_corpus.json"),
            "--embedder",
            "hash",
            "--reranker",
            "formulagate",
            "--audit",
            str(audit),
        ]
    )
    assert code == 0
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["action"] == "generate"
    assert payload["pipeline"].startswith("retrieve")
    assert "mod-congruence" in payload["retrieved_ids"]


def test_eval_cli_golden_floor() -> None:
    code = main(
        [
            "eval",
            "--golden",
            str(ROOT / "data" / "golden_cases.json"),
            "--corpus",
            str(ROOT / "data" / "sample_corpus.json"),
            "--embedder",
            "hash",
            "--reranker",
            "formulagate",
            "--min-n",
            "50",
            "--min-accuracy",
            "0.75",
            "--min-abstain-recall",
            "1.0",
        ]
    )
    assert code == 0

"""CLI: fit a calibration artifact from golden cases and reuse it downstream."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("numpy")
pytest.importorskip("faiss")

from formulagate import semantic_gate
from formulagate.calibration import get_calibration, set_calibration
from formulagate.cli import main

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_cases.json"
CORPUS = ROOT / "data" / "sample_corpus.json"


@pytest.fixture(autouse=True)
def _offline_gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(semantic_gate, "_get_model", lambda: None)
    before = get_calibration()
    yield
    set_calibration(before)


def test_calibrate_cli_writes_artifact(tmp_path: Path) -> None:
    out = tmp_path / "calibration.json"
    code = main(
        [
            "calibrate",
            "--golden", str(GOLDEN),
            "--corpus", str(CORPUS),
            "--embedder", "hash",
            "--out", str(out),
        ]
    )
    assert code == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert 0.0 < payload["params"]["threshold"] < 1.0
    assert payload["params"]["n_samples"] > 0
    assert 0.0 <= payload["report"]["expected_calibration_error"] <= 1.0
    assert 0.0 <= payload["report"]["brier_score"] <= 1.0
    assert payload["report"]["n_samples"] == payload["params"]["n_samples"]


def test_rag_cli_accepts_calibration_artifact(tmp_path: Path) -> None:
    out = tmp_path / "calibration.json"
    assert main(
        [
            "calibrate",
            "--golden", str(GOLDEN),
            "--corpus", str(CORPUS),
            "--embedder", "hash",
            "--out", str(out),
        ]
    ) == 0

    code = main(
        [
            "rag",
            "--brief", "modular residue congruence divisible number theory",
            "--draft", "n equiv 1 (mod 3)",
            "--corpus", str(CORPUS),
            "--embedder", "hash",
            "--reranker", "formulagate",
            "--calibration", str(out),
            "--json",
        ]
    )
    assert code == 0
    assert get_calibration().n_samples > 0

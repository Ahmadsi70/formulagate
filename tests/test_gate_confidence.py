"""Calibrated confidence must flow gate → RAG decision → audit payload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from formulagate import semantic_gate
from formulagate.audit import build_rag_audit
from formulagate.calibration import CalibrationParams, get_calibration, set_calibration
from formulagate.gate import run_gate
from formulagate.rag import LexicalStubRetriever, run_scientific_rag
from formulagate.semantic_gate import evaluate_semantic_gate

_CORPUS = [
    {
        "id": "mod-congruence",
        "english": (
            "equitable measure number modular residue congruence "
            "divisible arithmetic integers"
        ),
        "math_formula": r"a \equiv b \pmod{n}",
        "scientific_domain": "Number Theory",
    },
]

_BRIEF = "modular residue congruence divisible number theory integers"
_DRAFT = "n equiv 1 (mod 3); congruence classes cover the integers"


@pytest.fixture(autouse=True)
def _no_transformer(monkeypatch: pytest.MonkeyPatch):
    """Force the lexical fallback path: fast, offline, deterministic."""

    monkeypatch.setattr(semantic_gate, "_get_model", lambda: None)
    before = get_calibration()
    yield
    set_calibration(before)


def _corpus_file(tmp_path: Path) -> Path:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(_CORPUS), encoding="utf-8")
    return path


# ─── Semantic gate honours the calibrated threshold ───────────────────────────


def test_semantic_gate_uses_threshold_from_calibration() -> None:
    strict = CalibrationParams(a=8.0, b=-2.8, threshold=0.99)
    permissive = CalibrationParams(a=8.0, b=-2.8, threshold=0.01)

    rejected = evaluate_semantic_gate(
        brief=_BRIEF, draft=_DRAFT, candidate_text=r"a \equiv b \pmod{n}",
        lexical_score=8.0, calibration=strict,
    )
    accepted = evaluate_semantic_gate(
        brief=_BRIEF, draft=_DRAFT, candidate_text=r"a \equiv b \pmod{n}",
        lexical_score=8.0, calibration=permissive,
    )

    assert rejected.ok is False and rejected.action == "abstain"
    assert accepted.ok is True and accepted.action == "generate"
    assert rejected.threshold_used == pytest.approx(0.99)
    assert accepted.threshold_used == pytest.approx(0.01)
    assert rejected.confidence == pytest.approx(accepted.confidence)


def test_semantic_gate_confidence_is_a_probability() -> None:
    verdict = evaluate_semantic_gate(
        brief=_BRIEF, draft=_DRAFT, candidate_text=r"a \equiv b \pmod{n}",
        lexical_score=8.0,
    )
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.detail


# ─── Gate / RAG / audit propagation ───────────────────────────────────────────


def test_run_gate_exposes_calibrated_confidence(tmp_path: Path) -> None:
    evaluation = run_gate(
        brief=_BRIEF,
        candidate_text=_DRAFT,
        corpus_path=_corpus_file(tmp_path),
    )
    assert evaluation.ok is True
    assert evaluation.confidence is not None
    assert 0.0 <= evaluation.confidence <= 1.0
    assert evaluation.threshold is not None
    assert evaluation.combined_score is not None


def test_rag_decision_carries_confidence() -> None:
    decision = run_scientific_rag(
        brief=_BRIEF,
        draft=_DRAFT,
        retriever=LexicalStubRetriever(_CORPUS),
        retrieve_k=3,
    )
    assert decision.action == "generate"
    assert decision.gate.confidence is not None
    assert 0.0 <= decision.gate.confidence <= 1.0


def test_audit_payload_reports_confidence() -> None:
    decision = run_scientific_rag(
        brief=_BRIEF,
        draft=_DRAFT,
        retriever=LexicalStubRetriever(_CORPUS),
        retrieve_k=3,
    )
    audit = build_rag_audit(
        brief=_BRIEF,
        draft=_DRAFT,
        decision=decision,
        corpus_path="rag://candidates",
        embedder="hash",
        reranker="none",
    )
    assert 0.0 <= audit["confidence"] <= 1.0
    assert audit["threshold"] is not None
    assert audit["calibration"]["a"] == pytest.approx(get_calibration().a)


def test_global_calibration_shifts_gate_decision(tmp_path: Path) -> None:
    """A stricter loaded calibration must be able to flip accept → abstain."""

    corpus = _corpus_file(tmp_path)
    set_calibration(CalibrationParams(a=8.0, b=-2.8, threshold=0.01))
    permissive = run_gate(
        brief=_BRIEF, candidate_text=_DRAFT, corpus_path=corpus, use_semantic=True
    )
    set_calibration(CalibrationParams(a=8.0, b=-2.8, threshold=0.999))
    strict = run_gate(
        brief=_BRIEF, candidate_text=_DRAFT, corpus_path=corpus, use_semantic=True
    )

    assert permissive.ok is True
    assert strict.ok is False
    assert "confidence" in strict.detail

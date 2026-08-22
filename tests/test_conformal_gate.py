"""Test conformal prediction integration in the gate decision path.

Verifies that ConformalCalibration thresholds are correctly applied in
evaluate_gate_signals and through the Formulagate.check() SDK path.
"""

import json
import tempfile
from pathlib import Path

import pytest

from formulagate.conformal import ConformalCalibration
from formulagate.gate import (
    DiscoveryResult,
    GateSignals,
    LinkageEntry,
    evaluate_gate_signals,
)
from formulagate.domain import Domain

SAMPLE_CORPUS = [
    {"id": "1", "math_formula": "E = m c^2", "english": "energy mass equivalence",
     "scientific_domain": "physics", "title": "Relativity"},
]


def _make_result(score: int = 4, excerpt: str = "E = m c^2", domain: Domain = "physics") -> DiscoveryResult:
    return DiscoveryResult(
        domain=domain,
        corpus_path="<test>",
        entries=[
            LinkageEntry(
                record_id="1",
                score=score,
                math_relevance=3,
                formula_excerpt=excerpt,
                scientific_domain="physics",
                full_candidate_text=excerpt,
            )
        ],
    )


class TestConformalGate:
    def test_conformal_accepts_high_confidence(self):
        result = _make_result(score=9, excerpt="E = m c^2")
        conformal = ConformalCalibration(threshold=0.3, alpha=0.1)
        ok, detail, signals = evaluate_gate_signals(
            result, brief="energy", draft="E = m c^2",
            conformal=conformal, use_physics=False,
        )
        assert ok

    def test_conformal_rejects_low_confidence(self):
        result = _make_result(score=5, excerpt="some formula with physics notation E=mc^2")
        conformal = ConformalCalibration(threshold=0.95, alpha=0.05)
        ok, detail, signals = evaluate_gate_signals(
            result, brief="test", draft="test draft",
            conformal=conformal, use_physics=False,
        )
        # With very high threshold, low confidence should be rejected
        assert "conformal" in detail.lower() or "below" in detail.lower() or not ok

    def test_conformal_with_default_score_passes_threshold(self):
        result = _make_result(score=6, excerpt="E = m c^2")
        conformal = ConformalCalibration(threshold=0.51, alpha=0.1)
        ok, detail, _ = evaluate_gate_signals(
            result, brief="energy", draft="E = m c^2",
            conformal=conformal, use_physics=False,
        )
        assert ok

    def test_conformal_no_confidence_skipped(self):
        result = _make_result(score=0, excerpt="")
        conformal = ConformalCalibration(threshold=0.5, alpha=0.1)
        ok, detail, signals = evaluate_gate_signals(
            result, brief="test", draft="test",
            conformal=conformal, use_physics=False,
        )
        assert not ok  # empty excerpt fails

    def test_accepts_method_matches_threshold(self):
        cal = ConformalCalibration(threshold=0.6, alpha=0.1)
        assert cal.accepts(0.7) is True
        assert cal.accepts(0.5) is False
        assert cal.accepts(0.6) is True

    def test_conformal_sdk_path(self):
        from formulagate.sdk import Formulagate, Source

        conformal = ConformalCalibration(threshold=0.51, alpha=0.1)
        gate = Formulagate(
            sources=[Source(id="1", text="E = m c^2", formula="E = m c^2")],
            use_physics=False,
            conformal=conformal,
        )
        result = gate.check(brief="energy", draft="E = m c^2")
        assert result.action in ("generate", "abstain")
        assert result.detail

    def test_conformal_serialization(self):
        cal = ConformalCalibration(threshold=0.75, alpha=0.05)
        d = cal.to_dict()
        assert d["threshold"] == 0.75
        assert d["alpha"] == 0.05
        assert "n_cal" in d  # conformal calibration stores calibration data
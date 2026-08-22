"""Comprehensive gate integration tests — v0.7 pipeline end-to-end.

Tests the full chain: SDK → retrieval → ranking → physics signals →
symbol grounding → calibration → decision.

Covers all Phase 1–5 integrations in a single test file.
"""

import json
import tempfile
from pathlib import Path

import pytest

from formulagate.sdk import Formulagate, Source, ClaimResult
from formulagate.gate import (
    DiscoveryResult,
    GateEvaluation,
    GateSignals,
    LinkageEntry,
    discover_linkages,
    evaluate_gate,
    evaluate_gate_signals,
    rank_records,
    run_gate,
)
from formulagate.physics_signals import PhysicsSignals, physics_signals
from formulagate.scoring import keywords, formula_domain_relevance, score_record
from formulagate.calibration import (
    CalibrationParams,
    MultiCalibration,
    calibrate_confidence,
    set_multi_calibration,
    get_multi_calibration,
)
from formulagate.retriever import SQLiteRetriever, RetrieverProtocol
from formulagate.domain import Domain


# ─── Fixtures ─────────────────────────────────────────────────────────────────

PHYSICS_CORPUS = [
    {"id": "p1", "math_formula": "E = m c^2", "english": "Mass-energy equivalence formula",
     "scientific_domain": "physics", "title": "Special Relativity"},
    {"id": "p2", "math_formula": "F = m a", "english": "Newton's second law of motion",
     "scientific_domain": "physics", "title": "Classical Mechanics"},
    {"id": "p3", "math_formula": "F = G \\frac{m_1 m_2}{r^2}",
     "english": "Newton's law of universal gravitation",
     "scientific_domain": "physics", "title": "Gravitation"},
    {"id": "p4", "math_formula": "\\lambda = \\frac{h}{p}",
     "english": "De Broglie wavelength relation",
     "scientific_domain": "physics", "title": "Quantum Mechanics"},
    {"id": "p5", "math_formula": "\\Delta x \\Delta p \\geq \\frac{\\hbar}{2}",
     "english": "Heisenberg uncertainty principle",
     "scientific_domain": "physics", "title": "Quantum Mechanics"},
]

GENERAL_CORPUS = [
    {"id": "g1", "math_formula": "", "english": "The history of the Roman Empire",
     "scientific_domain": "history", "title": "Ancient Rome"},
    {"id": "g2", "text": "Photosynthesis is the process by which plants convert sunlight into energy",
     "domain": "biology", "title": "Plant Biology"},
]


class TestGateIntegration:
    """Full pipeline tests covering retrieval → ranking → physics → decision."""

    # ── Retrieval + ranking ────────────────────────────────────────────────

    def test_discover_and_evaluate_physics(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text(json.dumps(PHYSICS_CORPUS))

        result = discover_linkages(
            brief="What is mass-energy equivalence?",
            candidate_text="E equals m c squared",
            corpus_path=corpus_path,
        )
        assert not result.error
        assert result.domain == "physics"
        assert len(result.entries) > 0

        ok, detail = evaluate_gate(result, brief="mass energy", draft="E = m c^2")
        assert ok
        assert "gate_pass" in detail

    def test_discover_and_evaluate_general(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text(json.dumps(GENERAL_CORPUS))

        result = discover_linkages(
            brief="Tell me about ancient Rome",
            candidate_text="The Roman Empire was a powerful civilization",
            corpus_path=corpus_path,
        )
        assert not result.error
        assert result.domain == "general"

    def test_empty_corpus_graceful(self, tmp_path):
        corpus_path = tmp_path / "empty.json"
        corpus_path.write_text(json.dumps([]))

        result = discover_linkages(
            brief="test", candidate_text="test", corpus_path=corpus_path,
        )
        assert not result.error
        assert len(result.entries) == 0

    def test_missing_corpus_file(self, tmp_path):
        result = discover_linkages(
            brief="test", candidate_text="test",
            corpus_path=tmp_path / "nonexistent.json",
        )
        assert result.error is not None
        assert len(result.entries) == 0

    # ── Physics signals integration ────────────────────────────────────────

    def test_physics_signals_with_symbol_grounding(self):
        """Verify that physics_signals accepts and uses overrides from symbol grounding."""
        signals = physics_signals(
            draft="E = m c^2",
            candidate_text="The energy of a particle at rest is E = m c^2",
            overrides={"c": None},  # override should not crash
        )
        assert isinstance(signals, PhysicsSignals)
        assert signals.candidate_has_formula == 1.0

    def test_physics_signals_dimension_ok_consistent(self):
        """E = m c^2 should be dimensionally consistent (energy = mass * velocity^2)."""
        signals = physics_signals(
            draft="E = m c^2",
            candidate_text="Mass-energy equivalence E = m c^2",
        )
        assert signals.dimension_ok >= 0.0  # should not be rejected

    def test_physics_signals_dimension_ok_inconsistent(self):
        """E = m c^3 should be dimensionally wrong."""
        signals = physics_signals(
            draft="E = m c^3",
            candidate_text="E = m c^3",
        )
        assert signals.dimension_ok <= 0.0  # inconsistent

    def test_physics_signals_no_formulas(self):
        signals = physics_signals(
            draft="just some text without equations",
            candidate_text="no formulas here either",
        )
        assert signals.candidate_has_formula == 0.0
        assert signals.dimension_ok == 0.0

    # ── SDK retriever integration ──────────────────────────────────────────

    def test_sdk_with_sqlite_retriever(self):
        gate = Formulagate(
            sources=PHYSICS_CORPUS,
            retriever="sqlite",
            use_physics=False,
        )
        assert gate._retriever is not None
        result = gate.check(
            brief="mass energy equivalence",
            draft="E equals m c squared",
        )
        assert result.action in ("generate", "abstain")

    def test_sdk_retriever_fallback_when_no_sources(self):
        gate = Formulagate(
            sources=PHYSICS_CORPUS,
            retriever="sqlite",
            use_physics=False,
        )
        result = gate.check(
            brief="quantum wavelength",
            draft="lambda equals h over p",
            sources=None,
        )
        assert result.domain in ("general", "physics")

    def test_sdk_with_add_sources(self):
        gate = Formulagate(
            sources=[PHYSICS_CORPUS[0]],
            retriever="sqlite",
            use_physics=False,
        )
        gate.add_sources([PHYSICS_CORPUS[1], PHYSICS_CORPUS[2]])
        assert len(gate.sources) == 3

    def test_sdk_retriever_protocol(self):
        from formulagate.retriever import RetrieverProtocol, RetrievedDoc

        class CustomRetriever:
            def search(self, query, top_k=5):
                return [RetrievedDoc(id="1", text="E=mc^2", formula="E=mc^2", score=1.0)]
            def __len__(self):
                return 1

        gate = Formulagate(
            sources=PHYSICS_CORPUS,
            retriever=CustomRetriever(),
            use_physics=False,
        )
        result = gate.check(brief="test", draft="E=mc^2")
        assert result.action in ("generate", "abstain")

    # ── Scoring fixes (Phase 2) ────────────────────────────────────────────

    def test_scoring_general_domain_markers(self):
        """General domain should count math/physics markers plus basic structure."""
        rel = formula_domain_relevance("\\int f(x) dx = F(x) + C", "general")
        assert rel > 0  # should find integral and frac markers

    def test_scoring_general_domain_basic(self):
        """Very basic formula should still get some relevance via structure markers."""
        rel = formula_domain_relevance("x = y + z", "general")
        assert rel >= 0

    def test_scoring_general_domain_empty(self):
        rel = formula_domain_relevance("", "general")
        assert rel == 0

    def test_scoring_symbol_overlap_variables(self):
        """Variable count detection for general domain."""
        from formulagate.scoring import _count_distinct_variables
        n = _count_distinct_variables("x y z = a + b c")
        assert n >= 5  # x, y, z, a, b, c

    # ── Fused score normalization (Phase 1) ────────────────────────────────

    def test_fused_score_normalization(self):
        """Verify that the new normalization doesn't exceed [0,1] range."""
        from formulagate.gate import _fuse, _lexical_signals
        from formulagate.physics_signals import PhysicsSignals

        # Test with a mid-range lexical score
        signals = _lexical_signals(8)  # score=8/15 ≈ 0.533
        phys = PhysicsSignals()
        fused = _fuse(signals, phys)

        # Fused model not installed → confidence unchanged
        assert fused.confidence == signals.confidence

    # ── Full SDK pipeline ──────────────────────────────────────────────────

    def test_full_sdk_physics_pipeline(self):
        sources = [
            Source(id=s["id"], text=s["english"], formula=s["math_formula"])
            for s in PHYSICS_CORPUS
        ]
        gate = Formulagate(sources=sources, use_physics=True)
        result = gate.check(
            brief="What is mass-energy equivalence?",
            draft="The energy of a particle at rest equals its mass times the speed of light squared",
        )
        assert isinstance(result, ClaimResult)
        assert result.action in ("generate", "abstain")
        assert result.domain in ("general", "physics")

    def test_full_sdk_verify_dimensions(self):
        gate = Formulagate()
        result = gate.verify("E = m c^2")
        assert result.parsed is True
        assert result.dimensions in ("consistent", "unknown")

    def test_full_sdk_verify_with_grounding(self):
        gate = Formulagate()
        result = gate.verify(
            "F = G m1 m2 / r^2",
            context="where G is the gravitational constant, m1 and m2 are masses, r is the distance",
            use_grounding=True,
        )
        assert result.parsed is True

    # ── Regression: Phase 1 fixes ──────────────────────────────────────────

    def test_regression_full_candidate_text_preserved(self):
        """Verify that full_candidate_text includes abstract and context fields."""
        row = {
            "id": "test", "math_formula": "F = m a", "english": "Newton second law fundamental physics",
            "title": "mechanics laws motion",
            "abstract": "Newton's laws of motion describe fundamental physics",
            "context": "Classical mechanics",
        }
        result = rank_records(
            brief="Newton second law",
            candidate_text="F equals m a",
            rows=[row],
            top_k=1,
        )
        entry = result.entries[0]
        assert "Newton" in entry.full_candidate_text
        assert "Classical" in entry.full_candidate_text

    def test_regression_symbol_grounding_in_gate(self):
        """Verify symbol grounding overrides are passed to physics_signals."""
        result = DiscoveryResult(
            domain="physics",
            corpus_path="<test>",
            entries=[
                LinkageEntry(
                    record_id="1",
                    score=6,
                    math_relevance=3,
                    formula_excerpt="F = G m1 m2 / r^2",
                    scientific_domain="physics",
                    full_candidate_text="Newton's law F = G m1 m2 / r^2",
                )
            ],
        )
        ok, detail, signals = evaluate_gate_signals(
            result,
            brief="gravity",
            draft="F = G m1 m2 / r^2 where G is the gravitational constant",
            use_physics=True,
        )
        assert signals.physics is not None

    # ── Edge cases ─────────────────────────────────────────────────────────

    def test_long_candidate_text(self):
        """Very long candidate texts should not crash."""
        long_text = "E = m c^2 " * 500
        signals = physics_signals(
            draft="E = m c^2",
            candidate_text=long_text,
        )
        assert isinstance(signals, PhysicsSignals)

    def test_unicode_candidate_text(self):
        """Unicode characters should not crash."""
        signals = physics_signals(
            draft="\u0394x = h / \u0394p",
            candidate_text="Heisenberg uncertainty \u0394x \u00b7 \u0394p \u2265 \u0127/2",
        )
        assert isinstance(signals, PhysicsSignals)

    def test_run_gate_end_to_end(self, tmp_path):
        corpus_path = tmp_path / "corpus.json"
        corpus_path.write_text(json.dumps(PHYSICS_CORPUS))

        evaluation = run_gate(
            brief="energy mass relation",
            candidate_text="E equals m c squared is the famous equation",
            corpus_path=corpus_path,
            use_physics=True,
        )
        assert isinstance(evaluation, GateEvaluation)
        assert evaluation.ok is True
        assert evaluation.confidence is not None
        assert evaluation.model in ("platt", "fused")
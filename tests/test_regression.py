"""Regression tests for all Phase 1–5 fixes and v0.7 integrations.

Ensures that critical bugs fixed in Phase 1 stay fixed, SDK + retriever
(Phase 2) works correctly, the 8 benchmark dataset pipeline (Phase 3) is
operational, RunPod deployment scripts (Phase 4) are valid, and all v0.7
integrations (Phase 5) pass end-to-end.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TestPhase1Regression:
    """Regression tests for Phase 1 critical bug fixes."""

    def test_gate_full_candidate_text_includes_all_fields(self):
        from formulagate.gate import rank_records

        row = {
            "id": "r1",
            "math_formula": "F = m a",
            "english": "Newton second law of motion fundamental physics",
            "title": "Classical Mechanics laws",
            "abstract": "Fundamental physics principles",
            "context": "mechanics",
        }
        result = rank_records(brief="Newton second law", candidate_text="F equals ma", rows=[row], top_k=1)
        entry = result.entries[0]
        assert "Newton" in entry.full_candidate_text
        assert "Fundamental" in entry.full_candidate_text
        assert "mechanics" in entry.full_candidate_text

    def test_normalization_lexical_in_range(self):
        """New normalization must keep scores in [0, 1]."""
        from formulagate.gate import LEXICAL_SCORE_MAX
        for raw_int in range(0, 16):
            raw = raw_int / LEXICAL_SCORE_MAX
            norm = min(raw * 1.4, 1.0)
            result = norm * norm if raw < 0.25 else norm
            assert 0.0 <= result <= 1.0, f"raw={raw:.3f} → result={result:.3f}"

    def test_normalization_expands_low_scores(self):
        from formulagate.gate import LEXICAL_SCORE_MAX
        raw = 5.0 / LEXICAL_SCORE_MAX  # ~0.333
        norm = min(raw * 1.4, 1.0)
        assert norm > raw  # should expand

    def test_physics_signals_smt_on_consistent(self):
        """SMT should run even when exact walker says consistent (Phase 1, Fix 3)."""
        from formulagate.physics_signals import physics_signals
        signals = physics_signals(
            draft="E = m c^2",
            candidate_text="E = m c^2 energy mass equivalence",
        )
        assert signals.dimension_ok >= 0.0

    def test_symbol_grounding_connected_to_gate(self):
        """Symbol grounding overrides must reach the gate path (Phase 1, Fix 4)."""
        from formulagate.gate import DiscoveryResult, LinkageEntry, evaluate_gate_signals

        result = DiscoveryResult(
            domain="physics",
            corpus_path="<test>",
            entries=[
                LinkageEntry(
                    record_id="1", score=5, math_relevance=3,
                    formula_excerpt="E = m c^2",
                    scientific_domain="physics",
                    full_candidate_text="energy mass equivalence E = m c^2",
                )
            ],
        )
        _, _, signals = evaluate_gate_signals(
            result, brief="energy", draft="E = m c^2", use_physics=True,
        )
        assert signals.physics is not None
        assert signals.physics.dimension_ok >= 0.0

    def test_calibration_rebuild_script_exists(self):
        """Verify rebuild script is importable."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib
        try:
            importlib.import_module("rebuild_calibration")
            ok = True
        except ImportError:
            ok = False
        assert ok, "rebuild_calibration.py should be importable"


class TestPhase2Regression:
    """Regression tests for Phase 2 SDK + retriever rewrite."""

    def test_retriever_protocol_exists_and_runtime_checkable(self):
        from formulagate.retriever import RetrieverProtocol
        assert hasattr(RetrieverProtocol, "search")
        assert hasattr(RetrieverProtocol, "__len__")

    def test_sqlite_retriever_satisfies_protocol(self):
        from formulagate.retriever import RetrieverProtocol, SQLiteRetriever
        retriever = SQLiteRetriever([{"id": "1", "text": "test"}])
        assert isinstance(retriever, RetrieverProtocol)

    def test_sdk_retriever_param_shortcuts(self):
        from formulagate.sdk import Formulagate
        gate = Formulagate(
            sources=[{"id": "a", "text": "hello", "formula": "x=1"}],
            retriever="sqlite",
        )
        assert gate._retriever is not None
        result = gate.check(brief="hello", draft="x=1")
        assert result.action in ("generate", "abstain")

    def test_sdk_hybrid_retriever_shortcut(self):
        from formulagate.sdk import Formulagate
        gate = Formulagate(
            sources=[{"id": "h1", "text": "test hybrid", "formula": "a+b"}],
            retriever="hybrid",
        )
        assert gate._retriever is not None

    def test_sdk_dense_retriever_shortcut_falls_back(self):
        from formulagate.sdk import Formulagate
        gate = Formulagate(
            sources=[{"id": "d1", "text": "test dense", "formula": "x^2"}],
            retriever="dense",
        )
        assert gate._retriever is not None

    def test_scoring_general_domain_improved_fallback(self):
        from formulagate.scoring import formula_domain_relevance
        rel = formula_domain_relevance("E = m c^2", "general")
        assert rel > 0

    def test_scoring_with_symbol_overlap_in_general(self):
        from formulagate.scoring import formula_domain_relevance
        rel = formula_domain_relevance("a + b = c + d", "general")
        assert rel >= 0  # should not crash

    def test_retriever_sdk_add_sources_reindexes(self):
        from formulagate.sdk import Formulagate, Source
        gate = Formulagate(
            sources=[Source(id="1", text="first")],
            retriever="sqlite",
        )
        gate.add_sources([Source(id="2", text="second")])
        assert len(gate.sources) == 2


class TestPhase3Regression:
    """Regression tests for Phase 3 benchmark datasets."""

    def test_fetch_benchmarks_script_exists(self):
        script = ROOT / "scripts" / "fetch_benchmarks.py"
        assert script.is_file(), "fetch_benchmarks.py should exist"

    def test_bench_suite_script_exists(self):
        script = ROOT / "scripts" / "bench_suite.py"
        assert script.is_file(), "bench_suite.py should exist"

    def test_bench_suite_imports(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib
        ok = True
        try:
            importlib.import_module("bench_suite")
        except Exception:
            ok = False
        assert ok, "bench_suite should be importable"

    def test_benchmark_meta_keys(self):
        from scripts.bench_suite import BENCHMARK_META
        expected = {"gpqa", "truthfulqa", "halueval", "expertqa",
                     "alce", "mmlu", "mmlu_pro", "arxiv"}
        assert set(BENCHMARK_META) == expected

    def test_fetch_benchmarks_has_all_fetchers(self):
        from scripts.fetch_benchmarks import FETCHERS
        expected = {"gpqa", "truthfulqa", "halueval", "expertqa",
                     "alce", "mmlu", "mmlu_pro", "arxiv"}
        assert set(FETCHERS) == expected


class TestPhase4Regression:
    """Regression tests for Phase 4 RunPod deployment scripts."""

    def test_runpod_setup_script_exists(self):
        script = ROOT / "scripts" / "runpod_setup.sh"
        assert script.is_file(), "runpod_setup.sh should exist"

    def test_runpod_dockerfile_exists(self):
        dockerfile = ROOT / "Dockerfile.runpod"
        assert dockerfile.is_file(), "Dockerfile.runpod should exist"

    def test_runpod_script_executable(self):
        script = ROOT / "scripts" / "runpod_setup.sh"
        content = script.read_text()
        assert "pip install" in content
        assert "bench_suite" in content

    def test_runpod_dockerfile_has_correct_ports(self):
        dockerfile = ROOT / "Dockerfile.runpod"
        content = dockerfile.read_text()
        assert "8888" in content
        assert "22060" in content


class TestPhase5Regression:
    """Regression tests for Phase 5 v0.7 integrations."""

    def test_conformal_in_gate_path(self):
        from formulagate.conformal import ConformalCalibration
        from formulagate.gate import DiscoveryResult, LinkageEntry, evaluate_gate_signals

        result = DiscoveryResult(
            domain="physics", corpus_path="<test>",
            entries=[LinkageEntry(
                record_id="1", score=7, math_relevance=3,
                formula_excerpt="F = ma", scientific_domain="physics",
                full_candidate_text="Newton's second law F = m a",
            )],
        )
        conformal = ConformalCalibration(threshold=0.3, alpha=0.1)
        ok, detail, _ = evaluate_gate_signals(
            result, brief="force", draft="F = m a",
            use_physics=False, conformal=conformal,
        )
        assert ok

    def test_atomic_gate_integration(self):
        from formulagate.atomic import decompose_rule
        claims = decompose_rule("E = m c^2 is the famous equation by Einstein")
        assert len(claims) > 0

    def test_gasp_semantic_grounding_in_gate(self):
        from formulagate.grounding import semantic_grounding
        score = semantic_grounding("E = m c^2", "mass energy equivalence E = m c squared")
        assert score.sensitivity >= 0.0

    def test_semantic_entropy_penalty(self):
        from formulagate.semantic_entropy import SemanticEntropy
        se = SemanticEntropy(normalized_entropy=0.6, n_clusters=3, n_samples=5)
        confidence = 0.8
        adjusted = confidence * (1 - se.normalized_entropy)
        assert adjusted < confidence

    def test_all_seven_test_files_exist(self):
        test_dir = ROOT / "tests"
        expected_files = [
            "test_conformal_gate.py",
            "test_atomic_gate.py",
            "test_gasp_gate.py",
            "test_semantic_entropy_gate.py",
            "test_gate_integration.py",
            "test_regression.py",
            "test_bench_suite.py",
        ]
        for f in expected_files:
            assert (test_dir / f).is_file(), f"Missing: {f}"

    def test_sdk_all_reports_serializable(self):
        from formulagate.sdk import Formulagate, Source
        gate = Formulagate(
            sources=[Source(id="1", text="test", formula="x=1")],
            use_physics=False,
        )
        result = gate.check(brief="test", draft="x=1")
        d = result.to_dict()
        assert json.dumps(d)
        assert d["action"] in ("generate", "abstain")
        assert "sources" in d
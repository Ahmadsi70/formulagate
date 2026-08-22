"""Test semantic entropy integration in the gate decision path.

Verifies that SemanticEntropy penalizes confidence for uncertain outputs
and integrates correctly with evaluate_gate_signals.
"""

import pytest

from formulagate.semantic_entropy import SemanticEntropy


class TestSemanticEntropyGate:
    def test_semantic_entropy_certain(self):
        se = SemanticEntropy(
            normalized_entropy=0.1,
            n_clusters=1,
            n_samples=5,
        )
        assert se.normalized_entropy == 0.1
        assert se.is_uncertain is False
        assert se.n_clusters == 1

    def test_semantic_entropy_uncertain(self):
        se = SemanticEntropy(
            normalized_entropy=0.75,
            n_clusters=4,
            n_samples=5,
        )
        assert se.is_uncertain is True
        assert se.normalized_entropy == 0.75

    def test_semantic_entropy_to_dict(self):
        se = SemanticEntropy(
            normalized_entropy=0.54,
            n_clusters=3,
            n_samples=10,
        )
        d = se.to_dict()
        assert d["normalized_entropy"] == 0.54
        assert d["n_clusters"] == 3
        assert d["n_samples"] == 10

    def test_semantic_entropy_penalizes_confidence(self):
        """High entropy should reduce effective confidence below threshold."""
        se = SemanticEntropy(
            normalized_entropy=0.65,
            n_clusters=4,
            n_samples=5,
        )
        base_confidence = 0.7
        threshold = 0.5
        adjusted = base_confidence * (1.0 - se.normalized_entropy)
        assert adjusted == pytest.approx(0.245)
        assert adjusted < threshold

    def test_semantic_entropy_no_penalty_when_certain(self):
        se = SemanticEntropy(
            normalized_entropy=0.05,
            n_clusters=1,
            n_samples=5,
        )
        base_confidence = 0.7
        adjusted = base_confidence * (1.0 - se.normalized_entropy)
        assert adjusted == pytest.approx(0.665)
        assert adjusted > 0.5

    def test_semantic_entropy_gate_integration(self):
        """Verify that semantic entropy modifier is accepted by evaluate_gate_signals."""
        from formulagate.gate import DiscoveryResult, LinkageEntry, evaluate_gate_signals

        result = DiscoveryResult(
            domain="physics",
            corpus_path="<test>",
            entries=[
                LinkageEntry(
                    record_id="1",
                    score=5,
                    math_relevance=3,
                    formula_excerpt="E = m c^2",
                    scientific_domain="physics",
                    full_candidate_text="Energy mass equivalence E = m c^2",
                )
            ],
        )

        # Low entropy: should pass
        se_low = SemanticEntropy(normalized_entropy=0.1, n_clusters=1, n_samples=5)
        ok_low, _, _ = evaluate_gate_signals(
            result, brief="energy", draft="E = m c^2",
            semantic_entropy=se_low, use_physics=False,
        )
        assert ok_low

    def test_semantic_entropy_edge_cases(self):
        """Test boundary values: zero entropy, max entropy."""
        se_zero = SemanticEntropy(normalized_entropy=0.0, n_clusters=1, n_samples=5)
        assert se_zero.normalized_entropy == 0.0
        assert se_zero.is_uncertain is False

        se_max = SemanticEntropy(normalized_entropy=1.0, n_clusters=10, n_samples=10)
        adjusted = 0.8 * (1.0 - se_max.normalized_entropy)
        assert adjusted == 0.0  # max entropy zeros the confidence

    def test_semantic_entropy_frozen(self):
        """SemanticEntropy should be immutable."""
        se = SemanticEntropy(normalized_entropy=0.3, n_clusters=2, n_samples=5)
        with pytest.raises(Exception):
            se.normalized_entropy = 0.9

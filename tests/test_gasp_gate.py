"""Test GASP grounding integration in the gate path.

Verifies that grounding sensitivity flows from semantic_grounding through
physics_signals into the gate's fused confidence.
"""

import pytest

from formulagate.grounding import (
    GroundingScore,
    check_grounding,
    semantic_grounding,
    token_level_grounding,
    _jaccard_similarity,
    _overlap_coefficient,
    _jensen_shannon_divergence,
)


class TestGASPGate:
    def test_semantic_grounding_strong(self):
        draft = "The speed of light in vacuum is 299,792,458 meters per second"
        evidence = "The speed of light c is exactly 299,792,458 m/s"
        score = semantic_grounding(draft, evidence)
        assert score.sensitivity > 0.0
        assert score.sim_with_evidence > 0.0
        assert score.from_logprobs is False

    def test_semantic_grounding_none(self):
        draft = "Completely unrelated text about cats"
        evidence = "The Riemann zeta function has a pole at s=1"
        score = semantic_grounding(draft, evidence)
        assert score.sensitivity >= 0.0
        assert not score.is_grounded

    def test_semantic_grounding_no_evidence(self):
        score = semantic_grounding("some draft", "")
        assert score.sensitivity == 0.0

    def test_grounding_score_properties(self):
        score = GroundingScore(
            sensitivity=0.8,
            sim_with_evidence=0.9,
            sim_without_evidence=0.1,
        )
        assert score.is_grounded is True
        d = score.to_dict()
        assert d["sensitivity"] == 0.8
        assert d["sim_with_evidence"] == 0.9

    def test_grounding_score_not_grounded(self):
        score = GroundingScore(sensitivity=0.05)
        assert score.is_grounded is False

    def test_check_grounding_semantic_fallback(self):
        score = check_grounding(
            draft="E = m c^2",
            evidence="The energy-mass equivalence is E = m c^2",
        )
        assert score.sensitivity >= 0.0
        assert score.from_logprobs is False

    def test_check_grounding_with_logprobs(self):
        lp_with = [{"hello": -0.5, "world": -0.8}]
        lp_without = [{"hello": -1.2, "foo": -0.3}]
        score = check_grounding(
            draft="hello world",
            evidence="hello world",
            logprobs_with=lp_with,
            logprobs_without=lp_without,
        )
        assert score.from_logprobs is True
        assert score.sensitivity >= 0.0

    def test_jaccard_similarity(self):
        assert _jaccard_similarity("a b c", "a b c") == 1.0
        assert _jaccard_similarity("a b c", "d e f") == 0.0
        assert _jaccard_similarity("", "test") == 0.0

    def test_overlap_coefficient(self):
        assert _overlap_coefficient("a b", "a b c d") == 1.0
        assert _overlap_coefficient("a b c", "d e f") == 0.0
        assert _overlap_coefficient("", "test") == 0.0

    def test_js_divergence(self):
        p = {"a": 0.5, "b": 0.5}
        q = {"a": 0.5, "b": 0.5}
        js = _jensen_shannon_divergence(p, q)
        assert js == pytest.approx(0.0, abs=0.01)
        import math
        assert js <= math.log(2)

    def test_token_level_grounding_empty(self):
        score = token_level_grounding(None, None)
        assert score.from_logprobs is True
        assert score.sensitivity == 0.0

    def test_grounding_in_physics_signals(self):
        from formulagate.physics_signals import physics_signals
        signals = physics_signals(
            draft="E = m c^2",
            candidate_text="The energy-mass equivalence is E = m c squared",
        )
        assert signals.grounding_sensitivity >= 0.0
        assert isinstance(signals.is_hard_reject, bool)

    def test_gasp_through_gate_path(self):
        from formulagate.gate import evaluate_gate_signals
        from formulagate.gate import DiscoveryResult, LinkageEntry

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
                    full_candidate_text="energy mass equivalence E = m c^2",
                )
            ],
        )
        ok, detail, signals = evaluate_gate_signals(
            result, brief="mass energy", draft="E = m c^2",
            use_physics=True,
        )
        assert signals.physics is not None
        assert signals.physics.grounding_sensitivity >= 0.0
"""Tests for grounding sensitivity (GASP-style)."""

from __future__ import annotations

import math

import pytest

from formulagate.grounding import (
    GroundingScore,
    _jaccard_similarity,
    _jensen_shannon_divergence,
    _overlap_coefficient,
    _top_k_probs,
    check_grounding,
    semantic_grounding,
    token_level_grounding,
)


def test_jaccard_empty() -> None:
    assert _jaccard_similarity("", "") == 0.0
    assert _jaccard_similarity("hello", "") == 0.0


def test_jaccard_identical() -> None:
    assert _jaccard_similarity("hello world", "hello world") == 1.0


def test_jaccard_partial() -> None:
    sim = _jaccard_similarity("hello world", "hello there")
    assert 0.0 < sim < 1.0


def test_overlap_empty() -> None:
    assert _overlap_coefficient("", "") == 0.0


def test_overlap_subset() -> None:
    # Draft is a subset of evidence → overlap = 1.0
    sim = _overlap_coefficient("energy mass", "energy mass equivalence relativity")
    assert sim == 1.0


def test_overlap_partial() -> None:
    sim = _overlap_coefficient("energy mass light", "energy mass gravity")
    assert 0.0 < sim < 1.0


def test_semantic_grounding_strong() -> None:
    # Draft contains terms that appear in evidence → high sensitivity.
    score = semantic_grounding(
        draft="energy equals mass times speed of light squared",
        evidence="energy mass equivalence E equals m c squared special relativity",
    )
    assert score.sensitivity > 0.0
    assert score.sim_with_evidence > 0.0
    assert score.from_logprobs is False


def test_semantic_grounding_weak() -> None:
    # Draft has no overlap with evidence → zero sensitivity.
    score = semantic_grounding(
        draft="the sky is blue",
        evidence="energy mass equivalence special relativity",
    )
    assert score.sensitivity == 0.0
    assert score.from_logprobs is False


def test_semantic_grounding_empty_evidence() -> None:
    score = semantic_grounding(draft="energy mass", evidence="")
    assert score.sensitivity == 0.0
    assert score.sim_with_evidence == 0.0


def test_semantic_grounding_is_grounded_threshold() -> None:
    score = GroundingScore(sensitivity=0.20)
    assert score.is_grounded is True
    score2 = GroundingScore(sensitivity=0.05)
    assert score2.is_grounded is False


def test_top_k_probs_renormalizes() -> None:
    logprobs = {"a": math.log(0.5), "b": math.log(0.3), "c": math.log(0.2)}
    probs = _top_k_probs(logprobs, k=2)
    assert len(probs) == 2
    assert "a" in probs
    assert pytest.approx(sum(probs.values())) == 1.0


def test_js_divergence_identical() -> None:
    p = {"a": 0.5, "b": 0.5}
    q = {"a": 0.5, "b": 0.5}
    js = _jensen_shannon_divergence(p, q)
    assert js == pytest.approx(0.0)


def test_js_divergence_maximally_different() -> None:
    p = {"a": 1.0}
    q = {"b": 1.0}
    js = _jensen_shannon_divergence(p, q)
    # JS divergence max is ln(2) ≈ 0.693
    assert js > 0.5


def test_js_divergence_partial_overlap() -> None:
    p = {"a": 0.7, "b": 0.3}
    q = {"a": 0.3, "b": 0.7}
    js = _jensen_shannon_divergence(p, q)
    assert 0.0 < js < math.log(2)


def test_token_level_grounding_empty() -> None:
    score = token_level_grounding(None, None)
    assert score.from_logprobs is True
    assert score.sensitivity == 0.0


def test_token_level_grounding_identical() -> None:
    logprobs = [{"a": math.log(0.9), "b": math.log(0.1)}]
    score = token_level_grounding(logprobs, logprobs)
    assert score.from_logprobs is True
    assert score.sensitivity == pytest.approx(0.0, abs=0.01)


def test_token_level_grounding_different() -> None:
    with_evidence = [{"a": math.log(0.9), "b": math.log(0.1)}]
    without_evidence = [{"x": math.log(0.9), "y": math.log(0.1)}]
    score = token_level_grounding(with_evidence, without_evidence)
    assert score.from_logprobs is True
    assert score.sensitivity > 0.0


def test_check_grounding_prefers_logprobs() -> None:
    logprobs = [{"a": math.log(0.9)}]
    score = check_grounding(
        draft="test",
        evidence="test",
        logprobs_with=logprobs,
        logprobs_without=logprobs,
    )
    assert score.from_logprobs is True


def test_check_grounding_falls_back_to_semantic() -> None:
    score = check_grounding(
        draft="energy mass equivalence",
        evidence="energy mass relativity",
    )
    assert score.from_logprobs is False
    assert score.sensitivity > 0.0


def test_grounding_score_serialization() -> None:
    score = GroundingScore(sensitivity=0.75, sim_with_evidence=0.8, sim_without_evidence=0.2)
    d = score.to_dict()
    assert d["sensitivity"] == 0.75
    assert d["from_logprobs"] is False


def test_custom_sim_fn() -> None:
    def custom_sim(a: str, b: str) -> float:
        # Extremely simple: 1.0 if they share any word, else 0.0
        return 1.0 if set(a.split()) & set(b.split()) else 0.0

    score = semantic_grounding(
        draft="hello world",
        evidence="hello there",
        sim_fn=custom_sim,
    )
    assert score.sensitivity == 1.0
    assert score.sim_with_evidence == 1.0
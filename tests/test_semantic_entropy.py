"""Tests for semantic entropy computation."""

from __future__ import annotations

import math

import pytest

from formulagate.semantic_entropy import (
    SemanticEntropy,
    _entropy_from_counts,
    _jaccard,
    compute_semantic_entropy,
    jaccard_cluster,
)


def test_jaccard_identical() -> None:
    assert _jaccard("hello world", "hello world") == 1.0


def test_jaccard_no_overlap() -> None:
    assert _jaccard("hello world", "goodbye mars") == 0.0


def test_jaccard_partial() -> None:
    sim = _jaccard("energy mass equivalence", "energy mass relativity")
    assert 0.0 < sim < 1.0


def test_entropy_from_counts_uniform() -> None:
    # 3 clusters of equal size → maximum entropy = log(3)
    h = _entropy_from_counts([2, 2, 2], 6)
    assert h == pytest.approx(math.log(3))


def test_entropy_from_counts_single_cluster() -> None:
    h = _entropy_from_counts([10], 10)
    assert h == 0.0


def test_entropy_from_counts_empty() -> None:
    assert _entropy_from_counts([], 0) == 0.0


def test_jaccard_cluster_identical_texts() -> None:
    clusters = jaccard_cluster(["energy mass", "energy mass", "energy mass"])
    assert len(clusters) == 1
    assert len(list(clusters.values())[0]) == 3


def test_jaccard_cluster_different_texts() -> None:
    clusters = jaccard_cluster(["energy mass", "force acceleration", "light speed"])
    assert len(clusters) == 3


def test_jaccard_cluster_partial_overlap() -> None:
    # "energy mass equivalence" and "energy mass relativity" share "energy mass"
    clusters = jaccard_cluster([
        "energy mass equivalence",
        "energy mass relativity",
        "force acceleration",
    ])
    # First two should cluster together, third separate
    assert len(clusters) == 2


def test_compute_semantic_entropy_single_sample() -> None:
    se = compute_semantic_entropy(["hello"])
    assert se.entropy == 0.0
    assert se.n_samples == 1
    assert se.n_clusters == 1


def test_compute_semantic_entropy_all_same() -> None:
    se = compute_semantic_entropy(["hello world"] * 5)
    assert se.entropy == pytest.approx(0.0)
    assert se.n_clusters == 1
    assert se.normalized_entropy == pytest.approx(0.0)


def test_compute_semantic_entropy_all_different() -> None:
    samples = ["energy mass", "force acceleration", "light speed", "heat temperature"]
    se = compute_semantic_entropy(samples)
    # All different → maximum entropy
    assert se.n_clusters == 4
    assert se.entropy > 0.0
    assert se.normalized_entropy == pytest.approx(1.0)


def test_compute_semantic_entropy_mixed() -> None:
    samples = [
        "energy equals mass times c squared",
        "e equals m c squared",
        "force equals mass times acceleration",  # different meaning
        "energy is mass times speed of light squared",
    ]
    # With a higher Jaccard threshold, only the very similar ones cluster.
    se = compute_semantic_entropy(samples, jaccard_threshold=0.60)
    assert 1 < se.n_clusters <= 4  # some clustering, not all same


def test_semantic_entropy_confidence() -> None:
    se = SemanticEntropy(normalized_entropy=0.2)
    assert se.confidence == pytest.approx(0.8)
    assert se.is_uncertain is False


def test_semantic_entropy_is_uncertain() -> None:
    se = SemanticEntropy(normalized_entropy=0.6)
    assert se.is_uncertain is True


def test_semantic_entropy_serialization() -> None:
    se = compute_semantic_entropy(["hello world"] * 3)
    d = se.to_dict()
    assert d["entropy"] == 0.0
    assert d["n_samples"] == 3
    assert d["method"] == "jaccard"


def test_jaccard_cluster_custom_threshold() -> None:
    # At very high threshold, nothing clusters
    clusters = jaccard_cluster(
        ["energy mass equivalence", "energy mass relativity"],
        threshold=0.90,
    )
    assert len(clusters) == 2


def test_nli_cluster_falls_back_to_jaccard() -> None:
    from formulagate.semantic_entropy import nli_cluster

    # No NLI function → falls back to Jaccard
    clusters = nli_cluster(["energy mass", "energy mass"])
    assert len(clusters) == 1
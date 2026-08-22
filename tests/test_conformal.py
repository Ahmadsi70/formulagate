"""Tests for conformal calibration (risk-controlled thresholds)."""

from __future__ import annotations

import pytest

from formulagate.conformal import (
    ConformalCalibration,
    ConformalError,
    conformal_threshold,
    fit_conformal,
)


def test_conformal_rejects_empty_input() -> None:
    with pytest.raises(ConformalError):
        fit_conformal([], [])


def test_conformal_rejects_length_mismatch() -> None:
    with pytest.raises(ConformalError):
        fit_conformal([0.5, 0.6], [1])


def test_conformal_rejects_non_finite_confidences() -> None:
    with pytest.raises(ConformalError):
        fit_conformal([float("nan"), 0.5], [1, 0])


def test_conformal_rejects_out_of_range_confidences() -> None:
    with pytest.raises(ConformalError):
        fit_conformal([-0.1, 0.5], [1, 0])
    with pytest.raises(ConformalError):
        fit_conformal([0.5, 1.5], [1, 0])


def test_conformal_single_class_all_correct() -> None:
    """When all labels are 1, accept everything."""
    cal = fit_conformal([0.3, 0.5, 0.8], [1, 1, 1], alpha=0.05)
    assert cal.threshold <= min(0.3, 0.5, 0.8)
    assert cal.accepted_ratio == 1.0
    assert cal.empirical_fdr == 0.0


def test_conformal_single_class_all_wrong() -> None:
    """When all labels are 0, accept nothing."""
    cal = fit_conformal([0.3, 0.5, 0.8], [0, 0, 0], alpha=0.05)
    assert cal.threshold > max(0.3, 0.5, 0.8) or cal.accepted_ratio == 0.0


def test_conformal_perfectly_separable() -> None:
    """When correct and wrong are perfectly separated, find the boundary."""
    # Correct at high confidence, wrong at low.
    confidences = [0.9, 0.85, 0.8, 0.3, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0, 0]
    cal = fit_conformal(confidences, labels, alpha=0.05)
    # Should pick a threshold between 0.8 and 0.3 that accepts all the 1s.
    assert cal.threshold <= 0.8
    assert cal.empirical_fdr == 0.0


def test_conformal_respects_alpha() -> None:
    """Empirical FDR at the chosen threshold must not exceed alpha."""
    # Partially overlapping distributions.
    confidences = [0.9, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2]
    labels =      [1,   1,   0,   1,    0,   0,   1]
    cal = fit_conformal(confidences, labels, alpha=0.30)
    assert cal.empirical_fdr <= 0.30 + 1e-9
    assert cal.threshold > 0.0


def test_conformal_stricter_alpha_gives_higher_threshold() -> None:
    """Lower alpha → higher threshold (more conservative)."""
    confidences = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3]
    labels =      [1,   0,   1,   0,   1,   0]
    cal_loose = fit_conformal(confidences, labels, alpha=0.50)
    cal_tight = fit_conformal(confidences, labels, alpha=0.01)
    assert cal_tight.threshold >= cal_loose.threshold


def test_conformal_maximizes_acceptance() -> None:
    """Among thresholds that satisfy FDR ≤ α, pick the one with most accepts."""
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    labels =      [1,   1,   0,   1,   0,   0]
    cal = fit_conformal(confidences, labels, alpha=0.30)
    # Valid thresholds and their acceptance counts:
    #   τ=0.9: accept [0.9]           → 1 accept, FDR=0/1=0.00 ✓
    #   τ=0.8: accept [0.9,0.8]       → 2 accept, FDR=0/2=0.00 ✓
    #   τ=0.7: accept [0.9,0.8,0.7]   → 3 accept, FDR=1/3=0.33 ✗ (FDR > 0.30)
    #   τ=0.6: accept [0.9,0.8,0.7,0.6] → 4 accept, FDR=1/4=0.25 ✓  ← BEST
    #   τ=0.5: accept [0.9..0.5]      → 5 accept, FDR=2/5=0.40 ✗
    #   τ=0.4: accept all              → 6 accept, FDR=3/6=0.50 ✗
    assert cal.threshold == pytest.approx(0.6)
    assert cal.accepted_ratio == pytest.approx(4 / 6)
    assert cal.empirical_fdr == pytest.approx(1 / 4)


def test_conformal_threshold_convenience() -> None:
    tau = conformal_threshold([0.9, 0.3], [1, 0], alpha=0.05)
    assert isinstance(tau, float)
    assert 0.0 <= tau <= 1.0


def test_conformal_serialization_round_trip() -> None:
    cal = fit_conformal([0.9, 0.7, 0.3, 0.1], [1, 1, 0, 0], alpha=0.10)
    d = cal.to_dict()
    cal2 = ConformalCalibration.from_dict(d)
    assert cal2.alpha == cal.alpha
    assert cal2.threshold == cal.threshold
    assert cal2.n_cal == cal.n_cal
    assert cal2.empirical_fdr == cal.empirical_fdr


def test_conformal_accepts_method() -> None:
    cal = ConformalCalibration(alpha=0.05, threshold=0.60)
    assert cal.accepts(0.80) is True
    assert cal.accepts(0.60) is True
    assert cal.accepts(0.30) is False


def test_conformal_is_deterministic() -> None:
    xs = [0.9, 0.7, 0.6, 0.4, 0.3, 0.1]
    ys = [1,   1,   0,   0,   1,   0]
    first = fit_conformal(xs, ys, alpha=0.20)
    second = fit_conformal(list(xs), list(ys), alpha=0.20)
    assert first.threshold == second.threshold
    assert first.empirical_fdr == second.empirical_fdr
"""Confidence calibration: sigmoid fit, quality metrics, persistence."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import pytest

np = pytest.importorskip("numpy")

from formulagate.calibration import (
    CalibrationError,
    CalibrationParams,
    brier_score,
    calibrate_confidence,
    evaluate_calibration,
    expected_calibration_error,
    find_optimal_threshold,
    fit_calibration,
    get_calibration,
    load_calibration,
    save_calibration,
    set_calibration,
    sigmoid,
)


@pytest.fixture(autouse=True)
def _restore_global_calibration():
    """Global calibration is process state — restore it after every test."""

    before = get_calibration()
    yield
    set_calibration(before)


def _separable_dataset() -> tuple[list[float], list[int]]:
    """Deterministic two-cluster score set (no RNG → reproducible fits)."""

    positives = [0.50 + 0.01 * i for i in range(25)]  # 0.50 … 0.74
    negatives = [0.10 + 0.01 * i for i in range(25)]  # 0.10 … 0.34
    return positives + negatives, [1] * 25 + [0] * 25


# ─── Sigmoid / confidence mapping ─────────────────────────────────────────────


def test_sigmoid_is_bounded_and_overflow_safe() -> None:
    assert 0.0 <= float(sigmoid(-10_000)) < 1e-9
    assert 1.0 - 1e-9 < float(sigmoid(10_000)) <= 1.0
    assert float(sigmoid(0.0)) == pytest.approx(0.5)


def test_calibrate_confidence_is_monotonic_in_score() -> None:
    params = CalibrationParams(a=8.0, b=-2.8)
    confidences = [calibrate_confidence(s / 10, params) for s in range(11)]
    assert confidences == sorted(confidences)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_threshold_50_matches_sigmoid_midpoint() -> None:
    params = CalibrationParams(a=8.0, b=-2.8)
    assert calibrate_confidence(params.threshold_50, params) == pytest.approx(0.5)


# ─── Fitting ──────────────────────────────────────────────────────────────────


def test_fit_calibration_needs_no_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit must run on the declared dependency set (numpy only)."""

    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    scores, labels = _separable_dataset()
    params = fit_calibration(scores, labels)
    assert params.a > 0


def test_fit_calibration_separates_classes() -> None:
    scores, labels = _separable_dataset()
    params = fit_calibration(scores, labels)
    assert calibrate_confidence(0.70, params) > 0.80
    assert calibrate_confidence(0.15, params) < 0.20


def test_fit_calibration_is_deterministic() -> None:
    scores, labels = _separable_dataset()
    first = fit_calibration(scores, labels)
    second = fit_calibration(scores, labels)
    assert (first.a, first.b, first.threshold) == (second.a, second.b, second.threshold)


def test_fit_calibration_beats_default_params_on_ece() -> None:
    """Fitted sigmoid must be better calibrated than the hand-tuned default."""

    scores, labels = _separable_dataset()
    fitted = fit_calibration(scores, labels)
    default_ece = evaluate_calibration(scores, labels, CalibrationParams.default()).expected_calibration_error
    fitted_ece = evaluate_calibration(scores, labels, fitted).expected_calibration_error
    assert fitted_ece < default_ece


def test_fit_calibration_stores_optimal_threshold() -> None:
    scores, labels = _separable_dataset()
    params = fit_calibration(scores, labels)
    confidences = [calibrate_confidence(s, params) for s in scores]
    expected, _ = find_optimal_threshold(confidences, labels)
    assert params.threshold == pytest.approx(expected)
    assert 0.0 < params.threshold < 1.0


def test_fit_calibration_rejects_invalid_input() -> None:
    with pytest.raises(CalibrationError):
        fit_calibration([], [])
    with pytest.raises(CalibrationError):
        fit_calibration([0.5, 0.6], [1])


def test_fit_calibration_degrades_to_default_on_single_class() -> None:
    """One-class data carries no discrimination signal → keep defaults, never crash."""

    params = fit_calibration([0.4, 0.5, 0.6], [1, 1, 1])
    assert params.a == CalibrationParams.default().a
    assert params.b == CalibrationParams.default().b
    assert 0.0 < params.threshold < 1.0


def test_fitted_params_stay_finite_on_perfectly_separable_data() -> None:
    """L2 on the slope keeps logistic regression from diverging to infinity."""

    params = fit_calibration([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0])
    assert math.isfinite(params.a) and math.isfinite(params.b)
    assert params.a < 1e4


# ─── Quality metrics ──────────────────────────────────────────────────────────


def test_ece_includes_confidence_of_exactly_one() -> None:
    """Top-bin edge must be inclusive, otherwise saturated predictions vanish."""

    assert expected_calibration_error([1.0, 1.0, 1.0, 1.0], [0, 0, 0, 0]) == pytest.approx(1.0)


def test_ece_is_zero_for_perfectly_calibrated_bins() -> None:
    confidences = [1.0, 1.0, 0.0, 0.0]
    assert expected_calibration_error(confidences, [1, 1, 0, 0]) == pytest.approx(0.0)


def test_brier_score_extremes() -> None:
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_find_optimal_threshold_picks_separating_cut() -> None:
    threshold, f_score = find_optimal_threshold([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert f_score == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.8


def _overlapping_dataset() -> tuple[list[float], list[int]]:
    """Two overlapping bells matching the real arXiv gate-score distribution
    (grounded ~N(0.62, 0.14), distractor ~N(0.51, 0.11), AUC ≈ 0.71).

    Quantiles of a normal — not samples — so the fixture stays deterministic.
    """

    n = 40
    grounded = NormalDist(0.62, 0.14)
    distractor = NormalDist(0.51, 0.105)
    positives = [grounded.inv_cdf((i + 0.5) / n) for i in range(n)]
    negatives = [distractor.inv_cdf((i + 0.5) / n) for i in range(n)]
    return positives + negatives, [1] * n + [0] * n


def _balanced_accuracy(confidences, labels, threshold) -> float:
    tp = sum(1 for c, y in zip(confidences, labels) if c >= threshold and y == 1)
    fn = sum(1 for c, y in zip(confidences, labels) if c < threshold and y == 1)
    tn = sum(1 for c, y in zip(confidences, labels) if c < threshold and y == 0)
    fp = sum(1 for c, y in zip(confidences, labels) if c >= threshold and y == 0)
    return 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))


def test_balanced_accuracy_objective_protects_the_abstain_class() -> None:
    """F1 ignores true negatives, so on overlapping scores it drifts toward
    "always generate" — the exact failure a hallucination gate must not have."""

    scores, labels = _overlapping_dataset()
    f1_threshold, _ = find_optimal_threshold(scores, labels)
    bal_threshold, bal_value = find_optimal_threshold(
        scores, labels, objective="balanced_accuracy"
    )

    assert bal_threshold > f1_threshold
    assert bal_value == pytest.approx(_balanced_accuracy(scores, labels, bal_threshold))
    assert _balanced_accuracy(scores, labels, bal_threshold) > _balanced_accuracy(
        scores, labels, f1_threshold
    )


def test_fit_calibration_accepts_the_balanced_objective() -> None:
    scores, labels = _overlapping_dataset()
    f1_fit = fit_calibration(scores, labels)
    bal_fit = fit_calibration(scores, labels, objective="balanced_accuracy")

    assert (bal_fit.a, bal_fit.b) == (f1_fit.a, f1_fit.b)
    assert bal_fit.threshold > f1_fit.threshold


def test_find_optimal_threshold_rejects_unknown_objective() -> None:
    with pytest.raises(CalibrationError):
        find_optimal_threshold([0.9, 0.1], [1, 0], objective="accuracy_at_k")


def test_evaluate_calibration_reports_full_quality_panel() -> None:
    scores, labels = _separable_dataset()
    params = fit_calibration(scores, labels)
    report = evaluate_calibration(scores, labels, params)
    assert report.n_samples == 50
    assert report.accuracy == pytest.approx(1.0)
    assert report.f1 == pytest.approx(1.0)
    assert report.brier_score < 0.10
    # Platt targets cap confidence at (N+1)/(N+2), so a deterministic split
    # keeps a residual gap between confidence and the empirical rate.
    assert report.expected_calibration_error < 0.20
    assert report.reliability_gap >= 0.0
    payload = report.to_dict()
    assert payload["brier_score"] == pytest.approx(report.brier_score)
    assert payload["params"]["threshold"] == pytest.approx(params.threshold)


# ─── Persistence ──────────────────────────────────────────────────────────────


def test_save_load_roundtrip_preserves_threshold(tmp_path: Path) -> None:
    params = CalibrationParams(a=6.5, b=-2.1, threshold=0.62, n_samples=50, ece=0.04)
    path = tmp_path / "calibration.json"
    save_calibration(path, params)
    loaded = load_calibration(path)
    assert (loaded.a, loaded.b, loaded.threshold) == (6.5, -2.1, 0.62)
    assert loaded.n_samples == 50
    assert get_calibration() == loaded


def test_load_calibration_can_skip_global_apply(tmp_path: Path) -> None:
    before = get_calibration()
    path = tmp_path / "c.json"
    save_calibration(path, CalibrationParams(a=1.0, b=0.0, threshold=0.3))
    set_calibration(before)
    loaded = load_calibration(path, apply=False)
    assert loaded.a == 1.0
    assert get_calibration() == before


def test_load_calibration_accepts_legacy_v04_file(tmp_path: Path) -> None:
    """v0.4 artifacts stored only (a, b) — threshold must fall back to default."""

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"params": {"a": 5.0, "b": -2.0}}), encoding="utf-8")
    loaded = load_calibration(path, apply=False)
    assert (loaded.a, loaded.b) == (5.0, -2.0)
    assert loaded.threshold == CalibrationParams().threshold


def test_load_calibration_raises_on_corrupt_artifact(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_calibration(path, apply=False)


def test_shipped_artifact_is_loadable_and_evidence_backed() -> None:
    """The committed calibration must stay parseable and non-degenerate."""

    artifact = Path(__file__).resolve().parents[1] / "data" / "calibration.json"
    if not artifact.is_file():
        pytest.skip("no shipped calibration artifact")
    params = load_calibration(artifact, apply=False)
    assert params.a > 0
    assert 0.0 < params.threshold < 1.0
    assert params.n_samples >= 50
    assert params.ece is not None and params.ece < 0.20


def test_saved_artifact_is_byte_stable(tmp_path: Path) -> None:
    """Determinism: same params ⇒ identical artifact bytes (no timestamps)."""

    params = CalibrationParams(a=6.5, b=-2.1, threshold=0.62)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    save_calibration(first, params)
    save_calibration(second, params)
    assert first.read_bytes() == second.read_bytes()

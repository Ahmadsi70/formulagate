"""Multi-feature calibration: fuse similarity with deterministic physics signals."""

from __future__ import annotations

import json
from statistics import NormalDist

import pytest

from formulagate.calibration import (
    CalibrationError,
    MultiCalibration,
    fit_multi_calibration,
    load_multi_calibration,
    save_multi_calibration,
)

FEATURES = ("score", "dimension_ok")


def _dataset() -> tuple[list[list[float]], list[int]]:
    """Similarity alone overlaps (AUC ≈ 0.71, as measured on arXiv); the second
    feature is decisive. A useful fusion must lean on the second one."""

    grounded = NormalDist(0.62, 0.14)
    distractor = NormalDist(0.51, 0.105)
    rows: list[list[float]] = []
    labels: list[int] = []
    for i in range(40):
        rows.append([grounded.inv_cdf((i + 0.5) / 40), 1.0])
        labels.append(1)
        rows.append([distractor.inv_cdf((i + 0.5) / 40), -1.0])
        labels.append(0)
    return rows, labels


def test_fit_learns_a_weight_per_feature() -> None:
    rows, labels = _dataset()
    model = fit_multi_calibration(rows, labels, FEATURES)
    assert len(model.weights) == 2
    assert model.feature_names == FEATURES
    assert model.n_samples == len(labels)


def test_decisive_feature_dominates_the_noisy_one() -> None:
    rows, labels = _dataset()
    model = fit_multi_calibration(rows, labels, FEATURES)
    assert abs(model.weights[1]) > abs(model.weights[0])


def test_confidence_is_a_probability_and_monotonic() -> None:
    rows, labels = _dataset()
    model = fit_multi_calibration(rows, labels, FEATURES)
    low = model.confidence([0.3, -1.0])
    high = model.confidence([0.9, 1.0])
    assert 0.0 <= low < high <= 1.0


def test_fusion_beats_the_similarity_feature_alone() -> None:
    rows, labels = _dataset()
    fused = fit_multi_calibration(rows, labels, FEATURES)
    similarity_only = fit_multi_calibration([[r[0]] for r in rows], labels, ("score",))

    def accuracy(model, matrix) -> float:
        hits = sum(
            1
            for row, y in zip(matrix, labels)
            if (1 if model.confidence(row) >= model.threshold else 0) == y
        )
        return hits / len(labels)

    assert accuracy(fused, rows) > accuracy(similarity_only, [[r[0]] for r in rows])


def test_fit_is_deterministic() -> None:
    rows, labels = _dataset()
    assert fit_multi_calibration(rows, labels, FEATURES).to_dict() == fit_multi_calibration(
        rows, labels, FEATURES
    ).to_dict()


def test_single_class_data_degrades_to_a_neutral_model() -> None:
    model = fit_multi_calibration([[0.5, 1.0], [0.6, 1.0]], [1, 1], FEATURES)
    assert all(w == 0.0 for w in model.weights)
    assert model.confidence([0.5, 1.0]) == pytest.approx(0.5)


def test_rejects_ragged_or_empty_input() -> None:
    with pytest.raises(CalibrationError):
        fit_multi_calibration([], [], FEATURES)
    with pytest.raises(CalibrationError):
        fit_multi_calibration([[0.5, 1.0], [0.6]], [1, 0], FEATURES)
    with pytest.raises(CalibrationError):
        fit_multi_calibration([[0.5, 1.0]], [1], ("only_one_name",))


def test_confidence_rejects_a_wrong_width_vector() -> None:
    rows, labels = _dataset()
    model = fit_multi_calibration(rows, labels, FEATURES)
    with pytest.raises(CalibrationError):
        model.confidence([0.5])


def test_artifact_round_trips_through_disk(tmp_path) -> None:
    rows, labels = _dataset()
    model = fit_multi_calibration(rows, labels, FEATURES)
    path = tmp_path / "multi.json"
    save_multi_calibration(path, model)

    restored = load_multi_calibration(path)
    assert restored.to_dict() == model.to_dict()
    assert restored.confidence([0.7, 1.0]) == pytest.approx(model.confidence([0.7, 1.0]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "multi"
    assert payload["feature_names"] == list(FEATURES)


def test_load_rejects_a_single_feature_artifact(tmp_path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"params": {"a": 8.0, "b": -2.8}}), encoding="utf-8")
    with pytest.raises(CalibrationError):
        load_multi_calibration(path)


def test_model_is_a_frozen_dataclass() -> None:
    rows, labels = _dataset()
    model = fit_multi_calibration(rows, labels, FEATURES)
    assert isinstance(model, MultiCalibration)
    with pytest.raises(Exception):
        model.bias = 1.0  # type: ignore[misc]

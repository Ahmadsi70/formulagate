"""Confidence calibration Module Node — raw gate scores → probabilities.

Why this exists:
  ``combined_score`` produced by the gate is an ordering signal, not a
  probability: 0.42 does not mean "42% chance the citation is right". Every
  downstream consumer (abstain policy, audit trail, human review queue) needs a
  probability it can threshold, log and compare across corpora. Platt scaling
  gives that with two learned Information Points ``(a, b)`` plus a decision
  threshold learned from the same labeled set.

Model:
    confidence = sigmoid(a * combined_score + b)
    accept     ⟺ confidence >= threshold

Design constraints:
  - Pure standard library: the gate must stay importable without numpy/scipy.
  - Deterministic: same (scores, labels) ⇒ bit-identical params and artifact.
  - Total: invalid input raises ``CalibrationError``; degenerate-but-valid input
    (single class) degrades to documented defaults instead of failing.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

ARTIFACT_VERSION = "0.5"
ENV_CALIBRATION_PATH = "FORMULAGATE_CALIBRATION"
ENV_MULTI_CALIBRATION_PATH = "FORMULAGATE_CALIBRATION_MULTI"

DEFAULT_SLOPE = 8.0
DEFAULT_BIAS = -2.8
DEFAULT_THRESHOLD = 0.50

#: Extra confidence demanded when the draft is only loosely tied to the brief.
STRICT_THRESHOLD_MARGIN = 0.05
#: Upper bound used to normalise the integer lexical score into [0, 1].
LEXICAL_SCORE_MAX = 15.0

_THRESHOLD_GRID_LO = 0.10
_THRESHOLD_GRID_HI = 0.90
_THRESHOLD_GRID_STEPS = 81

_MAX_NEWTON_ITER = 100
_MAX_LINE_SEARCH = 20
_GRAD_TOL = 1e-10
_SLOPE_CLAMP = 1e3


class CalibrationError(ValueError):
    """Raised when calibration input or a stored artifact is unusable."""


# ─── Parameters ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationParams:
    """Learned sigmoid + decision threshold.

    ``n_samples`` and ``ece`` are provenance fields: they say how much evidence
    backs these numbers, which matters when an artifact is shipped between runs.
    """

    a: float = DEFAULT_SLOPE
    b: float = DEFAULT_BIAS
    threshold: float = DEFAULT_THRESHOLD
    n_samples: int = 0
    ece: float | None = None

    @property
    def threshold_50(self) -> float:
        """Raw score at which confidence crosses 0.50."""

        return -self.b / self.a if self.a != 0 else 0.5

    @property
    def strict_threshold(self) -> float:
        """Threshold applied to borderline drafts (weak brief↔draft coupling)."""

        return min(self.threshold + STRICT_THRESHOLD_MARGIN, 0.99)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "threshold": self.threshold,
            "threshold_50": self.threshold_50,
            "n_samples": self.n_samples,
            "ece": self.ece,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationParams":
        """Tolerant loader — v0.4 artifacts only carried ``a`` and ``b``."""

        return cls(
            a=float(d.get("a", DEFAULT_SLOPE)),
            b=float(d.get("b", DEFAULT_BIAS)),
            threshold=float(d.get("threshold", DEFAULT_THRESHOLD)),
            n_samples=int(d.get("n_samples", 0) or 0),
            ece=None if d.get("ece") is None else float(d["ece"]),
        )

    @classmethod
    def default(cls) -> "CalibrationParams":
        """Hand-tuned starting point used until a fit replaces it."""

        return cls()


_CALIBRATION = CalibrationParams.default()
_ENV_LOADED = False


def get_calibration() -> CalibrationParams:
    """Return the active calibration, loading ``$FORMULAGATE_CALIBRATION`` once."""

    global _ENV_LOADED
    if not _ENV_LOADED:
        _ENV_LOADED = True
        env_path = os.environ.get(ENV_CALIBRATION_PATH)
        if env_path:
            try:
                load_calibration(env_path)
            except CalibrationError as exc:
                logger.warning("Ignoring %s: %s", ENV_CALIBRATION_PATH, exc)
    return _CALIBRATION


def set_calibration(params: CalibrationParams) -> None:
    """Install ``params`` as the process-wide calibration."""

    global _CALIBRATION, _ENV_LOADED
    _CALIBRATION = params
    _ENV_LOADED = True
    logger.info(
        "Calibration set: a=%.3f b=%.3f threshold=%.3f n=%d",
        params.a, params.b, params.threshold, params.n_samples,
    )


# ─── Sigmoid mapping ──────────────────────────────────────────────────────────


def sigmoid(x: float) -> float:
    """Overflow-safe logistic function."""

    z = float(x)
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z)) if z < 700.0 else 1.0
    ex = math.exp(z) if z > -700.0 else 0.0
    return ex / (1.0 + ex)


def calibrate_confidence(
    combined_score: float,
    params: CalibrationParams | None = None,
) -> float:
    """Map a raw gate score to a calibrated probability in [0, 1]."""

    p = params or get_calibration()
    return sigmoid(p.a * float(combined_score) + p.b)


# ─── Fitting (Platt scaling via damped Newton) ────────────────────────────────


def _validate(scores: Sequence[float], labels: Sequence[int]) -> tuple[list[float], list[int]]:
    if len(scores) != len(labels):
        raise CalibrationError(
            f"scores/labels length mismatch: {len(scores)} vs {len(labels)}"
        )
    if not scores:
        raise CalibrationError("calibration needs at least one labeled sample")
    xs = [float(s) for s in scores]
    ys = [1 if int(y) else 0 for y in labels]
    if any(not math.isfinite(x) for x in xs):
        raise CalibrationError("scores must be finite")
    return xs, ys


def _platt_targets(labels: Sequence[int]) -> list[float]:
    """Platt (1999) smoothed targets — keep separable data from diverging."""

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    return [hi if y else lo for y in labels]


def _objective(
    a: float, b: float, xs: Sequence[float], ts: Sequence[float], l2: float
) -> float:
    total = 0.0
    for x, t in zip(xs, ts):
        p = min(max(sigmoid(a * x + b), 1e-15), 1.0 - 1e-15)
        total -= t * math.log(p) + (1.0 - t) * math.log(1.0 - p)
    return total / len(xs) + l2 * a * a


def fit_calibration(
    scores: Sequence[float],
    labels: Sequence[int],
    l2_penalty: float = 1e-3,
    beta: float = 1.0,
    objective: str = "fbeta",
) -> CalibrationParams:
    """Fit ``(a, b)`` by regularised logistic regression, then pick a threshold.

    The slope alone is penalised (``l2 * a²``): shrinking the bias would drag
    the whole curve toward 0.5 and destroy the base-rate information.

    Args:
        scores: Raw combined scores from the gate.
        labels: 1 = the accept is correct, 0 = the gate should abstain.
        l2_penalty: Slope shrinkage; larger ⇒ flatter, less confident curve.
        beta: F-beta weight for threshold search (β<1 favours precision).
        objective: Threshold criterion — see :func:`find_optimal_threshold`.

    Raises:
        CalibrationError: empty input or mismatched lengths.
    """

    xs, ys = _validate(scores, labels)
    if len(set(ys)) < 2:
        logger.warning("Single-class calibration data (n=%d) — keeping defaults", len(ys))
        return replace(CalibrationParams.default(), n_samples=len(ys))

    ts = _platt_targets(ys)
    a, b = DEFAULT_SLOPE, DEFAULT_BIAS
    n = float(len(xs))
    current = _objective(a, b, xs, ts, l2_penalty)

    for _ in range(_MAX_NEWTON_ITER):
        g_a = 2.0 * l2_penalty * a
        g_b = 0.0
        h_aa = 2.0 * l2_penalty
        h_ab = 0.0
        h_bb = 0.0
        for x, t in zip(xs, ts):
            p = sigmoid(a * x + b)
            err = (p - t) / n
            g_a += err * x
            g_b += err
            w = p * (1.0 - p) / n
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w

        if math.hypot(g_a, g_b) < _GRAD_TOL:
            break

        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:  # Degenerate curvature → plain gradient step.
            d_a, d_b = g_a, g_b
        else:
            d_a = (h_bb * g_a - h_ab * g_b) / det
            d_b = (h_aa * g_b - h_ab * g_a) / det

        step = 1.0
        improved = False
        for _ in range(_MAX_LINE_SEARCH):
            cand_a = a - step * d_a
            cand_b = b - step * d_b
            cand = _objective(cand_a, cand_b, xs, ts, l2_penalty)
            if cand < current:
                a, b, current, improved = cand_a, cand_b, cand, True
                break
            step *= 0.5
        if not improved:
            break

    a = max(-_SLOPE_CLAMP, min(_SLOPE_CLAMP, a))
    probe = CalibrationParams(a=a, b=b)
    confidences = [calibrate_confidence(x, probe) for x in xs]
    threshold, _ = find_optimal_threshold(confidences, ys, beta=beta, objective=objective)
    ece = expected_calibration_error(confidences, ys)

    params = CalibrationParams(
        a=a, b=b, threshold=threshold, n_samples=len(ys), ece=ece
    )
    logger.info(
        "Calibration fitted: a=%.3f b=%.3f threshold=%.3f ece=%.4f n=%d",
        a, b, threshold, ece, len(ys),
    )
    return params


def find_optimal_threshold(
    confidences: Sequence[float],
    labels: Sequence[int],
    beta: float = 1.0,
    objective: str = "fbeta",
) -> tuple[float, float]:
    """Scan a fixed grid for the best threshold under ``objective``.

    A fixed grid (not the observed score set) keeps the result stable when a
    single sample changes — thresholds shipped in artifacts must be reproducible.

    Args:
        confidences: Calibrated probabilities.
        labels: 1 = should generate, 0 = should abstain.
        beta: F-beta weight (only used by the ``fbeta`` objective).
        objective: ``"fbeta"`` optimises the generate class only; F-beta never
            looks at true negatives, so on overlapping scores it slides toward
            "always generate". ``"balanced_accuracy"`` weights the abstain class
            equally and is the right target for a hallucination gate.

    Raises:
        CalibrationError: unknown objective, or invalid input.
    """

    if objective not in ("fbeta", "balanced_accuracy"):
        raise CalibrationError(f"Unknown threshold objective: {objective!r}")

    xs, ys = _validate(confidences, labels)
    best_t, best_f = DEFAULT_THRESHOLD, 0.0
    span = _THRESHOLD_GRID_HI - _THRESHOLD_GRID_LO
    b2 = beta * beta

    for i in range(_THRESHOLD_GRID_STEPS):
        t = _THRESHOLD_GRID_LO + span * i / (_THRESHOLD_GRID_STEPS - 1)
        tp = sum(1 for c, y in zip(xs, ys) if c >= t and y == 1)
        fp = sum(1 for c, y in zip(xs, ys) if c >= t and y == 0)
        fn = sum(1 for c, y in zip(xs, ys) if c < t and y == 1)
        tn = sum(1 for c, y in zip(xs, ys) if c < t and y == 0)

        if objective == "balanced_accuracy":
            if tp + fn == 0 or tn + fp == 0:
                continue
            value = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        else:
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            if prec + rec == 0.0:
                continue
            value = (1.0 + b2) * prec * rec / (b2 * prec + rec)

        if value > best_f:
            best_f, best_t = value, t

    return best_t, best_f


# ─── Quality metrics ──────────────────────────────────────────────────────────


def expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Weighted |mean confidence − empirical rate| across equal-width bins.

    Bin index is clamped so a confidence of exactly 1.0 lands in the top bin
    instead of being silently dropped.
    """

    xs, ys = _validate(confidences, labels)
    bins = max(1, n_bins)
    sums = [0.0] * bins
    hits = [0] * bins
    counts = [0] * bins

    for c, y in zip(xs, ys):
        idx = min(int(c * bins), bins - 1)
        idx = max(0, idx)
        sums[idx] += c
        hits[idx] += y
        counts[idx] += 1

    total = len(xs)
    ece = 0.0
    for i in range(bins):
        if counts[i] == 0:
            continue
        mean_conf = sums[i] / counts[i]
        empirical = hits[i] / counts[i]
        ece += (counts[i] / total) * abs(mean_conf - empirical)
    return ece


def brier_score(confidences: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error of the probability forecast (0 = perfect)."""

    xs, ys = _validate(confidences, labels)
    return sum((c - y) ** 2 for c, y in zip(xs, ys)) / len(xs)


@dataclass(frozen=True)
class CalibrationReport:
    """Quality panel for a calibration fit."""

    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    expected_calibration_error: float
    brier_score: float
    calibration_params: CalibrationParams
    optimal_threshold: float
    confidence_mean_correct: float
    confidence_mean_wrong: float
    reliability_gap: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
            "optimal_threshold": self.optimal_threshold,
            "confidence_mean_correct": self.confidence_mean_correct,
            "confidence_mean_wrong": self.confidence_mean_wrong,
            "reliability_gap": self.reliability_gap,
            "params": self.calibration_params.to_dict(),
        }


def evaluate_calibration(
    scores: Sequence[float],
    labels: Sequence[int],
    params: CalibrationParams | None = None,
    n_bins: int = 10,
    optimize_threshold: bool = True,
) -> CalibrationReport:
    """Score a calibration on labeled data (ECE, Brier, F1 at the threshold)."""

    xs, ys = _validate(scores, labels)
    cal = params or get_calibration()
    confidences = [calibrate_confidence(x, cal) for x in xs]

    if optimize_threshold:
        threshold, _ = find_optimal_threshold(confidences, ys)
    else:
        threshold = cal.threshold

    preds = [1 if c >= threshold else 0 for c in confidences]
    tp = sum(1 for p, y in zip(preds, ys) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, ys) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, ys) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, ys) if p == 0 and y == 0)

    total = len(ys)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    right = [c for c, p, y in zip(confidences, preds, ys) if p == y]
    wrong = [c for c, p, y in zip(confidences, preds, ys) if p != y]
    conf_right = sum(right) / len(right) if right else 0.0
    conf_wrong = sum(wrong) / len(wrong) if wrong else 0.0

    return CalibrationReport(
        n_samples=total,
        accuracy=(tp + tn) / total,
        precision=precision,
        recall=recall,
        f1=f1,
        expected_calibration_error=expected_calibration_error(confidences, ys, n_bins),
        brier_score=brier_score(confidences, ys),
        calibration_params=cal,
        optimal_threshold=threshold,
        confidence_mean_correct=conf_right,
        confidence_mean_wrong=conf_wrong,
        reliability_gap=conf_right - conf_wrong,
    )


# ─── Persistence ──────────────────────────────────────────────────────────────


def save_calibration(
    path: Path | str,
    params: CalibrationParams | None = None,
    report: CalibrationReport | None = None,
) -> None:
    """Write a calibration artifact (sorted keys, no timestamp ⇒ byte-stable)."""

    p = params or get_calibration()
    payload: dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "model": "platt_sigmoid",
        "params": p.to_dict(),
    }
    if report is not None:
        payload["report"] = report.to_dict()
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Calibration saved to %s", target)


def load_calibration(path: Path | str, apply: bool = True) -> CalibrationParams:
    """Read a calibration artifact; ``apply`` installs it process-wide."""

    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration {target}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"malformed calibration {target}: {exc}") from exc
    if not isinstance(data, Mapping) or not isinstance(data.get("params"), Mapping):
        raise CalibrationError(f"calibration {target} has no 'params' object")

    params = CalibrationParams.from_dict(data["params"])
    if apply:
        set_calibration(params)
    return params


# ─── Multi-feature calibration ────────────────────────────────────────────────


@dataclass(frozen=True)
class MultiCalibration:
    """Logistic model over several gate features instead of one score.

    Platt scaling can only rescale a single number, so it inherits that number's
    discrimination — measured at AUC 0.711 on live arXiv data. Deterministic
    physics features (dimensional consistency, proven equivalence) carry signal
    that similarity structurally cannot, and only a multivariate fit can weigh
    them against it.
    """

    weights: tuple[float, ...]
    bias: float
    threshold: float = DEFAULT_THRESHOLD
    feature_names: tuple[str, ...] = ()
    n_samples: int = 0
    ece: float | None = None

    def confidence(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise CalibrationError(
                f"expected {len(self.weights)} features, got {len(features)}"
            )
        z = self.bias + sum(w * float(x) for w, x in zip(self.weights, features))
        return sigmoid(z)

    def accepts(self, features: Sequence[float]) -> bool:
        return self.confidence(features) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "bias": self.bias,
            "threshold": self.threshold,
            "feature_names": list(self.feature_names),
            "n_samples": self.n_samples,
            "ece": self.ece,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MultiCalibration":
        if "weights" not in d:
            raise CalibrationError("artifact has no 'weights' — not a multi-feature model")
        return cls(
            weights=tuple(float(w) for w in d["weights"]),
            bias=float(d.get("bias", 0.0)),
            threshold=float(d.get("threshold", DEFAULT_THRESHOLD)),
            feature_names=tuple(str(n) for n in d.get("feature_names", ())),
            n_samples=int(d.get("n_samples", 0) or 0),
            ece=None if d.get("ece") is None else float(d["ece"]),
        )


_MULTI_CALIBRATION: MultiCalibration | None = None
_MULTI_ENV_LOADED = False


def get_multi_calibration() -> MultiCalibration | None:
    """Active fused model, loading ``$FORMULAGATE_CALIBRATION_MULTI`` once.

    ``None`` means "not installed" — callers then fall back to scalar Platt, so
    the fused model is strictly opt-in and can never break an existing setup.
    """

    global _MULTI_ENV_LOADED, _MULTI_CALIBRATION
    if not _MULTI_ENV_LOADED:
        _MULTI_ENV_LOADED = True
        env_path = os.environ.get(ENV_MULTI_CALIBRATION_PATH)
        if env_path:
            try:
                _MULTI_CALIBRATION = load_multi_calibration(env_path)
            except CalibrationError as exc:
                logger.warning("Ignoring %s: %s", ENV_MULTI_CALIBRATION_PATH, exc)
    return _MULTI_CALIBRATION


def set_multi_calibration(model: MultiCalibration | None) -> None:
    """Install (or clear) the process-wide fused model."""

    global _MULTI_CALIBRATION, _MULTI_ENV_LOADED
    _MULTI_CALIBRATION = model
    _MULTI_ENV_LOADED = True


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gauss–Jordan with partial pivoting; ``None`` when the system is singular.

    Hand-rolled because the whole point of this module is to stay dependency-free
    and bit-reproducible — and a (k+1)×(k+1) solve with k≈5 is trivial work.
    """

    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row == col or aug[row][col] == 0.0:
                continue
            factor = aug[row][col]
            aug[row] = [v - factor * p for v, p in zip(aug[row], aug[col])]

    return [aug[i][n] for i in range(n)]


def _validate_matrix(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    feature_names: Sequence[str],
) -> tuple[list[list[float]], list[int]]:
    if len(rows) != len(labels):
        raise CalibrationError(f"rows/labels mismatch: {len(rows)} vs {len(labels)}")
    if not rows:
        raise CalibrationError("calibration needs at least one labeled sample")
    width = len(feature_names)
    matrix: list[list[float]] = []
    for row in rows:
        if len(row) != width:
            raise CalibrationError(
                f"every row must have {width} features, got {len(row)}"
            )
        values = [float(v) for v in row]
        if any(not math.isfinite(v) for v in values):
            raise CalibrationError("features must be finite")
        matrix.append(values)
    return matrix, [1 if int(y) else 0 for y in labels]


def fit_multi_calibration(
    rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    feature_names: Sequence[str],
    l2_penalty: float = 1e-3,
    beta: float = 1.0,
    objective: str = "balanced_accuracy",
) -> MultiCalibration:
    """Fit a regularised multivariate logistic model, then pick a threshold.

    Same numerics as the scalar fit — Platt-smoothed targets, damped Newton with
    a backtracking line search, L2 on the weights but never on the bias — lifted
    to k dimensions so the two fits stay comparable.

    Raises:
        CalibrationError: empty, ragged or non-finite input.
    """

    matrix, ys = _validate_matrix(rows, labels, feature_names)
    k = len(feature_names)
    names = tuple(str(n) for n in feature_names)

    if len(set(ys)) < 2:
        logger.warning("Single-class multi-feature data (n=%d) — neutral model", len(ys))
        return MultiCalibration(
            weights=tuple(0.0 for _ in range(k)),
            bias=0.0,
            threshold=DEFAULT_THRESHOLD,
            feature_names=names,
            n_samples=len(ys),
        )

    targets = _platt_targets(ys)
    weights = [0.0] * k
    bias = 0.0
    n = float(len(matrix))

    def loss(w: Sequence[float], b: float) -> float:
        total = 0.0
        for row, t in zip(matrix, targets):
            z = b + sum(wi * x for wi, x in zip(w, row))
            p = min(max(sigmoid(z), 1e-15), 1.0 - 1e-15)
            total -= t * math.log(p) + (1.0 - t) * math.log(1.0 - p)
        return total / n + l2_penalty * sum(wi * wi for wi in w)

    current = loss(weights, bias)

    for _ in range(_MAX_NEWTON_ITER):
        gradient = [2.0 * l2_penalty * w for w in weights] + [0.0]
        hessian = [[0.0] * (k + 1) for _ in range(k + 1)]
        for i in range(k):
            hessian[i][i] = 2.0 * l2_penalty

        for row, t in zip(matrix, targets):
            z = bias + sum(w * x for w, x in zip(weights, row))
            p = sigmoid(z)
            err = (p - t) / n
            weight = p * (1.0 - p) / n
            extended = list(row) + [1.0]
            for i, xi in enumerate(extended):
                gradient[i] += err * xi
                for j, xj in enumerate(extended):
                    hessian[i][j] += weight * xi * xj

        if math.sqrt(sum(g * g for g in gradient)) < _GRAD_TOL:
            break

        step_dir = _solve(hessian, gradient) or gradient  # Singular ⇒ gradient step.

        step = 1.0
        improved = False
        for _ in range(_MAX_LINE_SEARCH):
            cand_w = [w - step * d for w, d in zip(weights, step_dir[:k])]
            cand_b = bias - step * step_dir[k]
            cand = loss(cand_w, cand_b)
            if cand < current:
                weights, bias, current, improved = cand_w, cand_b, cand, True
                break
            step *= 0.5
        if not improved:
            break

    weights = [max(-_SLOPE_CLAMP, min(_SLOPE_CLAMP, w)) for w in weights]
    probe = MultiCalibration(tuple(weights), bias, DEFAULT_THRESHOLD, names, len(ys))
    confidences = [probe.confidence(row) for row in matrix]
    threshold, _ = find_optimal_threshold(confidences, ys, beta=beta, objective=objective)
    ece = expected_calibration_error(confidences, ys)

    model = MultiCalibration(
        weights=tuple(weights),
        bias=bias,
        threshold=threshold,
        feature_names=names,
        n_samples=len(ys),
        ece=ece,
    )
    logger.info(
        "Multi-feature calibration fitted: %s bias=%.3f threshold=%.3f ece=%.4f n=%d",
        dict(zip(names, (round(w, 3) for w in weights))), bias, threshold, ece, len(ys),
    )
    return model


def save_multi_calibration(
    path: Path | str,
    model: MultiCalibration,
    report: Mapping[str, Any] | None = None,
) -> None:
    """Write a multi-feature artifact (sorted keys ⇒ byte-stable)."""

    payload: dict[str, Any] = {
        "version": ARTIFACT_VERSION,
        "kind": "multi",
        "model": "multivariate_logistic",
        **model.to_dict(),
    }
    if report is not None:
        payload["report"] = dict(report)
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Multi-feature calibration saved to %s", target)


def load_multi_calibration(path: Path | str) -> MultiCalibration:
    """Read a multi-feature artifact, rejecting scalar Platt files."""

    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"malformed calibration {target}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise CalibrationError(f"calibration {target} is not an object")
    return MultiCalibration.from_dict(data)


# ─── Dataset collection ───────────────────────────────────────────────────────


def collect_calibration_samples(
    cases: Iterable[Mapping[str, Any]],
    *,
    corpus: Sequence[Mapping[str, Any]],
    make_retriever: Callable[[list[dict[str, Any]]], Any],
    retrieve_k: int = 3,
    reranker: Any | None = None,
    limit: int | None = None,
) -> tuple[list[float], list[int]]:
    """Replay golden cases through the RAG gate to harvest (score, label) pairs.

    Cases whose retrieval returns nothing contribute ``score = 0.0``: the gate
    really did see zero evidence there, so the low end of the curve must know it.
    """

    from formulagate.rag import run_scientific_rag  # lazy: avoids import cycle

    scores: list[float] = []
    labels: list[int] = []
    full_retriever: Any | None = None  # Reused: encoding the whole corpus dominates cost.

    for i, case in enumerate(cases):
        if limit is not None and i >= limit:
            break
        ids = case.get("corpus_ids")
        rows = (
            [dict(r) for r in corpus if str(r.get("id")) in set(ids)]
            if ids
            else []
        )
        if rows:
            retriever = make_retriever(rows)
        else:
            if full_retriever is None:
                full_retriever = make_retriever([dict(r) for r in corpus])
            retriever = full_retriever
        decision = run_scientific_rag(
            brief=str(case["brief"]),
            draft=str(case["draft"]),
            retriever=retriever,
            retrieve_k=retrieve_k,
            reranker=reranker,
        )
        combined = decision.gate.combined_score
        scores.append(float(combined) if combined is not None else 0.0)
        labels.append(1 if bool(case["expect_generate"]) else 0)

    return scores, labels

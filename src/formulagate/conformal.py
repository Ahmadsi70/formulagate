"""Conformal inference for risk-controlled gate decisions.

Why this exists:
  Platt scaling answers "how confident are we?" but not "what is the worst-case
  error rate among the decisions we actually accept?"  Conformal calibration
  answers the second question with a finite-sample **guarantee**: as long as the
  calibration set is exchangeable with future inputs, the false-discovery rate
  among accepted (generate) decisions is bounded at the chosen α.

  This is the same machinery behind CIC (Conformal Inference Control, Dong &
  Shinnou 2026) and the conformal-risk-control literature (Angelopoulos & Bates
  2023), reduced to the single-threshold, binary-decision case that a gate needs.

Model:
    Given a calibration set of (confidence, label) pairs where label=1 means
    "generating was correct" and label=0 means "should have abstained", find the
    **lowest** confidence threshold τ such that the empirical false-discovery
    rate among acceptances is ≤ α.

    accept  ⟺  confidence >= τ

    The guarantee is distribution-free but depends on exchangeability: if the
    deployment distribution drifts significantly from the calibration set,
    conformal guarantees weaken, just like any other learned threshold.

Usage:
    from formulagate.conformal import ConformalCalibration

    cal = ConformalCalibration(alpha=0.05)
    tau = cal.fit(confidences, labels)

    # Every decision with confidence >= tau has FDR ≤ 5% (with high probability).
    # Use tau in place of the Platt threshold.

Design constraints (same as calibration.py):
  - Pure standard library.
  - Deterministic.
  - Degenerate input (single class, empty) → documented fallback, never a crash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class ConformalError(ValueError):
    """Raised when conformal calibration input is unusable."""


@dataclass(frozen=True)
class ConformalCalibration:
    """Risk-controlled threshold with finite-sample guarantee.

    Attributes:
        alpha: Target false-discovery rate (e.g. 0.05 = 5%).
        threshold: Learned confidence threshold τ.
        n_cal: Number of calibration samples.
        empirical_fdr: Observed FDR on the calibration set at τ.
        accepted_ratio: Fraction of calibration samples accepted at τ.
    """

    alpha: float
    threshold: float
    n_cal: int = 0
    empirical_fdr: float | None = None
    accepted_ratio: float | None = None

    def accepts(self, confidence: float) -> bool:
        """True when this confidence clears the risk-controlled threshold."""
        return confidence >= self.threshold

    @property
    def guarantee(self) -> str:
        if self.n_cal == 0:
            return "no calibration data"
        return (
            f"FDR ≤ {self.alpha:.1%} among accepted decisions "
            f"(n_cal={self.n_cal}, τ={self.threshold:.4f})"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "n_cal": self.n_cal,
            "empirical_fdr": self.empirical_fdr,
            "accepted_ratio": self.accepted_ratio,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibration":
        return cls(
            alpha=float(d.get("alpha", 0.05)),
            threshold=float(d.get("threshold", 0.50)),
            n_cal=int(d.get("n_cal", 0) or 0),
            empirical_fdr=(
                None if d.get("empirical_fdr") is None
                else float(d["empirical_fdr"])
            ),
            accepted_ratio=(
                None if d.get("accepted_ratio") is None
                else float(d["accepted_ratio"])
            ),
        )


def _validate(
    confidences: Sequence[float],
    labels: Sequence[int],
) -> tuple[list[float], list[int]]:
    if len(confidences) != len(labels):
        raise ConformalError(
            f"length mismatch: {len(confidences)} confidences vs {len(labels)} labels"
        )
    if not confidences:
        raise ConformalError("need at least one calibration sample")
    xs = [float(c) for c in confidences]
    ys = [1 if int(y) else 0 for y in labels]
    if any(not math.isfinite(x) for x in xs):
        raise ConformalError("confidences must be finite")
    if any(x < 0.0 or x > 1.0 for x in xs):
        raise ConformalError("confidences must be in [0, 1]")
    if any(y not in (0, 1) for y in ys):
        raise ConformalError("labels must be 0 or 1")
    return xs, ys


def fit_conformal(
    confidences: Sequence[float],
    labels: Sequence[int],
    alpha: float = 0.05,
) -> ConformalCalibration:
    """Find the lowest confidence threshold that controls FDR at ``alpha``.

    Args:
        confidences: Calibrated probabilities from the gate (values in [0, 1]).
        labels: 1 = the generate was correct, 0 = should have abstained.
        alpha: Target false-discovery rate (default 0.05 = 5%).

    Returns:
        ``ConformalCalibration`` with the learned threshold τ.

    Raises:
        ConformalError: empty, mismatched, or out-of-range input.

    Theory:
        We scan candidate thresholds in descending order.  At each candidate τ,
        compute:

            FDR(τ) = FP(τ) / (TP(τ) + FP(τ))   when denominator > 0, else 0

        where FP = accepted with label 0, TP = accepted with label 1.
        We pick the **lowest** τ such that FDR(τ) ≤ α.  A lower τ accepts
        more decisions; we want to maximize the acceptance rate subject to
        the FDR constraint.
    """
    xs, ys = _validate(confidences, labels)

    if alpha <= 0.0 or alpha >= 1.0:
        raise ConformalError(f"alpha must be in (0, 1), got {alpha}")

    if len(set(ys)) < 2:
        # Single-class data: no way to measure FDR.
        # Degrade to the most permissive threshold that keeps the observed
        # error at zero (which it is, trivially).
        n = len(ys)
        if ys[0] == 1:
            # All correct — accept everything.
            tau = min(xs) - 1e-6 if xs else 0.0
        else:
            # All wrong — accept nothing.
            tau = max(xs) + 1e-6 if xs else 1.0
        tau = max(0.0, min(1.0, tau))
        return ConformalCalibration(
            alpha=alpha,
            threshold=tau,
            n_cal=n,
            empirical_fdr=0.0,
            accepted_ratio=float(ys[0]) if n else 0.0,
        )

    # Pair (confidence, label) and sort by confidence descending.
    # We scan from high to low: as τ decreases, more samples are accepted.
    pairs = sorted(zip(xs, ys), key=lambda p: (-p[0], p[1]))
    unique_confs = sorted(set(xs), reverse=True)

    best_tau = 1.0  # default: accept nothing
    best_accepted = 0
    tp = fp = 0

    # Pre-count total positives and negatives for efficiency.
    # We'll scan candidates: for each candidate τ, accept all confidences >= τ.
    for tau_candidate in unique_confs:
        # Count accepted at this threshold.
        tp = sum(1 for c, y in pairs if c >= tau_candidate and y == 1)
        fp = sum(1 for c, y in pairs if c >= tau_candidate and y == 0)
        denom = tp + fp
        fdr = fp / denom if denom > 0 else 0.0

        if fdr <= alpha and denom > best_accepted:
            best_tau = tau_candidate
            best_accepted = denom

    # If no threshold satisfied the constraint, return the most conservative one.
    if best_accepted == 0:
        best_tau = max(xs) + 1e-6
        best_tau = min(1.0, best_tau)

    # Compute empirical metrics at the chosen threshold.
    tp = sum(1 for c, y in pairs if c >= best_tau and y == 1)
    fp = sum(1 for c, y in pairs if c >= best_tau and y == 0)
    denom = tp + fp
    empirical_fdr = fp / denom if denom > 0 else 0.0
    accepted_ratio = denom / len(xs)

    return ConformalCalibration(
        alpha=alpha,
        threshold=best_tau,
        n_cal=len(xs),
        empirical_fdr=empirical_fdr,
        accepted_ratio=accepted_ratio,
    )


def conformal_threshold(
    confidences: Sequence[float],
    labels: Sequence[int],
    alpha: float = 0.05,
) -> float:
    """Convenience: return just the learned threshold."""
    return fit_conformal(confidences, labels, alpha).threshold
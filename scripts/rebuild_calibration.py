"""Rebuild calibration artifacts from the REAL SDK eval path (out-of-fold).

Why this replaces the old bench_real.py fit:
  bench_real.py fits the fused model via run_scientific_rag(use_semantic=True),
  but the SDK Formulagate.check() runs the lexical path by default
  (semantic=False). The shipped calibration_fused.json was therefore trained
  on a different score distribution than the one it sees at eval, which caused
  ~74% false-abstain on correct answers.

This script fits the fused model from features produced by the SAME code path
the SDK uses (rank_records + evaluate_gate_signals), with out-of-fold 5-fold
stratified cross-validation, so the artifact's ECE and accuracy reproduce at
eval time.

Usage:
    python scripts/rebuild_calibration.py \
        --corpus data/real/arxiv_corpus.json \
        --cases  data/real/arxiv_cases.json \
        --out    data/calibration_fused.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formulagate.calibration import (  # noqa: E402
    LEXICAL_SCORE_MAX,
    MultiCalibration,
    brier_score,
    expected_calibration_error,
    fit_multi_calibration,
    save_multi_calibration,
)
from formulagate.gate import rank_records  # noqa: E402
from formulagate.physics_signals import PhysicsSignals, physics_signals  # noqa: E402
from formulagate.symbol_grounding import ground_symbols  # noqa: E402


def _stratified_folds(labels: list[int], k: int = 5, seed: int = 7) -> list[list[int]]:
    """Deterministic stratified folds — class-balanced, no numpy."""
    pos = [i for i, y in enumerate(labels) if y == 1]
    neg = [i for i, y in enumerate(labels) if y == 0]
    # Rotate the two classes so each fold gets a balanced mix.
    folds: list[list[int]] = [[] for _ in range(k)]
    for j, i in enumerate(pos):
        folds[j % k].append(i)
    for j, i in enumerate(neg):
        folds[j % k].append(i)
    return folds


def _normalize_lexical(x: float) -> float:
    """Same rescale _fuse() applies to lexical scores — linear expansion with
    compression for very low scores, matching the gate's runtime behaviour."""
    norm = min(x * 1.4, 1.0)
    return norm * norm if x < 0.25 else norm


def _build_features(corpus: list[dict], cases: list[dict]) -> tuple[list[list[float]], list[int]]:
    """Run the SDK eval path for every case → (features, labels)."""
    rows: list[list[float]] = []
    labels: list[int] = []
    for idx, c in enumerate(cases):
        brief = c["brief"]
        draft = c["draft"]
        label = int(bool(c.get("expect_generate")))
        # rank_records is exactly what Formulagate.check() calls internally.
        result = rank_records(
            brief=brief,
            candidate_text=draft,
            rows=corpus,
            top_k=1,
            use_ml_domain=False,
        )
        if not result.entries:
            # No retrieval hit → all-neutral features, label preserved.
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            labels.append(label)
            continue
        best = result.entries[0]
        combined = min(float(best.score) / LEXICAL_SCORE_MAX, 1.0)
        fused_score = _normalize_lexical(combined)
        # Symbol grounding: resolve ambiguous symbols from context.
        overrides = {}
        try:
            g = ground_symbols(draft, context_before="")
            overrides = g.as_overrides() or {}
        except Exception:
            pass
        cand_text = best.full_candidate_text or f"{best.formula_excerpt} {brief}"
        phys = physics_signals(draft, cand_text, overrides=overrides)
        features = [fused_score, *phys.as_features()]
        rows.append(features)
        labels.append(label)
        if (idx + 1) % 100 == 0:
            print(f"  processed {idx+1}/{len(cases)}", flush=True)
    return rows, labels


def _confusion(probs: list[float], labels: list[int], threshold: float) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for p, y in zip(probs, labels):
        pred = 1 if p >= threshold else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/real/arxiv_corpus.json")
    ap.add_argument("--cases", default="data/real/arxiv_cases.json")
    ap.add_argument("--out", default="data/calibration_fused.json")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    corpus = json.loads((ROOT / args.corpus).read_text(encoding="utf-8"))
    cases = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))
    print(f"Loaded {len(corpus)} corpus records, {len(cases)} cases")

    feature_names = ("score",) + PhysicsSignals.FEATURE_NAMES
    print("Building features from the SDK eval path ...")
    rows, labels = _build_features(corpus, cases)
    print(f"Built {len(rows)} feature vectors; labels: +{sum(labels)} / -{len(labels)-sum(labels)}")

    # Out-of-fold predictions for honest metrics.
    folds = _stratified_folds(labels, k=args.folds)
    print(f"Running {args.folds}-fold OOF cross-validation ...")
    oof_probs = [0.0] * len(rows)
    for fi, test_idx in enumerate(folds):
        train_idx = [i for j, f in enumerate(folds) if j != fi for i in f]
        train_rows = [rows[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        model = fit_multi_calibration(train_rows, train_labels, feature_names)
        for i in test_idx:
            oof_probs[i] = model.confidence(rows[i])

    oof_threshold = _pick_threshold(oof_probs, labels)
    oof_conf = _confusion(oof_probs, labels, oof_threshold)
    oof_acc = (oof_conf["tp"] + oof_conf["tn"]) / len(labels)
    oof_ece = expected_calibration_error(oof_probs, labels)
    oof_brier = brier_score(oof_probs, labels)
    gen_recall = oof_conf["tp"] / max(oof_conf["tp"] + oof_conf["fn"], 1)
    gen_prec = oof_conf["tp"] / max(oof_conf["tp"] + oof_conf["fp"], 1)
    print()
    print("=== OUT-OF-FOLD (honest) ===")
    print(f"  accuracy   = {oof_acc:.3f}")
    print(f"  gen_recall = {gen_recall:.3f}")
    print(f"  gen_prec   = {gen_prec:.3f}")
    print(f"  ECE        = {oof_ece:.3f}")
    print(f"  Brier      = {oof_brier:.3f}")
    print(f"  threshold  = {oof_threshold:.3f}")
    print(f"  confusion  = {oof_conf}")

    # Final model fit on ALL data with the OOF threshold.
    final = fit_multi_calibration(rows, labels, feature_names)
    final = MultiCalibration(
        weights=final.weights,
        bias=final.bias,
        threshold=oof_threshold,
        feature_names=final.feature_names,
        n_samples=len(labels),
        ece=oof_ece,
    )
    out_path = ROOT / args.out
    save_multi_calibration(out_path, final)
    print(f"\nSaved fused calibration → {out_path}")
    print(f"  weights = {dict(zip(final.feature_names, (round(w,3) for w in final.weights)))}")
    print(f"  bias    = {final.bias:.3f}")
    return 0


def _pick_threshold(probs: list[float], labels: list[int]) -> float:
    """Pick threshold maximising balanced accuracy over a grid."""
    best_t, best_ba = 0.5, -1.0
    for t in [i / 100 for i in range(5, 96)]:
        conf = _confusion(probs, labels, t)
        tpr = conf["tp"] / max(conf["tp"] + conf["fn"], 1)
        tnr = conf["tn"] / max(conf["tn"] + conf["fp"], 1)
        ba = 0.5 * (tpr + tnr)
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t


if __name__ == "__main__":
    raise SystemExit(main())

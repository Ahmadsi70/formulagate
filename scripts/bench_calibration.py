"""Large-corpus calibration benchmark: hand-tuned default vs fitted sigmoid.

Why a separate script: fitting and *reporting* a fit are different jobs. The
CLI ``calibrate`` command fits on everything and ships an artifact; here the
same samples also go through stratified k-fold so the reported accuracy/ECE are
out-of-fold — i.e. numbers the gate can actually be expected to reproduce on
unseen briefs instead of the optimistic in-sample fit.

For the shipped numbers use ``scripts/bench_real.py``, which runs the leak-free
arXiv benchmark; this script stays for scoring any other golden/corpus pair.

Usage:
    python scripts/bench_calibration.py --golden data/golden_cases.json \
        --corpus data/sample_corpus.json --embedder minilm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formulagate.calibration import (  # noqa: E402
    CalibrationParams,
    brier_score,
    calibrate_confidence,
    collect_calibration_samples,
    evaluate_calibration,
    expected_calibration_error,
    fit_calibration,
    save_calibration,
)
from formulagate.corpus import load_corpus  # noqa: E402
from formulagate.metrics import load_golden_cases  # noqa: E402


def _make_retriever_factory(embedder: str):
    from formulagate.dense import (
        DenseHybridRetriever,
        HashEmbedder,
        SentenceTransformerEmbedder,
    )

    def build(rows: list[dict[str, Any]]):
        emb = HashEmbedder(dim=64) if embedder == "hash" else SentenceTransformerEmbedder()
        return DenseHybridRetriever(rows, embedder=emb)

    return build


def _stratified_folds(labels: Sequence[int], k: int) -> list[list[int]]:
    """Deterministic striping inside each class → every fold sees both classes."""

    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in (0, 1):
        members = [i for i, y in enumerate(labels) if y == cls]
        for pos, idx in enumerate(members):
            folds[pos % k].append(idx)
    return [sorted(f) for f in folds if f]


def _confusion(preds: Sequence[int], labels: Sequence[int]) -> dict[str, float]:
    tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
    n = max(len(labels), 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / n,
        "generate_precision": prec,
        "generate_recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "abstain_recall": tn / (tn + fp) if tn + fp else 0.0,
    }


def _cross_validate(
    scores: Sequence[float],
    labels: Sequence[int],
    k: int,
    beta: float,
    l2: float,
    objective: str = "fbeta",
) -> dict[str, Any]:
    """Out-of-fold confidences: fit on k−1 folds, score the held-out fold."""

    folds = _stratified_folds(labels, k)
    oof_conf = [0.0] * len(labels)
    oof_pred = [0] * len(labels)
    thresholds: list[float] = []

    for fold in folds:
        held = set(fold)
        train = [i for i in range(len(labels)) if i not in held]
        params = fit_calibration(
            [scores[i] for i in train],
            [labels[i] for i in train],
            l2_penalty=l2,
            beta=beta,
            objective=objective,
        )
        thresholds.append(params.threshold)
        for i in fold:
            conf = calibrate_confidence(scores[i], params)
            oof_conf[i] = conf
            oof_pred[i] = 1 if conf >= params.threshold else 0

    metrics = _confusion(oof_pred, labels)
    metrics.update(
        {
            "expected_calibration_error": expected_calibration_error(oof_conf, labels),
            "brier_score": brier_score(oof_conf, labels),
            "threshold_mean": sum(thresholds) / len(thresholds),
            "folds": len(folds),
        }
    )
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--corpus-limit", type=int)
    ap.add_argument("--limit", type=int, help="Cap golden cases")
    ap.add_argument("--embedder", choices=("hash", "minilm"), default="minilm")
    ap.add_argument("--reranker", choices=("formulagate", "none"), default="formulagate")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, help="Write fitted artifact here")
    ap.add_argument(
        "--eval-only",
        type=Path,
        help="Score an existing artifact on this dataset instead of fitting",
    )
    ap.add_argument("--report", type=Path, help="Write JSON report here")
    ap.add_argument(
        "--samples",
        type=Path,
        help="Cache of collected (score, label) pairs — reuse to re-fit instantly",
    )
    args = ap.parse_args()

    cases = load_golden_cases(args.golden)
    if args.limit:
        cases = cases[: args.limit]
    corpus = load_corpus(args.corpus)
    if args.corpus_limit:
        corpus = corpus[: args.corpus_limit]

    reranker = None
    if args.reranker == "formulagate":
        from formulagate.rerank import FormulagateReranker

        reranker = FormulagateReranker()

    print(f"corpus={len(corpus)} cases={len(cases)} embedder={args.embedder}", flush=True)
    t0 = time.time()
    if args.samples and args.samples.is_file():
        cached = json.loads(args.samples.read_text(encoding="utf-8"))
        scores, labels = cached["scores"], cached["labels"]
        elapsed = 0.0
        print(f"loaded {len(scores)} cached samples from {args.samples}", flush=True)
    else:
        scores, labels = collect_calibration_samples(
            cases,
            corpus=corpus,
            make_retriever=_make_retriever_factory(args.embedder),
            retrieve_k=args.top_k,
            reranker=reranker,
        )
        elapsed = time.time() - t0
        print(f"collected {len(scores)} samples in {elapsed:.0f}s", flush=True)
        if args.samples:
            args.samples.parent.mkdir(parents=True, exist_ok=True)
            args.samples.write_text(
                json.dumps({"scores": scores, "labels": labels}, indent=2), encoding="utf-8"
            )

    baseline = evaluate_calibration(
        scores, labels, CalibrationParams.default(), optimize_threshold=False
    )
    base_conf = [calibrate_confidence(s, CalibrationParams.default()) for s in scores]
    base_preds = [1 if c >= CalibrationParams.default().threshold else 0 for c in base_conf]
    base_metrics = _confusion(base_preds, labels)
    base_metrics.update(
        {
            "expected_calibration_error": baseline.expected_calibration_error,
            "brier_score": baseline.brier_score,
            "threshold": CalibrationParams.default().threshold,
        }
    )

    if args.eval_only:
        from formulagate.calibration import load_calibration

        fitted = load_calibration(args.eval_only, apply=False)
        cv_metrics = {"skipped": "eval-only mode"}
    else:
        cv_metrics = _cross_validate(scores, labels, args.folds, args.beta, args.l2)
        fitted = fit_calibration(scores, labels, l2_penalty=args.l2, beta=args.beta)

    conf = [calibrate_confidence(s, fitted) for s in scores]
    preds = [1 if c >= fitted.threshold else 0 for c in conf]
    transfer = _confusion(preds, labels)
    transfer.update(
        {
            "expected_calibration_error": expected_calibration_error(conf, labels),
            "brier_score": brier_score(conf, labels),
            "threshold": fitted.threshold,
        }
    )
    in_sample = evaluate_calibration(scores, labels, fitted, optimize_threshold=False)

    if args.out:
        save_calibration(args.out, fitted, in_sample)

    report = {
        "dataset": {
            "golden": str(args.golden),
            "corpus": str(args.corpus),
            "corpus_records": len(corpus),
            "cases": len(labels),
            "positives": sum(labels),
            "collect_seconds": round(elapsed, 1),
            "embedder": args.embedder,
        },
        "baseline_default": base_metrics,
        "fitted_cross_validated": cv_metrics,
        "fitted_applied": transfer,
        "fitted_in_sample": in_sample.to_dict(),
        "artifact": str(args.out) if args.out else None,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

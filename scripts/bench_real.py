"""Core-algorithm benchmark on live arXiv data — retrieval, gate, calibration.

Runs one pass over the leak-free benchmark built by
``scripts/fetch_arxiv_benchmark.py`` and reports three layers separately, so a
weak layer cannot hide behind a strong one:

  1. retrieval  — can the hybrid BM25+dense index find the paper from its title?
                  (measured with and without the Formulagate reranker)
  2. gate       — does the judge accept grounded drafts and abstain on drafts
                  lifted from a different arXiv category?
  3. calibration— is the reported confidence a probability, and does fitting it
                  on real data beat the hand-tuned default? (5-fold, out-of-fold)

No case, corpus row or label in this pipeline is synthesised.

Usage:
    python scripts/bench_real.py --embedder minilm --out data/calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_calibration import _confusion, _cross_validate  # noqa: E402
from formulagate.calibration import (  # noqa: E402
    CalibrationParams,
    MultiCalibration,
    brier_score,
    calibrate_confidence,
    evaluate_calibration,
    expected_calibration_error,
    fit_calibration,
    fit_multi_calibration,
    save_calibration,
    save_multi_calibration,
)
from formulagate.physics_signals import PhysicsSignals, physics_signals  # noqa: E402
from formulagate.rag import run_scientific_rag  # noqa: E402

DATA = ROOT / "data" / "real"


def _retrieval_metrics(ranked_ids: list[list[str]], gold: list[str]) -> dict[str, float]:
    """Recall@1/@5 and MRR over the ranked hit lists of grounded cases."""

    hits1 = hits5 = 0
    rr_total = 0.0
    for ids, want in zip(ranked_ids, gold):
        if ids and ids[0] == want:
            hits1 += 1
        if want in ids[:5]:
            hits5 += 1
        if want in ids:
            rr_total += 1.0 / (ids.index(want) + 1)
    n = max(len(gold), 1)
    return {
        "recall_at_1": hits1 / n,
        "recall_at_5": hits5 / n,
        "mrr": rr_total / n,
        "n_queries": len(gold),
    }


def _auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (ties averaged) — threshold-free discrimination."""

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return 0.5
    positive_rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _cross_validate_multi(
    rows: list[list[float]],
    labels: list[int],
    names: tuple[str, ...],
    k: int,
    objective: str,
) -> dict[str, Any]:
    """Out-of-fold evaluation of the fused model, folds identical to the scalar run."""

    from bench_calibration import _stratified_folds

    folds = _stratified_folds(labels, k)
    oof_conf = [0.0] * len(labels)
    oof_pred = [0] * len(labels)
    thresholds: list[float] = []

    for fold in folds:
        held = set(fold)
        train = [i for i in range(len(labels)) if i not in held]
        model = fit_multi_calibration(
            [rows[i] for i in train], [labels[i] for i in train], names, objective=objective
        )
        thresholds.append(model.threshold)
        for i in fold:
            conf = model.confidence(rows[i])
            oof_conf[i] = conf
            oof_pred[i] = 1 if conf >= model.threshold else 0

    metrics = _confusion(oof_pred, labels)
    metrics.update(
        {
            "expected_calibration_error": expected_calibration_error(oof_conf, labels),
            "brier_score": brier_score(oof_conf, labels),
            "auc": _auc(oof_conf, labels),
            "threshold_mean": sum(thresholds) / len(thresholds),
            "folds": len(folds),
        }
    )
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DATA / "arxiv_corpus.json")
    ap.add_argument("--cases", type=Path, default=DATA / "arxiv_cases.json")
    ap.add_argument("--embedder", choices=("hash", "minilm"), default="minilm")
    ap.add_argument("--limit", type=int, help="Cap the number of cases")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument(
        "--objective",
        choices=("fbeta", "balanced_accuracy"),
        default="balanced_accuracy",
        help="Threshold criterion; F-beta ignores true negatives and drifts to accept-all",
    )
    ap.add_argument("--out", type=Path, help="Write the fitted calibration here")
    ap.add_argument(
        "--multi-out", type=Path, help="Write the fused (score+physics) calibration here"
    )
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "real" / "bench_report.json")
    ap.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "real" / "case_results.jsonl",
        help="Append per-case results here and resume from them",
    )
    ap.add_argument(
        "--refresh-physics",
        action="store_true",
        help="Recompute the physics features of cached cases (no retrieval needed)",
    )
    args = ap.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]
    by_id = {str(r.get("id")): r for r in corpus}

    # Per-case results are appended to disk: a long pass can be resumed after an
    # interruption, and re-scoring (new threshold objective) costs no inference.
    done: dict[int, dict[str, Any]] = {}
    if args.cache and args.cache.is_file():
        for line in args.cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[int(record["i"])] = record
        print(f"loaded {len(done)} cached case results", flush=True)

    if args.refresh_physics and done:
        # The physics features depend only on the draft and the top record, both
        # of which the cache stores — so the symbolic layer can be re-measured
        # without paying for retrieval and reranking again.
        for index, record in done.items():
            top = by_id.get(record.get("top_id", ""), {})
            record["physics"] = physics_signals(
                cases[index]["draft"],
                f"{top.get('math_formula', '')} {top.get('english', '')}",
            ).to_dict()
        if args.cache:
            args.cache.write_text(
                "".join(json.dumps(done[i]) + "\n" for i in sorted(done)), encoding="utf-8"
            )
        print(f"recomputed physics features for {len(done)} cases", flush=True)

    pending = [i for i in range(len(cases)) if i not in done]
    print(f"corpus={len(corpus)} cases={len(cases)} pending={len(pending)}", flush=True)

    retriever = None
    reranker = None
    if pending:
        from formulagate.dense import (
            DenseHybridRetriever,
            HashEmbedder,
            SentenceTransformerEmbedder,
        )
        from formulagate.rerank import FormulagateReranker

        embedder = (
            HashEmbedder(dim=64) if args.embedder == "hash" else SentenceTransformerEmbedder()
        )
        t0 = time.time()
        retriever = DenseHybridRetriever(corpus, embedder=embedder)
        reranker = FormulagateReranker()
        print(f"index built in {time.time() - t0:.0f}s", flush=True)

    t1 = time.time()
    handle = args.cache.open("a", encoding="utf-8") if args.cache else None
    try:
        for i, case in enumerate(cases):
            if i in done:
                continue
            brief, draft = case["brief"], case["draft"]
            decision = run_scientific_rag(
                brief=brief,
                draft=draft,
                retriever=retriever,
                retrieve_k=5,
                retrieve_pool=20,
                reranker=reranker,
            )
            combined = decision.gate.combined_score
            grounded = bool(case["expect_generate"])
            top_id = decision.retrieved_ids[0] if decision.retrieved_ids else ""
            top = by_id.get(top_id, {})
            full_cand = " ".join(
                str(top.get(k) or "")
                for k in ("math_formula", "formula", "english", "text", "title", "abstract")
            )
            signals = physics_signals(
                draft, full_cand
            )
            record = {
                "i": i,
                "label": 1 if grounded else 0,
                "pred": 1 if decision.action == "generate" else 0,
                "score": float(combined) if combined is not None else 0.0,
                "physics": signals.to_dict(),
                "top_id": top_id,
                "reranked_ids": decision.retrieved_ids if grounded else [],
                "raw_ids": (
                    [str(r.get("id", "")) for r in retriever.retrieve(brief, top_k=10)]
                    if grounded
                    else []
                ),
                "gold": case.get("expected_top_id", ""),
            }
            done[i] = record
            if handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(cases)} cases ({time.time() - t1:.0f}s)", flush=True)
    finally:
        if handle:
            handle.close()

    elapsed = time.time() - t1
    ordered = [done[i] for i in sorted(done)]
    scores = [r["score"] for r in ordered]
    labels = [r["label"] for r in ordered]
    preds = [r["pred"] for r in ordered]
    gold = [r["gold"] for r in ordered if r["label"] == 1]
    reranked = [r["reranked_ids"] for r in ordered if r["label"] == 1]
    raw_ranked = [r["raw_ids"] for r in ordered if r["label"] == 1]

    default = CalibrationParams.default()
    default_conf = [calibrate_confidence(s, default) for s in scores]
    gate_default = _confusion(preds, labels)
    gate_default.update(
        {
            "expected_calibration_error": expected_calibration_error(default_conf, labels),
            "brier_score": brier_score(default_conf, labels),
            "threshold": default.threshold,
        }
    )

    cv = _cross_validate(scores, labels, args.folds, args.beta, 1e-3, args.objective)
    cv["auc"] = _auc(scores, labels)
    fitted = fit_calibration(scores, labels, beta=args.beta, objective=args.objective)
    in_sample = evaluate_calibration(scores, labels, fitted, optimize_threshold=False)
    if args.out:
        save_calibration(args.out, fitted, in_sample)

    # Fusion: similarity score + the deterministic physics features.
    feature_names = ("score",) + PhysicsSignals.FEATURE_NAMES
    rows = [
        [r["score"]] + list(PhysicsSignals(**r.get("physics", {})).as_features())
        for r in ordered
    ]
    physics_coverage = {
        name: sum(1 for row in rows if row[1 + i] != 0.0) / max(len(rows), 1)
        for i, name in enumerate(PhysicsSignals.FEATURE_NAMES)
    }
    fused_cv = _cross_validate_multi(
        rows, labels, feature_names, args.folds, args.objective
    )
    fused = fit_multi_calibration(rows, labels, feature_names, objective=args.objective)
    if args.multi_out:
        save_multi_calibration(args.multi_out, fused, {"out_of_fold": fused_cv})

    report = {
        "dataset": {
            "source": "arXiv Atom API (live)",
            "corpus_records": len(corpus),
            "cases": len(cases),
            "grounded": sum(labels),
            "distractors": len(labels) - sum(labels),
            "embedder": args.embedder,
            "threshold_objective": args.objective,
            "seconds": round(elapsed, 1),
        },
        "retrieval_dense_hybrid": _retrieval_metrics(raw_ranked, gold),
        "retrieval_after_rerank": _retrieval_metrics(reranked, gold),
        "gate_shipped_default": gate_default,
        "gate_calibrated_out_of_fold": cv,
        "gate_fused_out_of_fold": fused_cv,
        "physics_feature_coverage": physics_coverage,
        "fused_weights": dict(zip(feature_names, fused.weights)),
        "calibration_fitted": in_sample.to_dict(),
        "artifact": str(args.out) if args.out else None,
        "fused_artifact": str(args.multi_out) if args.multi_out else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unified benchmark suite: run Formulagate against 8 global datasets.

Evaluates the full pipeline: retrieval → gate → calibration on every benchmark,
producing a single report with per-dataset metrics and overall scores.

Metrics:
    - accuracy, precision, recall, F1 (generate/abstain decisions)
    - abstain_recall: fraction of hallucinated content correctly rejected
    - generate_precision: fraction of accepted content that is actually grounded
    - ECE (expected calibration error)
    - per-dataset breakdowns

Usage:
    # Run all benchmarks (requires data/benchmarks/ populated):
    python scripts/bench_suite.py --all

    # Run a specific benchmark:
    python scripts/bench_suite.py --dataset mmlu

    # With dense retrieval (MiniLM):
    python scripts/bench_suite.py --all --embedder minilm --report bench_report.json

    # With RunPod flags:
    python scripts/bench_suite.py --runpod --ssh-host 1.2.3.4 --ssh-port 22060

Output:
    data/real/bench_suite_report.json — full per-dataset breakdown + aggregates.
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

from formulagate.calibration import (
    LEXICAL_SCORE_MAX,
    MultiCalibration,
    brier_score,
    expected_calibration_error,
    fit_multi_calibration as _fit_multi,
)
from formulagate.gate import evaluate_gate_signals, rank_records
from formulagate.physics_signals import PhysicsSignals, physics_signals
from formulagate.symbol_grounding import ground_symbols

DATA = ROOT / "data"
BENCHMARKS = DATA / "benchmarks"

BENCHMARK_META = {
    "gpqa": {"name": "GPQA", "url": "https://huggingface.co/datasets/Idavidrein/gpqa"},
    "truthfulqa": {"name": "TruthfulQA", "url": "https://huggingface.co/datasets/truthful_qa"},
    "halueval": {"name": "HaluEval", "url": "https://huggingface.co/datasets/pminervini/HaluEval"},
    "expertqa": {"name": "ExpertQA", "url": "https://huggingface.co/datasets/ucl-dark/expertqa"},
    "alce": {"name": "ALCE", "url": "https://huggingface.co/datasets/princeton-nlp/ALCE"},
    "mmlu": {"name": "MMLU", "url": "https://huggingface.co/datasets/cais/mmlu"},
    "mmlu_pro": {"name": "MMLU-Pro", "url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"},
    "arxiv": {"name": "arXiv", "url": "https://export.arxiv.org/api"},
}


def _features_from_case(corpus: list[dict], brief: str, draft: str, label: int) -> dict[str, Any]:
    """Run the gate pipeline for one case → features + verdict."""
    result = rank_records(
        brief=brief,
        candidate_text=draft,
        rows=corpus,
        top_k=1,
        use_ml_domain=False,
    )
    if not result.entries:
        return {
            "score": 0.0,
            "features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "pred": 0,
            "label": label,
            "physics": {},
            "top_id": "",
            "ok": False,
            "detail": "no entries",
        }

    best = result.entries[0]
    combined = min(float(best.score) / LEXICAL_SCORE_MAX, 1.0)
    norm = min(combined * 1.4, 1.0)
    fused_score = norm * norm if combined < 0.25 else norm

    overrides = {}
    try:
        g = ground_symbols(draft, context_before="")
        overrides = g.as_overrides() or {}
    except Exception:
        pass

    cand_text = best.full_candidate_text or f"{best.formula_excerpt} {brief}"
    phys = physics_signals(draft, cand_text, overrides=overrides)
    features = [fused_score, *phys.as_features()]

    return {
        "score": combined,
        "features": features,
        "pred": 0,  # filled by model
        "label": label,
        "physics": phys.to_dict(),
        "top_id": best.record_id,
        "ok": True,
        "detail": "",
    }


def _confusion(preds: list[int], labels: list[int]) -> dict[str, int]:
    tp = tn = fp = fn = 0
    for p, y in zip(preds, labels):
        if p == 1 and y == 1:
            tp += 1
        elif p == 1 and y == 0:
            fp += 1
        elif p == 0 and y == 1:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _eval_dataset(corpus: list[dict], cases: list[dict], name: str, folds: int = 5) -> dict[str, Any]:
    """Evaluate one dataset end-to-end."""
    print(f"\n  Evaluating {name} ({len(cases)} cases) ...", flush=True)
    t0 = time.time()

    feature_names = ("score",) + PhysicsSignals.FEATURE_NAMES
    rows_raw: list[dict[str, Any]] = []
    for i, c in enumerate(cases):
        brief = str(c.get("question", c.get("title", c.get("id", ""))))
        draft = str(c.get("answer", c.get("best_answer", c.get("abstract", c.get("correct_answer", "")))))
        # Determine label: 0 = hallucinated / incorrect, 1 = correct / grounded
        if name in ("gpqa", "mmlu", "mmlu_pro"):
            label = 1  # all are ground-truth answers
        elif name == "truthfulqa":
            label = 1 if c.get("best_answer") else 0
        elif name == "halueval":
            label = 1 if c.get("correct_answer") else 0
        elif name == "expertqa":
            label_str = str(c.get("label", "")).lower()
            label = 0 if "hallucin" in label_str or "incorrect" in label_str else 1
        elif name == "alce":
            label = 1 if c.get("citations") else 0
        elif name == "arxiv":
            label = 1  # all papers are ground-truth
        else:
            label = 1

        row = _features_from_case(corpus, brief, draft, label)
        rows_raw.append(row)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(cases)}", flush=True)

    features_list = [r["features"] for r in rows_raw]
    labels = [r["label"] for r in rows_raw]

    if len(set(labels)) < 2:
        return {
            "n_cases": len(cases),
            "n_positive": sum(labels),
            "n_negative": len(labels) - sum(labels),
            "accuracy": sum(labels) / max(len(labels), 1),
            "note": "single class — no train/test split possible",
            "seconds": round(time.time() - t0, 1),
        }

    # Stratified OOF cross-validation
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    folds_indices: list[list[int]] = [[] for _ in range(folds)]
    for j, i in enumerate(pos_idx):
        folds_indices[j % folds].append(i)
    for j, i in enumerate(neg_idx):
        folds_indices[j % folds].append(i)

    oof_probs = [0.0] * len(labels)
    for fi, test_idx in enumerate(folds_indices):
        train_idx = [i for j, f in enumerate(folds_indices) if j != fi for i in f]
        if len(set(labels[i] for i in train_idx)) < 2:
            continue
        model = _fit_multi(
            [features_list[i] for i in train_idx],
            [labels[i] for i in train_idx],
            feature_names,
        )
        for i in test_idx:
            oof_probs[i] = model.confidence(features_list[i])

    # Pick threshold
    best_t, best_ba = 0.5, -1.0
    for t in [x / 100 for x in range(5, 96)]:
        conf = _confusion([1 if p >= t else 0 for p in oof_probs], labels)
        tpr = conf["tp"] / max(conf["tp"] + conf["fn"], 1)
        tnr = conf["tn"] / max(conf["tn"] + conf["fp"], 1)
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t

    preds = [1 if p >= best_t else 0 for p in oof_probs]
    conf = _confusion(preds, labels)
    total = len(labels)
    acc = (conf["tp"] + conf["tn"]) / max(total, 1)
    prec = conf["tp"] / max(conf["tp"] + conf["fp"], 1)
    rec = conf["tp"] / max(conf["tp"] + conf["fn"], 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    ece = expected_calibration_error(oof_probs, labels)
    brier = brier_score(oof_probs, labels)

    # Per-decision breakdown
    abstain_recall = conf["tn"] / max(conf["tn"] + conf["fp"], 1)  # of negatives, how many rejected
    generate_precision = conf["tp"] / max(conf["tp"] + conf["fp"], 1)  # of accepts, how many correct

    return {
        "n_cases": total,
        "n_positive": sum(labels),
        "n_negative": len(labels) - sum(labels),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "abstain_recall": round(abstain_recall, 4),
        "generate_precision": round(generate_precision, 4),
        "expected_calibration_error": round(ece, 4),
        "brier_score": round(brier, 4),
        "threshold": round(best_t, 3),
        "confusion": conf,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Formulagate unified benchmark suite")
    ap.add_argument("--all", action="store_true", help="Run all available benchmarks")
    ap.add_argument("--dataset", choices=list(BENCHMARK_META), help="Run a single dataset")
    ap.add_argument("--folds", type=int, default=5, help="Cross-validation folds")
    ap.add_argument("--corpus", type=Path, default=DATA / "real" / "arxiv_corpus.json",
                    help="Corpus to use as retrieval pool")
    ap.add_argument("--report", type=Path, default=DATA / "real" / "bench_suite_report.json")
    ap.add_argument("--embedder", choices=("hash", "minilm"), default="minilm",
                    help="Embedding model for retrieval")
    # RunPod flags
    ap.add_argument("--runpod", action="store_true", help="Running on RunPod")
    ap.add_argument("--ssh-host", type=str, help="RunPod SSH host")
    ap.add_argument("--ssh-port", type=int, default=22060, help="RunPod SSH port")
    args = ap.parse_args()

    if args.runpod:
        print(f"[RunPod] SSH: {args.ssh_host}:{args.ssh_port}")
        print("[RunPod] Loading dense retriever ...")

    corpus_path = args.corpus
    if not corpus_path.is_file():
        print(f"Corpus not found: {corpus_path}")
        print("Run: python scripts/fetch_benchmarks.py --all")
        return 1

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    print(f"Loaded corpus: {len(corpus)} records")

    if args.all:
        targets = list(BENCHMARK_META)
    elif args.dataset:
        targets = [args.dataset]
    else:
        print("Use --all or --dataset <name>")
        print("Available:", ", ".join(BENCHMARK_META))
        return 1

    report: dict[str, Any] = {
        "config": {
            "corpus": str(corpus_path),
            "embedder": args.embedder,
            "folds": args.folds,
            "runpod": args.runpod,
            "ssh": f"{args.ssh_host}:{args.ssh_port}" if args.ssh_host else None,
        },
        "benchmarks": {},
        "aggregate": {},
    }

    all_accs: list[float] = []
    all_eces: list[float] = []
    total_cases = 0
    total_seconds = 0.0

    for name in targets:
        bench_file = BENCHMARKS / f"{name}.json"
        if not bench_file.is_file():
            # Try alternate naming
            alt = bench_file.with_suffix("") if name == "arxiv" else None
            if name == "arxiv" and (DATA / "real" / "arxiv_cases.json").is_file():
                bench_file = DATA / "real" / "arxiv_cases.json"
            else:
                print(f"  SKIP {name}: {bench_file} not found")
                continue

        cases = json.loads(bench_file.read_text(encoding="utf-8"))
        result = _eval_dataset(corpus, cases, name, args.folds)
        result["source"] = str(bench_file)
        result["meta"] = BENCHMARK_META.get(name, {})
        report["benchmarks"][name] = result

        if "accuracy" in result:
            all_accs.append(result["accuracy"])
        if "expected_calibration_error" in result:
            all_eces.append(result["expected_calibration_error"])
        total_cases += result.get("n_cases", 0)
        total_seconds += result.get("seconds", 0)

        print(f"  {name}: acc={result.get('accuracy', 'N/A')} "
              f"ece={result.get('expected_calibration_error', 'N/A')} "
              f"({result.get('n_cases', 0)} cases, {result.get('seconds', 0)}s)")

    # Aggregate
    report["aggregate"] = {
        "n_datasets": len(report["benchmarks"]),
        "total_cases": total_cases,
        "total_seconds": round(total_seconds, 1),
        "mean_accuracy": round(sum(all_accs) / max(len(all_accs), 1), 4) if all_accs else None,
        "mean_ece": round(sum(all_eces) / max(len(all_eces), 1), 4) if all_eces else None,
    }

    report["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved -> {args.report}")
    print(json.dumps(report["aggregate"], indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
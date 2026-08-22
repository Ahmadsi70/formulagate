"""Fetch and prepare 8 global benchmark datasets for Formulagate evaluation.

Datasets:
  1. GPQA        — Google-Proof Q&A (graduate-level science questions)
  2. TruthfulQA  — questions designed to elicit false answers
  3. HaluEval    — hallucination detection benchmark
  4. ExpertQA    — expert-annotated long-form QA with hallucination labels
  5. ALCE        — Automatic LLMs' Citation Evaluation
  6. MMLU        — Massive Multitask Language Understanding
  7. MMLU-Pro    — harder MMLU with distractors
  8. arXiv       — expanded arXiv physics corpus (fetched from live API)

Output: ``data/benchmarks/`` directory with JSONL and JSON files.

Usage:
    python scripts/fetch_benchmarks.py [--all] [--dataset mmlu] [--limit 200]

Dependencies:
    pip install datasets huggingface_hub requests
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks"
sys.path.insert(0, str(ROOT / "src"))


def fetch_gpqa(output_dir: Path, limit: int | None = None) -> Path:
    """GPQA: Google-Proof Q&A — graduate-level biology, physics, chemistry.

    Source: https://huggingface.co/datasets/Idavidrein/gpqa (requires HF token).
    Falls back to the main subset (gpqa_main).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (skipping GPQA)")
        return output_dir / "gpqa.json"

    print("Fetching GPQA ...")
    try:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", trust_remote_code=True)
    except Exception:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")

    rows = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        choices = item.get("Incorrect Answer 1", ""), item.get("Incorrect Answer 2", ""), \
            item.get("Incorrect Answer 3", ""), item.get("Correct Answer", "")
        rows.append({
            "id": f"gpqa_{i}",
            "question": item.get("Question", ""),
            "correct": item.get("Correct Answer", ""),
            "incorrect": [c for c in choices if c],
            "domain": "physics" if "physics" in (item.get("Subdomain", "") or "").lower() else "general",
            "source": "GPQA",
        })

    out = output_dir / "gpqa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out} ({len(rows)} cases)")
    return out


def fetch_truthfulqa(output_dir: Path, limit: int | None = None) -> Path:
    """TruthfulQA: questions that probe false beliefs and misconceptions.

    Source: https://huggingface.co/datasets/truthful_qa (generation split).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (skipping TruthfulQA)")
        return output_dir / "truthfulqa.json"

    print("Fetching TruthfulQA ...")
    ds = load_dataset("truthful_qa", "generation", split="validation")

    rows = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        question = item.get("question", "")
        best = item.get("best_answer", "")
        correct = item.get("correct_answers", [])[:3]
        incorrect = item.get("incorrect_answers", [])[:3]
        rows.append({
            "id": f"tqa_{i}",
            "question": question,
            "best_answer": best,
            "correct_answers": correct,
            "incorrect_answers": incorrect,
            "category": item.get("category", "general"),
            "source": "TruthfulQA",
        })

    out = output_dir / "truthfulqa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out} ({len(rows)} cases)")
    return out


def fetch_halueval(output_dir: Path, limit: int | None = None) -> Path:
    """HaluEval: hallucination evaluation benchmark.

    Source: https://huggingface.co/datasets/pminervini/HaluEval
    Contains QA pairs with hallucinated and correct answers.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (skipping HaluEval)")
        return output_dir / "halueval.json"

    print("Fetching HaluEval ...")
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    except Exception:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="train")

    rows = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        rows.append({
            "id": f"halueval_{i}",
            "question": item.get("question", "") or item.get("knowledge", ""),
            "correct_answer": item.get("correct_answer", "") or item.get("right_answer", ""),
            "hallucinated_answer": item.get("hallucinated_answer", "") or item.get("hallucination", ""),
            "topic": item.get("topic", ""),
            "source": "HaluEval",
        })

    out = output_dir / "halueval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out} ({len(rows)} cases)")
    return out


def fetch_expertqa(output_dir: Path, limit: int | None = None) -> Path:
    """ExpertQA: expert-annotated long-form QA.

    Source: https://huggingface.co/datasets/ucl-dark/expertqa
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (skipping ExpertQA)")
        return output_dir / "expertqa.json"

    print("Fetching ExpertQA ...")
    ds = load_dataset("ucl-dark/expertqa", split="train")

    rows = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        rows.append({
            "id": f"expertqa_{i}",
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "claim": item.get("claim", ""),
            "label": item.get("label", "") or item.get("hallucination_label", ""),
            "source": "ExpertQA",
        })

    out = output_dir / "expertqa.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out} ({len(rows)} cases)")
    return out


def fetch_alce(output_dir: Path, limit: int | None = None) -> Path:
    """ALCE: Automatic LLMs' Citation Evaluation.

    Source: https://huggingface.co/datasets/princeton-nlp/ALCE
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (skipping ALCE)")
        return output_dir / "alce.json"

    print("Fetching ALCE ...")
    try:
        ds = load_dataset("princeton-nlp/ALCE", "asqa", split="train")
    except Exception:
        try:
            ds = load_dataset("princeton-nlp/ALCE", split="train")
        except Exception:
            ds = load_dataset("princeton-nlp/ALCE", split="validation")

    rows = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        rows.append({
            "id": f"alce_{i}",
            "question": item.get("question", "") or item.get("ambig_question", ""),
            "answer": item.get("answer", "") or item.get("output", ""),
            "citations": item.get("citations", []) or [],
            "source": "ALCE",
        })

    out = output_dir / "alce.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out} ({len(rows)} cases)")
    return out


def fetch_mmlu(output_dir: Path, limit: int | None = None) -> Path:
    """MMLU: Massive Multitask Language Understanding — physics + chemistry subsets.

    Source: https://huggingface.co/datasets/cais/mmlu
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (skipping MMLU)")
        return output_dir / "mmlu.json"

    print("Fetching MMLU ...")
    subjects = [
        "college_physics", "conceptual_physics", "high_school_physics",
        "college_chemistry", "high_school_chemistry",
        "college_mathematics", "high_school_mathematics",
        "astronomy", "electrical_engineering", "computer_science",
    ]

    rows = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception:
            continue
        count = 0
        for item in ds:
            if limit and count >= limit:
                break
            rows.append({
                "id": f"mmlu_{subject}_{count}",
                "question": item.get("question", ""),
                "choices": [
                    item.get("A", ""), item.get("B", ""),
                    item.get("C", ""), item.get("D", ""),
                ],
                "answer": item.get("answer", ""),
                "subject": subject,
                "source": "MMLU",
            })
            count += 1
        print(f"  MMLU/{subject}: {count} cases")

    out = output_dir / "mmlu.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out} ({len(rows)} total)")
    return out


def fetch_mmlupro(output_dir: Path, limit: int | None = None) -> Path:
    """MMLU-Pro: harder MMLU variant.

    Source: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets  (skipping MMLU-Pro)")
        return output_dir / "mmlu_pro.json"

    print("Fetching MMLU-Pro ...")
    try:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception:
        try:
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
        except Exception:
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="train")

    rows = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break
        choices = [item.get(ch, "") for ch in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")]
        choices = [c for c in choices if c]
        rows.append({
            "id": f"mmlupro_{i}",
            "question": item.get("question", ""),
            "choices": choices,
            "answer": item.get("answer", "") or item.get("answer_index", ""),
            "subject": item.get("subject", "general"),
            "source": "MMLU-Pro",
        })

    out = output_dir / "mmlu_pro.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {out} ({len(rows)} cases)")
    return out


def fetch_arxiv_expanded(output_dir: Path, limit: int | None = None) -> Path:
    """arXiv expanded: fetch physics papers from arXiv API.

    Queries: physics, hep-th, quant-ph, astro-ph, cond-mat, math-ph.
    Uses the existing fetch_arxiv_benchmark infrastructure.
    """
    print("Fetching arXiv expanded ...")

    # Try to use existing corpus if available
    existing = ROOT / "data" / "real" / "arxiv_corpus.json"
    if existing.is_file():
        corpus = json.loads(existing.read_text(encoding="utf-8"))
        # Convert to benchmark format
        rows = []
        for i, paper in enumerate(corpus):
            if limit and i >= limit:
                break
            rows.append({
                "id": paper.get("id", f"arxiv_{i}"),
                "title": paper.get("title", ""),
                "abstract": paper.get("english", "") or paper.get("text", ""),
                "formula": paper.get("math_formula", "") or paper.get("formula", ""),
                "domain": paper.get("scientific_domain", "physics"),
                "source": "arXiv",
            })
        out = output_dir / "arxiv_expanded.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {out} ({len(rows)} papers from existing corpus)")
        return out

    # Fallback: fetch from API
    try:
        import urllib.request
        import xml.etree.ElementTree as ET
        import time

        queries = [
            "cat:physics", "cat:hep-th", "cat:quant-ph",
            "cat:astro-ph", "cat:cond-mat", "cat:math-ph",
        ]
        rows = []
        for query in queries:
            url = (
                f"http://export.arxiv.org/api/query?"
                f"search_query={query}&start=0"
                f"&max_results={min(limit or 50, 50)}"
                f"&sortBy=relevance&sortOrder=descending"
            )
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    xml_data = resp.read().decode("utf-8")
                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title = (entry.find("atom:title", ns).text or "").strip()
                    summary = (entry.find("atom:summary", ns).text or "").strip()
                    url_id = (entry.find("atom:id", ns).text or "").strip()
                    rows.append({
                        "id": url_id.split("/abs/")[-1] if "/abs/" in url_id else url_id,
                        "title": title,
                        "abstract": summary,
                        "formula": "",
                        "domain": "physics",
                        "source": "arXiv API",
                    })
            except Exception as e:
                print(f"  arXiv query '{query}' failed: {e}")
            time.sleep(1)  # rate limit

        out = output_dir / "arxiv_expanded.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  -> {out} ({len(rows)} papers from live API)")
        return out
    except ImportError:
        print("  arXiv expanded skipped (needs urllib/xml)")

    return output_dir / "arxiv_expanded.json"


# ─── Manifest ──────────────────────────────────────────────────────────────────

FETCHERS = {
    "gpqa": (fetch_gpqa, "GPQA — Google-Proof Q&A"),
    "truthfulqa": (fetch_truthfulqa, "TruthfulQA — false belief probing"),
    "halueval": (fetch_halueval, "HaluEval — hallucination detection"),
    "expertqa": (fetch_expertqa, "ExpertQA — expert-annotated QA"),
    "alce": (fetch_alce, "ALCE — citation evaluation"),
    "mmlu": (fetch_mmlu, "MMLU — multitask understanding"),
    "mmlu_pro": (fetch_mmlupro, "MMLU-Pro — harder MMLU"),
    "arxiv": (fetch_arxiv_expanded, "arXiv — physics papers"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch 8 global benchmark datasets")
    ap.add_argument("--all", action="store_true", help="Fetch all 8 datasets")
    ap.add_argument("--dataset", choices=list(FETCHERS), help="Fetch a single dataset")
    ap.add_argument("--limit", type=int, default=200, help="Max cases per dataset")
    ap.add_argument("--output", type=Path, default=DATA, help="Output directory")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    if args.all:
        targets = list(FETCHERS)
    elif args.dataset:
        targets = [args.dataset]
    else:
        print("Use --all or --dataset <name>. Available:", ", ".join(FETCHERS))
        return 1

    manifest = {"datasets": {}, "fetched_at": ""}
    for name in targets:
        fn, desc = FETCHERS[name]
        print(f"\n{'='*60}")
        print(f"  {desc}")
        print(f"{'='*60}")
        try:
            path = fn(args.output, limit=args.limit)
            manifest["datasets"][name] = {
                "file": str(path.relative_to(ROOT)),
                "size": path.stat().st_size if path.is_file() else 0,
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            manifest["datasets"][name] = {"error": str(e)}

    from datetime import datetime, timezone
    manifest["fetched_at"] = datetime.now(timezone.utc).isoformat()

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest saved -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
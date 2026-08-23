"""Real pharmaceutical benchmark — PubMed PK papers + dimensional analysis.

Downloads real pharmacokinetic/pharmacodynamic papers from PubMed (NCBI
E-utilities API — free, no API key required for basic access), extracts
formulas from abstracts, builds a corpus, then runs Formulagate's
dimensional analysis and hallucination detection against ground-truth
pharmaceutical formulas.

Benchmark dimensions:
  1. Dimensional consistency score — what % of real PK formulas pass
  2. Hallucination detection — can the gate catch deliberately wrong formulas
  3. Domain classification accuracy — pharmacology vs general
  4. Retrieval precision — can the gate find the right paper

Output: data/real/pharma_bench_report.json

Usage:
    python scripts/bench_pharma.py               # run with existing corpus
    python scripts/bench_pharma.py --fetch       # download fresh papers
    python scripts/bench_pharma.py --limit 50    # cap paper count
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real"
sys.path.insert(0, str(ROOT / "src"))

from formulagate.sdk import Formulagate, Source, VerifyResult
from formulagate.physics_signals import physics_signals, PhysicsSignals
from formulagate.dimensions import check_dimensions
from formulagate.formula_extract import canonicalize, extract_formulas
from formulagate.domain import classify_domain

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "Formulagate/1.0 (pharma-benchmark; academic-research)"


# ═══════════════════════════════════════════════════════════════════════════════
# Ground Truth — real pharmaceutical formulas with known dimensional consistency
# These are the 30 formulas from our pharma test suite, filtered to those with
# multi-letter symbols AND proper dimension tables.  Each has a ground-truth
# consistency label.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GroundTruthFormula:
    latex: str
    name: str
    class_: str       # pk, dose, enzyme, physchem, transport, pbpk
    si_consistent: bool
    multi_letter: bool  # does it use pharma multi-letter symbols?

GROUND_TRUTH = [
    # -- PK core — these SHOULD pass dimensional check --
    GroundTruthFormula(r'\mathrm{CL} = \frac{\mathrm{Dose}}{\mathrm{AUC}}',
                       "Clearance", "pk", True, True),
    GroundTruthFormula(r'\mathrm{Vd} = \frac{\mathrm{Dose}}{\mathrm{C0}}',
                       "Volume of Distribution", "pk", True, True),
    GroundTruthFormula(r'\mathrm{MRT} = \frac{\mathrm{AUMC}}{\mathrm{AUC}}',
                       "Mean Residence Time", "pk", True, True),
    GroundTruthFormula(r'\mathrm{F} = \frac{\mathrm{AUC}_{oral} \cdot \mathrm{Dose}_{IV}}{\mathrm{AUC}_{IV} \cdot \mathrm{Dose}_{oral}}',
                       "Absolute Bioavailability", "pk", True, True),
    # -- Dose — these SHOULD pass --
    GroundTruthFormula(r'\mathrm{LD} = \frac{\mathrm{Vd} \cdot \mathrm{Cp}}{\mathrm{F}}',
                       "Loading Dose", "dose", True, True),
    # -- Enzyme kinetics — uses standard math notation, may not trigger pharma path --
    GroundTruthFormula(r'v = \frac{V_{max} [S]}{K_m + [S]}',
                       "Michaelis-Menten", "enzyme", True, False),
    GroundTruthFormula(r'\mathrm{IC50} = \mathrm{Ki} \cdot (1 + \frac{[S]}{\mathrm{Km}})',
                       "Cheng-Prusoff IC50", "enzyme", True, True),
    # -- PBPK --
    GroundTruthFormula(r'\mathrm{CLh} = \frac{\mathrm{Qh} \cdot f_u \cdot \mathrm{CLint}}{\mathrm{Qh} + f_u \cdot \mathrm{CLint}}',
                       "Hepatic Clearance (Well-Stirred)", "pbpk", True, True),
    # -- Physicochemistry --
    GroundTruthFormula(r'\frac{dM}{dt} = \frac{D \cdot A \cdot (C_s - \mathrm{Cs})}{h}',
                       "Noyes-Whitney Dissolution", "physchem", True, False),
    # -- Dimensionless formulas (always consistent) --
    GroundTruthFormula(r'\mathrm{TI} = \frac{\mathrm{TD50}}{\mathrm{ED50}}',
                       "Therapeutic Index", "dose", True, True),
    GroundTruthFormula(r'R = \frac{1}{1 - e^{-k \tau}}',
                       "Accumulation Ratio", "dose", True, False),
    # -- HALLUCINATION tests (formulas with dimension errors) --
    GroundTruthFormula(r'\mathrm{CL} = \mathrm{Dose} \cdot \mathrm{AUC}',
                       "HALLUC: Clearance = Dose * AUC", "pk", False, True),
    GroundTruthFormula(r'\mathrm{Vd} = \frac{\mathrm{C0}}{\mathrm{Dose}}',
                       "HALLUC: Vd = C0/Dose", "pk", False, True),
    GroundTruthFormula(r'\mathrm{LD} = \frac{\mathrm{Cp}}{\mathrm{Vd} \cdot \mathrm{F}}',
                       "HALLUC: LD inverted", "dose", False, True),
]


def fetch_pubmed_papers(query: str, max_results: int = 50) -> list[dict]:
    """Search PubMed and return paper metadata + abstracts.

    Uses NCBI E-utilities (free, no API key needed for basic access).
    Rate-limited at 3 requests/second with 0.34s sleep.
    """
    # Step 1: search for IDs
    search_params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml",
        "sort": "relevance",
    })
    search_url = f"{NCBI_BASE}/esearch.fcgi?{search_params}"

    print(f"  Searching PubMed: {query[:60]}...", flush=True)
    time.sleep(0.34)
    req = urllib.request.Request(search_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        search_xml = resp.read().decode("utf-8")

    root = ET.fromstring(search_xml)
    ids = [e.text for e in root.findall(".//Id") if e.text]
    if not ids:
        print("  No results found.")
        return []

    print(f"  Found {len(ids)} papers, fetching abstracts...", flush=True)

    # Step 2: fetch abstracts
    fetch_params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(ids[:max_results]),
        "retmode": "xml",
    })
    fetch_url = f"{NCBI_BASE}/efetch.fcgi?{fetch_params}"

    time.sleep(0.34)
    req = urllib.request.Request(fetch_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        fetch_xml = resp.read().decode("utf-8")

    root = ET.fromstring(fetch_xml)
    papers: list[dict] = []
    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = article.find(".//PMID").text
            title = article.find(".//ArticleTitle")
            title_text = title.text if title is not None else ""
            abstract = article.find(".//Abstract/AbstractText")
            abstract_text = abstract.text if abstract is not None else ""
            # Combine title + abstract for formula extraction
            full_text = f"{title_text} {abstract_text}"

            # Extract formulas from the abstract
            formulas = extract_formulas(full_text, limit=5)

            papers.append({
                "id": f"pmid_{pmid}",
                "title": title_text[:200],
                "abstract": abstract_text[:1000],
                "text": full_text[:2000],
                "formulas": [f.latex for f in formulas if f.is_usable],
                "scientific_domain": "pharmacology",
                "source": "PubMed",
                "query": query,
            })
        except Exception:
            continue

    return papers


def fetch_arxiv_pharma_papers(limit: int = 50) -> list[dict]:
    """Fetch pharmaceutical papers from arXiv q-bio categories."""
    import urllib.request
    import urllib.parse

    queries = [
        "cat:q-bio.QM",    # Quantitative Methods
        "cat:q-bio.TO",    # Toxicology
        "cat:q-bio.BM",    # Biomolecules
    ]
    papers: list[dict] = []
    for cat in queries:
        params = urllib.parse.urlencode({
            "search_query": cat,
            "start": 0, "max_results": min(limit // 3, 20),
            "sortBy": "relevance", "sortOrder": "descending",
        })
        url = f"http://export.arxiv.org/api/query?{params}"
        time.sleep(0.5)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(data)
            for entry in root.findall("atom:entry", ns):
                pid = entry.find("atom:id", ns).text.strip().split("/")[-1]
                title = (entry.find("atom:title", ns).text or "").strip()
                summary = (entry.find("atom:summary", ns).text or "").strip()
                full = f"{title} {summary}"
                formulas = extract_formulas(full, limit=5)
                papers.append({
                    "id": f"arxiv_{pid}",
                    "title": title[:200],
                    "abstract": summary[:1000],
                    "text": full[:2000],
                    "formulas": [f.latex for f in formulas if f.is_usable],
                    "scientific_domain": "pharmacology",
                    "source": "arXiv",
                    "query": cat,
                })
        except Exception as e:
            print(f"  arXiv {cat}: {e}")
    return papers


def build_benchmark_corpus(papers: list[dict]) -> list[dict]:
    """Convert paper records to Formulagate corpus format."""
    corpus: list[dict] = []
    for p in papers:
        corpus.append({
            "id": p["id"],
            "title": p.get("title", ""),
            "english": p.get("abstract", p.get("text", "")),
            "math_formula": " ".join(p.get("formulas", [])[:3]),
            "scientific_domain": p.get("scientific_domain", "pharmacology"),
            "text": p.get("text", ""),
            "source": p.get("source", "unknown"),
        })
    return corpus


def run_gate_benchmark(corpus: list[dict]) -> dict:
    """Run Formulagate gate against ground-truth pharma formulas."""
    sources = [
        Source(id=c["id"], text=c.get("english", ""), formula=c.get("math_formula", ""),
               domain=c.get("scientific_domain", ""))
        for c in corpus
    ]
    gate = Formulagate(sources=sources, use_physics=True, top_k=3)

    results = {
        "total": len(GROUND_TRUTH),
        "pk_consistent_pass": 0,
        "pk_consistent_fail": 0,
        "hallucination_caught": 0,
        "hallucination_missed": 0,
        "details": [],
    }

    for gt in GROUND_TRUTH:
        # Dimensional check
        canon = canonicalize(gt.latex)
        dim = check_dimensions(canon, None)
        dim_ok = dim.status == "consistent"

        # Gate check
        gate_result = gate.check(brief=gt.name, draft=gt.latex)

        # Physics signals
        phys = physics_signals(gt.latex, " ".join(
            c.get("english", "") or c.get("text", "") for c in corpus[:5]
        ))

        record = {
            "name": gt.name,
            "formula": gt.latex,
            "class": gt.class_,
            "multi_letter": gt.multi_letter,
            "expected_consistent": gt.si_consistent,
            "is_hallucination": not gt.si_consistent,
            "dimensions": dim.status,
            "dim_ok": dim_ok,
            "gate_action": gate_result.action,
            "gate_confidence": gate_result.confidence,
            "physics": phys.to_dict(),
        }
        results["details"].append(record)

        if gt.si_consistent:
            if dim_ok:
                results["pk_consistent_pass"] += 1
            else:
                results["pk_consistent_fail"] += 1
        else:
            if not dim_ok or gate_result.action == "abstain":
                results["hallucination_caught"] += 1
            else:
                results["hallucination_missed"] += 1

    return results


def run_retrieval_benchmark(corpus: list[dict]) -> dict:
    """Test whether the gate retrieves the correct paper for each formula."""
    sources = [
        Source(id=c["id"], text=c.get("english", ""), formula=c.get("math_formula", ""),
               domain=c.get("scientific_domain", ""))
        for c in corpus
    ]
    gate = Formulagate(sources=sources, use_physics=False, top_k=5)

    results = {"total_queries": 0, "recall_at_1": 0, "recall_at_3": 0,
               "mrr": 0.0, "domain_accuracy": 0}

    # Test retrieval: for formulas found in papers, check if gate finds the paper
    pk_corpus = [c for c in corpus if c.get("formulas") or c.get("math_formula")]
    for paper in pk_corpus[:20]:
        title = paper.get("title", "")
        if not title or len(title) < 10:
            continue
        results["total_queries"] += 1

        brief = title[:100]
        gate_result = gate.check(brief=brief, draft=title, top_k=3)
        # Domain classification
        domain = classify_domain(brief, use_ml=False)
        if domain == "pharmacology":
            results["domain_accuracy"] += 1

        # Check if paper id is in top results
        sources = gate_result.sources
        if sources and any(s.id == paper["id"] for s in sources[:1]):
            results["recall_at_1"] += 1
        if sources and any(s.id == paper["id"] for s in sources[:3]):
            results["recall_at_3"] += 1
        # MRR
        for rank, s in enumerate(sources, 1):
            if s.id == paper["id"]:
                results["mrr"] += 1.0 / rank
                break

    n = max(results["total_queries"], 1)
    results["recall_at_1"] = results["recall_at_1"] / n
    results["recall_at_3"] = results["recall_at_3"] / n
    results["mrr"] = results["mrr"] / n
    results["domain_accuracy"] = results["domain_accuracy"] / n

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Formulagate Pharmaceutical Benchmark")
    ap.add_argument("--fetch", action="store_true",
                    help="Download fresh papers from PubMed + arXiv")
    ap.add_argument("--limit", type=int, default=50,
                    help="Max papers to download (default 50)")
    ap.add_argument("--corpus", type=Path, default=DATA / "pharma_corpus.json",
                    help="Path to corpus file")
    ap.add_argument("--report", type=Path, default=DATA / "pharma_bench_report.json",
                    help="Path to benchmark report")
    ap.add_argument("--existing", type=Path,
                    help="Use existing corpus file instead of downloading")
    args = ap.parse_args()

    print("=" * 65)
    print("  FORMULAGATE PHARMACEUTICAL BENCHMARK — Real-World Data")
    print("=" * 65)
    print(f"  No LLM key required — all checks are deterministic algebra\n")

    # -- Load or download corpus -------------------------------------------
    corpus: list[dict] = []
    if args.existing and args.existing.is_file():
        corpus = json.loads(args.existing.read_text(encoding="utf-8"))
        print(f"Loaded {len(corpus)} records from {args.existing}")
    elif args.corpus.is_file() and not args.fetch:
        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        print(f"Loaded {len(corpus)} records from cache")
    elif args.fetch:
        print("Downloading pharmaceutical papers ...")
        papers: list[dict] = []

        # PubMed — pharmacokinetics
        print("\n[1/3] PubMed: pharmacokinetic drug clearance ...")
        papers += fetch_pubmed_papers(
            "pharmacokinetics drug clearance bioavailability",
            max_results=args.limit // 3,
        )

        # PubMed — enzyme kinetics / drug metabolism
        print("\n[2/3] PubMed: enzyme kinetics drug metabolism ...")
        papers += fetch_pubmed_papers(
            "enzyme kinetics michaelis menten drug metabolism CYP450",
            max_results=args.limit // 3,
        )

        # arXiv — q-bio / tox / biomolecules
        print("\n[3/3] arXiv: q-bio papers ...")
        papers += fetch_arxiv_pharma_papers(limit=args.limit // 3)

        print(f"\nTotal papers fetched: {len(papers)}")
        corpus = build_benchmark_corpus(papers)

        # Cache
        args.corpus.parent.mkdir(parents=True, exist_ok=True)
        args.corpus.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Corpus cached -> {args.corpus}")
    else:
        print("No corpus found. Run with --fetch to download papers.")
        return 1

    n_corpus = len(corpus)
    n_formulas = sum(len(c.get("formulas", [])) for c in corpus if isinstance(c.get("formulas"), list))
    n_with_formula = sum(1 for c in corpus if c.get("math_formula") or c.get("formulas"))
    print(f"\nCorpus: {n_corpus} papers, {n_formulas} formulas, {n_with_formula} with equations")

    # -- Run gate benchmark ------------------------------------------------
    print("\n" + "=" * 65)
    print("  [1/3] DIMENSIONAL ANALYSIS + HALLUCINATION DETECTION")
    print("=" * 65)
    t0 = time.time()
    gate_results = run_gate_benchmark(corpus)
    gate_elapsed = time.time() - t0

    valid = len([g for g in GROUND_TRUTH if g.si_consistent])
    hall = len([g for g in GROUND_TRUTH if not g.si_consistent])
    dim_pass = gate_results["pk_consistent_pass"]
    hall_catch = gate_results["hallucination_caught"]

    print(f"\n  Ground-truth formulas tested:   {len(GROUND_TRUTH)}")
    print(f"  Valid PK formulas:              {valid}")
    print(f"  Hallucination formulas:         {hall}")
    print(f"  -------------------------------------")
    print(f"  Dimensional consistency pass:   {dim_pass}/{valid}  ({dim_pass/max(valid,1):.0%})")
    print(f"  Hallucinations caught:          {hall_catch}/{hall}  ({hall_catch/max(hall,1):.0%})")
    print(f"  Gate decisions time:            {gate_elapsed:.2f}s")

    # Detail by formula class
    print(f"\n  By drug discovery phase:")
    for cls in sorted(set(g.class_ for g in GROUND_TRUTH)):
        items = [g for g in GROUND_TRUTH if g.class_ == cls]
        dim_results = [r for r in gate_results["details"]
                       if r["class"] == cls and r["expected_consistent"] and not r["is_hallucination"]]
        dim_ok = sum(1 for r in dim_results if r["dim_ok"])
        n_ok = sum(1 for g in items if g.si_consistent)
        print(f"    {cls:12s}: {dim_ok}/{n_ok} consistent"
              + (f"  ✓" if n_ok > 0 and dim_ok == n_ok else ""))

    # -- Retrieval benchmark -----------------------------------------------
    print("\n" + "=" * 65)
    print("  [2/3] RETRIEVAL + DOMAIN CLASSIFICATION")
    print("=" * 65)
    t1 = time.time()
    retrieval = run_retrieval_benchmark(corpus)
    ret_elapsed = time.time() - t1

    print(f"\n  Queries tested:                 {retrieval['total_queries']}")
    print(f"  Domain classification accuracy: {retrieval['domain_accuracy']:.0%}")
    print(f"  Recall@1 (paper retrieval):     {retrieval['recall_at_1']:.0%}")
    print(f"  Recall@3:                       {retrieval['recall_at_3']:.0%}")
    print(f"  MRR:                            {retrieval['mrr']:.3f}")
    print(f"  Retrieval time:                 {ret_elapsed:.2f}s")

    # -- Combined report ---------------------------------------------------
    print("\n" + "=" * 65)
    print("  [3/3] FINAL REPORT")
    print("=" * 65)

    report = {
        "metadata": {
            "model": "Formulagate v1.0.1 + Pharma Extension",
            "no_llm_key_required": True,
            "method": "deterministic dimensional analysis + Z3 SMT",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "corpus": {
            "total_papers": n_corpus,
            "total_formulas": n_formulas,
            "papers_with_equations": n_with_formula,
            "sources": list(set(c.get("source", "unknown") for c in corpus)),
        },
        "dimensional_analysis": {
            "valid_formulas_tested": valid,
            "dimensional_pass": dim_pass,
            "dimensional_pass_rate": round(dim_pass / max(valid, 1), 3),
            "hallucinations_tested": hall,
            "hallucinations_caught": hall_catch,
            "hallucination_detection_rate": round(hall_catch / max(hall, 1), 3),
            "seconds": round(gate_elapsed, 2),
            "details": gate_results["details"],
        },
        "retrieval": {
            "total_queries": retrieval["total_queries"],
            "domain_accuracy": round(retrieval["domain_accuracy"], 3),
            "recall_at_1": round(retrieval["recall_at_1"], 3),
            "recall_at_3": round(retrieval["recall_at_3"], 3),
            "mrr": round(retrieval["mrr"], 3),
            "seconds": round(ret_elapsed, 2),
        },
    }

    # Save
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Report saved -> {args.report}")
    print(f"\n  SUMMARY: {dim_pass}/{valid} PK formulas dimensionally consistent")
    print(f"           {hall_catch}/{hall} hallucinations caught by the gate\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


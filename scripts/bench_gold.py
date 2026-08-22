"""Gold-set diagnostic: report every anchored equation with context and verdicts.

Produces a human-readable JSON that shows, for every anchored equation in the
top-N papers: the equation text, surrounding prose, grounded symbols with
dimensions and sources, SMT verdict, and the list of ungrounded free symbols
that block decidability. This is the input for manual spot-checking of
grounding quality — the honest way to measure precision without a hand-labeled
corpus.

Usage:
  python scripts/bench_gold.py --top-papers 10 --report data/real/diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "real"
FULLTEXT = DATA / "fulltext"

import sys
sys.path.insert(0, str(ROOT))

from formulagate.formula_extract import canonicalize
from formulagate.symbol_grounding import ground_paper
from formulagate.tex_ingest import extract_from_tex_file
from formulagate.verify_smt import check_dimensions_smt, is_dimension_anchored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-papers", type=int, default=10)
    ap.add_argument("--precomputed", type=Path, default=DATA / "bench_phase4_cases.json")
    ap.add_argument("--report", type=Path, default=DATA / "diagnostics.json")
    args = ap.parse_args()

    rows = json.loads(args.precomputed.read_text(encoding="utf-8"))
    paper_anchored = Counter()
    for r in rows:
        if any(g.get("source") in ("context", "constant") for g in r.get("symbols", [])):
            paper_anchored[r.get("paper", "")] += 1
    top_papers = [p for p, _ in paper_anchored.most_common(args.top_papers)]

    diagnostics = []
    t0 = time.time()

    for pid in top_papers:
        eqs = []
        for tex in sorted((FULLTEXT / pid).glob("*.tex")) + sorted((FULLTEXT / pid).glob("*.ltx")):
            try:
                eqs.extend(extract_from_tex_file(tex))
            except Exception:
                pass
        if not eqs:
            continue
        groundings = ground_paper(eqs)

        for eq, grd in zip(eqs, groundings):
            f = canonicalize(eq.latex)
            if not f.is_usable:
                continue
            ctx = [g for g in grd.grounded if g.source in ("context", "constant")]
            if not ctx:
                continue
            ov = {g.symbol: g.dimension for g in ctx if g.dimension}
            certain = [g.symbol for g in ctx]
            anchored, anchors = is_dimension_anchored(f, overrides=ov, certain_symbols=certain)
            if not anchored:
                continue

            smt = check_dimensions_smt(f, overrides=ov, certain_symbols=certain)
            ungrounded = set(f.symbols)
            for g in grd.grounded:
                if g.dimension is not None:
                    ungrounded.discard(g.symbol)

            diagnostics.append({
                "paper": pid,
                "eq": eq.latex,
                "context_before": eq.context_before[-200:] if eq.context_before else "",
                "context_after": eq.context_after[:200] if eq.context_after else "",
                "anchors": sorted(anchors),
                "smt_verdict": smt.status,
                "smt_detail": smt.detail[:100] if smt.detail else "",
                "grounded": [
                    {"symbol": g.symbol, "quantity": g.quantity,
                     "dimension": str(g.dimension) if g.dimension else None,
                     "source": g.source}
                    for g in grd.grounded
                ],
                "ungrounded": sorted(ungrounded - {"pi", "e", "i"}),
            })

    elapsed = time.time() - t0

    summary = {
        "papers": top_papers,
        "total_anchored": len(diagnostics),
        "by_verdict": {
            v: sum(1 for d in diagnostics if d["smt_verdict"] == v)
            for v in ("consistent", "inconsistent", "unknown")
        },
        "elapsed_s": round(elapsed, 1),
    }
    output = {"summary": summary, "diagnostics": diagnostics}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

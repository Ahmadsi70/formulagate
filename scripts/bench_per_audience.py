"""Per-audience benchmark: runs the full pipeline on 142 real arXiv papers,
split by domain/audience.  Honest numbers per profession.

Usage: python scripts/bench_per_audience.py --report data/real/bench_audience.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "real"
FULLTEXT = DATA / "fulltext"

import sys
sys.path.insert(0, str(ROOT))

from formulagate.domain_tables import classify_subdomain
from formulagate.formula_extract import canonicalize
from formulagate.symbol_grounding import ground_paper
from formulagate.tex_ingest import extract_from_tex_file
from formulagate.verify_smt import check_dimensions_smt, is_dimension_anchored
from formulagate.dimensions import check_dimensions

# Map subdomain to audience
_DOMAIN_TO_AUDIENCE = {
    "cosmology_gr": "National Labs (CERN/NASA)",
    "quantum_field": "National Labs (CERN/NASA)",
    "nuclear": "National Labs (CERN/NASA)",
    "fluid_dynamics": "Oil & Gas / Aerospace",
    "electromagnetism": "Academic Journals",
    "thermodynamics": "Oil & Gas / Aerospace",
    "condensed_matter": "R&D Semiconductor",
    "general": "Academic Journals",
}

# Paper IDs that were downloaded with explicit audience labels
_AUDIENCE_LABELS: dict[str, str] = {}

PAPER_CACHE: dict[str, str] = {}


def _sympy_add_args(formula):
    import sympy
    try:
        expr = sympy.sympify(formula.canonical)
    except Exception:
        return []
    if isinstance(expr, sympy.Add):
        return list(expr.args)
    return [expr]


def _read_paper(pid: str) -> str:
    if pid in PAPER_CACHE:
        return PAPER_CACHE[pid]
    texts = []
    pdir = FULLTEXT / pid
    for tex in sorted(pdir.glob("*.tex")) + sorted(pdir.glob("*.ltx")):
        try:
            texts.append(tex.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    result = " ".join(texts)
    PAPER_CACHE[pid] = result
    return result


def _corrupt(latex: str):
    import re, random
    syms = re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", latex)
    syms = [s for s in syms if s not in ("sin","cos","tan","exp","log","ln","int","sum","prod","frac","sqrt")]
    if len(syms) >= 2:
        a, b = random.sample(syms, 2)
        return latex.replace(a, b, 1), f"{a}->{b}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=DATA / "bench_audience.json")
    args = ap.parse_args()

    manifest = json.loads((FULLTEXT / "manifest.json").read_text(encoding="utf-8"))
    paper_ids = [pid for pid, e in manifest.items() if e.get("status") == "ok"]

    # Metrics per audience
    metrics: dict[str, dict] = defaultdict(lambda: {
        "papers": 0, "equations": 0, "parsable": 0,
        "decidable_exact": 0, "decidable_smt": 0,
        "fp_exact": 0, "fp_smt": 0,
        "anchored_scope": 0, "anchored_decidable": 0,
        "anchored_fp": 0, "corruptions": 0,
        "detected_exact": 0, "detected_smt": 0,
        "scope_exact": 0, "scope_smt": 0,
        "grounding_sum": 0.0,
    })

    t0 = time.time()
    for pid in paper_ids:
        eqs = []
        pdir = FULLTEXT / pid
        for tex in sorted(pdir.glob("*.tex")) + sorted(pdir.glob("*.ltx")):
            try:
                eqs.extend(extract_from_tex_file(tex))
            except Exception:
                pass
        if not eqs:
            continue

        # Classify paper domain — prefer manifest label for explicitly downloaded papers
        pid_meta = manifest.get(pid, {})
        if pid_meta.get("audience"):
            audience = pid_meta["audience"]
        else:
            text = _read_paper(pid)
            subdomain = classify_subdomain(text)
            audience = _DOMAIN_TO_AUDIENCE.get(subdomain, "Academic Journals")
        m = metrics[audience]
        m["papers"] += 1

        groundings = ground_paper(eqs)
        for eq, grd in zip(eqs, groundings):
            f = canonicalize(eq.latex)
            if not f.is_usable:
                continue
            m["equations"] += 1
            m["parsable"] += 1

            ctx = [g for g in grd.grounded if g.source in ("context", "constant")]
            ov = {g.symbol: g.dimension for g in ctx if g.dimension}
            certain = [g.symbol for g in ctx]
            anchored, _ = is_dimension_anchored(f, overrides=ov, certain_symbols=certain)
            if anchored:
                m["anchored_scope"] += 1

            # Exact walker
            v = check_dimensions(f)
            if v.status == "inconsistent":
                m["fp_exact"] += 1
                m["decidable_exact"] += 1
            elif v.status == "consistent":
                m["decidable_exact"] += 1

            # SMT
            smt = check_dimensions_smt(f, overrides=ov, certain_symbols=certain)
            if smt.status == "inconsistent":
                m["fp_smt"] += 1
                m["decidable_smt"] += 1
                if anchored:
                    m["anchored_fp"] += 1
            elif smt.status == "consistent":
                m["decidable_smt"] += 1
                if anchored:
                    m["anchored_decidable"] += 1

            m["grounding_sum"] += grd.coverage

            # Corruption detection
            terms = len([a for a in _sympy_add_args(f) if not a.is_number])
            if terms < 2:
                continue
            mutation = _corrupt(eq.latex)
            if mutation is None:
                continue
            corrupted_latex, how = mutation
            cf = canonicalize(corrupted_latex)
            if not cf.is_usable or cf.structure_hash == f.structure_hash:
                continue
            m["corruptions"] += 1

            if v.status == "consistent":
                m["scope_exact"] += 1
                cv = check_dimensions(cf)
                if cv.status == "inconsistent":
                    m["detected_exact"] += 1
            if smt.status == "consistent":
                m["scope_smt"] += 1
                csv = check_dimensions_smt(cf, overrides=ov, certain_symbols=certain)
                if csv.status == "inconsistent":
                    m["detected_smt"] += 1

    elapsed = time.time() - t0

    # Compile report
    report = {"elapsed_s": round(elapsed, 1), "audiences": {}}
    for audience, m in sorted(metrics.items()):
        parsable = max(m["parsable"], 1)
        dec_exact = max(m["decidable_exact"], 1)
        dec_smt = max(m["decidable_smt"], 1)
        report["audiences"][audience] = {
            "papers": m["papers"],
            "equations_parsable": m["parsable"],
            "decidable_smt": m["decidable_smt"],
            "decidable_exact": m["decidable_exact"],
            "FPR_smt_pct": round(m["fp_smt"] / max(dec_smt, 1) * 100, 2),
            "FPR_exact_pct": round(m["fp_exact"] / max(dec_exact, 1) * 100, 2),
            "anchored_in_scope": m["anchored_scope"],
            "anchored_decidable": m["anchored_decidable"],
            "anchored_fp": m["anchored_fp"],
            "anchored_fp_rate": round(m["anchored_fp"] / max(m["anchored_decidable"], 1), 3),
            "grounding_pct": round(m["grounding_sum"] / parsable * 100, 1),
            "corruptions": m["corruptions"],
            "detected_smt": m["detected_smt"],
            "detection_rate_pct": round(m["detected_smt"] / max(m["scope_smt"], 1) * 100, 1),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

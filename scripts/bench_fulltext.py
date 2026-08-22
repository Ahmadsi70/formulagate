"""Real full-text benchmark: complete equations lifted from arXiv LaTeX sources.

This is Phase 1's benchmark. It measures exactly what the abstract-level
benchmark could not: can the dimensional veto decide on *complete equations*
that real papers actually typeset?

Pipeline (all real data, nothing synthesised):
  1. ``scripts/fetch_arxiv_sources.py`` downloads the LaTeX sources.
  2. :mod:`formulagate.tex_ingest` extracts complete equations + prose context.
  3. :mod:`formulagate.symbol_grounding` grounds symbols from the prose.
  4. :mod:`formulagate.verify_smt` proves/refutes dimensional consistency.
  5. Corruption (deterministic, from real equations) measures detection.

Reported numbers:
  coverage       — share of real equations the veto can now *decide*
  false_positive — untouched real equations wrongly rejected (must stay ~0)
  detection      — corrupted real equations actually caught
  in_scope       — equations with >=2 comparable terms (the checkable set)

Usage:
    python scripts/bench_fulltext.py --papers 60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formulagate.dimensions import check_dimensions  # noqa: E402
from formulagate.formula_extract import canonicalize  # noqa: E402
from formulagate.symbol_grounding import ground_paper  # noqa: E402
from formulagate.tex_ingest import extract_from_tex_file  # noqa: E402
from formulagate.verify_smt import check_dimensions_smt, is_dimension_anchored  # noqa: E402

FULLTEXT = ROOT / "data" / "real" / "fulltext"

# Deterministic corruption of real equations (no generated text).
_SYMBOL_SWAPS = (("t", "x"), ("m", "v"), ("c", "m"), ("v", "t"), ("E", "p"), ("a", "v"))
_POWER = re.compile(r"\^\{?(-?\d)\}?|\^\{?[a-zA-Z]")


def _corrupt(latex: str) -> tuple[str, str] | None:
    """Return (corrupted latex, description) or None if nothing applies.

    Prefers an exponent bump (the classic LLM error), else a dimension-changing
    symbol swap. Only ever touches real formulas.
    """
    m = _POWER.search(latex)
    if m:
        raw = m.group(0)
        if raw[1:].lstrip("{}").isdigit():
            bumped = str(int(raw[1:].lstrip("{}")) + 1)
            corrupted = latex[: m.start()] + f"^{bumped}" + latex[m.end() :]
            return corrupted, f"exponent {raw[1:]}->{bumped}"
    for source, target in _SYMBOL_SWAPS:
        pattern = re.compile(rf"(?<![A-Za-z_\\]){re.escape(source)}(?![A-Za-z_])")
        if pattern.search(latex):
            return pattern.sub(target, latex, count=1), f"symbol {source}->{target}"
    return None


def _parses(latex: str) -> bool:
    return canonicalize(latex).is_usable


def _has_math(content: str) -> bool:
    return "$" in content or "\\begin{equation" in content or "\\begin{align" in content


def _paper_equations(paper_dir: Path) -> list[Any]:
    """All equations from every .tex file of one paper (real source)."""
    out: list[Any] = []
    for tex in sorted(paper_dir.glob("*.tex")) + sorted(paper_dir.glob("*.ltx")):
        try:
            out.extend(extract_from_tex_file(tex, source_file=tex.name))
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", type=int, default=0, help="Cap papers examined (0 = all)")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "real" / "bench_fulltext.json")
    ap.add_argument("--per-case", type=Path, help="Dump per-equation rows (debug)")
    args = ap.parse_args()

    manifests: list[dict] = []
    manifest_path = FULLTEXT / "manifest.json"
    if manifest_path.is_file():
        manifests = json.loads(manifest_path.read_text(encoding="utf-8"))
    paper_ids = [
        pid for pid, entry in manifests.items()
        if entry.get("status") == "ok" and _has_math(_join_tex(FULLTEXT / pid))
    ]
    if args.papers:
        paper_ids = paper_ids[: args.papers]
    print(f"papers with LaTeX sources: {len(paper_ids)}", flush=True)

    t0 = time.time()
    n_equations = 0
    parsable = 0
    decidable_exact = 0
    decidable_smt = 0
    grounded_ok = 0
    decidable_with_grounding = 0
    two_terms = 0
    false_positives_exact = 0
    false_positives_smt = 0
    corrupted_total = 0
    corrupted_in_scope = 0
    scope_exact = 0
    scope_smt = 0
    scope_combined = 0
    detected_exact = 0
    detected_smt = 0
    detected_combined = 0
    in_scope_total = 0
    detect_anchored_scope = 0
    detect_anchored_total = 0
    detect_anchored_hit = 0
    fp_anchored_only = 0
    decidable_anchored_smt = 0
    in_scope_anchored = 0
    coverage_rows: list[dict[str, Any]] = []
    examples: list[dict[str, str]] = []

    for pid in paper_ids:
        paper_equations = _paper_equations(FULLTEXT / pid)
        paper_groundings = ground_paper(paper_equations)
        for eq, grounding in zip(paper_equations, paper_groundings):
            n_equations += 1
            formula = canonicalize(eq.latex)
            if not formula.is_usable:
                continue
            parsable += 1
            overrides = grounding.as_overrides()
            certain = [g.symbol for g in grounding.grounded if g.source in ("context", "constant")]
            anchored, _anchors = is_dimension_anchored(formula, overrides=overrides,
                                                       certain_symbols=certain)
            if anchored:
                in_scope_anchored += 1

            verdict = check_dimensions(formula)
            if verdict.status == "inconsistent":
                false_positives_exact += 1
                decidable_exact += 1
            elif verdict.status == "consistent":
                decidable_exact += 1

            smt = check_dimensions_smt(formula, overrides=overrides, certain_symbols=certain)
            if smt.status == "inconsistent":
                false_positives_smt += 1
                decidable_smt += 1
                if anchored:
                    fp_anchored_only += 1
            elif smt.status == "consistent":
                decidable_smt += 1
                if anchored:
                    decidable_anchored_smt += 1

            # Combined decision: exact walker with grounding overrides.
            combined = check_dimensions(formula, overrides=overrides)
            if combined.status == "inconsistent":
                decidable_with_grounding += 1
            elif combined.status == "consistent":
                decidable_with_grounding += 1

            # Whether the exact walker with full grounding decided the ORIGINAL.
            original_decided = combined.status in ("consistent", "inconsistent")

            # Two comparable terms — the checkable set.
            terms = len([a for a in _sympy_add_args(formula) if not a.is_number])
            is_multi = terms >= 2
            two_terms += is_multi

            row = {
                "paper": pid,
                "eq": eq.latex[:120],
                "symbols": [
                    {"symbol": g.symbol, "quantity": g.quantity,
                     "dimension": str(g.dimension) if g.dimension else None,
                     "source": g.source}
                    for g in grounding.grounded
                ],
                "cov": grounding.coverage,
                "exact": verdict.status,
                "smt": smt.status,
                "combined": combined.status,
                "multi": is_multi,
            }
            coverage_rows.append(row)

            if grounding.coverage > 0:
                grounded_ok += 1

            # Only the checkable set is corrupted (single-term spans are
            # trivially consistent, so no corruption of them is detectable).
            if not is_multi:
                continue
            in_scope_total += 1
            mutation = _corrupt(eq.latex)
            if mutation is None:
                continue
            corrupted_latex, how = mutation
            c_formula = canonicalize(corrupted_latex)
            if not c_formula.is_usable or c_formula.structure_hash == formula.structure_hash:
                continue
            corrupted_total += 1
            corrupted_in_scope += 1

            caught_exact = (
                verdict.status == "consistent"
                and check_dimensions(c_formula).status == "inconsistent"
            )
            caught_smt = (
                smt.status == "consistent"
                and check_dimensions_smt(c_formula, overrides=overrides,
                                         require_grounded=True).status == "inconsistent"
            )
            caught_combined = (
                combined.status == "consistent"
                and check_dimensions(c_formula, overrides=overrides).status == "inconsistent"
            )
            detected_exact += caught_exact
            detected_smt += caught_smt
            detected_combined += caught_combined
            if anchored:
                detect_anchored_total += 1
                if caught_smt:
                    detect_anchored_hit += 1
                if smt.status == "consistent":
                    detect_anchored_scope += 1
            # Detection denominators: corruptions whose ORIGINAL the method decided.
            if verdict.status == "consistent":
                scope_exact += 1
            if smt.status == "consistent":
                scope_smt += 1
            if combined.status == "consistent":
                scope_combined += 1
            if (caught_exact or caught_smt) and len(examples) < 6:
                examples.append(
                    {
                        "paper": pid,
                        "original": eq.latex[:140],
                        "corrupted": corrupted_latex[:140],
                        "how": how,
                        "exact": caught_exact,
                        "smt": caught_smt,
                    }
                )

    elapsed = time.time() - t0
    denom_dec = max(decidable_with_grounding, 1)
    report = {
        "pipeline": "real arXiv LaTeX sources -> tex_ingest -> symbol_grounding -> dimensions/SMT",
        "papers": len(paper_ids),
        "equations_extracted": n_equations,
        "equations_parsable": parsable,
        "decision_coverage_exact": decidable_exact / max(parsable, 1),
        "decision_coverage_smt": decidable_smt / max(parsable, 1),
        "decision_coverage_grounded": decidable_with_grounding / max(parsable, 1),
        "decidable_exact": decidable_exact,
        "decidable_smt": decidable_smt,
        "decidable_grounded": decidable_with_grounding,
        "two_term_equations": two_terms,
        "two_term_share": two_terms / max(parsable, 1),
        "symbol_grounding_coverage": grounded_ok / max(parsable, 1),
        "false_positive_rate_exact": false_positives_exact / denom_dec,
        "false_positive_rate_smt": false_positives_smt / max(decidable_smt, 1),
        "corrupted_equations": corrupted_total,
        "corrupted_in_scope": corrupted_in_scope,
        "detection_scope_exact": scope_exact,
        "detection_scope_smt": scope_smt,
        "detection_scope_grounded": scope_combined,
        "detection_rate_exact": detected_exact / max(scope_exact, 1),
        "detection_rate_smt": detected_smt / max(scope_smt, 1),
        "detection_rate_grounded": detected_combined / max(scope_combined, 1),
"detected_exact": detected_exact,
        "detected_smt": detected_smt,
        "detected_grounded": detected_combined,
        # Honest scoped metrics: only equations carrying a reliable dimension
        # anchor (integer constants / prose-stated symbols) are *falsifiable*.
        # Metrics over the full set are dominated by dimensionaless identities
        # and combinatorics that cannot be proven wrong by dimensional analysis.
        "anchored_in_scope": in_scope_anchored,
        "anchored_decidable_smt": decidable_anchored_smt + fp_anchored_only,
        "anchored_coverage": (decidable_anchored_smt + fp_anchored_only)
        / max(in_scope_anchored, 1),
        "anchored_fp_smt": fp_anchored_only,
        "anchored_fp_rate": fp_anchored_only / max(decidable_anchored_smt + fp_anchored_only, 1),
        "anchored_detection_corruptions": detect_anchored_total,
        "anchored_detection_scope_smt": detect_anchored_scope,
        "anchored_detection_hit": detect_anchored_hit,
        "anchored_detection_rate": detect_anchored_hit / max(detect_anchored_scope, 1),
        "examples_caught": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_case:
        args.per_case.write_text(
            json.dumps(coverage_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _sympy_add_args(formula):
    import sympy

    expr = sympy.sympify(formula.canonical)
    return expr.args if isinstance(expr, sympy.Add) else [expr]


def _join_tex(paper_dir: Path) -> str:
    out = []
    for tex in list(paper_dir.glob("*.tex")) + list(paper_dir.glob("*.ltx")):
        try:
            out.append(tex.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())

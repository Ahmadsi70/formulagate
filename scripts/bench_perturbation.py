"""Measure the dimensional veto: does it catch corrupted physics, and does it
ever reject real physics?

The arXiv benchmark cannot answer the first question — every draft there is a
verbatim author sentence, so there is no wrong physics to catch. This script
supplies the missing half **without inventing text**: it takes real formulas out
of the real corpus and applies one deterministic corruption (an exponent bump or
a symbol swap), which is exactly the error class a language model produces when
it half-remembers an equation.

Two numbers come out:
  detection rate     — corrupted formulas the veto rejects
  false-positive rate— untouched real formulas the veto rejects (must stay ~0)

Usage:
    python scripts/bench_perturbation.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from formulagate.dimensions import check_dimensions  # noqa: E402
from formulagate.formula_extract import canonicalize, extract_latex_spans  # noqa: E402

# Swaps between symbols of provably different dimension: the corrupted formula
# is guaranteed wrong, so a miss is a real miss.
_SYMBOL_SWAPS = (("t", "x"), ("m", "v"), ("c", "m"), ("v", "t"), ("E", "p"))
_POWER = re.compile(r"\^\{?(\d)\}?")


def _corrupt(latex: str) -> tuple[str, str] | None:
    """Return (corrupted latex, description), or None if nothing applies."""

    match = _POWER.search(latex)
    if match:
        bumped = int(match.group(1)) + 1
        corrupted = latex[: match.start()] + f"^{bumped}" + latex[match.end() :]
        return corrupted, f"exponent {match.group(1)}->{bumped}"

    for source, target in _SYMBOL_SWAPS:
        pattern = re.compile(rf"(?<![A-Za-z\\]){re.escape(source)}(?![A-Za-z])")
        if pattern.search(latex):
            return pattern.sub(target, latex, count=1), f"symbol {source}->{target}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "real" / "arxiv_corpus.json")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "real" / "perturbation.json")
    args = ap.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))

    import sympy

    parsable = 0           # real formulas SymPy could read
    checkable = 0          # …of those, ones the analyser can actually decide
    in_scope = 0           # …of those, ones with ≥2 comparable terms
    false_positives = 0    # real formulas wrongly rejected
    corrupted_total = 0
    corrupted_in_scope = 0
    detected = 0
    examples: list[dict[str, str]] = []

    def comparable_terms(formula) -> int:
        """Dimensional analysis needs two dimensional terms to compare.

        A single-term span (``E_0``, ``\\alpha\\beta``) is trivially consistent, so
        no corruption of it can ever be detected — counting those as misses
        would understate a limit that is structural, not a bug.
        """

        expr = sympy.sympify(formula.canonical)
        return len([a for a in expr.args if not a.is_number]) if expr.is_Add else 1

    for record in corpus:
        for span in extract_latex_spans(record.get("english", "")):
            original = canonicalize(span)
            if not original.is_usable:
                continue
            parsable += 1
            verdict = check_dimensions(original)
            if verdict.status == "inconsistent":
                false_positives += 1
                checkable += 1
                continue
            if verdict.status != "consistent":
                continue
            checkable += 1
            multi_term = comparable_terms(original) >= 2
            in_scope += multi_term

            mutation = _corrupt(span)
            if mutation is None:
                continue
            corrupted_latex, how = mutation
            corrupted = canonicalize(corrupted_latex)
            if not corrupted.is_usable or corrupted.structure_hash == original.structure_hash:
                continue
            corrupted_total += 1
            corrupted_in_scope += multi_term
            caught = check_dimensions(corrupted).status == "inconsistent"
            detected += caught
            if caught and len(examples) < 5:
                examples.append({"original": span, "corrupted": corrupted_latex, "how": how})

    report = {
        "corpus": str(args.corpus),
        "method": "deterministic corruption of real arXiv formulas (no generated text)",
        "real_formulas_parsable": parsable,
        "real_formulas_decidable": checkable,
        "decidable_with_two_terms": in_scope,
        "false_positive_rate": false_positives / max(checkable, 1),
        "corrupted_formulas": corrupted_total,
        "corrupted_in_scope": corrupted_in_scope,
        "detection_rate_in_scope": detected / max(corrupted_in_scope, 1),
        "detection_rate_overall": detected / max(corrupted_total, 1),
        "examples_caught": examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

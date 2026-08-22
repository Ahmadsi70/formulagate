"""Gate accept/reject threshold tests."""

from __future__ import annotations

import json
from pathlib import Path

from formulagate.gate import discover_linkages, evaluate_gate


def _corpus(path: Path) -> Path:
    rows = [
        {
            "id": "decoy-integral",
            "english": "count field creation light measure",
            "math_formula": r"E(t) = \int c \, dt",
            "scientific_domain": None,
        },
        {
            "id": "mod-congruence",
            "english": (
                "equitable measure number modular residue congruence "
                "divisible arithmetic integers"
            ),
            "math_formula": r"a \equiv b \pmod{n}",
            "scientific_domain": "Number Theory",
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_gate_passes_math_brief_with_mod_formula(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path / "corpus.json")
    brief = "Count n with 1<=n<=100 where n^2+n+1 divisible by 3 using modular residues congruence"
    draft = "n ≡ 1 (mod 3); there are 34 such integers; congruence classes."
    result = discover_linkages(brief=brief, candidate_text=draft, corpus_path=corpus, top_k=3)
    ok, detail = evaluate_gate(result)
    assert result.domain == "mathematics"
    assert ok is True
    assert result.entries[0].record_id == "mod-congruence"
    assert "pmod" in result.entries[0].formula_excerpt or "equiv" in result.entries[0].formula_excerpt


def test_gate_rejects_when_score_below_threshold(tmp_path: Path) -> None:
    corpus = tmp_path / "c.json"
    # Formula too short → score_record returns (0,0) → no entries → reject.
    corpus.write_text(
        json.dumps(
            [
                {
                    "id": "weak",
                    "english": "zzz",
                    "math_formula": "x=1",
                    "scientific_domain": None,
                }
            ]
        ),
        encoding="utf-8",
    )
    result = discover_linkages(
        brief="completely unrelated cooking recipe pasta",
        candidate_text="boil water add salt",
        corpus_path=corpus,
    )
    ok, detail = evaluate_gate(result)
    assert ok is False
    assert "matched" in detail.lower() or "minimum" in detail or "not found" in detail

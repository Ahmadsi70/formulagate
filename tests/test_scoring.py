"""Exact IR scoring formula tests (overlap + 2*rel)."""

from __future__ import annotations

from formulagate.domain import classify_domain
from formulagate.scoring import formula_domain_relevance, keywords, score_record


def test_keywords_drop_stopwords_and_short_tokens() -> None:
    keys = keywords("the modular residue of n with gcd")
    assert "the" not in keys
    assert "with" not in keys
    assert "modular" in keys
    assert "residue" in keys
    assert "gcd" not in keys  # length < 4


def test_math_domain_and_rel_markers() -> None:
    assert classify_domain("n^2+n+1 divisible by 3 modular residues") == "mathematics"
    rel = formula_domain_relevance(r"a \equiv b \pmod{n}", "mathematics")
    assert rel >= 1


def test_score_formula_overlap_plus_two_times_rel() -> None:
    """score = keyword_overlap + 2 * formula_domain_relevance."""

    keys = {"modular", "residue", "congruence", "number"}
    row = {
        "english": "equitable measure number modular residue congruence",
        "math_formula": r"a \equiv b \pmod{n}",
        "scientific_domain": "Number Theory",
        "arabic": "",
    }
    score, rel = score_record(keys, row, "mathematics")
    # overlap: modular, residue, congruence, number → 4
    # markers in formula: mod, equiv, pmod → rel=3 → +6
    assert rel == 3
    assert score == 4 + 2 * 3


def test_math_row_with_zero_rel_is_hard_zero() -> None:
    keys = {"energy", "field", "count"}
    row = {
        "english": "energy field count light",
        "math_formula": "abcdef",  # 6 chars, no math markers, no equation structure
        "scientific_domain": None,
    }
    score, rel = score_record(keys, row, "mathematics")
    assert rel == 0
    assert score == 0

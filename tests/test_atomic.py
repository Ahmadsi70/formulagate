"""Tests for atomic fact decomposition and verification."""

from __future__ import annotations

import pytest

from formulagate.atomic import (
    AtomicClaim,
    AtomicReport,
    AtomicVerdict,
    AtomicVerifier,
    _classify_claim,
    _guess_claim_type,
    decompose_rule,
)


def test_empty_draft_returns_empty() -> None:
    assert decompose_rule("") == []
    assert decompose_rule("   ") == []


def test_single_sentence_no_formula() -> None:
    claims = decompose_rule("The sky is blue.")
    assert len(claims) == 1
    assert claims[0].text == "The sky is blue."
    assert claims[0].claim_type == "statement"


def test_multi_sentence_split() -> None:
    claims = decompose_rule("First claim. Second claim. Third claim.")
    assert len(claims) == 3
    assert claims[0].text == "First claim."
    assert claims[1].text == "Second claim."
    assert claims[2].text == "Third claim."


def test_formula_detection_inline() -> None:
    claims = decompose_rule("Einstein proposed $E = m c^2$ which is fundamental.")
    assert len(claims) >= 2
    formulas = [c for c in claims if c.claim_type == "formula"]
    assert len(formulas) == 1
    assert formulas[0].formula is not None
    assert "E = m c^2" in formulas[0].formula


def test_formula_detection_plaintext() -> None:
    claims = decompose_rule("Newton said F = m a, his second law.")
    formulas = [c for c in claims if c.claim_type == "formula"]
    assert len(formulas) >= 1


def test_attribution_detection() -> None:
    claims = decompose_rule(
        "According to Einstein, energy and mass are equivalent."
    )
    assert len(claims) >= 1
    assert claims[0].claim_type == "attribution"


def test_definition_detection() -> None:
    claims = decompose_rule("Force is defined as mass times acceleration.")
    assert len(claims) >= 1
    assert claims[0].claim_type == "definition"


def test_guess_claim_type_heuristics() -> None:
    assert _guess_claim_type("According to Newton, force is F = ma.") == "attribution"
    assert _guess_claim_type("Energy is the capacity to do work.") == "definition"
    assert _guess_claim_type("The ball rolls down the hill.") == "statement"


def test_classify_claim_splits_on_formula() -> None:
    claims = _classify_claim("The formula $E = m c^2$ is famous.")
    types = [c.claim_type for c in claims]
    assert "formula" in types
    assert len(claims) >= 2


def test_atomic_claim_serialization() -> None:
    claim = AtomicClaim(text="E = mc^2", claim_type="formula", formula="E = m c^2")
    d = claim.to_dict()
    assert d["text"] == "E = mc^2"
    assert d["claim_type"] == "formula"


def test_verifier_no_sources() -> None:
    verifier = AtomicVerifier(sources=(), use_physics=False)
    claim = AtomicClaim(text="Some claim", claim_type="statement")
    verdict = verifier.verify(claim)
    assert verdict.supported is False
    assert "no sources" in verdict.reason


def test_verifier_prose_insufficient_match() -> None:
    sources = ({"id": "1", "english": "dogs are mammals", "text": "", "title": ""},)
    verifier = AtomicVerifier(sources=sources, use_physics=False)
    claim = AtomicClaim(text="The sky is made of cheese", claim_type="statement")
    verdict = verifier.verify(claim)
    assert verdict.supported is False


def test_verifier_prose_sufficient_match() -> None:
    sources = (
        {
            "id": "1",
            "english": "energy mass equivalence special relativity",
            "text": "Einstein showed energy equals mass",
            "title": "Relativity",
        },
    )
    verifier = AtomicVerifier(sources=sources, use_physics=False)
    claim = AtomicClaim(
        text="Einstein showed energy and mass are related in relativity",
        claim_type="statement",
    )
    verdict = verifier.verify(claim)
    # "energy", "mass", "relativity" should match
    assert verdict.supported is True


def test_verifier_formula_dimensional_check() -> None:
    verifier = AtomicVerifier(sources=(), use_physics=True)
    claim = AtomicClaim(
        text="$E = m c^2$", claim_type="formula", formula="E = m c^2"
    )
    verdict = verifier.verify(claim)
    assert verdict.supported is True
    assert "dimensionally consistent" in verdict.reason


def test_verifier_formula_broken_dimensions() -> None:
    verifier = AtomicVerifier(sources=(), use_physics=True)
    claim = AtomicClaim(
        text="$E = m c^3$", claim_type="formula", formula="E = m c^3"
    )
    verdict = verifier.verify(claim)
    assert verdict.supported is False
    assert "inconsistent" in verdict.reason


def test_verifier_formula_with_sources() -> None:
    sources = (
        {"id": "rel-1", "formula": "E = m c^2", "english": "energy mass equivalence"},
    )
    verifier = AtomicVerifier(sources=sources, use_physics=True)
    claim = AtomicClaim(
        text="$E = m c^2$", claim_type="formula", formula="E = m c^2"
    )
    verdict = verifier.verify(claim)
    assert verdict.supported is True
    assert "equivalent to source" in verdict.reason
    assert verdict.evidence == "rel-1"


def test_verifier_formula_contradicted_by_source() -> None:
    sources = (
        {"id": "rel-1", "formula": "E = m c^2", "english": "energy-mass equivalence"},
    )
    verifier = AtomicVerifier(sources=sources, use_physics=True)
    # E = m v^2 shares {E, m} with source E=mc^2.  Both are dimensionally
    # consistent (energy), but the formulas differ → contradicted.
    claim = AtomicClaim(
        text="E = m v^2", claim_type="formula", formula="E = m v^2"
    )
    verdict = verifier.verify(claim)
    assert verdict.supported is False
    assert verdict.reason == "contradicted"


def test_verify_all_produces_report() -> None:
    verifier = AtomicVerifier(sources=(), use_physics=True)
    claims = [
        AtomicClaim(text="$E = m c^2$", claim_type="formula", formula="E = m c^2"),
        AtomicClaim(text="$E = m c^3$", claim_type="formula", formula="E = m c^3"),
        AtomicClaim(text="Some prose", claim_type="statement"),
    ]
    report = verifier.verify_all(claims)
    assert report.n_total == 3
    assert report.n_supported == 1  # only E=mc^2 (formula) passes; prose has no sources
    assert report.n_contradicted == 1  # E=mc^3
    assert report.precision == pytest.approx(1 / 3)
    assert report.action == "abstain"


def test_atomic_report_action_threshold() -> None:
    report = AtomicReport(
        claims=(), precision=0.75, n_supported=3, n_total=4,
    )
    assert report.action == "generate"
    report2 = AtomicReport(
        claims=(), precision=0.33, n_supported=1, n_total=3,
    )
    assert report2.action == "abstain"


def test_custom_verify_fn() -> None:
    def always_support(claim, sources):
        return AtomicVerdict(claim=claim, supported=True, reason="custom")

    verifier = AtomicVerifier(sources=(), verify_fn=always_support)
    claim = AtomicClaim(text="anything", claim_type="statement")
    verdict = verifier.verify(claim)
    assert verdict.supported is True
    assert verdict.reason == "custom"
"""Test atomic claim decomposition in the gate decision path.

Verifies that AtomicVerifier correctly decomposes drafts into independently
verifiable claims and feeds atomic evidence into the gate path.
"""

import pytest

from formulagate.atomic import AtomicClaim, AtomicReport, AtomicVerifier, AtomicVerdict
from formulagate.sdk import Formulagate, Source


class TestAtomicGate:
    def test_atomic_decomposition_basic(self):
        from formulagate.atomic import decompose_rule
        claims = decompose_rule("The speed of light is constant. Energy equals mass times c squared.")
        assert len(claims) > 0
        assert all(isinstance(c, AtomicClaim) for c in claims)

    def test_atomic_decomposition_empty(self):
        from formulagate.atomic import decompose_rule
        claims = decompose_rule("")
        assert len(claims) == 0

    def test_atomic_decomposition_single_claim(self):
        from formulagate.atomic import decompose_rule
        claims = decompose_rule("F = m a")
        assert len(claims) >= 1

    def test_atomic_verdict_structure(self):
        claim = AtomicClaim(text="E = m c^2", claim_type="formula", formula="E = m c^2")
        verdict = AtomicVerdict(
            claim=claim,
            supported=True,
            confidence=0.9,
            reason="dimension_check",
        )
        d = verdict.to_dict()
        assert d["supported"] is True
        assert d["confidence"] == 0.9
        assert d["claim"]["text"] == "E = m c^2"

    def test_atomic_verifier_verify_single(self):
        sources = [{"id": "1", "text": "energy mass equivalence", "math_formula": "E = m c^2"}]
        verifier = AtomicVerifier(sources=tuple(sources))
        claim = AtomicClaim(text="E = m c^2", claim_type="formula", formula="E = m c^2")
        verdict = verifier.verify(claim)
        assert isinstance(verdict, AtomicVerdict)
        assert verdict.claim.text == "E = m c^2"

    def test_atomic_report_aggregation(self):
        verdicts = [
            AtomicVerdict(claim=AtomicClaim(text="a"), supported=True, confidence=0.9, reason="ok"),
            AtomicVerdict(claim=AtomicClaim(text="b"), supported=False, confidence=0.3, reason="no evidence"),
            AtomicVerdict(claim=AtomicClaim(text="c"), supported=True, confidence=0.7, reason="ok"),
        ]
        report = AtomicReport(claims=tuple(verdicts), precision=2/3, n_total=3, n_supported=2)
        assert report.n_total == 3
        assert report.n_supported == 2
        assert report.precision == pytest.approx(2/3)
        d = report.to_dict()
        assert d["n_total"] == 3
        assert d["n_supported"] == 2

    def test_atomic_gate_integration(self):
        gate = Formulagate(
            sources=[
                Source(id="1", text="Newton's second law", formula="F = m a"),
                Source(id="2", text="energy relation", formula="E = m c^2"),
            ],
            use_physics=False,
        )
        result = gate.check(
            brief="What is Newton's second law?",
            draft="F equals m a. This is a fundamental law of classical mechanics.",
        )
        assert result.domain in ("general", "physics")
        assert result.action in ("generate", "abstain")

    def test_atomic_verifier_multiple_sources(self):
        sources = [
            {"id": "1", "math_formula": "F = m a", "english": "Newton's second law"},
            {"id": "2", "math_formula": "E = m c^2", "english": "mass energy equivalence"},
        ]
        verifier = AtomicVerifier(sources=tuple(sources))
        claim = AtomicClaim(text="F = m a", claim_type="formula", formula="F = m a")
        verdict = verifier.verify(claim)
        assert verdict.claim.text == "F = m a"
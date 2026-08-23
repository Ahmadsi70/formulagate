"""End-to-end integration tests for Formulagate.

Gate 6 of MVP roadmap: comprehensive E2E tests covering:
- Complete check flow from SDK through gate to decision
- Integration with all layers (semantic, physics, calibration)
- Real-world usage patterns
- Error handling and edge cases
"""

from __future__ import annotations

import json
from pathlib import Path

from formulagate.gate import discover_linkages, evaluate_gate
from formulagate.sdk import Formulagate, Source


def _create_test_corpus(path: Path) -> Path:
    """Create a test corpus with diverse scientific records."""
    rows = [
        {
            "id": "formula-1",
            "english": "The speed of light in vacuum is constant and equal to 299792458 meters per second",
            "math_formula": r"c = 299792458 \text{ m/s}",
            "scientific_domain": None,
        },
        {
            "id": "formula-2",
            "english": "The area of a circle is pi times the radius squared: A = πr²",
            "math_formula": r"A = \pi r^2",
            "scientific_domain": None,
        },
        {
            "id": "formula-3",
            "english": "Newton's second law: Force equals mass times acceleration, F = ma",
            "math_formula": r"F = m a",
            "scientific_domain": None,
        },
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_e2e_discover_linkages(tmp_path: Path) -> None:
    """Test complete linkage discovery flow."""
    corpus = _create_test_corpus(tmp_path / "corpus.json")
    brief = "What is the speed of light in vacuum?"
    draft = "The speed of light is 299,792,458 meters per second."

    result = discover_linkages(brief=brief, candidate_text=draft, corpus_path=corpus, top_k=3)

    assert result.domain == "physics"
    assert len(result.entries) > 0
    assert result.entries[0].record_id == "formula-1"
    assert "299792458" in result.entries[0].formula_excerpt


def test_e2e_gate_passes_with_strong_match(tmp_path: Path) -> None:
    """Test gate passes when strong match is found."""
    corpus = _create_test_corpus(tmp_path / "corpus.json")
    brief = "Calculate the area of a circle with radius 5."
    draft = "The area of a circle with radius 5 is A = π(5)² = 25π."

    result = discover_linkages(brief=brief, candidate_text=draft, corpus_path=corpus, top_k=3)
    ok, detail = evaluate_gate(result)

    assert result.domain == "physics"
    assert ok is True
    assert result.entries[0].record_id == "formula-2"


def test_e2e_cross_domain_query(tmp_path: Path) -> None:
    """Test that cross-domain queries are handled correctly."""
    corpus = _create_test_corpus(tmp_path / "corpus.json")
    brief = "What is the Pythagorean theorem?"
    draft = "The Pythagorean theorem states that a² + b² = c² for right triangles."

    result = discover_linkages(brief=brief, candidate_text=draft, corpus_path=corpus, top_k=3)

    assert result.domain in ["general", "physics", "mathematics"]


def test_e2e_empty_corpus(tmp_path: Path) -> None:
    """Test handling of empty corpus."""
    corpus = tmp_path / "empty.json"
    corpus.write_text(json.dumps([]), encoding="utf-8")

    result = discover_linkages(
        brief="some question",
        candidate_text="some answer",
        corpus_path=corpus,
    )

    assert len(result.entries) == 0
    ok, detail = evaluate_gate(result)
    assert ok is False


def test_e2e_gate_rejects_when_no_match(tmp_path: Path) -> None:
    """Test gate rejects when no domain-relevant formula is found."""
    corpus = tmp_path / "c.json"
    corpus.write_text(
        json.dumps([{"id": "weak", "english": "zzz", "math_formula": "x=1", "scientific_domain": None}]),
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


def test_e2e_sdk_check(tmp_path: Path) -> None:
    """Test SDK check flow with supported formula."""
    gate = Formulagate(
        sources=[
            {"id": "1", "text": "Speed of light is 299,792,458 m/s", "formula": r"c = 299792458 m/s"}
        ],
    )

    brief = "What is the speed of light in vacuum?"
    draft = "The speed of light is 299,792,458 meters per second."

    result = gate.check(brief=brief, draft=draft)

    assert result.action == "generate"
    assert result.ok is True
    assert result.sources[0].id == "1"


def test_e2e_sdk_check_multiple_sources(tmp_path: Path) -> None:
    """Test SDK check flow with multiple candidate sources."""
    gate = Formulagate(
        sources=[
            {"id": "1", "text": "Speed of light is 299,792,458 m/s", "formula": r"c = 299792458 m/s"},
            {"id": "2", "text": "Area of circle is πr²", "formula": r"A = \pi r^2"},
            {"id": "3", "text": "Newton's second law F=ma", "formula": r"F = m a"},
        ],
    )

    brief = "Calculate the area of a circle with radius 5."
    draft = "The area of a circle with radius 5 is A = π(5)² = 25π."

    result = gate.check(brief=brief, draft=draft)

    assert result.action == "generate"
    assert result.ok is True
    assert result.sources[0].id == "2"


def test_e2e_sdk_check_without_physics(tmp_path: Path) -> None:
    """Test SDK check flow with physics layer disabled."""
    gate = Formulagate(
        sources=[
            {"id": "1", "text": "Mass and energy are equivalent: E = mc²", "formula": r"E = m c^2"},
        ],
        use_physics=False,
    )

    brief = "What is mass-energy equivalence?"
    draft = "Mass and energy are equivalent: E = mc²."

    result = gate.check(brief=brief, draft=draft)

    assert result.action == "generate"
    assert result.ok is True
    assert result.physics is None


def test_e2e_sdk_source_objects(tmp_path: Path) -> None:
    """Test SDK check flow using Source objects."""
    gate = Formulagate(
        sources=[
            Source(id="1", text="Mass and energy are equivalent: E = mc²", formula=r"E = m c^2", domain="Physics"),
            Source(id="2", text="Force equals mass times acceleration: F = ma", formula=r"F = m a", domain="Physics"),
        ],
        use_physics=True,
    )

    brief = "What is mass-energy equivalence?"
    draft = "Mass and energy are equivalent: E = mc²."

    result = gate.check(brief=brief, draft=draft)

    assert result.action == "generate"
    assert result.ok is True
    assert result.sources[0].id == "1"


def test_e2e_multiple_checks_same_gate(tmp_path: Path) -> None:
    """Test multiple check calls on same gate instance."""
    gate = Formulagate(
        sources=[
            {"id": "1", "text": "Mass and energy are equivalent: E = mc²", "formula": r"E = m c^2"},
            {"id": "2", "text": "Force equals mass times acceleration: F = ma", "formula": r"F = m a"},
        ],
    )

    # First check
    result1 = gate.check(brief="What is mass-energy equivalence?", draft="Mass and energy are equivalent: E = mc².")
    assert result1.action == "generate"

    # Second check
    result2 = gate.check(brief="What is Newton's second law?", draft="Force equals mass times acceleration: F = ma.")
    assert result2.action == "generate"


def test_e2e_sdk_add_sources(tmp_path: Path) -> None:
    """Test adding sources after gate initialization."""
    gate = Formulagate(sources=[])

    brief = "What is mass-energy equivalence?"
    draft = "Mass and energy are equivalent: E = mc²."

    result = gate.check(brief=brief, draft=draft)
    assert result.action == "abstain"  # No sources initially

    gate.add_sources([{"id": "1", "text": "Mass and energy are equivalent: E = mc²", "formula": r"E = m c^2"}])

    result2 = gate.check(brief=brief, draft=draft)
    assert result2.action == "generate"


def test_e2e_calibrated_confidence(tmp_path: Path) -> None:
    """Test that calibrated confidence is provided."""
    gate = Formulagate(
        sources=[
            {"id": "1", "text": "Mass and energy are equivalent: E = mc²", "formula": r"E = m c^2"},
        ],
        semantic=False,  # Disable semantic to avoid model loading
    )

    brief = "What is mass-energy equivalence?"
    draft = "Mass and energy are equivalent: E = mc²."

    result = gate.check(brief=brief, draft=draft)

    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0


def test_e2e_threshold_is_provided(tmp_path: Path) -> None:
    """Test that threshold is provided in results."""
    gate = Formulagate(
        sources=[
            {"id": "1", "text": "Mass and energy are equivalent: E = mc²", "formula": r"E = m c^2"},
        ],
    )

    brief = "What is mass-energy equivalence?"
    draft = "Mass and energy are equivalent: E = mc²."

    result = gate.check(brief=brief, draft=draft)

    assert result.threshold is not None


def test_e2e_gate_with_no_physics(tmp_path: Path) -> None:
    """Test gate behavior with physics layer disabled."""
    gate = Formulagate(use_physics=False)

    brief = "What is the speed of light in vacuum?"
    draft = "The speed of light is 299,792,458 meters per second."

    result = gate.check(brief=brief, draft=draft)

    assert result.physics is None


def test_e2e_check_with_source_override(tmp_path: Path) -> None:
    """Test checking with temporary source override."""
    gate = Formulagate(sources=[])

    brief = "What is mass-energy equivalence?"
    draft = "E = mc²."

    # Check with empty sources
    result1 = gate.check(brief=brief, draft=draft)
    assert result1.action == "abstain"

    # Check with sources in same call
    result2 = gate.check(
        brief=brief,
        draft="Mass and energy are equivalent: E = mc²",
        sources=[{"id": "1", "text": "Mass and energy are equivalent: E = mc²", "formula": r"E = m c^2"}],
    )
    assert result2.action == "generate"


def test_e2e_calibrated_threshold_applied(tmp_path: Path) -> None:
    """Test that calibrated threshold is applied in check flow."""
    gate = Formulagate(
        sources=[
            {"id": "1", "text": "Mass and energy are equivalent: E = mc²", "formula": r"E = m c^2"},
        ],
        semantic=False,
    )

    brief = "What is mass-energy equivalence?"
    draft = "Mass and energy are equivalent: E = mc²."

    result = gate.check(brief=brief, draft=draft)

    assert result.threshold is not None


def test_lexical_domain_fallback_classifies_fundamental_constants() -> None:
    # The lexical (no-embedding) fallback used to score zero physics markers
    # for "speed of light in vacuum" — a canonical physics brief — so CI runs
    # without sentence-transformers classified it as "general".
    from formulagate.domain import classify_domain

    assert classify_domain("What is the speed of light in vacuum?", use_ml=False) == "physics"
    assert classify_domain("What is the capital of France?", use_ml=False) == "general"

"""Scientific RAG sketch: retrieve → Formulagate judge → accept/abstain."""

from __future__ import annotations

from formulagate.rag import LexicalStubRetriever, RagDecision, run_scientific_rag

_CORPUS = [
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
    {
        "id": "harmonic-bridge",
        "english": "periodic motion vibration oscillation frequency period",
        "math_formula": r"N(t) = A\sin(2\pi t / T) + B\cos(2\pi t / T)",
        "scientific_domain": "Classical Mechanics",
    },
]


def test_rag_accepts_when_retriever_surfaces_congruent_formula() -> None:
    """Judge accepts after lexical recall finds the number-theory hit."""

    retriever = LexicalStubRetriever(_CORPUS)
    brief = (
        "Count n with 1<=n<=100 where n^2+n+1 divisible by 3 "
        "using modular residues congruence"
    )
    draft = "n ≡ 1 (mod 3); congruence classes."
    decision = run_scientific_rag(
        brief=brief,
        draft=draft,
        retriever=retriever,
        retrieve_k=5,
    )
    assert isinstance(decision, RagDecision)
    assert decision.ok is True
    assert decision.action == "generate"
    assert decision.retrieved_ids[0] == "mod-congruence"
    assert decision.gate.result.entries[0].record_id == "mod-congruence"


def test_rag_abstains_on_weak_analog_after_retrieval() -> None:
    """Recall may return a decoy; judge still abstains for mathematics."""

    decoy_only = [_CORPUS[0]]
    retriever = LexicalStubRetriever(decoy_only)
    decision = run_scientific_rag(
        brief="modular residue congruence divisible number theory",
        draft="use integral of energy over time as analogy",
        retriever=retriever,
        retrieve_k=3,
    )
    assert decision.ok is False
    assert decision.action == "abstain"
    assert "matched" in decision.gate.detail.lower() or decision.gate.detail


def test_retriever_ranks_by_query_not_draft_alone() -> None:
    """Search stage uses the brief (scientific query), not the LLM draft."""

    retriever = LexicalStubRetriever(_CORPUS)
    ids = [r["id"] for r in retriever.retrieve("modular congruence residue", top_k=2)]
    assert ids[0] == "mod-congruence"

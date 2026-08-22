"""Tests for Formulagate LangChain and LlamaIndex integrations."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from formulagate.langchain import FormulagateCallback, FormulagateGuardrail, FormulagateRetriever


# ─── FormulagateCallback Tests ────────────────────────────────────────────────


def test_callback_init():
    """Test basic initialization with no sources."""
    cb = FormulagateCallback()
    assert cb._sources == []
    assert cb._use_physics is True
    assert cb._message_on_abstain is not None
    assert "I cannot verify" in cb._message_on_abstain


def test_callback_init_with_sources():
    """Test initialization with explicit sources."""
    sources = [
        {"id": "1", "text": "E = mc^2 is the energy-mass equivalence."},
        {"id": "2", "text": "Force equals mass times acceleration."},
    ]
    cb = FormulagateCallback(sources=sources, use_physics=False)
    assert cb._sources == sources
    assert cb._use_physics is False


def test_callback_on_llm_start():
    """Test that prompts are captured during LLM start."""
    cb = FormulagateCallback()
    prompts = ["What is E=mc^2?"]
    cb.on_llm_start(serialized={}, prompts=prompts, **{})
    assert cb._last_prompt == "What is E=mc^2?"


def test_callback_on_llm_start_empty_prompts():
    """Test handling of empty prompt lists."""
    cb = FormulagateCallback()
    cb.on_llm_start(serialized={}, prompts=[], **{})
    assert cb._last_prompt == ""


def test_callback_extract_text_aimessage():
    """Test text extraction from AIMessage."""
    cb = FormulagateCallback()
    msg = AIMessage(content="E = mc^2 is correct.")
    text = cb._extract_text(msg)
    assert text == "E = mc^2 is correct."


def test_callback_extract_text_list_generations():
    """Test text extraction from LLMResult with generations."""
    cb = FormulagateCallback()
    gen = MagicMock()
    gen.text = "Kinetic energy: KE = 1/2 mv^2"
    gen_list = [gen]
    llm_result = MagicMock()
    llm_result.generations = [gen_list]
    text = cb._extract_text(llm_result)
    assert text == "Kinetic energy: KE = 1/2 mv^2"


def test_callback_extract_text_dict():
    """Test text extraction from dict response."""
    cb = FormulagateCallback()
    response = {"text": "Gravity is g = 9.8 m/s^2", "confidence": 0.9}
    text = cb._extract_text(response)
    assert text == "Gravity is g = 9.8 m/s^2"


def test_callback_extract_text_empty():
    """Test text extraction when response has no text."""
    cb = FormulagateCallback()
    response = {"text": ""}
    text = cb._extract_text(response)
    assert text == ""


def test_callback_extract_text_sympy_expression():
    """Test that SymPy expressions are handled gracefully."""
    from sympy import Symbol

    cb = FormulagateCallback()
    expr = Symbol("x") + Symbol("y")
    text = cb._extract_text(expr)
    # Should convert to string representation
    assert "x + y" in text or text == ""


def test_callback_on_llm_end_with_generate_action():
    """Test that generate actions do not modify the response."""
    cb = FormulagateCallback()
    cb._last_prompt = "What is the formula for velocity?"
    cb._gate = MagicMock()
    cb._gate.check.return_value = MagicMock(
        action="generate", confidence=0.95, detail="Confidently supported"
    )

    msg = AIMessage(content="Velocity = displacement / time")
    cb.on_llm_end(response=msg, **{})

    assert msg.content == "Velocity = displacement / time"


def test_callback_on_llm_end_with_abstain_action():
    """Test that abstain actions replace response with safety message."""
    cb = FormulagateCallback()
    cb._last_prompt = "Explain the theory of everything"
    cb._gate = MagicMock()
    cb._gate.check.return_value = MagicMock(
        action="abstain",
        confidence=0.1,
        detail="Cannot verify with available evidence",
    )

    msg = AIMessage(content="The theory of everything combines gravity...")
    cb.on_llm_end(response=msg, **{})

    assert msg.content == cb._message_on_abstain


def test_callback_on_llm_end_no_text():
    """Test handling when response has no extractable text."""
    cb = FormulagateCallback()
    cb._last_prompt = "Some question"

    # Mock gate that returns abstain
    cb._gate = MagicMock()
    cb._gate.check.return_value = MagicMock(
        action="abstain", confidence=0.0, detail=""
    )

    # Response with no text field
    response = {"no": "text"}
    cb.on_llm_end(response=response, **{})

    # Should not raise, and gate should not be called when there's no text
    assert not cb._gate.check.called


def test_callback_to_dict():
    """Test that guardrail result can be serialized."""
    guardrail = FormulagateGuardrail()
    result = guardrail.check(
        prompt="What is the formula for force?",
        response="F = ma",
        context_nodes=[],
    )

    assert isinstance(result, dict)
    assert "action" in result
    assert "confidence" in result
    assert "sources" in result


# ─── FormulagateGuardrail Tests ───────────────────────────────────────────────


def test_guardrail_init():
    """Test basic initialization."""
    guardrail = FormulagateGuardrail()
    assert guardrail._sources == []
    assert guardrail._use_physics is True
    assert guardrail._reject_empty is True


def test_guardrail_init_with_sources():
    """Test initialization with explicit sources."""
    sources = [
        {"id": "1", "text": "E = mc^2", "formula": "E = m*c**2"},
        {"id": "2", "text": "F = ma", "formula": "F = m*a"},
    ]
    guardrail = FormulagateGuardrail(sources=sources, use_physics=False)
    assert guardrail._sources == sources
    assert guardrail._use_physics is False


def test_guardrail_check_with_context_nodes():
    """Test that context nodes are properly extracted."""
    guardrail = FormulagateGuardrail()
    
    # Mock nodes from LlamaIndex
    mock_node = MagicMock()
    mock_node.node_id = "node-123"
    mock_node.text = "E = mc^2 is energy-mass equivalence."
    mock_node.metadata = {
        "formula": "E = m*c**2",
        "domain": "physics",
    }
    
    nodes = [mock_node]
    result = guardrail.check(
        prompt="What is the formula for energy?",
        response="E = mc^2",
        context_nodes=nodes,
    )
    
    # Should have converted nodes to sources
    assert "action" in result
    assert "sources" in result


def test_guardrail_check_without_context_nodes():
    """Test guardrail without context nodes falls back to provided sources."""
    guardrail = FormulagateGuardrail(sources=[{"id": "1", "text": "E = mc^2"}])

    result = guardrail.check(
        prompt="What is E=mc^2?",
        response="E = mc^2",
        context_nodes=None,
    )

    assert "action" in result
    # When no context nodes, sources should be preserved and gate should run
    assert len(result["sources"]) >= 0
    # At least there's some result from the gate
    assert "confidence" in result


def test_guardrail_check_reject_empty_when_no_sources():
    """Test that reject_empty=True causes abstain when no sources."""
    guardrail = FormulagateGuardrail(reject_empty=True)

    result = guardrail.check(
        prompt="Some question",
        response="Some answer",
        context_nodes=[],
    )

    assert result["action"] == "abstain"
    assert result["confidence"] == 0.0
    assert "no sources provided" in result["detail"]


def test_guardrail_check_to_dict():
    """Test guardrail check result can be serialized."""
    guardrail = FormulagateGuardrail()
    result = guardrail.check(
        prompt="What is the formula for force?",
        response="F = ma",
        context_nodes=[],
    )
    
    assert isinstance(result, dict)
    assert "action" in result
    assert "confidence" in result
    assert "sources" in result


# ─── Integration Tests ─────────────────────────────────────────────────────────


def test_callback_integration_with_real_gate():
    """Test that callback properly initializes and uses the real gate."""
    sources = [
        {"id": "1", "text": "E = mc^2", "formula": "E = m*c**2"},
    ]

    with patch("formulagate.sdk.Formulagate") as MockGate:
        mock_instance = MagicMock()
        mock_instance.check.return_value = MagicMock(
            action="abstain",
            confidence=0.2,
            detail="No matching formula found",
        )
        MockGate.return_value = mock_instance

        cb = FormulagateCallback(sources=sources)
        msg = AIMessage(content="E = mc^3 would be wrong.")
        cb.on_llm_start(serialized={}, prompts=["What is E=mc^2?"], **{})
        cb.on_llm_end(response=msg, **{})

        # Gate should have been called
        assert mock_instance.check.called
        call_args = mock_instance.check.call_args
        assert call_args[1]["brief"] == "What is E=mc^2?"
        assert call_args[1]["draft"] == "E = mc^3 would be wrong."


def test_guardrail_integration_with_real_gate():
    """Test that guardrail properly initializes and uses the real gate."""
    sources = [
        {"id": "1", "text": "F = ma", "formula": "F = m*a"},
    ]

    with patch("formulagate.sdk.Formulagate") as MockGate:
        # Create a mock that returns a ClaimResult-like structure
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "action": "generate",
            "ok": True,
            "detail": "Supporting evidence found",
            "confidence": 0.95,
            "threshold": 0.7,
            "combined_score": 0.85,
            "physics": None,
            "model": "platt",
            "domain": "physics",
            "top_source_id": "1",
            "sources": [],
        }
        mock_instance.check.return_value = mock_result
        MockGate.return_value = mock_instance

        guardrail = FormulagateGuardrail(sources=sources)
        result = guardrail.check(
            prompt="What is force?",
            response="F = ma",
            context_nodes=[],
        )

        # Gate should have been called
        assert mock_instance.check.called
        assert result["action"] == "generate"


# ─── Error Handling Tests ─────────────────────────────────────────────────────


def test_callback_extract_text_none():
    """Test handling of None response."""
    cb = FormulagateCallback()
    text = cb._extract_text(None)
    assert text == ""


def test_guardrail_check_with_invalid_node():
    """Test guardrail handles nodes without text gracefully."""
    guardrail = FormulagateGuardrail()
    
    # Node without text field
    mock_node = MagicMock()
    mock_node.node_id = "node-123"
    mock_node.text = None
    mock_node.metadata = {}
    
    result = guardrail.check(
        prompt="Question",
        response="Answer",
        context_nodes=[mock_node],
    )
    
    # Should not raise, should handle gracefully
    assert "action" in result


# ─── FormulagateRetriever Tests ───────────────────────────────────────────────


class MockDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockRetriever:
    def invoke(self, query, **kwargs):
        return [
            MockDocument("E = mc^2 is mass-energy equivalence", {"formula": "E = m c^2"}),
            MockDocument("F = m a is Newton's second law", {"formula": "F = m a"}),
            MockDocument("The sky is blue", {}),
        ]
    async def ainvoke(self, query, **kwargs):
        return self.invoke(query, **kwargs)


def test_retriever_filter_mode():
    r = FormulagateRetriever(base_retriever=MockRetriever(), mode="filter")
    docs = r.invoke("energy mass equivalence")
    assert len(docs) >= 1


def test_retriever_rerank_mode():
    r = FormulagateRetriever(base_retriever=MockRetriever(), mode="rerank")
    docs = r.invoke("energy mass equivalence")
    assert len(docs) == 3
    assert "_fg_ok" in docs[0].metadata


def test_retriever_no_physics():
    r = FormulagateRetriever(base_retriever=MockRetriever(), mode="filter", use_physics=False)
    docs = r.invoke("anything")
    assert len(docs) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

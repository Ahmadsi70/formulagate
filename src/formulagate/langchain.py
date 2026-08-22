"""LangChain and LlamaIndex integrations — drop-in guardrails for RAG pipelines.

Gate 4 of the MVP roadmap.  Provides:

  ``FormulagateCallback`` — LangChain callback that intercepts
      LLM responses and verifies them against a knowledge base.  When the gate
      abstains, the response text is replaced with a safety message.

  ``FormulagateGuardrail`` — LlamaIndex guardrail that sits
      between the retriever and the response synthesizer.

  ``FormulagateRouter`` — LangChain router that routes queries through
      the Formulagate semantic gate before answering.

  ``FormulagateRetriever`` — wraps any LangChain BaseRetriever and filters
      or re-ranks retrieved documents by dimensional consistency.  A document
      whose formula is dimensionally inconsistent with the query gets demoted
      or dropped.

Usage (LangChain Callback):
    from formulagate.langchain import FormulagateCallback

    llm = ChatOpenAI(callbacks=[FormulagateCallback(sources=docs)])
    response = llm.invoke("What is E=mc^2?")

Usage (LangChain Retriever):
    from formulagate.langchain import FormulagateRetriever

    retriever = FormulagateRetriever(
        base_retriever=vectorstore.as_retriever(),
        mode="filter",  # or "rerank"
    )
    docs = retriever.invoke("F = m a")

Usage (LlamaIndex):
    from formulagate.langchain import FormulagateGuardrail
    guardrail = FormulagateGuardrail(sources=docs)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

# ─── Data models ──────────────────────────────────────────────────────────────


@dataclass
class GuardrailResult:
    """Result of a Formulagate guardrail check."""

    action: str  # "generate" or "abstain"
    confidence: float | None = None
    detail: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    verdict: str = ""  # "pass", "abstain", "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "confidence": self.confidence,
            "detail": self.detail,
            "sources": self.sources,
            "elapsed_ms": self.elapsed_ms,
            "verdict": self.verdict,
        }


# ─── LangChain Callback ──────────────────────────────────────────────────────


class FormulagateCallback:
    """LangChain callback that verifies every LLM response against a knowledge base.

    Args:
        sources: Records the gate may cite (same format as ``Formulagate``).
        use_physics: Enable dimensional veto (default True).
        message_on_abstain: Text to replace the LLM response with when the
            gate abstains.
        on_verify: Optional callback called with GuardrailResult after each check.
        semantic: Use embedding-based scoring (requires dense extras).

    Example:
        from langchain.chat_models import ChatOpenAI

        llm = ChatOpenAI(callbacks=[FormulagateCallback(sources=my_docs)])
        response = llm.invoke("What is the formula for kinetic energy?")
    """

    def __init__(
        self,
        sources: list[dict[str, Any]] | None = None,
        *,
        use_physics: bool = True,
        message_on_abstain: str | None = None,
        on_verify: Callable[[GuardrailResult], None] | None = None,
        semantic: bool = False,
    ) -> None:
        self._sources = sources or []
        self._use_physics = use_physics
        self._message_on_abstain = message_on_abstain or (
            "I cannot verify this answer with the available evidence. "
            "Please rephrase your question or consult a trusted source."
        )
        self._last_prompt: str = ""
        self._gate = None
        self._on_verify = on_verify
        self._semantic = semantic
        self._verify_count: int = 0
        self._abstain_count: int = 0

    @property
    def _get_gate(self):
        """Lazy-init the gate to avoid importing heavy deps until needed."""
        if self._gate is None:
            from formulagate.sdk import Formulagate

            self._gate = Formulagate(
                sources=self._sources,
                use_physics=self._use_physics,
                semantic=self._semantic,
            )
        return self._gate

    @property
    def verify_count(self) -> int:
        return self._verify_count

    @property
    def abstain_count(self) -> int:
        return self._abstain_count

    # ── LangChain callback interface ─────────────────────────────────────

    def on_llm_start(
        self, serialized: dict, prompts: list[str], **kwargs: Any
    ) -> None:
        """Capture the user prompt for later verification."""
        self._last_prompt = prompts[0] if prompts else ""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Verify the LLM response and modify it if the gate abstains."""
        if not self._last_prompt:
            return

        t0 = time.monotonic()
        draft = self._extract_text(response)
        if not draft:
            return

        gate = self._get_gate
        result = gate.check(
            brief=self._last_prompt,
            draft=draft,
            sources=self._sources if self._sources else None,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        guardrail_result = GuardrailResult(
            action=result.action,
            confidence=result.confidence,
            detail=result.detail,
            sources=[s.to_dict() if hasattr(s, "to_dict") else {"id": s.get("id", "")} for s in result.sources] if hasattr(result, 'sources') else [],
            elapsed_ms=elapsed_ms,
            verdict="abstain" if result.action == "abstain" else "pass",
        )

        if result.action == "abstain":
            self._set_text(response, self._message_on_abstain)
            self._abstain_count += 1

        self._verify_count += 1

        if self._on_verify:
            try:
                self._on_verify(guardrail_result)
            except Exception as exc:
                logger.warning("on_verify callback failed: %s", exc)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull text from LangChain's various generation output formats."""
        if response is None:
            return ""

        # LLMResult with generations
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                if gen_list:
                    first = gen_list[0]
                    if hasattr(first, "text"):
                        return first.text
                    if hasattr(first, "message") and hasattr(first.message, "content"):
                        return first.message.content
        # AIMessage
        if hasattr(response, "content"):
            return str(response.content)
        # Simple dict
        if isinstance(response, dict):
            return str(response.get("text") or response.get("content") or "")
        return str(response)

    @staticmethod
    def _set_text(response: Any, text: str) -> None:
        """Replace text in LangChain response object."""
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "text"):
                        gen.text = text
                    if hasattr(gen, "message") and hasattr(gen.message, "content"):
                        gen.message.content = text
        if hasattr(response, "content"):
            response.content = text


# ─── LlamaIndex Guardrail ────────────────────────────────────────────────────


class FormulagateGuardrail:
    """LlamaIndex guardrail that verifies retrieved context against LLM output.

    Place this between the retriever and the response synthesizer in a
    LlamaIndex query pipeline.

    Args:
        sources: Knowledge base records.
        use_physics: Enable dimensional veto.
        reject_empty: When True, reject responses with no supporting evidence.
        message_on_abstain: Text to return when gate abstains.

    Example:
        from llama_index.core.query_pipeline import QueryPipeline

        pipeline = QueryPipeline(
            modules=[
                my_retriever,
                FormulagateGuardrail(sources=docs),
                my_synthesizer,
            ]
        )
    """

    def __init__(
        self,
        sources: list[dict[str, Any]] | None = None,
        *,
        use_physics: bool = True,
        reject_empty: bool = True,
        message_on_abstain: str | None = None,
    ) -> None:
        self._sources = sources or []
        self._use_physics = use_physics
        self._reject_empty = reject_empty
        self._message_on_abstain = message_on_abstain or (
            "I cannot verify this answer with the available evidence."
        )
        self._gate = None

    @property
    def _get_gate(self):
        if self._gate is None:
            from formulagate.sdk import Formulagate

            self._gate = Formulagate(
                sources=self._sources, use_physics=self._use_physics
            )
        return self._gate

    def check(
        self, prompt: str, response: str, context_nodes: list[Any] | None = None
    ) -> dict[str, Any]:
        """Run the gate and return a verdict dict.

        Returns:
            ``{"action": "generate" | "abstain", "confidence": float, "detail": str, "sources": [...]}``
        """
        # Check if we should reject (no sources at all)
        has_sources = bool(self._sources) or (context_nodes is not None and len(context_nodes) > 0)

        if self._reject_empty and not has_sources:
            return {
                "action": "abstain",
                "confidence": 0.0,
                "detail": "no sources provided",
                "sources": [],
            }

        # Initialize sources
        sources = self._sources

        # Process context nodes if provided
        if context_nodes and len(context_nodes) > 0:
            sources = []
            for node in context_nodes:
                src: dict[str, Any] = {
                    "id": node.node_id if hasattr(node, "node_id") else str(id(node))
                }
                if hasattr(node, "text"):
                    src["text"] = node.text
                if hasattr(node, "metadata"):
                    meta = node.metadata or {}
                    src["formula"] = meta.get("formula", "")
                    src.update(meta)
                sources.append(src)

        gate = self._get_gate
        result = gate.check(brief=prompt, draft=response, sources=sources if sources else None)
        return result.to_dict()


# ─── LangChain Router ────────────────────────────────────────────────────────


class FormulagateRouter:
    """LangChain router that routes queries through Formulagate before answering.

    This router can be used with LangChain's ``RouterOutputParser`` or as a
    standalone pre-answer verification step.

    Args:
        sources: Knowledge base records.
        use_physics: Enable dimensional veto.
        semantic: Use embedding-based scoring.

    Example:
        from formulagate.langchain import FormulagateRouter

        router = FormulagateRouter(sources=docs)
        if router.should_answer("What is E=mc^2?"):
            answer = llm.invoke("What is E=mc^2?")
    """

    def __init__(
        self,
        sources: list[dict[str, Any]] | None = None,
        *,
        use_physics: bool = True,
        semantic: bool = False,
    ) -> None:
        self._sources = sources or []
        self._use_physics = use_physics
        self._semantic = semantic
        self._gate = None

    @property
    def _get_gate(self):
        if self._gate is None:
            from formulagate.sdk import Formulagate

            self._gate = Formulagate(
                sources=self._sources,
                use_physics=self._use_physics,
                semantic=self._semantic,
            )
        return self._gate

    def should_answer(self, query: str, draft: str | None = None) -> bool:
        """Check if the query has supporting evidence in the knowledge base.

        Args:
            query: The user's question.
            draft: Optional pre-generated answer to verify.

        Returns:
            True if the gate passes (evidence found), False otherwise.
        """
        if draft:
            result = self._get_gate.check(brief=query, draft=draft)
            return result.action == "generate"

        # For queries without draft, check if any sources exist
        return bool(self._sources)

    def verify_answer(self, query: str, answer: str) -> GuardrailResult:
        """Verify an answer against the knowledge base.

        Args:
            query: The user's question.
            answer: The LLM's answer to verify.

        Returns:
            GuardrailResult with verification details.
        """
        t0 = time.monotonic()
        result = self._get_gate.check(brief=query, draft=answer)
        elapsed_ms = (time.monotonic() - t0) * 1000

        return GuardrailResult(
            action=result.action,
            confidence=result.confidence,
            detail=result.detail,
            sources=[s.to_dict() if hasattr(s, "to_dict") else s for s in result.sources] if hasattr(result, 'sources') else [],
            elapsed_ms=elapsed_ms,
            verdict="abstain" if result.action == "abstain" else "pass",
        )


# ─── LangChain Retriever ──────────────────────────────────────────────────────


class FormulagateRetriever:
    """LangChain retriever that wraps any BaseRetriever with dimensional filtering.

    In ``filter`` mode, documents whose formula is dimensionally inconsistent
    with the query are dropped.  In ``rerank`` mode, consistent documents are
    boosted above inconsistent ones (but all are returned).

    Args:
        base_retriever: Any LangChain BaseRetriever (vector store, BM25, etc.).
        mode: ``"filter"`` (drop inconsistent) or ``"rerank"`` (boost consistent).
        use_physics: Enable the dimensional veto.
        score_field: When using rerank mode, the metadata key to store the
            Formulagate score in (default ``"_fg_score"``).

    Example:
        from langchain_community.vectorstores import Chroma
        from formulagate.langchain import FormulagateRetriever

        vectorstore = Chroma(...)
        retriever = FormulagateRetriever(
            base_retriever=vectorstore.as_retriever(),
            mode="filter",
        )
        docs = retriever.invoke("F = m a")
    """

    def __init__(
        self,
        base_retriever,
        *,
        mode: str = "filter",
        use_physics: bool = True,
        score_field: str = "_fg_score",
    ) -> None:
        self._base = base_retriever
        self._mode = mode
        self._use_physics = use_physics
        self._score_field = score_field

    def invoke(self, query: str, **kwargs) -> list[Any]:
        docs = self._base.invoke(query, **kwargs)
        if not docs or not self._use_physics:
            return docs

        from formulagate.sdk import Formulagate

        results = []
        for doc in docs:
            text = getattr(doc, "page_content", "") or ""
            formula = ""
            if hasattr(doc, "metadata") and doc.metadata:
                formula = doc.metadata.get("formula", "") or doc.metadata.get("math_formula", "")

            src = {"id": str(hash(text))[:12], "text": text, "formula": formula}
            gate = Formulagate(sources=[src], use_physics=True)
            result = gate.check(brief=query, draft=text)

            # Store Formulagate result in metadata
            if hasattr(doc, "metadata"):
                doc.metadata[self._score_field] = result.to_dict()

            if self._mode == "filter":
                if result.action == "generate":
                    results.append(doc)
            else:  # rerank
                doc.metadata["_fg_ok"] = result.action == "generate"
                results.append(doc)

        if self._mode == "rerank":
            # Boost consistent documents to the top
            results.sort(key=lambda d: (
                not d.metadata.get("_fg_ok", False),
                -(d.metadata.get(self._score_field, {}).get("confidence", 0) or 0),
            ))

        return results

    async def ainvoke(self, query: str, **kwargs):
        docs = await self._base.ainvoke(query, **kwargs)
        if not docs or not self._use_physics:
            return docs
        return self.invoke(query, **kwargs)  # fall back to sync for simplicity

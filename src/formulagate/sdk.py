"""Stable public surface of Formulagate — the layer integrators build against.

Two calls cover the product:

    verify(latex, against=...)      # is this formula possible, and is it *that* formula?
    check(brief, draft, sources)    # is this claim actually supported? generate or abstain

Everything here is intentionally boring: plain dataclasses, JSON-serialisable
results, no exceptions from the symbolic layer, and no import of a model unless
the caller explicitly asks for semantic scoring. The internals (``gate``,
``calibration``, ``dimensions``, ``equivalence``) stay free to change; this
module is what does not.

    from formulagate.sdk import Formulagate

    gate = Formulagate(sources=my_records, fused_calibration="calibration_fused.json")
    result = gate.check(brief=user_question, draft=llm_answer)
    if result.action == "abstain":
        ...   # ask the model to try again, or say "not enough evidence"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from formulagate.conformal import ConformalCalibration

from formulagate import __version__ as __sdk_version__

__all__ = [
    "ClaimResult",
    "Formulagate",
    "RankedSource",
    "RetrieverMode",
    "Source",
    "VerifyResult",
    "__sdk_version__",
    "check",
    "verify",
]


# ─── Inputs ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """One record the gate may cite.

    ``formula`` is what the deterministic layer reads; ``text`` is what the
    lexical scorer reads. Either may be empty, but a record with neither can
    never support a claim.
    """

    id: str
    text: str = ""
    formula: str = ""
    domain: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        """Internal corpus shape (kept private so field names can evolve)."""

        record: dict[str, Any] = {
            "id": self.id,
            "english": self.text,
            "math_formula": self.formula,
            "scientific_domain": self.domain,
        }
        record.update(self.meta)
        return record


def _coerce(source: Source | Mapping[str, Any]) -> Source:
    if isinstance(source, Source):
        return source
    if not isinstance(source, Mapping):
        raise TypeError(f"source must be a Source or mapping, got {type(source).__name__}")
    identifier = source.get("id")
    if not identifier:
        raise ValueError("every source needs a non-empty 'id'")
    return Source(
        id=str(identifier),
        text=str(source.get("text") or source.get("english") or ""),
        formula=str(source.get("formula") or source.get("math_formula") or ""),
        domain=(
            str(source["domain"])
            if source.get("domain")
            else (str(source["scientific_domain"]) if source.get("scientific_domain") else None)
        ),
        meta={
            k: v
            for k, v in source.items()
            if k
            not in {"id", "text", "english", "formula", "math_formula", "domain", "scientific_domain"}
        },
    )


def _resolve_retriever(kind: str | Any, sources: list[Source]) -> "RetrieverProtocol | None":
    """Resolve a retriever string shortcut or pass-through an existing instance."""
    from formulagate.retriever import RetrieverProtocol, SQLiteRetriever

    if isinstance(kind, RetrieverProtocol):
        return kind
    if not isinstance(kind, str):
        return None

    records = [s.as_record() for s in sources]
    kind = kind.lower().strip()

    if kind == "sqlite":
        return SQLiteRetriever(records)
    if kind == "hybrid":
        try:
            from formulagate.hybrid import HybridRrfRetriever
            return HybridRrfRetriever(records)
        except ImportError:
            return SQLiteRetriever(records)
    if kind == "dense":
        try:
            from formulagate.dense import DenseHybridRetriever
            return DenseHybridRetriever(records)
        except ImportError:
            try:
                from formulagate.hybrid import HybridRrfRetriever
                return HybridRrfRetriever(records)
            except ImportError:
                return SQLiteRetriever(records)
    return None


# ─── Outputs ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VerifyResult:
    """Verdict of the deterministic layer on a single formula.

    ``ok`` is False only when something was *proven* wrong — an unparsable or
    undecidable formula is not evidence of correctness either, so it reports
    ``ok=False`` with ``dimensions="unknown"``. Callers who want "not refuted"
    should test ``result.refuted`` instead.
    """

    formula: str
    parsed: bool
    ok: bool
    dimensions: str
    reason: str = ""
    symbols: tuple[str, ...] = ()
    structure_hash: str | None = None
    equivalence: str | None = None
    equivalence_method: str | None = None

    @property
    def refuted(self) -> bool:
        """True when algebra proved the formula wrong (not merely unproven)."""

        return self.dimensions == "inconsistent" or self.equivalence == "different"

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "parsed": self.parsed,
            "ok": self.ok,
            "refuted": self.refuted,
            "dimensions": self.dimensions,
            "reason": self.reason,
            "symbols": list(self.symbols),
            "structure_hash": self.structure_hash,
            "equivalence": self.equivalence,
            "equivalence_method": self.equivalence_method,
        }


@dataclass(frozen=True)
class RankedSource:
    """A candidate record with the evidence score that ranked it."""

    id: str
    score: int
    relevance: int
    formula: str
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "relevance": self.relevance,
            "formula": self.formula,
            "domain": self.domain,
        }


@dataclass(frozen=True)
class ClaimResult:
    """Decision for one (brief, draft) pair: cite the sources, or abstain."""

    action: str
    detail: str
    confidence: float | None = None
    threshold: float | None = None
    combined_score: float | None = None
    physics: dict[str, float] | None = None
    model: str = "platt"
    sources: tuple[RankedSource, ...] = ()
    domain: str = "general"

    @property
    def ok(self) -> bool:
        return self.action == "generate"

    @property
    def top_source_id(self) -> str | None:
        return self.sources[0].id if self.sources else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "ok": self.ok,
            "detail": self.detail,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "combined_score": self.combined_score,
            "physics": self.physics,
            "model": self.model,
            "domain": self.domain,
            "top_source_id": self.top_source_id,
            "sources": [s.to_dict() for s in self.sources],
        }


# ─── Client ───────────────────────────────────────────────────────────────────


class Formulagate:
    """Configured gate: sources and calibration are supplied once, then reused.

    Args:
        sources: records the gate may cite (``Source`` objects or mappings).
        calibration: path to a Platt artifact (``formulagate calibrate``).
        fused_calibration: path to a similarity+physics artifact; when present it
            decides, because installing it is the act of asking it to.
        use_physics: enable the dimensional veto and the physics features.
        semantic: allow embeddings (domain classification and semantic scoring).
            Left at ``False``, no neural model is ever loaded and a decision
            costs milliseconds on CPU.
        top_k: how many ranked sources to consider and report.
        conformal: ``ConformalCalibration`` for risk-controlled thresholds.
            When set, every ``check()`` applies the conformal threshold on top
            of the calibrated confidence.
        retriever: built-in retriever to use when ``check()`` is called without
            explicit ``sources``.  Accepts a string shortcut (``"sqlite"``,
            ``"hybrid"``, ``"dense"``) or a ``RetrieverProtocol`` instance.
            When a string is given and ``sources`` are supplied, they are
            auto-indexed at construction time.
    """

    RetrieverMode = str  # "sqlite" | "hybrid" | "dense"


    def __init__(
        self,
        sources: Iterable[Source | Mapping[str, Any]] | None = None,
        *,
        calibration: str | Path | None = None,
        fused_calibration: str | Path | None = None,
        use_physics: bool = True,
        semantic: bool = False,
        top_k: int = 3,
        conformal: "ConformalCalibration | None" = None,
        retriever: RetrieverMode | "RetrieverProtocol | None" = None,
    ) -> None:
        self._sources: list[Source] = [_coerce(s) for s in (sources or [])]
        self.use_physics = use_physics
        self.semantic = semantic
        self.top_k = top_k
        self._conformal = conformal
        self._retriever: RetrieverProtocol | None = None

        if retriever is not None:
            self._retriever = _resolve_retriever(retriever, self._sources)

        if calibration is not None:
            from formulagate.calibration import load_calibration

            load_calibration(calibration)
        if fused_calibration is not None:
            from formulagate.calibration import load_multi_calibration, set_multi_calibration

            set_multi_calibration(load_multi_calibration(fused_calibration))

    @property
    def sources(self) -> tuple[Source, ...]:
        return tuple(self._sources)

    def add_sources(self, sources: Iterable[Source | Mapping[str, Any]]) -> None:
        """Extend the citable set (ranking is per call, so this is cheap).
        Also re-indexes the retriever if one is installed."""
        new = [_coerce(s) for s in sources]
        self._sources.extend(new)
        if self._retriever is not None and hasattr(self._retriever, 'index'):
            self._retriever.index([s.as_record() for s in new])

    def verify(
        self,
        latex: str,
        against: str | None = None,
        *,
        context: str = "",
        use_grounding: bool = False,
    ) -> VerifyResult:
        """Dimensional analysis, and optionally equivalence against a reference.

        Pure algebra: no corpus, no model, no network. Never raises on malformed
        LaTeX — it comes back as ``parsed=False``.

        Args:
            latex: the equation to verify.
            against: optional reference equation for equivalence checking.
            context: prose around the equation (e.g. an abstract). Used only when
                ``use_grounding=True`` to resolve ambiguous symbols whose
                dimension the prose states (``c`` = speed of light, ``G`` =
                gravitational constant, …).  Without grounding the table of
                certain constants alone decides, and many classical equations
                return ``unknown``.
            use_grounding: when ``True`` and ``context`` is given, run the
                4-layer symbol resolver (:mod:`formulagate.symbol_grounding`)
                and feed its overrides to the dimensional walker.  This is the
                path that lifts classic equations (``F = G m1 m2 / r^2``,
                ``lambda = h / p``) from ``unknown`` to ``consistent``.
        """

        from formulagate.dimensions import check_dimensions
        from formulagate.equivalence import check_equivalence
        from formulagate.formula_extract import canonicalize

        formula = canonicalize(latex)

        overrides = None
        if use_grounding and context:
            try:
                from formulagate.symbol_grounding import ground_symbols

                grounding = ground_symbols(latex, context_before=context)
                overrides = grounding.as_overrides() or None
            except Exception:
                overrides = None  # never let grounding crash verify()

        dimensions = check_dimensions(formula, overrides=overrides)

        equivalence = method = None
        if against is not None:
            verdict = check_equivalence(formula, canonicalize(against))
            equivalence, method = verdict.status, verdict.method

        refuted = dimensions.status == "inconsistent" or equivalence == "different"
        proven = dimensions.status == "consistent" or equivalence == "equivalent"
        return VerifyResult(
            formula=formula.latex,
            parsed=formula.is_usable,
            ok=bool(proven and not refuted),
            dimensions=dimensions.status,
            reason=dimensions.detail or (formula.parse_error or ""),
            symbols=formula.symbols,
            structure_hash=formula.structure_hash,
            equivalence=equivalence,
            equivalence_method=method,
        )

    def check(
        self,
        brief: str,
        draft: str,
        sources: Iterable[Source | Mapping[str, Any]] | None = None,
        top_k: int | None = None,
    ) -> ClaimResult:
        """Decide whether ``draft`` is supported by the sources.

        Args:
            brief: the question or task the draft answers (drives retrieval).
            draft: the text to be published — the thing under judgement.
            sources: overrides the client's records for this call.  When
                ``None`` AND a ``retriever`` is installed, uses the retriever
                to fetch relevant documents from the indexed corpus.
            top_k: how many ranked records to judge.

        Returns:
            ``ClaimResult`` with ``action`` = ``"generate"`` or ``"abstain"``.
        """

        from formulagate.gate import evaluate_gate_signals, rank_records

        if sources is not None:
            records = [_coerce(s) for s in sources]
        elif self._retriever is not None:
            retrieved = self._retriever.search(brief, top_k=top_k or self.top_k * 3)
            records = [_coerce(r.to_dict()) for r in retrieved]
        else:
            records = self._sources

        result = rank_records(
            brief=brief,
            candidate_text=draft,
            rows=[s.as_record() for s in records],
            top_k=top_k or self.top_k,
            use_ml_domain=self.semantic,
        )
        ok, detail, signals = evaluate_gate_signals(
            result,
            brief=brief,
            draft=draft,
            use_semantic=self.semantic,
            use_physics=self.use_physics,
            conformal=self._conformal,
        )
        return ClaimResult(
            action="generate" if ok else "abstain",
            detail=detail,
            confidence=signals.confidence,
            threshold=signals.threshold,
            combined_score=signals.combined_score,
            # None means "the layer did not run"; a dict of zeros would read as
            # "it ran and found nothing", which is a different fact.
            physics=(
                signals.physics.to_dict() if self.use_physics and signals.physics else None
            ),
            model=signals.model,
            sources=tuple(
                RankedSource(
                    id=e.record_id,
                    score=e.score,
                    relevance=e.math_relevance,
                    formula=e.formula_excerpt,
                    domain=e.scientific_domain,
                )
                for e in result.entries
            ),
            domain=result.domain,
        )


# ─── Module-level convenience (one-off calls, default configuration) ──────────

_DEFAULT = Formulagate()


def verify(
    latex: str,
    against: str | None = None,
    *,
    context: str = "",
    use_grounding: bool = False,
) -> VerifyResult:
    """One-off :meth:`Formulagate.verify` with default configuration."""

    return _DEFAULT.verify(
        latex, against=against, context=context, use_grounding=use_grounding
    )


def check(
    brief: str,
    draft: str,
    sources: Sequence[Source | Mapping[str, Any]],
    top_k: int = 3,
) -> ClaimResult:
    """One-off :meth:`Formulagate.check` with default configuration."""

    return _DEFAULT.check(brief, draft, sources=sources, top_k=top_k)

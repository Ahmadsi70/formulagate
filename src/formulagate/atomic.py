"""Atomic fact decomposition for fine-grained verification.

Why this exists:
  The gate currently makes a single binary decision for an entire draft.  But a
  draft can contain multiple claims — some correct, some wrong, some irrelevant.
  A single accept/reject cannot say "the formula is right but the attribution is
  wrong".  Atomic decomposition breaks a draft into independently verifiable
  claims (like SAFE / FactScore) so each can be checked separately, and the
  aggregate verdict reflects how much of the draft is actually supported.

Design:
  - Rule-based decomposition first (regex patterns, sentence splitting) so the
    core path stays dependency-free and deterministic.
  - Optional LLM decomposition via a pluggable callable.
  - Each claim is verified independently against the same gate infrastructure.
  - The aggregate is a FactScore-style precision/recall panel.

References:
  - SAFE (Google DeepMind, 2024): Search-Augmented Factuality Evaluator
  - FactScore (EMNLP 2023): Fine-grained Atomic Evaluation of Factual Precision
  - ARE framework (Yan et al. 2024): Atomic fact decomposition for attributed QA
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

# ─── Data types ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AtomicClaim:
    """One independently verifiable fact extracted from a draft.

    Attributes:
        text: The claim text (a single sentence or clause).
        claim_type: ``"formula"``, ``"definition"``, ``"attribution"``, or
            ``"statement"``.
        formula: LaTeX body if the claim is a formula, else ``None``.
        source_span: Character range in the original draft (for traceability).
    """

    text: str
    claim_type: str = "statement"
    formula: str | None = None
    source_span: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "claim_type": self.claim_type,
            "formula": self.formula,
            "source_span": list(self.source_span) if self.source_span else None,
        }


@dataclass(frozen=True)
class AtomicVerdict:
    """Verification result for a single atomic claim."""

    claim: AtomicClaim
    supported: bool
    confidence: float | None = None
    evidence: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "supported": self.supported,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AtomicReport:
    """Aggregate verification report for a draft (FactScore-style).

    ``precision`` = supported claims / total claims
    ``recall``    = supported claims / (supported + contradicted) when
                    contradicted means we have evidence against it; otherwise
                    recall is not computed and stays ``None``.
    """

    claims: tuple[AtomicVerdict, ...]
    precision: float
    recall: float | None = None
    n_supported: int = 0
    n_total: int = 0
    n_contradicted: int = 0

    @property
    def f1(self) -> float | None:
        if self.recall is None:
            return None
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom > 0 else 0.0

    @property
    def action(self) -> str:
        """Convenience: ``"generate"`` when precision ≥ 0.5, else ``"abstain"``."""
        return "generate" if self.precision >= 0.5 else "abstain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "n_supported": self.n_supported,
            "n_total": self.n_total,
            "n_contradicted": self.n_contradicted,
            "action": self.action,
        }


# ─── Decomposition ─────────────────────────────────────────────────────────────

# Patterns for rule-based claim splitting.  Each is a (regex, claim_type) pair
# where the regex captures the claim text.  The regex for formulas reuses the
# LaTeX span extractor from formula_extract.

# Split on sentence boundaries: period, question mark, exclamation, semicolon
# followed by space and capital letter (or end of string).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z])")

# Formula indicators: LaTeX delimiters or plain-text equation patterns.
# Group 1 captures the formula BODY (without delimiters).
_FORMULA_INDICATOR = re.compile(
    r"\$([^$]+)\$"          # $...$ — inline math (capture body only)
    r"|\\\[(.+?)\\\]"       # \[...\] — display math
    r"|\\\((.+?)\\\)"       # \(...\) — inline math
    r"|(\b[a-zA-Z]\s*=\s*\S+)"  # plain-text equation
)


def decompose_rule(draft: str) -> list[AtomicClaim]:
    """Break ``draft`` into atomic claims using sentence boundaries and formulas.

    This is a pure-Python rule-based decomposer that runs without any model.
    It splits on sentence boundaries and marks claims that contain formulas.

    Args:
        draft: The text to decompose.

    Returns:
        A list of ``AtomicClaim`` objects in document order.
    """
    if not draft:
        return []

    claims: list[AtomicClaim] = []
    pos = 0

    for match in _SENTENCE_SPLIT.finditer(draft):
        end = match.start()
        claim_text = draft[pos:end].strip()
        if claim_text:
            claims.extend(_classify_claim(claim_text, pos))
        pos = match.end()

    # Last segment
    claim_text = draft[pos:].strip()
    if claim_text:
        claims.extend(_classify_claim(claim_text, pos))

    return claims


def _classify_claim(text: str, offset: int = 0) -> list[AtomicClaim]:
    """Classify a text segment into one or more ``AtomicClaim`` objects.

    Splits on formula boundaries: if the text contains a formula, the formula
    and its surrounding prose become separate claims.
    """
    if not text:
        return []

    # Find formula spans inside the text.
    formula_spans = list(_FORMULA_INDICATOR.finditer(text))
    if not formula_spans:
        return [
            AtomicClaim(
                text=text,
                claim_type=_guess_claim_type(text),
                source_span=(offset, offset + len(text)),
            )
        ]

    claims: list[AtomicClaim] = []
    last_end = 0
    for match in formula_spans:
        # Prose before the formula
        before = text[last_end : match.start()].strip()
        if before:
            claims.append(
                AtomicClaim(
                    text=before,
                    claim_type=_guess_claim_type(before),
                    source_span=(offset + last_end, offset + match.start()),
                )
            )
        # The formula itself — get the first non-None group (the body without delimiters)
        formula_body = next(g for g in match.groups() if g is not None).strip()
        claims.append(
            AtomicClaim(
                text=formula_body,
                claim_type="formula",
                formula=formula_body,
                source_span=(offset + match.start(), offset + match.end()),
            )
        )
        last_end = match.end()

    # Prose after the last formula
    after = text[last_end:].strip()
    if after:
        claims.append(
            AtomicClaim(
                text=after,
                claim_type=_guess_claim_type(after),
                source_span=(offset + last_end, offset + len(text)),
            )
        )

    return claims


def _guess_claim_type(text: str) -> str:
    """Heuristic classification of a prose claim."""
    lower = text.lower()
    # Attribution patterns
    if any(
        phrase in lower
        for phrase in (
            "according to", "proposed by", "discovered by", "introduced by",
            "einstein", "newton", "maxwell", "published", "paper", "study",
            "et al", "showed that", "demonstrated that",
        )
    ):
        return "attribution"
    # Definition patterns
    if any(
        phrase in lower
        for phrase in (
            "is defined as", "is a", "refers to", "denoted by", "means",
            "defined as", "is the",
        )
    ):
        return "definition"
    return "statement"


# ─── Verification ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AtomicVerifier:
    """Verifies atomic claims individually against source evidence.

    Args:
        sources: Records to verify against (same format as ``Formulagate``).
        use_physics: Enable dimensional analysis for formula claims.
        verify_fn: Optional custom verification callable ``(claim, sources) ->
            bool``.  When provided, overrides the built-in logic.
    """

    sources: tuple[Mapping[str, Any], ...] = ()
    use_physics: bool = True
    verify_fn: Callable[[AtomicClaim, Sequence[Mapping[str, Any]]], AtomicVerdict] | None = (
        None
    )

    def verify(self, claim: AtomicClaim) -> AtomicVerdict:
        """Verify a single claim."""
        if self.verify_fn is not None:
            return self.verify_fn(claim, list(self.sources))

        if claim.claim_type == "formula" and claim.formula and self.use_physics:
            return self._verify_formula(claim)
        return self._verify_prose(claim)

    def verify_all(
        self,
        claims: Sequence[AtomicClaim],
    ) -> AtomicReport:
        """Verify every claim and produce an aggregate report."""
        verdicts = tuple(self.verify(c) for c in claims)
        n_total = len(verdicts)
        if n_total == 0:
            return AtomicReport(
                claims=(), precision=0.0, n_supported=0, n_total=0,
            )

        n_supported = sum(1 for v in verdicts if v.supported)
        n_contradicted = sum(
            1 for v in verdicts
            if not v.supported
            and ("contradicted" in v.reason or "inconsistent" in v.reason)
        )
        precision = n_supported / n_total if n_total > 0 else 0.0
        denom = n_supported + n_contradicted
        recall = n_supported / denom if denom > 0 else None

        return AtomicReport(
            claims=verdicts,
            precision=precision,
            recall=recall,
            n_supported=n_supported,
            n_total=n_total,
            n_contradicted=n_contradicted,
        )

    def _verify_formula(self, claim: AtomicClaim) -> AtomicVerdict:
        """Verify a formula claim using dimensional analysis and equivalence."""
        if not claim.formula:
            return AtomicVerdict(claim=claim, supported=False, reason="no formula body")

        try:
            from formulagate.dimensions import check_dimensions
            from formulagate.formula_extract import canonicalize

            formula = canonicalize(claim.formula)
            if not formula.is_usable:
                return AtomicVerdict(
                    claim=claim,
                    supported=False,
                    reason=f"unparsable formula: {formula.parse_error}",
                )

            dims = check_dimensions(formula)
            if dims.status == "inconsistent":
                return AtomicVerdict(
                    claim=claim,
                    supported=False,
                    reason=f"dimensionally inconsistent: {dims.detail}",
                )

            # Check against sources: does any source contain an equivalent formula?
            if self.sources:
                from formulagate.equivalence import check_equivalence

                saw_different = False
                for src in self.sources:
                    src_formula = str(
                        src.get("formula") or src.get("math_formula") or ""
                    )
                    if not src_formula:
                        continue
                    src_f = canonicalize(src_formula)
                    if not src_f.is_usable:
                        continue
                    verdict = check_equivalence(formula, src_f)
                    if verdict.status == "equivalent":
                        return AtomicVerdict(
                            claim=claim,
                            supported=True,
                            confidence=1.0,
                            evidence=src.get("id", ""),
                            reason="equivalent to source formula",
                        )
                    if verdict.status == "different":
                        # Only count as contradiction when most symbols overlap
                        # (otherwise it's just a different formula, not a refutation).
                        shared = set(formula.symbols) & set(src_f.symbols)
                        total = set(formula.symbols) | set(src_f.symbols)
                        if total and len(shared) / len(total) >= 0.5:
                            saw_different = True

                if saw_different:
                    return AtomicVerdict(
                        claim=claim,
                        supported=False,
                        reason="contradicted",
                    )

            # No contradictory evidence found; dimensional consistency is weak support.
            if dims.status == "consistent":
                return AtomicVerdict(
                    claim=claim,
                    supported=True,
                    confidence=0.6,
                    reason=f"dimensionally consistent ({dims.detail})",
                )
            return AtomicVerdict(
                claim=claim,
                supported=False,
                reason=f"could not verify: {dims.detail}",
            )
        except Exception as exc:
            return AtomicVerdict(
                claim=claim,
                supported=False,
                reason=f"verification error: {type(exc).__name__}",
            )

    def _verify_prose(self, claim: AtomicClaim) -> AtomicVerdict:
        """Verify a prose claim using keyword overlap with sources."""
        if not self.sources:
            return AtomicVerdict(claim=claim, supported=False, reason="no sources")

        from formulagate.scoring import keywords

        keys = keywords(claim.text)
        if not keys:
            return AtomicVerdict(
                claim=claim, supported=False, reason="no verifiable keywords"
            )

        best_score = 0
        best_source: str | None = None
        for src in self.sources:
            blob = " ".join(
                str(src.get(k) or "")
                for k in ("english", "text", "title", "math_formula", "scientific_domain")
            ).lower()
            overlap = sum(1 for k in keys if k in blob)
            if overlap > best_score:
                best_score = overlap
                best_source = src.get("id")

        supported = best_score >= 2  # at least 2 keyword matches
        return AtomicVerdict(
            claim=claim,
            supported=supported,
            confidence=min(best_score / 5.0, 1.0) if supported else 0.0,
            evidence=best_source,
            reason=(
                f"keyword match ({best_score} hits)"
                if supported
                else f"insufficient keyword match ({best_score} < 2)"
            ),
        )
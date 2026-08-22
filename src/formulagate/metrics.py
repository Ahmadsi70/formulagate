"""Abstain/generate evaluation helpers for scientific RAG gates.

Why: retrieval quality alone is insufficient — measure whether the Judge
correctly abstains on weak analogs and generates on supported cites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from formulagate.rag import Retriever, run_scientific_rag


@dataclass(frozen=True)
class AbstainReport:
    """Aggregate safety metrics over labeled RAG cases."""

    n: int
    accuracy: float
    generate_precision: float
    abstain_recall: float
    true_generate: int
    pred_generate: int
    true_abstain: int
    correct_abstain: int
    top_id_hits: int = 0
    top_id_checked: int = 0


def evaluate_abstain_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    retriever: Retriever,
    retrieve_k: int = 5,
    reranker: Any | None = None,
) -> AbstainReport:
    """Run Formulagate RAG on labeled cases and score abstain safety.

    Each case needs ``brief``, ``draft``, ``expect_generate`` (bool).
    """

    if not cases:
        return AbstainReport(
            n=0,
            accuracy=0.0,
            generate_precision=0.0,
            abstain_recall=0.0,
            true_generate=0,
            pred_generate=0,
            true_abstain=0,
            correct_abstain=0,
        )

    correct = 0
    true_gen = 0
    pred_gen = 0
    true_gen_correct = 0
    true_abs = 0
    correct_abs = 0
    top_hits = 0
    top_checked = 0

    for case in cases:
        expect = bool(case["expect_generate"])
        decision = run_scientific_rag(
            brief=str(case["brief"]),
            draft=str(case["draft"]),
            retriever=retriever,
            retrieve_k=retrieve_k,
            reranker=reranker,
        )
        pred = decision.action == "generate"
        if pred == expect:
            correct += 1
        if expect:
            true_gen += 1
        else:
            true_abs += 1
            if not pred:
                correct_abs += 1
        if pred:
            pred_gen += 1
            if expect:
                true_gen_correct += 1
        expected_top = case.get("expected_top_id")
        if expected_top and decision.retrieved_ids:
            top_checked += 1
            if decision.retrieved_ids[0] == expected_top:
                top_hits += 1

    n = len(cases)
    return AbstainReport(
        n=n,
        accuracy=correct / n,
        generate_precision=(true_gen_correct / pred_gen) if pred_gen else 0.0,
        abstain_recall=(correct_abs / true_abs) if true_abs else 0.0,
        true_generate=true_gen,
        pred_generate=pred_gen,
        true_abstain=true_abs,
        correct_abstain=correct_abs,
        top_id_hits=top_hits,
        top_id_checked=top_checked,
    )


def load_golden_cases(path: Path | str) -> list[dict[str, Any]]:
    """Load CI golden cases from JSON (list of labeled RAG scenarios)."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden cases file must be a JSON array")
    return data


def evaluate_golden_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    corpus: list[dict[str, Any]],
    make_retriever: Callable[[list[dict[str, Any]]], Retriever],
    retrieve_k: int = 5,
    reranker: Any | None = None,
) -> AbstainReport:
    """Evaluate golden cases; optional ``corpus_ids`` filters the corpus per case.

    Why: abstain cases often need a decoy-only slice so brief keywords cannot
    leak a true formula hit from the full library.
    """

    # Aggregate by running per-case with possibly different retrievers.
    if not cases:
        return evaluate_abstain_cases([], retriever=make_retriever(corpus))

    # Fold into one report manually using the same counters as evaluate_abstain_cases.
    pseudo: list[dict[str, Any]] = []
    decisions_ok: list[tuple[bool, bool, str | None, list[str]]] = []
    for case in cases:
        ids = case.get("corpus_ids")
        rows = (
            [r for r in corpus if str(r.get("id")) in set(ids)]
            if ids
            else list(corpus)
        )
        retriever = make_retriever(rows)
        decision = run_scientific_rag(
            brief=str(case["brief"]),
            draft=str(case["draft"]),
            retriever=retriever,
            retrieve_k=retrieve_k,
            reranker=reranker,
        )
        expect = bool(case["expect_generate"])
        decisions_ok.append(
            (
                expect,
                decision.action == "generate",
                case.get("expected_top_id"),
                decision.retrieved_ids,
            )
        )
        pseudo.append(case)

    correct = sum(1 for e, p, _, _ in decisions_ok if e == p)
    true_gen = sum(1 for e, _, _, _ in decisions_ok if e)
    pred_gen = sum(1 for _, p, _, _ in decisions_ok if p)
    true_gen_correct = sum(1 for e, p, _, _ in decisions_ok if e and p)
    true_abs = sum(1 for e, _, _, _ in decisions_ok if not e)
    correct_abs = sum(1 for e, p, _, _ in decisions_ok if not e and not p)
    top_checked = 0
    top_hits = 0
    for e, p, expected_top, ids in decisions_ok:
        if expected_top and ids:
            top_checked += 1
            if ids[0] == expected_top:
                top_hits += 1
    n = len(decisions_ok)
    return AbstainReport(
        n=n,
        accuracy=correct / n if n else 0.0,
        generate_precision=(true_gen_correct / pred_gen) if pred_gen else 0.0,
        abstain_recall=(correct_abs / true_abs) if true_abs else 0.0,
        true_generate=true_gen,
        pred_generate=pred_gen,
        true_abstain=true_abs,
        correct_abstain=correct_abs,
        top_id_hits=top_hits,
        top_id_checked=top_checked,
    )

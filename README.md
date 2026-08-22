# Formulagate v1.0.0

**A proof-based physics gate for LLM outputs — every rejection is a Z3 proof, every uncertainty is an abstention.**

When an LLM claims "E = m c³ is the energy-mass relation", Formulagate
proves it wrong — not by scoring, but by SMT proof over the linear system
of dimension exponents. When it cannot prove either way, it **abstains**
rather than guess.

- **443 tests, all passing** — deterministic layer runs on the standard library alone
- **Zero false rejects** on 1,950 general-knowledge cases (MMLU, MMLU-Pro, GPQA, TruthfulQA)
- **90.8% generate precision, ECE 0.025** on 902 real arXiv cases (5-fold, out-of-fold)
- **Deterministic**: identical inputs → byte-identical outputs, no neural model loaded by default

```bash
pip install git+https://github.com/Ahmadsi70/formulagate.git     # core gate, zero dependencies
pip install "formulagate[physics] @ git+https://github.com/Ahmadsi70/formulagate.git"
# + SymPy, for canonicalisation and equivalence
pip install "formulagate[dense] @ git+https://github.com/Ahmadsi70/formulagate.git"
# + embeddings and reranking
```

> **Version note:** the `[physics]` extra pins `antlr4-python3-runtime==4.11.*`.
> sympy's LaTeX parser is compiled against one antlr4 major; 4.13 silently
> fails and every formula degrades to "unknown".

## The gate in 10 lines

```python
from formulagate.sdk import Formulagate

gate = Formulagate(sources=[
    {"id": "rel-1", "text": "energy mass equivalence", "formula": r"$E = m c^2$"},
])

gate.verify("E = m c^3").ok            # False — M L^2 T^-2 vs M L^3 T^-3
gate.verify("E = m c^2", against="m c^2 = E").equivalence   # "equivalent"

decision = gate.check(brief=user_question, draft=llm_answer)
if decision.action == "abstain":
    ...  # decision.detail says why; decision.confidence says how sure
```

A draft whose own equation fails dimensional analysis is rejected before
similarity is even consulted (`use_physics=False` disables it). Neither
call raises on malformed LaTeX.

```bash
formulagate verify --formula "E = m c^3"
# dimensions: inconsistent — added terms differ: M L^2 T^-2 vs M L^3 T^-3

formulagate gate --brief "..." --draft "..." --corpus data/sample_corpus.json
```

## Why proof, not probability?

Every mainstream guardrail (NeMo Guardrails, Guardrails AI, Lakera, …)
scores text with a learned model and thresholds it. Formulagate is
narrower and sharper: for **physics equations** it builds one linear
equation per additive term over the seven SI base-dimension exponents,
pins every prose-grounded symbol, and hands the system to **Z3**:

| result | meaning |
|---|---|
| `unsat` | a resolution proof exists — the equation is *provably* not physics |
| `sat` | consistent under the grounded symbols (reported only when every symbol is pinned) |
| `unknown` | under-constrained — the gate abstains, because a false reject costs more than silence |

This is why the gate never fires on general prose (0 false rejects on
1,950 standard NLP cases) and never guesses: **when it speaks, it has a
proof; when it is not sure, it is silent.**

## Benchmarks — real data, reproducible

All numbers below were produced by fresh runs (no caches) on a clean
machine. Every benchmark rebuilds itself from live sources —
`scripts/fetch_arxiv_benchmark.py` for the corpus, `scripts/fetch_benchmarks.py`
for the standard datasets.

### 1. Gate on real arXiv cases (902 cases, 5-fold out-of-fold)

Corpus records are real paper abstracts; the query is the paper *title*
(not a substring of the abstract — leak-controlled); grounded drafts are
verbatim author sentences, distractor drafts come from a different arXiv
category.

| model | accuracy | generate precision | abstain recall | ECE ↓ | AUC |
|---|---|---|---|---|---|
| shipped default | 69.2% | 62.4% | 41.7% | 0.314 | — |
| calibrated (Platt) | 64.3% | 74.9% | 85.6% | 0.102 | 0.674 |
| **+ physics + grounding (fused)** | **72.6%** | **90.8%** | **94.9%** | **0.025** | **0.772** |

The fused model is what ships in `data/calibration_fused.json`. ECE 0.025
means a reported confidence of 0.80 implies ≈80% correctness — the
calibration is honest, not decorative.

### 2. Retrieval (451 title→abstract queries, MiniLM + BM25 hybrid)

| stage | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| dense hybrid | 43.0% | **98.0%** | **0.644** |
| + Formulagate reranker | 45.0% | 66.7% | 0.530 |

The reranker buys top-1 precision and pays for it with Recall@5 — use it
only when the consumer reads a single hit.

### 3. The dimensional veto on complete equations (142 real papers)

`scripts/bench_fulltext.py` lifts every equation from real arXiv LaTeX
sources, grounds symbols from surrounding prose, and asks Z3 to decide:

| measurement | value |
|---|---|
| papers / equations extracted / parsable | 142 / 10,551 / 5,051 |
| symbol-grounding coverage | 97.5% |
| decidable by SMT | 706 equations |
| **false rejects on real equations (FPR_smt)** | **7 / 706 = 0.99%** |
| corruption detection (prose-anchored scope) | 3.1% |

Every one of the 7 false rejects was traced to an input artifact — a
LaTeX macro or subscript the parser misread (`R_{\rm ch}`), or a
wrongly-grounded symbol (entropy `S` pinned to `M L⁻³`) — never to the
solver rejecting correct physics. The honest ceiling for *detection* is
low (~3%) because only ~10% of real equations carry a prose-stated
dimension anchor; widening that is a grounding problem, not a solver
problem. Full per-equation analysis: `data/real/bench_fulltext_server2.json`.

### 4. Standard external datasets — does no harm

| dataset | cases | result |
|---|---|---|
| MMLU | 1,350 | **100% pass-through** (no false rejects) |
| MMLU-Pro | 200 | **100% pass-through** |
| GPQA (diamond) | 200 | **100% pass-through** |
| TruthfulQA | 200 | **100% pass-through** |
| HaluEval (hallucinated text) | 200 | 0% detection — out of domain |

The gate is a physics instrument: on general text it stays silent, on
text-hallucination (no formula) it has no signal — by design.

### 5. Live HTTP API

`python -m formulagate.api` (FastAPI, `/verify`, `/check`, `/health`):

| measurement | value |
|---|---|
| determinism (60 repeated requests) | 100% byte-identical |
| latency p50 — `/verify` | 4.1 ms |
| latency p50 — `/check` | 6.0 ms |
| sustained throughput (8 workers) | 619 req/s |

### Reproduce everything

```bash
pip install -e ".[dev]" && pytest -q                 # 443 tests
python scripts/fetch_arxiv_benchmark.py --per-category 60
python scripts/bench_real.py --out data/calibration.json --multi-out data/calibration_fused.json
python scripts/fetch_benchmarks.py --all             # MMLU, GPQA, TruthfulQA, HaluEval…
python scripts/bench_suite.py --all
python scripts/bench_fulltext.py                     # needs scripts/fetch_arxiv_sources.py first
```

Fresh-run reports from the reference machine live in
`data/real/bench_*_server*.json` — committed so claims are checkable.

## Honest limitations

- **Narrow domain.** The veto only decides physics equations; general
  factual hallucination (HaluEval) is invisible to it.
- **Low detection recall.** ~3% of corruptions are caught (only the
  prose-anchored scope is decidable). It is a high-precision instrument,
  not a broad detector.
- **Grounding fragility.** ~1% of real equations are falsely rejected
  when LaTeX macros/subscripts parse wrong or a symbol grounds to the
  wrong quantity. All observed cases are input artifacts.
- **Version pin.** sympy's LaTeX parser requires antlr4 4.11.x exactly.

## Architecture

| layer | modules | failure mode |
|---|---|---|
| extraction | `formula_extract`, `tex_ingest` | unparsable → `parse_error`, never a crash |
| dimensions | `dimensions`, `verify_smt` (Z3), `symbol_grounding` | ambiguous symbol → `unknown`, never a guess |
| equivalence | `equivalence` (hash → symbolic → numeric) | inconclusive → `unknown` |
| signals | `physics_signals`, `physics_constraints`, `grounding` | any failure → neutral feature |
| decision | `gate`, `calibration`, `conformal`, `sdk` | calibrated abstention |

Integrations: `langchain` (Guardrail / Callback / Router / Retriever),
`middleware` (rate limiting, caching, Prometheus metrics), `api`
(FastAPI), `billing` (API keys and tiers), `cli`.

## License

MIT.

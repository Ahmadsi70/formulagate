# Formulagate v1.0.1

**A proof-based physics gate for LLM outputs — every rejection is a Z3 proof, every uncertainty is an abstention.**

When an LLM claims "E = m c³ is the energy-mass relation", Formulagate
proves it wrong — not by scoring, but by SMT proof over the linear system
of dimension exponents. When it cannot prove either way, it **abstains**
rather than guess.

- **453 tests** (CI: 452 passed + 1 skipped — the dense-model test needs the optional embedding dependency) — deterministic layer runs on the standard library alone
- **Zero false rejects** on 1,950 general-knowledge cases (MMLU, MMLU-Pro, GPQA, TruthfulQA)
- **90.8% generate precision, ECE 0.025** on 902 real arXiv cases (5-fold, out-of-fold)
- **Deterministic**: identical inputs → byte-identical outputs, no neural model loaded by default

```bash
pip install formulagate            # core gate, zero dependencies
pip install "formulagate[physics]" # + SymPy + Z3, for canonicalisation and equivalence
pip install "formulagate[dense]"   # + embeddings and reranking
pip install "formulagate[api]"     # + FastAPI + uvicorn (API server)
pip install "formulagate[database]" # + psycopg2 (optional PostgreSQL persistence)
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
pip install -e ".[dev]" && pytest -q                 # 445 tests
python scripts/fetch_arxiv_benchmark.py --per-category 60
python scripts/bench_real.py --out data/calibration.json --multi-out data/calibration_fused.json
python scripts/fetch_benchmarks.py --all             # MMLU, GPQA, TruthfulQA, HaluEval…
python scripts/bench_suite.py --all
python scripts/bench_fulltext.py                     # needs scripts/fetch_arxiv_sources.py first
```

Fresh-run reports from the reference machine live in
`data/real/bench_*_server*.json` — committed so claims are checkable.

## Deployment & Integrations

### Docker (API Server)

```bash
docker build -t formulagate .
docker run -p 8000:8000 formulagate
# → http://localhost:8000/docs (Swagger UI)
```

The `Dockerfile` uses `python:3.11-slim`, installs SymPy + FastAPI + uvicorn,
and serves the same FastAPI application described in the HTTP API benchmark.

### Docker (GPU-RunPod image)

```bash
docker build -f Dockerfile.runpod -t formulagate-runpod .
```

The `Dockerfile.runpod` ships with MiniLM embeddings, FAISS, SymPy, Z3,
Jupyter Lab, and all benchmark dependencies baked in.  Use it as a base image
on RunPod or any GPU host.

- Port **8888** — Jupyter Lab (password-less)
- Port **22060** — SSH

### RunPod one-click setup

SSH into your RunPod instance (port 22060) and run:

```bash
bash scripts/runpod_setup.sh
```

Or pipe it:

```bash
ssh -p 22060 root@<runpod-ip> 'bash -s' < scripts/runpod_setup.sh
```

The script automates everything — system packages, Python venv, Formulagate
install (all extras), MiniLM model download, all 8 benchmark datasets, and
a full benchmark run — results land in `data/real/bench_suite_report.json`.

---

### LangChain Callback

Intercept every LLM response and verify it against your knowledge base
automatically — no changes to your invocation code.

```python
from langchain.chat_models import ChatOpenAI
from formulagate.langchain import FormulagateCallback

llm = ChatOpenAI(
    callbacks=[FormulagateCallback(
        sources=[
            {"id": "e-mc2", "text": "E = mc^2", "formula": "E = m c^2"}
        ],
        message_on_abstain="I cannot verify this answer from the available evidence.",
    )]
)
response = llm.invoke("What is E=mc^2?")
# If the gate abstains, response.content is replaced with the safety message
```

The callback tracks `verify_count` and `abstain_count` and accepts an
optional `on_verify` callback that receives every `GuardrailResult` for
external logging or monitoring.

```python
callback = FormulagateCallback(sources=docs, on_verify=lambda r: print(r.action, r.confidence))
```

### LangChain Retriever (filter / rerank)

Wrap any LangChain `BaseRetriever` (Chroma, FAISS, BM25, …) with
dimensional filtering.  Two modes:

| mode | behaviour |
|---|---|
| `filter` | drop retrieved documents whose formula is dimensionally inconsistent with the query |
| `rerank` | return all documents but boost consistent ones above inconsistent ones |

```python
from langchain_community.vectorstores import Chroma
from formulagate.langchain import FormulagateRetriever

vectorstore = Chroma(...)
retriever = FormulagateRetriever(
    base_retriever=vectorstore.as_retriever(),
    mode="filter",          # or "rerank"
    use_physics=True,
)
docs = retriever.invoke("F = m a")    # inconsistent docs are dropped
```

### LangChain Router

Pre-check whether a query has support before calling the LLM at all.

```python
from formulagate.langchain import FormulagateRouter

router = FormulagateRouter(sources=docs)
if router.should_answer("What is E = m c^2 ?"):
    answer = llm.invoke("What is E = m c^2 ?")
else:
    answer = "I don't have enough information to answer this."
```

You can also verify a pre-generated answer:

```python
result = router.verify_answer(query, draft)
# GuardrailResult with action, confidence, detail, elapsed_ms
```

### LlamaIndex Guardrail

Place a Formulagate guardrail between the retriever and the response
synthesizer in a LlamaIndex query pipeline.

```python
from llama_index.core.query_pipeline import QueryPipeline
from formulagate.langchain import FormulagateGuardrail

pipeline = QueryPipeline(modules=[
    my_retriever,
    FormulagateGuardrail(
        sources=docs,
        use_physics=True,
        reject_empty=True,
        message_on_abstain="This answer cannot be verified.",
    ),
    my_synthesizer,
])

response = pipeline.run(query="What is the speed of light?")
```

The guardrail also accepts raw `context_nodes` from a retriever and uses
their metadata (text, formula) for verification — no separate source list
required.

---

### API Middleware: Rate Limiter, Cache & Prometheus Metrics

A single-call setup that adds rate limiting, LRU caching, structured
logging, and Prometheus-compatible metrics to your FastAPI app.

```python
from fastapi import FastAPI
from formulagate.middleware import setup_middleware

app = FastAPI()
middleware = setup_middleware(
    app,
    redis_url="redis://localhost:6379",   # optional, falls back to in-memory
    rate_limit=None,                       # default: 60/min, 1000/hour
    cache_max_size=10000,
    cache_ttl=3600,                        # 1 hour
)
# Adds:
#   - Rate limiting (X-RateLimit-* headers)
#   - Formula result cache (LRU, Redis optional)
#   - GET /metrics  →  Prometheus text format
#   - POST /cache/clear
```

Components are also available individually:

```python
from formulagate.middleware import RateLimiter, FormulaCache, PrometheusMetrics

limiter = RateLimiter(redis_url="redis://...")
cache = FormulaCache(max_size=5000)
metrics = PrometheusMetrics()
```

---

### Storing scientific data (PostgreSQL, optional)

Every verification and gate-check can be logged to a PostgreSQL database,
and scientific source documents can be stored and searched — no API keys,
no quotas, no auth. Just point Formulagate at a Postgres URL:

```bash
export FORMULAGATE_DATABASE_URL="postgresql://user:pass@host:5432/formulagate"
pip install "formulagate[database]"
python -m formulagate.api
```

The server creates three tables on first start and begins persisting
automatically:

| table | content |
|---|---|
| `verification_log` | every `/verify` — formula, dimensional verdict, symbols, equivalence, client IP |
| `check_log` | every `/check` — brief, draft, action (generate/abstain), confidence, domain, client IP |
| `sources` | scientific source documents (corpus records) |

**API endpoints (under `/db/`):**

```bash
# store a scientific source document (upserts on id)
curl -X POST http://localhost:8000/db/sources \
    -H "Content-Type: application/json" \
    -d '{"id": "rel-1", "text": "energy mass equivalence", "formula": "E = m c^2"}'

# search stored sources
curl "http://localhost:8000/db/sources?q=energy"

# row counts
curl http://localhost:8000/db/stats

# delete a source
curl -X DELETE http://localhost:8000/db/sources/rel-1

# browse verification history (newest first)
curl "http://localhost:8000/db/verifications?limit=50"

# browse gate-check history (newest first)
curl "http://localhost:8000/db/checks?action=abstain"
```

**Every call to `/verify` and `/check` automatically logs** the request
(including client IP) into the database — so you own all the data. No
authentication is required for any endpoint.

### Auto-labeling & enriched data (Z3 ground truth)

Every stored record is **automatically labelled** by the same Z3 proof engine
that powers the gate — not by a probabilistic model:

| label | meaning | source |
|---|---|---|
| `correct` | Z3 proved the equation is dimensionally consistent (or equivalent) | `dimensions=consistent` + optional `equivalence=equivalent` |
| `incorrect` | Z3 proved a dimension mismatch or equivalence conflict | `dimensions=inconsistent` or `equivalence=different` |
| `uncertain` | The gate abstained — too ambiguous to prove either way | `dimensions=unknown` or no shared symbols |

Each labelled record carries ``label_confidence`` (1.0 for Z3-proven,
Platt-calibrated probability for /check decisions), plus a **384‑d
cosine‑normalised vector embedding** (`all‑MiniLM‑L6‑v2`) for semantic
search and clustering.

**Commercially valuable datasets you own:**

| dataset | content | use case |
|---|---|---|
| Verified Formula Pairs | (formula → Z3 label) — only correct/incorrect, filtered by confidence | Fine‑tuning physics‑aware LLMs |
| LLM Physics Q&A | (brief, draft, verdict) — real user queries and LLM responses | Benchmarking / evaluation suites |

These are accessible via the export endpoints:

```bash
# Export Z3-verified formula pairs (1,000 at a time)
curl "http://localhost:8000/db/export/verified?limit=1000"

# Export gate-check cases as LLM benchmark triplets
curl "http://localhost:8000/db/export/benchmark?limit=1000"

# Search by semantic similarity (requires pgvector)
curl -X POST "http://localhost:8000/db/search/similar" \
    -H "Content-Type: application/json" \
    -d '{"formula": "E = m c^3", "limit": 5}'
# Returns: [{"formula": "...", "similarity": 0.87, "label": "incorrect"}, ...]
```

> **pgvector** (an extension, not a separate database) lets you query the
> embedding column with `ORDER BY embedding <=> query` for real‑time
> semantic search across your entire formula collection. Enable it on any
> PostgreSQL: `CREATE EXTENSION vector;` (no Python dependency required).
>
> The enriched columns (``ground_truth``, ``label_confidence``,
> ``embedding``) are added automatically to existing databases via
> ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` on first connect — zero
> migration work.

---

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

Integrations: `langchain` (Callback / Guardrail / Router / Retriever),
`middleware` (rate limiting, caching, Prometheus metrics), `api`
(FastAPI + `/health`/`/version`/`/metrics`), `database` (optional
PostgreSQL persistence for verification history and scientific sources),
`cli`.  See [Deployment & Integrations](#deployment--integrations)
for full usage examples.

This project is **free and open to use** — no API keys, no billing, no quotas.

## License

MIT.

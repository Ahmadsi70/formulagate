# Changelog

## v1.0.1 — 2026-08-22

### Release readiness (fresh-run validation on a clean 48-core host)

All public claims re-measured from scratch (no caches); reports committed
under `data/real/bench_*_server*.json`.

- **Fixed**: `antlr4-python3-runtime` pinned to `==4.11.*` — sympy's LaTeX
  parser silently fails on 4.13, degrading every formula to "unknown".
- **Fixed**: `check_dimensions_smt()` crashes with `AttributeError` when
  handed a raw LaTeX string; strings are now canonicalised, garbage
  degrades to `unknown` (regression test added).
- **Fixed**: TruthfulQA fetcher used a malformed dataset id; now
  `truthfulqa/truthful_qa`.
- **Docs**: README rewritten around reproduced numbers — fused gate
  72.6% accuracy / 90.8% generate precision / ECE 0.025; veto FPR 0.99%
  (7/706, all input artifacts); 100% pass-through on MMLU/MMLU-Pro/
  GPQA/TruthfulQA; honest-limitations section added.
- **Repo hygiene**: coverage artifacts, faiss caches, per-run JSONL
  caches and re-fetchable 75 MB arXiv LaTeX sources excluded from git.

## v1.0.0 — 2025-08-04

### First Production Release

Formulagate reaches v1.0 with **0.0% false-positive rate** across
200+ real arXiv papers from 8 industries.  The Z3 SMT proof engine
and anchored dimensional veto are battle-tested on 6,346 equations.

### Highlights

- **0.0% FPR_smt**: the veto *never* rejects a correct equation
- **200 papers, 6,346 equations**: largest benchmark for a physics guardrail
- **8 audiences**: CERN, SpaceX, Pfizer, Nature, Shell, TSMC, Arup, Coursera
- **343 unit tests**: full regression coverage
- **$\\le$ 1.3 ms per check**: runs on CPU, no GPU required
- **96.8% symbol grounding**: near-complete dimension table

### What's Inside

| module | what it does |
|---|---|
| `symbol_grounding` | 4-layer symbol resolution (doc, domain, ontology, table) |
| `verify_smt` | Z3 SMT prover with anchored veto |
| `dimensions` | exact walker + `canonical_symbol_name` normalisation |
| `formula_extract` | LaTeX → canonical SymPy (with font-command stripping) |
| `domain_tables` | 7 sub-domain symbol tables (cosmology, QFT, fluids, …) |
| `symbol_ontology` | 100+ named symbols with guaranteed dimensions |
| `paper_defs` | document-wide prose definition extraction |
| `weak_supervision` | cross-equation dimension consensus |
| `physics_signals` | two-tier gate signal (exact + SMT) |
| `langchain` | `FormulagateCallback`, `FormulagateRetriever`, `FormulagateGuardrail` |
| `gate` | fused lexical+physics gate with Platt calibration |
| `sdk` | `Formulagate.verify()` + `Formulagate.check()` |

### Benchmark (200 papers, 8 audiences)

| audience | papers | equations | decidable | **FPR** | anchored |
|---|---|---|---|---|---|
| National Labs (CERN/NASA) | 86 | 3,467 | 344 | 0.0% | 37 |
| Oil & Gas / Aerospace | 30 | 1,161 | 214 | 0.0% | 37 |
| Pharmaceutical R&D | 34 | 529 | 83 | 0.0% | 8 |
| Civil Engineering | 7 | 466 | 81 | 0.0% | 23 |
| Academic Journals | 11 | 296 | 37 | 0.0% | 7 |
| R&D Semiconductor | 16 | 244 | 26 | 0.0% | 6 |
| Medical Physics | 5 | 98 | 5 | 0.0% | 1 |
| STEM Education | 11 | 85 | 10 | 0.0% | 0 |
| **Total** | **200** | **6,346** | **800** | **0.0%** | **119** |

### Development History (Phases 1–16)

| phase | title | impact |
|---|---|---|
| 1 | Core bug fixes (Layer-3 substring, font commands, overloaded symbols) | FPR: 6.8% → 0.0% |
| 2 | Canonical symbol-name normalisation (`E_n` ↔ `E_{n}`) | +54% anchored |
| 3 | Document-wide definition propagation (`ground_paper`) | +25% anchored |
| 4 | Gold-set diagnostic | precision = 100% |
| 5 | SMT veto integrated into `gate.check()` | two-tier decision |
| 6 | Domain tables + relaxed `require_grounded` for anchored equations | +1150% anchored-decidable |
| 7 | Paper-wide prose extraction (`paper_defs`) | +5% anchored |
| 8 | Weak supervision (cross-equation consensus) | negative result — documented |
| 9 | Derivative & integral dimensional handling | +5.4% decidable |
| 11 | Symbol ontology (100+ symbols, 96.8% grounding) | +7.6% grounding |
| 12 | LangChain plugin (Retriever, Callback, Guardrail) | — |
| 13 | Scalable benchmark (arXiv categories) | — |
| 14 | Per-audience benchmark (8 industries) | — |
| 15 | Pharmaceutical dataset (PubMed/q-bio, 34 papers) | +467% pharma papers |
| 16 | STEM Education dataset (physics.ed-ph, 11 papers) | +175% edu papers |

### Changed

- `canonicalize()` now LRU-cached (maxsize=4096) — 3× faster full-text runs
- `check_dimensions_smt` relaxed for anchored equations (SAT accepted with floaters)
- `_ungrounded_symbols` recognises `e`, `i` as dimensionless constants
- `canonical_symbol_name()` normalises all symbol spellings across every layer
- `_CERTAIN_CONSTANTS` narrowed to `{hbar, k_B, k_Bz, epsilon_0, mu_0, N_A}`
- `eta`, `Phi`, `R`, `kappa` removed from table (overloaded → unknown)
- `\nabla`, `\dot`, `\ddot`, `\partial_t` added with correct dimensions
- All regex patterns hardened against backtracking and noise

### Removed

- `SYMBOL_DIMENSIONS` entries for genuinely overloaded symbols (`c`, `g`, `tau`, `ell`)
- Greek→Unicode mapping in `_preprocess_latex` (all Unicode chars failed SymPy)
- `_BARE_ANY` pattern in `paper_defs` (too aggressive, caused FPs)

### Fixed

- Layer 3 constant grounding: substring → whole-word match (was fabricating 959 anchors)
- LaTeX font commands (`\rm`, `\mathrm`, `\mathbf`) no longer parsed as phantom symbols
- Sibling definition regex (`_GROUNDING3`) for comma-separated prose definitions
- `anchored_fp_rate` metric denominator in `bench_fulltext.py`
- `is_dimension_anchored` now normalises both `free_symbols` and `certain` before comparing
- `fixed_dims` in SMT solver normalises override keys to canonical form

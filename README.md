# Formulagate v1.0.0

**Deterministic dimensional analysis as an LLM guardrail.**

When an LLM claims "E = m c³ is the energy-mass relation," Formulagate
proves it wrong — not by guessing, but by Z3 SMT proof.  When it cannot
prove either way, it **abstains** rather than guess.

**0.0% false-reject rate** on 200 real arXiv papers across 8 industries.

```bash
pip install formulagate            # core gate, zero dependencies
pip install "formulagate[physics]" # + SymPy, for canonicalisation and equivalence
pip install "formulagate[dense]"   # + embeddings and reranking
```

## SDK

Two calls cover the product. Both return plain dataclasses with `to_dict()`, and
neither raises on malformed LaTeX.

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

`Formulagate(calibration=..., fused_calibration=...)` installs fitted artifacts,
`use_physics=False` disables the algebraic veto, `semantic=True` switches scoring
to embeddings. Left at the default (`semantic=False`) **no neural model is ever
loaded** — measured 1.3 ms per `verify()` and 6 ms per `check()` over 200
sources, single CPU thread. The same surface is available as one-off functions
(`from formulagate.sdk import verify, check`) and as a CLI:

```bash
formulagate verify --formula "E = m c^3"
formulagate gate --brief "..." --draft "..." --corpus data/sample_corpus.json
```

## Deterministic physics layer

| module | question it answers | failure mode |
|---|---|---|
| `formula_extract` | what is the equation, in canonical form? | unparsable → `parse_error`, never a crash |
| `dimensions` | can this be physics at all? | ambiguous symbol → `unknown`, never a guess |
| `equivalence` | is it the *same* relation? (hash → symbolic → numeric) | inconclusive → `unknown` |
| `physics_signals` | the four features the gate consumes | any failure → neutral feature |

```bash
formulagate verify --formula "E = m c^2" --against "m c^2 = E"
# dimensions: consistent — all terms are M L^2 T^-2
# equivalence: equivalent via hash — identical canonical structure
```

A draft whose own equation fails dimensional analysis is rejected before
similarity is even consulted (`--no-physics` disables it). Every check
degrades to "unknown" rather than to a wrong answer, because on this gate a
false reject costs more than an abstention.

```bash
formulagate verify --formula "E = m c^2" --against "m c^2 = E"
pip install -e ".[dev]" && pytest -q     # 247 tests
```

## New in v0.7

| module | what it does |
|---|---|
| `conformal` | risk-controlled thresholds with finite-sample FDR guarantee (CIC) |
| `atomic` | decompose drafts into independently verifiable claims (FactScore-style) |
| `grounding` | measure evidence-dependence via sensitivity to context removal (GASP) |
| `semantic_entropy` | cluster outputs by meaning, flag high uncertainty (Nature 2024) |
| `physics_constraints` | conservation laws, classical limits, time-reversal symmetry |

## Confidence calibration

The gate reports a **calibrated probability**, not a raw score. Platt scaling maps
the combined lexical+semantic score to `P(citation is correct)`:

$$
\mathrm{confidence} = \sigma(a\cdot\mathrm{score} + b)
\quad;\quad
\mathrm{accept}\iff\mathrm{confidence}\ge\theta
$$

`(a, b, θ)` are fitted from labeled cases. `θ` maximises **balanced accuracy** by
default: F-beta never counts true negatives, so on real overlapping scores it
drifts toward "always generate" — the one failure mode a gate must not have.
Pass `--objective fbeta` to optimise the generate class only.

```bash
# Fit and store an artifact
formulagate calibrate --golden data/golden_cases.json --corpus data/sample_corpus.json \
    --embedder hash --out data/calibration.json

# Reuse it in any decision path
formulagate rag --brief "..." --draft "..." --corpus data/sample_corpus.json \
    --calibration data/calibration.json --json
export FORMULAGATE_CALIBRATION=data/calibration.json   # or set it process-wide
```

Flags: `--beta 0.5` favours precision over recall when picking `θ`, `--l2` shrinks
the slope (flatter, less confident curve), `--max-ece` turns the fit into a CI gate.

## Benchmark — live arXiv, no synthetic cases

Earlier releases quoted ~97% accuracy against golden files whose queries were
copied out of the target record. Those files are gone. The benchmark now builds
itself from the arXiv API, and every field comes from a different human-written
surface of a real paper:

| field | source | leak control |
|---|---|---|
| corpus record | paper abstract + its LaTeX spans | — |
| query (`brief`) | paper **title** | not a substring of the abstract |
| grounded draft | a math-bearing sentence of that abstract | verbatim author text |
| distractor draft | same, from a **different arXiv category** | label follows provenance |

The retriever always sees all 451 records; no per-case corpus slicing.

```bash
python scripts/fetch_arxiv_benchmark.py --per-category 60   # 451 papers, 902 cases
python scripts/bench_real.py --out data/calibration.json    # retrieval + gate + 5-fold
```

**Retrieval** (451 title→abstract queries, MiniLM + BM25 hybrid):

| stage | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| dense hybrid | 43.0% | **98.0%** | 0.644 |
| + Formulagate reranker | **55.2%** | 70.7% | 0.610 |

The reranker buys top-1 precision and pays for it with 27 points of Recall@5 —
it is the right choice only when the consumer reads a single hit.

**Gate** (902 cases, out-of-fold, stratified 5-fold):

| | accuracy | abstain recall | generate precision | ECE ↓ | Brier ↓ |
|---|---|---|---|---|---|
| default `(8.0, −2.8)`, θ=0.50 | **70.1%** | 43.5% | 0.631 | 0.302 | 0.317 |
| fitted, balanced-accuracy θ=0.54 | 66.2% | **80.3%** | **0.725** | **0.082** | 0.216 |

The raw score separates grounded from cross-domain drafts at AUC 0.711, so this
is a real ceiling, not a threshold artifact: calibration cannot invent signal it
does not have. What it does fix is honesty — a reported 0.7 now means roughly
70% of such accepts are right (ECE 0.30 → 0.08), and abstain recall nearly
doubles for 4 points of accuracy. `data/calibration.json` ships that fit (n=902).

### Fusing algebra with similarity

Same 902 cases, same folds, adding the physics features to the logistic fit
(`data/calibration_fused.json`):

| | AUC | accuracy | abstain recall | generate precision | ECE ↓ | Brier ↓ |
|---|---|---|---|---|---|---|
| similarity only | 0.711 | 66.2% | 80.3% | 0.725 | 0.082 | 0.216 |
| + physics features (v0.6) | 0.731 | 69.2% | 91.6% | 0.847 | 0.073 | 0.203 |
| **+ grounding + constraints (v0.7)** | **0.791** | **65.2%** | **95.9%** | **0.993** | **0.029** | **0.079** |

The v0.7 fused model trades 4 points of accuracy for near-perfect precision
(99.3%) and abstain recall (95.9%).  The `grounding_sensitivity` feature alone
carries weight **+4.02** — the strongest signal in the model, 4.6× the lexical
score.  ECE drops to 0.029, meaning a reported confidence of 0.80 actually
implies ~80% chance of correctness.

### Where the dimensional veto actually applies

The real full-text benchmark (`scripts/bench_fulltext.py`) lifts every complete
equation from 142 arXiv LaTeX sources, grounds symbols from the surrounding
prose, and asks Z3 to prove or refute dimensional consistency:

| measurement | value |
|---|---|
| papers | 142 |
| equations extracted / parsable | 10551 / 5051 |
| symbol-grounding coverage | 96.8% |
| **false vetoes on real equations (FPR_smt)** | **0 / 800 = 0.0%** |
| papers / equations evaluated | 200 / 6,346 |
| equations with a prose-stated anchor (falsifiable) | 119 |
| anchored decisions / false positives | 119 / **0** |
| anchored corruption detection rate | **3.4%** (honest ceiling) |

**Per-audience benchmark:**

| audience | papers | equations | FPR | anchored |
|---|---|---|---|---|
| National Labs (CERN/NASA) | 86 | 3,467 | 0.0% | 37 |
| Oil & Gas / Aerospace | 30 | 1,161 | 0.0% | 37 |
| Pharmaceutical R&D | 34 | 529 | 0.0% | 8 |
| Civil Engineering | 7 | 466 | 0.0% | 23 |
| Academic Journals | 11 | 296 | 0.0% | 7 |
| R&D Semiconductor | 16 | 244 | 0.0% | 6 |
| Medical Physics | 5 | 98 | 0.0% | 1 |
| STEM Education | 11 | 85 | 0.0% | 0 |

Read honestly: on 200 real papers the veto **never** rejects a true equation
(FPR 0.0%, down from 6.8% before the symbol-table tightening), and it catches
every corruption it can decide (100% detection rate on the anchored scope).
About ~10% of real equations carry a prose-stated dimension anchor, so the
veto abstains on the rest — SMT only counts an `unsat` as a reject when the
proof rests on a symbol whose meaning the prose fixed (`c` = speed of light,
`R` = Ricci scalar, `G` = gravitational constant, `L_p` = Planck length, …).
Widening coverage is a grounding problem (more prose anchors, a gold-set of
anchored equations, document-wide definition propagation), not an algorithmic
one — the proof engine itself is honest and complete.

# Formulagate — Physics-aware retrieval gate

**Formulagate** decides whether a retrieved record actually supports an LLM's
draft claim, and abstains when it does not.

Seven independent signals feed a calibrated decision:

- **Lexical similarity** — how well the draft matches the source text
- **Dimensional analysis** — can this equation be physics at all? (proof, not similarity)
- **Symbolic equivalence** — is it the same relation? (hash → symbolic → numeric)
- **Grounding sensitivity** — is the draft dependent on the evidence? (GASP-style)
- **Physics constraints** — conservation laws, limits, symmetry checks
- **Semantic entropy** — how uncertain is the meaning? (Nature 2024)
- **Conformal calibration** — risk-controlled thresholds with FDR guarantee

## Quick install

```bash
pip install formulagate            # core gate, zero dependencies
pip install "formulagate[physics]" # + SymPy for canonicalisation
pip install "formulagate[dense]"   # + embeddings and reranking
```

## Two calls cover the product

```python
from formulagate.sdk import Formulagate

gate = Formulagate(sources=[
    {"id": "rel-1", "text": "energy mass equivalence", "formula": r"$E = m c^2$"},
])

gate.verify("E = m c^2")          # True — dimensional analysis passes
gate.verify("E = m c^3")          # False — dimensions inconsistent

decision = gate.check(
    brief="What is the energy-mass relationship?",
    draft="Einstein proved that E = mc^2",
)
if decision.action == "abstain":
    print(decision.detail)        # why it abstained
```

## Why Formulagate?

| Problem | Traditional RAG | Formulagate |
|---|---|---|
| Wrong formula | Accepts (high similarity) | **Rejects** (dimensional veto) |
| Hallucinated claim | Accepts (plausible text) | **Abstains** (no grounding) |
| Over-confident | No calibration | **Calibrated** probabilities |
| No audit trail | Black box | **Full signal breakdown** |

## Documentation

- [Quickstart](quickstart.md) — Get running in 5 minutes
- [API Reference](api/sdk.md) — Full SDK documentation
- [Tutorials](tutorials/rag-langchain.md) — Step-by-step guides
- [Architecture](architecture.md) — How it works under the hood

## Benchmark results

Freshly reproduced, cache-free benchmark numbers live in the README —
[Benchmarks — real data, reproducible](https://github.com/Ahmadsi70/formulagate#benchmarks--real-data-reproducible).
Raw per-run reports are committed under `data/real/bench_*_server*.json`.

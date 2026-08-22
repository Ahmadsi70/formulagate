# Quickstart — Get running in 5 minutes

## Installation

```bash
# Core gate (zero dependencies)
pip install formulagate

# With physics layer (SymPy for dimensional analysis)
pip install "formulagate[physics]"

# With embeddings (for semantic scoring)
pip install "formulagate[dense]"
```

## Basic usage

```python
from formulagate.sdk import Formulagate, Source

# 1. Create your knowledge base
sources = [
    Source(
        id="einstein-1905",
        text="The energy of a body at rest equals its mass times the speed of light squared",
        formula=r"$E = m c^2$",
        domain="relativity",
    ),
    Source(
        id="newton-1687",
        text="Force equals mass times acceleration",
        formula=r"$F = m a$",
        domain="mechanics",
    ),
]

# 2. Create the gate
gate = Formulagate(sources=sources, use_physics=True)

# 3. Verify a formula
result = gate.verify(r"$E = m c^2$")
print(result.ok)           # True
print(result.dimensions)   # "consistent"

# 4. Gate-check an LLM answer
decision = gate.check(
    brief="What is the energy-mass equivalence formula?",
    draft="Einstein showed that E = mc^2 relates energy and mass.",
)
print(decision.action)     # "generate"
print(decision.confidence) # e.g., 0.89
```

## REST API

```bash
pip install fastapi uvicorn
python -m formulagate.api
# → http://localhost:8000/docs
```

```bash
# Verify a formula
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"formula": "E = m c^2"}'

# Check an LLM answer
curl -X POST http://localhost:8000/check \
  -H "Content-Type: application/json" \
  -d '{
    "brief": "What is E=mc^2?",
    "draft": "Energy equals mass times speed of light",
    "sources": [{"id": "1", "text": "...", "formula": "E = mc^2"}]
  }'
```

## Docker

```bash
docker build -t formulagate .
docker run -p 8000:8000 formulagate
```

## Next steps

- [API Reference](api/sdk.md) — Full SDK documentation
- [RAG with LangChain](tutorials/rag-langchain.md) — Integration guide
- [Production Deployment](tutorials/production.md) — Scaling guide

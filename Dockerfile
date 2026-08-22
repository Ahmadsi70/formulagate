# ── Formulagate API Server ──────────────────────────────────────────────
# Build:   docker build -t formulagate .
# Run:     docker run -p 8000:8000 formulagate
# Docs:    http://localhost:8000/docs

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Formulagate"
LABEL org.opencontainers.image.description="Physics-aware retrieval gate for LLM outputs"
LABEL org.opencontainers.image.version="0.7.0"

# Install SymPy for the physics layer (dimensional analysis + equivalence).
# Kept separate so the base image stays small when only the core gate is needed.
RUN pip install --no-cache-dir sympy fastapi uvicorn

WORKDIR /app
COPY src/ /app/src/
COPY data/ /app/data/
COPY pyproject.toml /app/

RUN pip install --no-cache-dir -e ".[physics]"

EXPOSE 8000

# Run with 1 worker (CPU-bound gate, no parallelism benefit from more).
CMD ["python", "-m", "uvicorn", "formulagate.api:app", "--host", "0.0.0.0", "--port", "8000"]
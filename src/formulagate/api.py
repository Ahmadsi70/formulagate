"""Formulagate REST API — FastAPI server for formula verification and gate checks.

Usage:
    pip install fastapi uvicorn
    python -m formulagate.api
    # → http://localhost:8000/docs

Or via the CLI:
    formulagate serve --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from formulagate import __version__
from formulagate.sdk import Formulagate, Source
from formulagate.calibration import get_calibration, get_multi_calibration

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Formulagate",
    description="Physics-aware retrieval gate — verify formulas and gate-check LLM outputs",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialised gate (sources loaded from env or request body).
_gate: Formulagate | None = None


def _get_gate(sources: list[dict] | None = None) -> Formulagate:
    """Return a configured gate, reusing a global one when no sources are given."""
    global _gate
    if sources:
        return Formulagate(sources=sources, use_physics=True)
    if _gate is None:
        _gate = Formulagate(use_physics=True)
    return _gate


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {
        "service": "Formulagate",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/verify", "/check", "/health", "/version"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/version")
def version():
    return {"version": __version__}


@app.post("/verify")
def verify_formula(payload: dict[str, Any]):
    """Verify a single formula (dimensional analysis + optional equivalence).

    Request body:
        {"formula": "E = m c^2", "against": "m c^2 = E"}   // against is optional

    Response:
        {"ok": true, "dimensions": "consistent", "equivalence": "equivalent", ...}
    """
    formula = payload.get("formula", "")
    if not formula:
        raise HTTPException(400, "field 'formula' is required")

    against = payload.get("against")
    result = _get_gate().verify(formula, against=against)
    return JSONResponse(result.to_dict())


@app.post("/check")
def check_claim(payload: dict[str, Any]):
    """Gate-check an LLM draft against source documents.

    Request body:
        {
            "brief": "What is the energy-mass relationship?",
            "draft": "Einstein proved that $E = m c^2$ ...",
            "sources": [
                {"id": "1", "text": "...", "formula": "E = m c^2"},
                ...
            ]
        }

    Response:
        {"action": "generate", "confidence": 0.89, "sources": [...], ...}
    """
    brief = payload.get("brief", "")
    draft = payload.get("draft", "")
    sources = payload.get("sources", [])

    if not brief:
        raise HTTPException(400, "field 'brief' is required")
    if not draft:
        raise HTTPException(400, "field 'draft' is required")

    result = _get_gate(sources=sources if sources else None).check(
        brief=brief,
        draft=draft,
        sources=sources if sources else None,
    )
    return JSONResponse(result.to_dict())


@app.get("/calibration")
def calibration_info():
    """Return the active calibration parameters (if any)."""
    cal = get_calibration()
    multi = get_multi_calibration()
    return {
        "platt": cal.to_dict() if cal else None,
        "fused": multi.to_dict() if multi else None,
    }


# ─── CLI entry point ─────────────────────────────────────────────────────────


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Formulagate REST API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "formulagate.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
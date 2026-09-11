"""Formulagate REST API — FastAPI server for formula verification and gate checks.

Usage:
    pip install fastapi uvicorn
    python -m formulagate.api
    # → http://localhost:8000/docs

Or via the CLI:
    formulagate serve --port 8000

Persistence (optional):
    Set FORMULAGATE_DATABASE_URL to a PostgreSQL URL and every /verify and
    /check result is **auto-labelled** (correct/incorrect by Z3 proof) and
    stored with vector embeddings for semantic search. Without the env var
    nothing is stored and the API behaves exactly the same.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from formulagate import __version__
from formulagate.sdk import Formulagate, Source
from formulagate.calibration import get_calibration, get_multi_calibration
from formulagate.auth import get_current_user, UserContext

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
_db = None


def _get_gate(sources: list[dict] | None = None) -> Formulagate:
    """Return a configured gate, reusing a global one when no sources are given."""
    global _gate
    if sources:
        return Formulagate(sources=sources, use_physics=True)
    if _gate is None:
        _gate = Formulagate(use_physics=True)
    return _gate


def _get_db():
    """Lazily-created DatabaseManager; never raises, None when disabled."""
    global _db
    if _db is None:
        from formulagate.database import get_database

        _db = get_database()
    return _db


def _make_embedding(text: str) -> list[float] | None:
    """Encode text into a 384-d embedding vector, or None."""
    from formulagate.database import _make_embedding as _embed

    return _embed(text)


def _label_check_result(data: dict) -> tuple[str, float]:
    """Assign ground_truth + label_confidence from a /check result dict."""
    physics = data.get("physics") or {}
    from formulagate.database import _label_check

    return _label_check(
        action=data.get("action", "abstain"),
        physics_ok=physics.get("dimension_ok"),
        confidence=data.get("confidence"),
    )


def _label_verify_result(data: dict) -> tuple[str, float]:
    """Assign ground_truth + label_confidence from a /verify result dict."""
    from formulagate.database import _label_verification

    return _label_verification(
        dimensions=data.get("dimensions", "unknown"),
        equivalence=data.get("equivalence"),
    )


# ─── Usage tracking helper ─────────────────────────────────────────────────────


def _track(user_id: int | None, endpoint: str) -> None:
    """Fire-and-forget usage tracking.  Never raises."""
    if user_id is None:
        return
    try:
        _get_db().track_usage(user_id, endpoint)
    except Exception:
        pass


# ─── Request schemas ───────────────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UpgradeRequest(BaseModel):
    plan: str = Field(..., pattern=r"^(pro|team)$")
    success_url: str = Field(..., min_length=1)
    cancel_url: str = Field(..., min_length=1)


class PortalRequest(BaseModel):
    return_url: str = Field(..., min_length=1)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {
        "service": "Formulagate",
        "version": __version__,
        "docs": "/docs",
        "endpoints": [
            "/verify", "/check", "/health", "/version",
            "/auth/register", "/auth/key", "/account", "/account/usage",
            "/billing/upgrade", "/billing/portal", "/billing/webhook",
            "/plans",
            "/db/stats", "/db/sources",
            "/db/verifications", "/db/checks",
            "/db/export/verified", "/db/export/benchmark",
            "/db/search/similar",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/version")
def version():
    return {"version": __version__}


@app.post("/verify")
async def verify_formula(
    request: Request,
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(get_current_user),
):
    """Verify a single formula (dimensional analysis + optional equivalence).

    Request body:
        {"formula": "E = m c^2", "against": "m c^2 = E"}   // against is optional

    Response:
        {"ok": true, "dimensions": "consistent", "equivalence": "equivalent", ...}
    """
    # ── Quota check ───────────────────────────────────────────────────────
    if user.is_authenticated and user.remaining is not None and user.remaining <= 0:
        raise HTTPException(
            429,
            "Monthly quota exceeded. Upgrade your plan at /billing/upgrade",
        )

    formula = payload.get("formula", "")
    if not formula:
        raise HTTPException(400, "field 'formula' is required")

    against = payload.get("against")
    result = _get_gate().verify(formula, against=against)
    data = result.to_dict()

    # ── Auto-label: Z3 proof → ground_truth ──────────────────────────────
    gt, gt_conf = _label_verify_result(data)

    # ── Embed: build vector for semantic search ──────────────────────────
    embedding = _make_embedding(formula)

    client_ip = request.client.host if request.client else ""
    try:
        _get_db().save_verification(
            formula=data["formula"],
            against=against,
            parsed=data["parsed"],
            ok=data["ok"],
            refuted=data["refuted"],
            dimensions=data["dimensions"],
            equivalence=data["equivalence"],
            symbols=data["symbols"],
            context=payload.get("context", ""),
            client_ip=client_ip,
            ground_truth=gt,
            label_confidence=gt_conf,
            embedding=embedding,
        )
    except Exception:
        pass

    # ── Track usage (fire-and-forget) ─────────────────────────────────────
    if user.is_authenticated:
        background_tasks.add_task(_track, user.user_id, "verify")

    # Attach labels to response for transparency.
    data["ground_truth"] = gt
    data["label_confidence"] = gt_conf
    return JSONResponse(data)


@app.post("/check")
async def check_claim(
    request: Request,
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
    user: UserContext = Depends(get_current_user),
):
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
    # ── Quota check ───────────────────────────────────────────────────────
    if user.is_authenticated and user.remaining is not None and user.remaining <= 0:
        raise HTTPException(
            429,
            "Monthly quota exceeded. Upgrade your plan at /billing/upgrade",
        )

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
    data = result.to_dict()

    # ── Auto-label ───────────────────────────────────────────────────────
    gt, gt_conf = _label_check_result(data)

    # ── Embed both brief and draft ───────────────────────────────────────
    emb_brief = _make_embedding(brief)
    emb_draft = _make_embedding(draft)

    client_ip = request.client.host if request.client else ""
    try:
        physics = data.get("physics") or {}
        _get_db().save_check(
            brief=brief,
            draft=draft,
            action=data["action"],
            confidence=data.get("confidence"),
            domain=data.get("domain", ""),
            physics_ok=physics.get("dimension_ok") if physics else None,
            detail=data.get("detail", ""),
            sources=data.get("sources", []),
            client_ip=client_ip,
            ground_truth=gt,
            label_confidence=gt_conf,
            embedding_brief=emb_brief,
            embedding_draft=emb_draft,
        )
    except Exception:
        pass

    # ── Track usage (fire-and-forget) ─────────────────────────────────────
    if user.is_authenticated:
        background_tasks.add_task(_track, user.user_id, "check")

    # Attach labels to response.
    data["ground_truth"] = gt
    data["label_confidence"] = gt_conf
    return JSONResponse(data)


@app.get("/calibration")
def calibration_info():
    """Return the active calibration parameters (if any)."""
    cal = get_calibration()
    multi = get_multi_calibration()
    return {
        "platt": cal.to_dict() if cal else None,
        "fused": multi.to_dict() if multi else None,
    }


# ─── Database browsing (optional; 404 when no Postgres configured) ───────────


@app.get("/db/stats")
def db_stats():
    """Row counts for verification_log, check_log and sources, plus labels."""
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")
    return {"enabled": True, **db.get_stats()}


@app.get("/db/sources")
def db_sources(
    q: str | None = Query(default=None, description="Text/formula substring filter"),
    limit: int = Query(default=20, ge=1, le=500),
):
    """List stored scientific source documents (optionally filtered)."""
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")
    return {"sources": db.search_sources(query=q or "", limit=limit)}


@app.post("/db/sources")
def db_sources_add(payload: dict[str, Any]):
    """Store one or more scientific source documents.

    Request body (single or list):
        {"id": "rel-1", "text": "energy mass equivalence", "formula": "E = m c^2"}
    Each record upserts on ``id``.
    """
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")

    records = payload if isinstance(payload, list) else [payload]
    for record in records:
        sid = record.get("id")
        if not sid:
            raise HTTPException(400, "every source needs a non-empty 'id'")
        db.save_source(
            id=str(sid),
            english=str(record.get("text") or record.get("english") or ""),
            math_formula=str(record.get("formula") or record.get("math_formula") or ""),
            scientific_domain=record.get("domain") or record.get("scientific_domain"),
            meta={k: v for k, v in record.items() if k not in {
                "id", "text", "english", "formula", "math_formula", "domain", "scientific_domain"
            }},
        )
    return {"stored": len(records)}


@app.delete("/db/sources/{source_id}")
def db_sources_delete(source_id: str):
    """Delete a stored source document by id."""
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")
    try:
        with db._conn() as conn:  # noqa: SLF001 — internal helper
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sources WHERE id = %s", (source_id,))
            conn.commit()
    except Exception as exc:
        raise HTTPException(500, f"delete failed: {exc}") from exc
    return {"deleted": source_id}


@app.get("/db/verifications")
def db_verifications(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    formula: str = Query(default="", description="Filter by formula text"),
    client_ip: str = Query(default="", description="Filter by client IP"),
    ground_truth: str = Query(default="", description="Filter by label (correct/incorrect/uncertain)"),
):
    """Browse verification history (newest first)."""
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")
    return {"verifications": db.query_verifications(
        limit=limit, offset=offset,
        formula_filter=formula, client_ip=client_ip,
        ground_truth_filter=ground_truth,
    )}


@app.get("/db/checks")
def db_checks(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    brief: str = Query(default="", description="Filter by brief text"),
    action: str = Query(default="", description="Filter by action (generate/abstain)"),
    client_ip: str = Query(default="", description="Filter by client IP"),
    ground_truth: str = Query(default="", description="Filter by label (correct/incorrect/uncertain)"),
):
    """Browse gate-check history (newest first)."""
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")
    return {"checks": db.query_checks(
        limit=limit, offset=offset,
        brief_filter=brief, action_filter=action, client_ip=client_ip,
        ground_truth_filter=ground_truth,
    )}


# ─── Data export endpoints (commercially valuable datasets) ──────────────────


@app.get("/db/export/verified")
def export_verified(
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """Export Z3-labelled formula pairs for training/fine-tuning.

    Only returns records where ``ground_truth`` is ``correct`` or
    ``incorrect`` (not ``uncertain``). Use ``min_confidence`` to filter
    for only the most certain labels (default: all).
    """
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")
    records = db.export_verified_formulas(
        limit=limit, offset=offset, min_confidence=min_confidence,
    )
    return {
        "dataset": "formulagate-verified-formulas",
        "description": "Formula pairs labelled correct/incorrect by Z3 proof",
        "total": len(records),
        "records": records,
    }


@app.get("/db/export/benchmark")
def export_benchmark(
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
):
    """Export gate-check cases as an LLM benchmark dataset.

    Each record is a (brief, draft, verdict) triplet usable for evaluating
    physics awareness of language models.
    """
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")
    records = db.export_benchmark_cases(limit=limit, offset=offset)
    return {
        "dataset": "formulagate-benchmark-cases",
        "description": "Physics gate-check cases: (brief, draft, verdict) triplets",
        "total": len(records),
        "records": records,
    }


@app.post("/db/search/similar")
def search_similar_formulas(payload: dict[str, Any]):
    """Find semantically similar formulas in the verified dataset.

    Request body:
        {"formula": "E = m c^3", "limit": 10, "min_similarity": 0.5}
    """
    db = _get_db()
    if not db.enabled:
        raise HTTPException(404, "PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)")

    text = payload.get("formula", "")
    if not text:
        raise HTTPException(400, "field 'formula' is required")

    embedding = _make_embedding(text)
    if embedding is None:
        raise HTTPException(503, "embedding model unavailable (install sentence-transformers)")

    results = db.search_similar_formulas(
        embedding,
        limit=payload.get("limit", 10),
        min_similarity=payload.get("min_similarity", 0.5),
    )
    # Build a response with the semantic neighbours.
    return {
        "query": text,
        "similarity_metric": "cosine",
        "results": results,
    }


# ─── Auth & account management ────────────────────────────────────────────────


@app.post("/auth/register")
def auth_register(payload: RegisterRequest):
    """Register a new user account.

    Request body:
        {"email": "user@example.com"}

    Response:
        {"api_key": "fg_live_...", "plan": "free", ...}

    The raw API key is shown **only once** in this response.  Store it
    securely — it cannot be retrieved again.
    """
    from formulagate.auth import generate_api_key, hash_api_key

    db = _get_db()
    if not db.enabled:
        raise HTTPException(
            503,
            "Registration unavailable — PostgreSQL not configured (set FORMULAGATE_DATABASE_URL)",
        )

    email = payload.email.lower().strip()

    # Check if email already exists.
    existing = db.get_user_by_email(email)
    if existing:
        raise HTTPException(409, "A user with this email already exists")

    api_key = generate_api_key()
    hashed = hash_api_key(api_key)
    prefix = api_key[:15]  # "fg_live_" + first 8 hex chars

    user = db.create_user(
        email=email,
        api_key_hash=hashed,
        api_key_prefix=prefix,
        plan="free",
    )
    if user is None:
        raise HTTPException(409, "A user with this email already exists")

    return JSONResponse(
        {
            "api_key": api_key,
            "api_key_prefix": prefix,
            "user_id": user["id"],
            "email": user["email"],
            "plan": user["plan"],
            "created_at": user["created_at"],
        },
        status_code=201,
    )


@app.get("/auth/key")
def auth_key(user: "UserContext" = Depends(get_current_user)):
    """Show the API key prefix and plan for the authenticated user.

    The full key cannot be retrieved — only its prefix for identification.
    """
    if not user.is_authenticated:
        raise HTTPException(401, "Authentication required")

    db = _get_db()
    full_user = db.get_user_by_id(user.user_id) if user.user_id else None
    return JSONResponse(
        {
            "api_key_prefix": full_user.get("api_key_prefix", "") if full_user else "",
            "email": user.email,
            "plan": user.plan,
        }
    )


@app.get("/account")
def account(user: "UserContext" = Depends(get_current_user)):
    """Return the authenticated user's account status.

    Includes plan, quota, and current-month usage.
    """
    from formulagate.plans import PLANS, plan_for_upgrade

    d = user.to_dict()

    # Enrich with plan details.
    plan_obj = PLANS.get(user.plan, PLANS["free"])
    d["plan_name"] = plan_obj.name
    d["features"] = sorted(plan_obj.features)
    d["rate_limit_per_minute"] = plan_obj.rate_limit_per_minute
    d["available_upgrades"] = plan_for_upgrade(user.plan) if user.is_authenticated else []

    # Subscription info.
    if user.is_authenticated and user.user_id:
        db = _get_db()
        sub = db.get_active_subscription(user.user_id)
        d["subscription"] = sub

    return JSONResponse(d)


@app.get("/account/usage")
def account_usage(user: "UserContext" = Depends(get_current_user)):
    """Return detailed usage breakdown for the current month."""
    if not user.is_authenticated:
        raise HTTPException(401, "Authentication required")

    db = _get_db()
    monthly = db.get_monthly_usage(user.user_id)
    return JSONResponse({"usage": monthly, "total": monthly["verify"] + monthly["check"]})


# ─── Billing (Dodo Payments) ──────────────────────────────────────────────────


@app.post("/billing/upgrade")
def billing_upgrade(
    payload: UpgradeRequest,
    user: UserContext = Depends(get_current_user),
):
    """Start a Dodo Payments checkout session to upgrade to Pro or Team.

    Request body:
        {"plan": "pro", "success_url": "https://...", "cancel_url": "https://..."}

    Response:
        {"checkout_url": "https://checkout.dodopayments.com/..."}
    """
    if not user.is_authenticated:
        raise HTTPException(401, "Authentication required")

    from formulagate.billing import create_checkout_session, is_billing_enabled
    from formulagate.plans import PURCHASABLE

    if payload.plan not in PURCHASABLE:
        raise HTTPException(400, f"Plan must be one of: {', '.join(sorted(PURCHASABLE))}")

    if not is_billing_enabled():
        raise HTTPException(
            503,
            "Billing is not configured (set FORMULAGATE_DODO_API_KEY)",
        )

    url = create_checkout_session(
        user_id=user.user_id,
        email=user.email,
        plan=payload.plan,
        success_url=payload.success_url,
        cancel_url=payload.cancel_url,
    )
    if url is None:
        raise HTTPException(500, "Failed to create checkout session")

    return JSONResponse({"checkout_url": url})


@app.post("/billing/portal")
def billing_portal(
    payload: PortalRequest,
    user: UserContext = Depends(get_current_user),
):
    """Get a payment management portal link for your subscription.

    Request body:
        {"return_url": "https://..."}

    Response:
        {"portal_url": "https://app.dodopayments.com/..."}
    """
    if not user.is_authenticated:
        raise HTTPException(401, "Authentication required")

    from formulagate.billing import create_portal_session, is_billing_enabled

    if not is_billing_enabled():
        raise HTTPException(
            503,
            "Billing is not configured (set FORMULAGATE_DODO_API_KEY)",
        )

    url = create_portal_session(
        user_id=user.user_id,
        return_url=payload.return_url,
    )
    if url is None:
        raise HTTPException(
            400,
            "No payment provider link available. Contact support for billing management.",
        )

    return JSONResponse({"portal_url": url})


@app.post("/billing/webhook")
async def billing_webhook(request: Request):
    """Receive Dodo Payments webhook events.

    Dodo sends events to this endpoint.  The signature header is validated
    against ``FORMULAGATE_DODO_WEBHOOK_SECRET``.

    Returns 200 on success (even for unhandled event types) so Dodo does
    not retry.
    """
    from formulagate.billing import handle_webhook

    payload = await request.body()
    signature = request.headers.get("Dodo-Webhook-Signature", "")

    ok = handle_webhook(payload, signature)
    if not ok:
        raise HTTPException(400, "Invalid webhook signature or billing not configured")

    return JSONResponse({"received": True})


# ─── Plans (public) ────────────────────────────────────────────────────────────


@app.get("/plans")
def list_plans():
    """Return all available plans and their features."""
    from formulagate.plans import plan_list

    return JSONResponse({"plans": plan_list()})


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
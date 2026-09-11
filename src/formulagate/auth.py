"""Authentication and API-key management for Formulagate.

Provides:

- ``hash_api_key`` / ``verify_api_key`` — SHA256-based key hashing
- ``generate_api_key`` — human-friendly key generation (``fg_live_…``)
- ``get_current_user`` — FastAPI dependency that extracts a ``UserContext``
  from the ``X-API-Key`` header
- ``UserContext`` — Pydantic model carrying plan, quota, and usage info

Security properties:

- API keys are **never stored in plaintext** — only SHA256 hashes
- The raw key is returned exactly **once** (on registration) and never again
- **Fail-open**: when the database is unreachable, unauthenticated requests
  are treated as anonymous ``free``-tier — the gate stays available
- When ``FORMULAGATE_REQUIRE_AUTH`` is ``"true"``, missing or invalid keys
  receive a 401; otherwise auth is best-effort

Usage::

    from formulagate.auth import get_current_user, UserContext, generate_api_key

    @app.get("/account")
    def account(user: UserContext = Depends(get_current_user)):
        return user.to_dict()
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field


# ─── Environment control ───────────────────────────────────────────────────────

def _require_auth() -> bool:
    """Read ``FORMULAGATE_REQUIRE_AUTH`` (case-insensitive truthy)."""
    val = os.environ.get("FORMULAGATE_REQUIRE_AUTH", "").strip().lower()
    return val in ("true", "1", "yes", "on")


# ─── Key generation & hashing ──────────────────────────────────────────────────


def generate_api_key(prefix: str = "fg_live") -> str:
    """Produce a human-friendly API key.

    Format: ``fg_live_<32 random alphanumeric chars>``
    """
    raw = secrets.token_hex(16)  # 32 hex chars
    return f"{prefix}_{raw}"


def hash_api_key(key: str) -> str:
    """Return the SHA256 hex digest of *key*."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def verify_api_key(key: str, hashed: str) -> bool:
    """Constant-time comparison of *key* against a stored *hashed* value."""
    return hmac.compare_digest(hash_api_key(key), hashed)


# ─── User context ──────────────────────────────────────────────────────────────


class UserContext(BaseModel):
    """Carried through request handlers after authentication."""

    user_id: int | None = None
    email: str = "anonymous"
    plan: str = "free"
    monthly_quota: int | None = None
    usage_this_month: int = 0
    is_authenticated: bool = False

    @property
    def remaining(self) -> int | None:
        if self.monthly_quota is None:
            return None  # unlimited
        return max(0, self.monthly_quota - self.usage_this_month)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "plan": self.plan,
            "monthly_quota": self.monthly_quota,
            "usage_this_month": self.usage_this_month,
            "remaining": self.remaining,
            "is_authenticated": self.is_authenticated,
        }


# ─── FastAPI dependency ────────────────────────────────────────────────────────


async def get_current_user(request: Request) -> UserContext:
    """FastAPI dependency: resolve a ``UserContext`` from ``X-API-Key``.

    Behaviour (see module docstring for the full matrix):

    - DB unreachable → anonymous ``free`` (fail-open), unless
      ``FORMULAGATE_REQUIRE_AUTH=true``
    - Valid key → authenticated user with their plan + quota
    - Invalid key → 401
    - Missing key + ``REQUIRE_AUTH`` → 401
    - Missing key + no ``REQUIRE_AUTH`` → anonymous ``free``

    """
    api_key = request.headers.get("X-API-Key", "").strip()

    # ── Resolve user from database ────────────────────────────────────────
    user: dict[str, Any] | None = None
    db_ok = True
    if api_key:
        try:
            from formulagate.database import get_database

            db = get_database()
            if db.enabled:
                hashed = hash_api_key(api_key)
                user = db.get_user_by_api_key(hashed)
                # Audit the access (fire-and-forget; best-effort).
                _audit(db, user, hashed, request)
            else:
                db_ok = False
        except Exception:
            db_ok = False

    # ── Fail-open when DB is down ─────────────────────────────────────────
    if api_key and not db_ok:
        if _require_auth():
            raise HTTPException(
                status_code=503,
                detail="Authentication service unavailable — please try again later",
            )
        return UserContext()  # anonymous free

    # ── Valid key ──────────────────────────────────────────────────────────
    if user is not None:
        return _build_user_context(user)

    # ── Invalid key ────────────────────────────────────────────────────────
    if api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ── No key given ───────────────────────────────────────────────────────
    if _require_auth():
        raise HTTPException(
            status_code=401,
            detail="X-API-Key header required. Register at /auth/register",
        )

    return UserContext()  # anonymous free


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _build_user_context(user: dict[str, Any]) -> UserContext:
    """Populate a ``UserContext`` from a database user row."""
    from formulagate.plans import PLANS

    plan_key = user.get("plan", "free")
    plan = PLANS.get(plan_key, PLANS["free"])

    usage = {"verify": 0, "check": 0}
    try:
        from formulagate.database import get_database

        db = get_database()
        usage = db.get_monthly_usage(user["id"])
    except Exception:
        pass

    total_usage = usage["verify"] + usage["check"]
    return UserContext(
        user_id=user["id"],
        email=user.get("email", "anonymous"),
        plan=plan_key,
        monthly_quota=plan.monthly_quota,
        usage_this_month=total_usage,
        is_authenticated=True,
    )


def _audit(db: Any, user: dict[str, Any] | None, hashed: str, request: Request) -> None:
    """Best-effort audit logging — never raises."""
    try:
        db.log_api_key_usage(
            user_id=user["id"] if user else None,
            api_key_hash=hashed,
            endpoint=request.url.path,
            client_ip=request.client.host if request.client else "",
            user_agent=request.headers.get("User-Agent", ""),
        )
    except Exception:
        pass
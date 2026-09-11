"""Subscription plans and tier definitions for Formulagate.

Each plan defines a monthly usage quota, per-minute rate limit, feature
flags, and optional Dodo Payments product mapping.  These definitions are the
single source of truth — the API, middleware, and billing layers all
consume them.

Usage::

    from formulagate.plans import PLANS, Plan, get_plan, resolve_plan

    plan = resolve_plan(user=None)            # → "free"
    plan = resolve_plan(user={"plan": "pro"})  # → "pro"
    info = PLANS[plan]                        # → Plan dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Plan:
    """A subscription plan definition."""

    key: str
    name: str
    monthly_quota: int | None
    """None = unlimited."""
    rate_limit_per_minute: int
    features: frozenset[str]
    dodo_product_id: str | None = None
    price_monthly_usd: int | None = None
    display_order: int = 0
    description: str = ""

    def has_feature(self, feature: str) -> bool:
        return feature in self.features

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "monthly_quota": self.monthly_quota,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "features": sorted(self.features),
            "price_monthly_usd": self.price_monthly_usd,
            "description": self.description,
        }


# ─── Plan definitions ──────────────────────────────────────────────────────────

PLANS: dict[str, Plan] = {
    "free": Plan(
        key="free",
        name="Free",
        monthly_quota=1_000,
        rate_limit_per_minute=10,
        features=frozenset({"verify", "check", "health"}),
        dodo_product_id=None,
        price_monthly_usd=0,
        display_order=0,
        description="1,000 requests/month — perfect for experimentation and prototyping.",
    ),
    "pro": Plan(
        key="pro",
        name="Pro",
        monthly_quota=50_000,
        rate_limit_per_minute=60,
        features=frozenset({"verify", "check", "health", "db_access", "export"}),
        dodo_product_id=None,  # set via FORMULAGATE_DODO_PRODUCT_PRO env
        price_monthly_usd=49,
        display_order=1,
        description=(
            "50,000 requests/month with database access, auto-labelling, "
            "and data export — for production pipelines."
        ),
    ),
    "team": Plan(
        key="team",
        name="Team",
        monthly_quota=250_000,
        rate_limit_per_minute=300,
        features=frozenset({"verify", "check", "health", "db_access", "export", "priority"}),
        dodo_product_id=None,  # set via FORMULAGATE_DODO_PRODUCT_TEAM env
        price_monthly_usd=299,
        display_order=2,
        description=(
            "250,000 requests/month with priority routing and multi-user "
            "support — for teams shipping AI features."
        ),
    ),
    "enterprise": Plan(
        key="enterprise",
        name="Enterprise",
        monthly_quota=None,  # unlimited
        rate_limit_per_minute=1_000,
        features=frozenset({
            "verify", "check", "health", "db_access", "export",
            "priority", "on_premise", "custom_domain_tables", "sla",
        }),
        dodo_product_id=None,  # enterprise is direct-sales only
        price_monthly_usd=None,  # custom pricing
        display_order=3,
        description=(
            "Unlimited requests, on-premise deployment, custom domain tables, "
            "and 24/7 SLA — contact sales for pricing."
        ),
    ),
}

# Plans that are purchasable via Dodo Payments self-service checkout.
PURCHASABLE = frozenset({"pro", "team"})


def get_plan(key: str) -> Plan:
    """Return a Plan by its key; falls back to ``free`` when unknown."""
    return PLANS.get(key, PLANS["free"])


def resolve_plan(user: dict[str, Any] | None) -> Plan:
    """Resolve which plan a user is on.

    When ``user`` is ``None``, returns the free plan (anonymous / fail-open).
    Otherwise reads ``user["plan"]`` and falls back to ``free``.
    """
    if user is None:
        return PLANS["free"]
    return PLANS.get(user.get("plan", "free"), PLANS["free"])


def plan_list() -> list[dict[str, Any]]:
    """Return an ordered list of plan dicts for public display."""
    return sorted(
        (p.to_dict() for p in PLANS.values()),
        key=lambda p: p.get("display_order", 0),
    )


def plan_for_upgrade(current_plan: str) -> list[str]:
    """Return upgrade paths available from *current_plan*."""
    current = PLANS.get(current_plan, PLANS["free"])
    return sorted(
        [
            p.key
            for p in PLANS.values()
            if p.display_order > current.display_order and p.key in PURCHASABLE
        ],
        key=lambda k: PLANS[k].display_order,
    )


# ─── Dodo Payments product id resolution ──────────────────────────────────────

def _dodo_product_env(plan_key: str) -> str | None:
    """Read a Dodo Payments product id from the environment, if set."""
    import os

    env_map = {
        "pro": "FORMULAGATE_DODO_PRODUCT_PRO",
        "team": "FORMULAGATE_DODO_PRODUCT_TEAM",
    }
    return os.environ.get(env_map.get(plan_key, ""))


def dodo_product_id(plan_key: str) -> str | None:
    """Return the effective Dodo Payments product id for *plan_key*.

    Resolution order:
    1. Environment variable (``FORMULAGATE_DODO_PRODUCT_PRO``, etc.)
    2. The plan's built-in ``dodo_product_id``.
    """
    env_val = _dodo_product_env(plan_key)
    if env_val:
        return env_val
    plan = PLANS.get(plan_key)
    return plan.dodo_product_id if plan else None
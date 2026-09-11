"""Dodo Payments billing integration for Formulagate.

Handles self-service checkout, customer portal redirects, and webhook
event processing.  All Dodo calls are lazy — nothing is initialised
unless ``FORMULAGATE_DODO_API_KEY`` is set.

Events processed:

- ``subscription.active`` — provision Plan → user
- ``subscription.cancelled`` — revert to ``free``
- ``payment.succeeded`` — confirm payment

Usage::

    from formulagate.billing import (
        create_checkout_session, create_portal_session,
        handle_webhook, is_billing_enabled,
    )
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── Dodo client (lazy) ───────────────────────────────────────────────────────

_dodo: Any = None


def _get_dodo():
    """Return a ``DodoPayments`` client when the API key is set, else None."""
    global _dodo
    if _dodo is None:
        key = os.environ.get("FORMULAGATE_DODO_API_KEY", "").strip()
        if key:
            try:
                from dodopayments import DodoPayments

                env = os.environ.get("FORMULAGATE_DODO_ENVIRONMENT", "live_mode").strip()
                _dodo = DodoPayments(bearer_token=key, environment=env)
                logger.info("Dodo Payments client initialised (env=%s)", env)
            except ImportError:
                logger.warning(
                    "dodopayments package not installed — "
                    "pip install formulagate[billing]"
                )
                _dodo = False
            except Exception as exc:
                logger.warning("Dodo Payments init failed: %s", exc)
                _dodo = False
        else:
            _dodo = False
    return _dodo if _dodo is not False else None


def is_billing_enabled() -> bool:
    """True when Dodo Payments can process payments."""
    return _get_dodo() is not None


# ─── Webhook event types ───────────────────────────────────────────────────────


class WebhookEvent(str, Enum):
    SUBSCRIPTION_ACTIVE = "subscription.active"
    SUBSCRIPTION_CANCELLED = "subscription.cancelled"
    SUBSCRIPTION_UPDATED = "subscription.updated"
    PAYMENT_SUCCEEDED = "payment.succeeded"
    PAYMENT_FAILED = "payment.failed"


# ─── Checkout ──────────────────────────────────────────────────────────────────


def create_checkout_session(
    *,
    user_id: int,
    email: str,
    plan: str,
    success_url: str,
    cancel_url: str,
) -> str | None:
    """Start a Dodo Payments checkout session and return its URL.

    Returns ``None`` when billing is not configured or the product id is unknown.
    """
    from formulagate.plans import dodo_product_id

    dodo = _get_dodo()
    if dodo is None:
        return None

    product_id = dodo_product_id(plan)
    if not product_id:
        logger.warning("No Dodo product id for plan %r", plan)
        return None

    try:
        session = dodo.checkout_sessions.create(
            product_cart=[{"product_id": product_id, "quantity": 1}],
            customer={"email": email},
            return_url=success_url,
            cancel_url=cancel_url,
        )
        return session.checkout_url
    except Exception as exc:
        logger.error("Failed to create checkout session: %s", exc)
        return None


def create_portal_session(
    *,
    user_id: int,
    return_url: str,
) -> str | None:
    """Return a URL for the Dodo Payments customer portal, or None.

    Dodo Payments does not currently expose a programmatic "customer portal"
    link via the REST API.  As a fallback, we return a generic billing
    management link that points to the Dodo-hosted dashboard.
    """
    dodo = _get_dodo()
    if dodo is None:
        return None

    # Dodo doesn't have a direct portal-session creation endpoint like
    # Stripe's billing_portal.Session.create().  We return the base
    # checkout/customer URL so the user can manage their subscription there.
    base = (
        "https://test.dodopayments.com"
        if os.environ.get("FORMULAGATE_DODO_ENVIRONMENT") == "test_mode"
        else "https://app.dodopayments.com"
    )
    return f"{base}/portal?return_url={return_url}"


# ─── Webhook ───────────────────────────────────────────────────────────────────


def handle_webhook(payload: bytes, signature: str) -> bool:
    """Process a Dodo Payments webhook event.

    Dodo uses a ``Dodo-Webhook-Signature`` header for verification.
    The webhook secret is read from ``FORMULAGATE_DODO_WEBHOOK_SECRET``.

    Returns ``True`` on success, ``False`` when the signature is invalid
    or billing is not configured.
    """
    dodo = _get_dodo()
    if dodo is None:
        logger.warning("Webhook received but Dodo Payments is not configured")
        return False

    secret = os.environ.get("FORMULAGATE_DODO_WEBHOOK_SECRET", "").strip()
    if not secret:
        logger.warning(
            "Webhook received but FORMULAGATE_DODO_WEBHOOK_SECRET not set"
        )
        return False

    # Verify the webhook signature using HMAC-SHA256.
    if not _verify_signature(payload, signature, secret):
        logger.warning("Invalid webhook signature")
        return False

    try:
        event = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid webhook payload JSON: %s", exc)
        return False

    event_type = event.get("type", event.get("event", ""))
    data = event.get("data", event)

    logger.info("Dodo webhook: %s", event_type)

    try:
        if event_type == WebhookEvent.SUBSCRIPTION_ACTIVE:
            _on_subscription_active(data)
        elif event_type == WebhookEvent.SUBSCRIPTION_CANCELLED:
            _on_subscription_cancelled(data)
        elif event_type == WebhookEvent.PAYMENT_SUCCEEDED:
            _on_payment_succeeded(data)
        elif event_type == WebhookEvent.PAYMENT_FAILED:
            _on_payment_failed(data)
        elif event_type == WebhookEvent.SUBSCRIPTION_UPDATED:
            _on_subscription_updated(data)
        else:
            logger.debug("Unhandled webhook event type: %s", event_type)
    except Exception as exc:
        logger.error("Webhook handler failed for %s: %s", event_type, exc)

    return True


# ─── Signature verification ────────────────────────────────────────────────────


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify a Dodo Payments webhook signature using HMAC-SHA256.

    Dodo signs webhook payloads with HMAC-SHA256 using the webhook secret.
    The signature header typically looks like: ``t=<timestamp>,v1=<hmac>``
    """
    import hmac
    import hashlib

    if not signature:
        return False

    try:
        # Try the Stripe-style format first: t=...,v1=...
        parts = {}
        for part in signature.split(","):
            k, _, v = part.partition("=")
            parts[k.strip()] = v.strip()

        if "v1" in parts:
            expected = parts["v1"]
        else:
            # Fallback: treat the whole header as the raw HMAC hex.
            expected = signature

        computed = hmac.new(
            secret.encode("utf-8"),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, expected)
    except Exception:
        return False


# ─── Event handlers ────────────────────────────────────────────────────────────


def _on_subscription_active(data: dict[str, Any]) -> None:
    """Provision the paid plan when a subscription becomes active."""
    user_id = _extract_user_id(data)
    plan = _extract_plan(data) or "pro"

    if user_id is None:
        logger.warning("subscription.active: no user_id in webhook data")
        return

    from formulagate.database import get_database

    db = get_database()
    db.update_user_plan(user_id, plan)

    subscription_id = data.get("subscription_id") or data.get("id", "")
    db.save_subscription(
        user_id=user_id,
        plan=plan,
        payment_subscription_id=str(subscription_id),
        status="active",
    )
    logger.info("Subscription active: user=%d plan=%s", user_id, plan)


def _on_subscription_updated(data: dict[str, Any]) -> None:
    """Sync plan and status from subscription changes."""
    user_id = _extract_user_id(data)
    subscription_id = data.get("subscription_id") or data.get("id", "")
    status = data.get("status", "active")

    from formulagate.database import get_database

    db = get_database()
    if subscription_id:
        db.update_subscription_status(str(subscription_id), status)

    if user_id:
        plan = _extract_plan(data) or "free"
        db.update_user_plan(user_id, plan)
        logger.info(
            "Subscription updated: user=%d plan=%s status=%s",
            user_id, plan, status,
        )


def _on_subscription_cancelled(data: dict[str, Any]) -> None:
    """Revert user to free when a subscription ends."""
    user_id = _extract_user_id(data)
    subscription_id = data.get("subscription_id") or data.get("id", "")

    from formulagate.database import get_database

    db = get_database()
    if subscription_id:
        db.update_subscription_status(str(subscription_id), "canceled")

    if user_id:
        db.update_user_plan(user_id, "free")
        logger.info("Subscription cancelled: user=%d reverted to free", user_id)


def _on_payment_succeeded(_data: dict[str, Any]) -> None:
    """Confirm payment — subscription status already reflects this."""
    pass


def _on_payment_failed(data: dict[str, Any]) -> None:
    """Mark subscription as past_due on payment failure."""
    subscription_id = data.get("subscription_id") or data.get("id", "")
    if subscription_id:
        from formulagate.database import get_database

        db = get_database()
        db.update_subscription_status(str(subscription_id), "past_due")
        logger.warning("Payment failed for subscription %s", subscription_id)


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _extract_user_id(data: dict[str, Any]) -> int | None:
    """Try multiple paths to get the user_id from webhook data."""
    # Direct field.
    uid = data.get("user_id")
    if uid is not None:
        return int(uid)

    # Nested in metadata (like Stripe's metadata object).
    meta = data.get("metadata") or data.get("meta") or {}
    uid = meta.get("user_id")
    if uid is not None:
        return int(uid)

    # Customer email → user lookup.
    customer = data.get("customer", {})
    email = customer.get("email") if isinstance(customer, dict) else None
    if email:
        from formulagate.database import get_database

        db = get_database()
        user = db.get_user_by_email(email)
        if user:
            return user["id"]

    return None


def _extract_plan(data: dict[str, Any]) -> str | None:
    """Try multiple paths to get the plan key from webhook data."""
    # Direct field.
    plan = data.get("plan")
    if plan:
        return plan

    # Metadata.
    meta = data.get("metadata") or data.get("meta") or {}
    plan = meta.get("plan")
    if plan:
        return plan

    # Product id → plan resolution.
    product_id = data.get("product_id") or ""
    if not product_id:
        items = data.get("items") or data.get("product_cart") or []
        if isinstance(items, list) and items:
            product_id = items[0].get("product_id", "") if isinstance(items[0], dict) else ""

    if product_id:
        from formulagate.plans import dodo_product_id

        for key in ("pro", "team"):
            if dodo_product_id(key) == product_id:
                return key

    return None
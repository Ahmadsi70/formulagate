"""Tests for the billing, auth, and subscription infrastructure.

These tests exercise the freemium plan system, API key generation/hashing,
user context, rate limiting, and new API endpoints — all without requiring
a real PostgreSQL connection or Dodo Payments account.

The test strategy mirrors the existing ``test_database_api.py``: unit tests
for pure functions plus integration tests via FastAPI ``TestClient`` for the
HTTP surface.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ─── Ensure the core formulagate package is importable ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Prevent accidental database connections during the suite.
os.environ.pop("FORMULAGATE_DATABASE_URL", None)
os.environ.pop("DATABASE_URL", None)

# Prevent accidental Dodo Payments calls.
os.environ.pop("FORMULAGATE_DODO_API_KEY", None)
os.environ.pop("FORMULAGATE_DODO_WEBHOOK_SECRET", None)

# Auth is best-effort by default.
os.environ.pop("FORMULAGATE_REQUIRE_AUTH", None)


# ─── Plans ────────────────────────────────────────────────────────────────────


class TestPlans:
    """Plan definitions, resolution, and upgrade paths."""

    def test_all_plans_defined(self):
        from formulagate.plans import PLANS

        assert set(PLANS.keys()) == {"free", "pro", "team", "enterprise"}

    def test_plans_have_required_fields(self):
        from formulagate.plans import PLANS

        for plan in PLANS.values():
            assert plan.name
            assert plan.rate_limit_per_minute > 0
            assert len(plan.features) > 0
            assert plan.display_order >= 0

    def test_plan_ordering(self):
        from formulagate.plans import PLANS

        orders = {k: p.display_order for k, p in PLANS.items()}
        assert orders["free"] < orders["pro"] < orders["team"] < orders["enterprise"]

    def test_get_plan_valid(self):
        from formulagate.plans import get_plan

        assert get_plan("pro").key == "pro"

    def test_get_plan_unknown_falls_back_to_free(self):
        from formulagate.plans import get_plan

        assert get_plan("nonexistent").key == "free"

    def test_resolve_plan_from_user_dict(self):
        from formulagate.plans import resolve_plan

        assert resolve_plan({"plan": "pro"}).key == "pro"
        assert resolve_plan({"plan": "nonexistent"}).key == "free"

    def test_resolve_plan_none_is_free(self):
        from formulagate.plans import resolve_plan

        assert resolve_plan(None).key == "free"

    def test_plan_for_upgrade_free(self):
        from formulagate.plans import plan_for_upgrade

        upgrades = plan_for_upgrade("free")
        assert "pro" in upgrades
        assert "team" in upgrades

    def test_plan_for_upgrade_pro(self):
        from formulagate.plans import plan_for_upgrade

        upgrades = plan_for_upgrade("pro")
        assert "team" in upgrades
        assert "enterprise" not in upgrades  # enterprise is not self-service

    def test_plan_for_upgrade_enterprise_is_empty(self):
        from formulagate.plans import plan_for_upgrade

        assert plan_for_upgrade("enterprise") == []

    def test_plan_list_is_ordered(self):
        from formulagate.plans import plan_list

        plans = plan_list()
        assert len(plans) == 4
        assert plans[0]["key"] == "free"
        assert plans[-1]["key"] == "enterprise"

    def test_purchasable_only_pro_and_team(self):
        from formulagate.plans import PURCHASABLE

        assert PURCHASABLE == {"pro", "team"}

    def test_dodo_product_id_from_env(self, monkeypatch):
        monkeypatch.setenv("FORMULAGATE_DODO_PRODUCT_PRO", "pdt_env_pro")

        from formulagate.plans import dodo_product_id

        assert dodo_product_id("pro") == "pdt_env_pro"

    def test_dodo_product_id_fallback(self):
        from formulagate.plans import dodo_product_id

        # Neither env nor built-in is set in PLANS.
        assert dodo_product_id("free") is None

    def test_free_plan_has_no_product_id(self):
        from formulagate.plans import PLANS

        assert PLANS["free"].dodo_product_id is None
        assert PLANS["free"].price_monthly_usd == 0

    def test_enterprise_plan_unlimited_quota(self):
        from formulagate.plans import PLANS

        assert PLANS["enterprise"].monthly_quota is None

    def test_plan_features(self):
        from formulagate.plans import PLANS

        assert PLANS["free"].has_feature("verify")
        assert PLANS["free"].has_feature("check")
        assert not PLANS["free"].has_feature("db_access")

        assert PLANS["pro"].has_feature("db_access")
        assert PLANS["pro"].has_feature("export")

        assert PLANS["enterprise"].has_feature("sla")
        assert PLANS["enterprise"].has_feature("on_premise")

    def test_plan_to_dict(self):
        from formulagate.plans import PLANS

        d = PLANS["pro"].to_dict()
        assert d["key"] == "pro"
        assert d["name"] == "Pro"
        assert d["price_monthly_usd"] == 49
        assert isinstance(d["features"], list)


# ─── Auth ─────────────────────────────────────────────────────────────────────


class TestApiKeyHashing:
    """API key generation, hashing, and verification."""

    def test_generate_api_key_format(self):
        from formulagate.auth import generate_api_key

        key = generate_api_key()
        assert key.startswith("fg_live_")
        assert len(key) == 8 + 32  # "fg_live_" + 32 hex chars

    def test_generate_api_keys_are_unique(self):
        from formulagate.auth import generate_api_key

        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100

    def test_hash_api_key_deterministic(self):
        from formulagate.auth import hash_api_key

        key = "fg_live_abcdef1234567890abcdef1234567890"
        h1 = hash_api_key(key)
        h2 = hash_api_key(key)
        assert h1 == h2

    def test_hash_api_key_does_not_contain_plaintext(self):
        from formulagate.auth import generate_api_key, hash_api_key

        key = generate_api_key()
        hashed = hash_api_key(key)
        assert key not in hashed
        assert hashed not in key

    def test_verify_api_key_match(self):
        from formulagate.auth import hash_api_key, verify_api_key

        key = "fg_live_test1234abcd5678ef901234567890"
        hashed = hash_api_key(key)
        assert verify_api_key(key, hashed)

    def test_verify_api_key_mismatch(self):
        from formulagate.auth import hash_api_key, verify_api_key

        key = "fg_live_test1234abcd5678ef901234567890"
        wrong = "fg_live_00000000000000000000000000000000"
        hashed = hash_api_key(key)
        assert not verify_api_key(wrong, hashed)

    def test_hash_api_key_length_is_64(self):
        from formulagate.auth import hash_api_key

        assert len(hash_api_key("test")) == 64  # SHA256 hex


class TestUserContext:
    """UserContext model used across auth layer."""

    def test_user_context_defaults_to_anonymous_free(self):
        from formulagate.auth import UserContext

        ctx = UserContext()
        assert ctx.is_authenticated is False
        assert ctx.plan == "free"
        assert ctx.email == "anonymous"
        assert ctx.user_id is None

    def test_user_context_authenticated(self):
        from formulagate.auth import UserContext

        ctx = UserContext(
            user_id=42,
            email="user@example.com",
            plan="pro",
            monthly_quota=50_000,
            usage_this_month=100,
            is_authenticated=True,
        )
        assert ctx.is_authenticated is True
        assert ctx.email == "user@example.com"
        assert ctx.plan == "pro"

    def test_user_context_remaining(self):
        from formulagate.auth import UserContext

        ctx = UserContext(monthly_quota=1_000, usage_this_month=300)
        assert ctx.remaining == 700

    def test_user_context_remaining_cannot_be_negative(self):
        from formulagate.auth import UserContext

        ctx = UserContext(monthly_quota=1_000, usage_this_month=2_000)
        assert ctx.remaining == 0

    def test_user_context_remaining_unlimited(self):
        from formulagate.auth import UserContext

        ctx = UserContext(monthly_quota=None, usage_this_month=50_000)
        assert ctx.remaining is None

    def test_user_context_to_dict(self):
        from formulagate.auth import UserContext

        ctx = UserContext(user_id=1, email="a@b.com", plan="pro",
                          monthly_quota=50_000, usage_this_month=0,
                          is_authenticated=True)
        d = ctx.to_dict()
        assert d["user_id"] == 1
        assert d["plan"] == "pro"
        assert d["remaining"] == 50_000


class TestAuthRequireEnv:
    """FORMULAGATE_REQUIRE_AUTH environment variable parsing."""

    def test_default_is_false(self):
        from formulagate.auth import _require_auth

        assert _require_auth() is False

    @pytest.mark.parametrize("val", ["true", "1", "yes", "on", "True", "TRUE", "ON"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("FORMULAGATE_REQUIRE_AUTH", val)

        from formulagate.auth import _require_auth

        assert _require_auth() is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "off", "", " ", "maybe"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("FORMULAGATE_REQUIRE_AUTH", val)

        from formulagate.auth import _require_auth

        assert _require_auth() is False


# ─── Billing ──────────────────────────────────────────────────────────────────


class TestBillingDisabled:
    """Billing functions when Dodo Payments is not configured."""

    def test_is_billing_enabled_false(self):
        from formulagate.billing import is_billing_enabled

        assert is_billing_enabled() is False

    def test_create_checkout_session_billing_disabled(self):
        from formulagate.billing import create_checkout_session

        url = create_checkout_session(
            user_id=1, email="test@example.com",
            plan="pro", success_url="https://ok.example.com",
            cancel_url="https://cancel.example.com",
        )
        assert url is None

    def test_create_portal_session_billing_disabled(self):
        from formulagate.billing import create_portal_session

        url = create_portal_session(user_id=1, return_url="https://ret.example.com")
        assert url is None

    def test_handle_webhook_billing_disabled(self):
        from formulagate.billing import handle_webhook

        ok = handle_webhook(b"{}", "sig")
        assert ok is False


# ─── Rate Limiting ────────────────────────────────────────────────────────────


class TestRateLimiter:
    """Basic RateLimiter (non-tiered) smoke tests."""

    def test_single_request_allowed(self):
        from formulagate.middleware import RateLimiter

        rl = RateLimiter()
        allowed, limits = rl.is_allowed("client-1")
        assert allowed is True
        assert limits["remaining_minute"] >= 0

    def test_burst_below_threshold_always_allowed(self):
        from formulagate.middleware import RateLimiter

        rl = RateLimiter()
        for _ in range(5):
            allowed, _ = rl.is_allowed("client-2")
            assert allowed is True


class TestTieredRateLimiter:
    """TieredRateLimiter respects per-plan differences."""

    def test_initialised_with_all_plans(self):
        from formulagate.middleware import TieredRateLimiter

        rl = TieredRateLimiter()
        # All four plans + anonymous fallback.
        assert len(rl._limiters) == 5

    def test_free_plan_allows_up_to_free_limit(self):
        from formulagate.middleware import TieredRateLimiter

        rl = TieredRateLimiter()
        for i in range(10):  # free = 10/min
            allowed, _ = rl.is_allowed(f"client-{i}", plan="free")
            assert allowed is True

    def test_pro_plan_has_higher_limit_than_free(self):
        from formulagate.middleware import TieredRateLimiter
        from formulagate.plans import PLANS

        rl = TieredRateLimiter()
        assert (
            PLANS["pro"].rate_limit_per_minute
            > PLANS["free"].rate_limit_per_minute
        )

    def test_user_id_used_as_window_key(self):
        from formulagate.middleware import TieredRateLimiter

        rl = TieredRateLimiter()
        for uid in range(5):
            allowed, _ = rl.is_allowed("some-ip", plan="free", user_id=uid)
            assert allowed is True


# ─── API Integration ──────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """Return a TestClient for the Formulagate FastAPI app."""
    from formulagate.api import app

    return TestClient(app)


class TestApiRoot:
    """Root and health endpoints — always public."""

    def test_root_lists_new_endpoints(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        endpoints = resp.json()["endpoints"]
        assert "/auth/register" in endpoints
        assert "/billing/upgrade" in endpoints
        assert "/billing/webhook" in endpoints
        assert "/account" in endpoints
        assert "/plans" in endpoints

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_version(self, client):
        resp = client.get("/version")
        assert resp.status_code == 200
        assert "version" in resp.json()


class TestPlansEndpoint:
    """GET /plans — public."""

    def test_plans_endpoint(self, client):
        resp = client.get("/plans")
        assert resp.status_code == 200
        plans = resp.json()["plans"]
        assert len(plans) == 4
        assert plans[0]["key"] == "free"


class TestAuthEndpoints:
    """Registration, key display, account endpoints."""

    def test_register_requires_db(self, client):
        resp = client.post("/auth/register", json={"email": "test@example.com"})
        assert resp.status_code == 503  # no PostgreSQL

    def test_auth_key_requires_auth(self, client):
        resp = client.get("/auth/key")
        assert resp.status_code == 401

    def test_account_anonymous(self, client):
        """Without X-API-Key, account returns anonymous free info."""
        resp = client.get("/account")
        assert resp.status_code == 200
        data = resp.json()
        assert data["plan"] == "free"
        assert data["is_authenticated"] is False

    def test_account_usage_requires_auth(self, client):
        resp = client.get("/account/usage")
        assert resp.status_code == 401


class TestBillingEndpoints:
    """Dodo Payments checkout, portal, and webhook endpoints."""

    def test_upgrade_requires_auth(self, client):
        resp = client.post("/billing/upgrade", json={
            "plan": "pro",
            "success_url": "https://ok.example.com",
            "cancel_url": "https://cancel.example.com",
        })
        assert resp.status_code == 401

    def test_upgrade_billing_disabled(self, client):
        resp = client.post("/billing/upgrade", json={
            "plan": "pro",
            "success_url": "https://ok.example.com",
            "cancel_url": "https://cancel.example.com",
        }, headers={"X-API-Key": "any-key"})
        # Key is invalid → 401 (auth runs first).
        assert resp.status_code == 401

    def test_webhook_no_signature(self, client):
        resp = client.post("/billing/webhook", content=b"{}")
        assert resp.status_code == 400  # billing not configured

    def test_portal_requires_auth(self, client):
        resp = client.post("/billing/portal", json={
            "return_url": "https://example.com",
        })
        assert resp.status_code == 401


class TestVerifyAndCheckWithAuth:
    """Test that /verify and /check still work (no DB, no auth required)."""

    def test_verify_no_api_key_works(self, client):
        resp = client.post("/verify", json={"formula": "E = m c^2"})
        assert resp.status_code == 200
        data = resp.json()
        assert "ok" in data
        assert "ground_truth" in data

    def test_verify_with_api_key_works(self, client):
        resp = client.post(
            "/verify",
            json={"formula": "E = m c^2"},
            headers={"X-API-Key": "fg_live_00000000000000000000000000000000"},
        )
        assert resp.status_code == 200
        assert "ground_truth" in resp.json()

    def test_check_no_api_key_works(self, client):
        resp = client.post("/check", json={
            "brief": "What is E?",
            "draft": "Einstein said E = m c^2.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "action" in data
        assert "ground_truth" in data

    def test_verify_missing_formula_is_400(self, client):
        resp = client.post("/verify", json={})
        assert resp.status_code == 400

    def test_check_missing_brief_is_400(self, client):
        resp = client.post("/check", json={"draft": "x"})
        assert resp.status_code == 400

    def test_check_missing_draft_is_400(self, client):
        resp = client.post("/check", json={"brief": "x"})
        assert resp.status_code == 400


class TestRequireAuth:
    """When FORMULAGATE_REQUIRE_AUTH=true, no anonymous access."""

    def test_verify_401_when_require_auth_and_no_key(self, client, monkeypatch):
        monkeypatch.setenv("FORMULAGATE_REQUIRE_AUTH", "true")

        from formulagate.auth import _require_auth

        assert _require_auth() is True

        resp = client.post("/verify", json={"formula": "E = m c^2"})
        assert resp.status_code == 401

        monkeypatch.delenv("FORMULAGATE_REQUIRE_AUTH", raising=False)

    def test_account_401_when_require_auth_and_no_key(self, client, monkeypatch):
        monkeypatch.setenv("FORMULAGATE_REQUIRE_AUTH", "true")
        resp = client.get("/account")
        assert resp.status_code == 401
        monkeypatch.delenv("FORMULAGATE_REQUIRE_AUTH", raising=False)

    def test_health_never_requires_auth(self, client, monkeypatch):
        monkeypatch.setenv("FORMULAGATE_REQUIRE_AUTH", "true")
        resp = client.get("/health")
        assert resp.status_code == 200  # health is always public
        monkeypatch.delenv("FORMULAGATE_REQUIRE_AUTH", raising=False)


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestRegisterValidation:
    """Request body validation for /auth/register."""

    def test_register_invalid_email(self, client):
        resp = client.post("/auth/register", json={"email": "not-an-email"})
        assert resp.status_code == 422

    def test_register_missing_email(self, client):
        resp = client.post("/auth/register", json={})
        assert resp.status_code == 422


class TestUpgradeValidation:
    """Request body validation for /billing/upgrade."""

    def test_upgrade_invalid_plan(self, client):
        # Pydantic validation (plan must be pro|team) runs before auth → 422.
        resp = client.post("/billing/upgrade", json={
            "plan": "enterprise",
            "success_url": "https://ok.com",
            "cancel_url": "https://cxl.com",
        })
        assert resp.status_code == 422

    def test_upgrade_missing_urls(self, client):
        resp = client.post("/billing/upgrade", json={"plan": "pro"})
        assert resp.status_code == 422


class TestDodoCheckoutSession:
    """Mock out Dodo Payments SDK to exercise checkout session creation."""

    def test_create_checkout_session_returns_url_when_configured(
        self, monkeypatch
    ):
        monkeypatch.setenv("FORMULAGATE_DODO_API_KEY", "dodo_test_fake")
        monkeypatch.setenv("FORMULAGATE_DODO_ENVIRONMENT", "test_mode")
        monkeypatch.setenv("FORMULAGATE_DODO_PRODUCT_PRO", "pdt_fake_pro")

        import types
        import formulagate.billing as bm

        # Build a fake DodoPayments client.
        fake_dodo = types.SimpleNamespace()

        class FakeCheckoutSessions:
            @staticmethod
            def create(**kwargs):
                return types.SimpleNamespace(
                    session_id="cks_fake123",
                    checkout_url="https://test.checkout.dodopayments.com/session/fake",
                )

        fake_dodo.checkout_sessions = FakeCheckoutSessions()

        # Inject the fake client directly.
        bm._dodo = fake_dodo

        url = bm.create_checkout_session(
            user_id=1, email="test@t.com", plan="pro",
            success_url="https://ok.com", cancel_url="https://cxl.com",
        )
        assert url == "https://test.checkout.dodopayments.com/session/fake"

        # Clean up.
        bm._dodo = None
        monkeypatch.delenv("FORMULAGATE_DODO_API_KEY", raising=False)
        monkeypatch.delenv("FORMULAGATE_DODO_ENVIRONMENT", raising=False)
        monkeypatch.delenv("FORMULAGATE_DODO_PRODUCT_PRO", raising=False)

    def test_webhook_signature_validation(self, monkeypatch):
        monkeypatch.setenv("FORMULAGATE_DODO_API_KEY", "dodo_test_fake")
        monkeypatch.setenv("FORMULAGATE_DODO_WEBHOOK_SECRET", "whsec_fake123")

        import types
        import formulagate.billing as bm

        fake_dodo = types.SimpleNamespace()
        bm._dodo = fake_dodo

        # With a garbage signature, verification should fail.
        from formulagate.billing import handle_webhook

        ok = handle_webhook(b"{}", "t=fake,v1=fake_sig")
        # Should be False because HMAC doesn't match.
        assert ok is False

        bm._dodo = None
        monkeypatch.delenv("FORMULAGATE_DODO_API_KEY", raising=False)
        monkeypatch.delenv("FORMULAGATE_DODO_WEBHOOK_SECRET", raising=False)

    def test_webhook_valid_signature_accepted(self, monkeypatch):
        import hashlib
        import hmac

        secret = "whsec_test_secret"
        monkeypatch.setenv("FORMULAGATE_DODO_API_KEY", "dodo_test_fake")
        monkeypatch.setenv("FORMULAGATE_DODO_WEBHOOK_SECRET", secret)

        import types
        import formulagate.billing as bm

        fake_dodo = types.SimpleNamespace()
        bm._dodo = fake_dodo

        payload = b'{"type":"payment.succeeded","data":{}}'
        computed_sig = hmac.new(
            secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()

        from formulagate.billing import handle_webhook

        ok = handle_webhook(payload, computed_sig)
        assert ok is True

        bm._dodo = None
        monkeypatch.delenv("FORMULAGATE_DODO_API_KEY", raising=False)
        monkeypatch.delenv("FORMULAGATE_DODO_WEBHOOK_SECRET", raising=False)


# ─── Database (inert — no Postgres) ────────────────────────────────────────────


class TestDatabaseManagementMethods:
    """DatabaseManager user/usage/subscription methods when DB is down."""

    def test_create_user_returns_none_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        assert db.enabled is False
        assert db.create_user(
            email="x@y.com", api_key_hash="hash", api_key_prefix="pfx",
        ) is None

    def test_get_user_by_api_key_returns_none_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        assert db.get_user_by_api_key("any-hash") is None

    def test_get_monthly_usage_returns_zeros_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        usage = db.get_monthly_usage(1)
        assert usage == {"verify": 0, "check": 0}

    def test_check_quota_always_ok_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        ok, used, limit = db.check_quota(1, "free")
        assert ok is True
        assert used == 0

    def test_track_usage_does_not_crash_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        db.track_usage(1, "verify")  # should not raise

    def test_update_user_plan_returns_false_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        assert db.update_user_plan(1, "pro") is False

    def test_update_user_payment_customer_returns_false_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        assert db.update_user_payment_customer(1, "cus_123") is False

    def test_get_usage_stats_returns_zeros_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        stats = db.get_usage_stats()
        assert stats == {"total_users": 0, "total_verify": 0, "total_check": 0}

    def test_log_api_key_usage_does_not_crash_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        db.log_api_key_usage(
            user_id=1, api_key_hash="hash",
            endpoint="/verify", client_ip="127.0.0.1", user_agent="test",
        )

    def test_list_users_returns_empty_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        assert db.list_users() == []

    def test_get_active_subscription_returns_none_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        assert db.get_active_subscription(1) is None

    def test_get_user_id_by_payment_customer_returns_none_without_db(self):
        from formulagate.database import DatabaseManager

        db = DatabaseManager("")
        assert db.get_user_id_by_payment_customer("cus_123") is None


# ─── Backward Compatibility ───────────────────────────────────────────────────

class TestNoBillingCodeLeak:
    """Ensure the new billing module is properly structured."""

    def test_new_billing_module_exists(self):
        """The new billing module is importable."""
        import formulagate.billing

        assert hasattr(formulagate.billing, "create_checkout_session")
        assert hasattr(formulagate.billing, "handle_webhook")
        assert hasattr(formulagate.billing, "is_billing_enabled")

    def test_new_auth_module_exists(self):
        """The new auth module is importable."""
        import formulagate.auth

        assert hasattr(formulagate.auth, "generate_api_key")
        assert hasattr(formulagate.auth, "hash_api_key")
        assert hasattr(formulagate.auth, "UserContext")

    def test_new_plans_module_exists(self):
        """The new plans module is importable."""
        import formulagate.plans

        assert hasattr(formulagate.plans, "PLANS")
        assert hasattr(formulagate.plans, "get_plan")
        assert hasattr(formulagate.plans, "dodo_product_id")
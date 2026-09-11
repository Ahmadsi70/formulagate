"""Tests for the optional PostgreSQL persistence layer and the open API.

Includes coverage of auto-labeling (ground_truth) and export endpoints.
"""

from __future__ import annotations

import os

import pytest

# Run every test with the database env vars cleared so the disabled path is
# always exercised (no live Postgres in CI).
os.environ.pop("FORMULAGATE_DATABASE_URL", None)
os.environ.pop("DATABASE_URL", None)


def teardown_module():
    from formulagate.database import close_database

    close_database()


def test_database_disabled_without_env():
    """Without FORMULAGATE_DATABASE_URL the manager is inert, never a crash."""
    from formulagate.database import DatabaseManager

    db = DatabaseManager()
    assert db.enabled is False

    # save methods are no-ops
    db.save_verification(
        formula="E = mc^2",
        parsed=True,
        ok=True,
        refuted=False,
        dimensions="consistent",
        ground_truth="correct",
        label_confidence=1.0,
    )
    db.save_check(
        brief="q", draft="a", action="generate",
        ground_truth="correct", label_confidence=0.95,
    )
    db.save_source(id="x", english="t", math_formula="f")

    assert db.get_stats() == {
        "verification_log": 0, "check_log": 0, "sources": 0,
        "verified_correct": 0, "verified_incorrect": 0,
        "check_correct": 0, "check_incorrect": 0,
    }
    assert db.search_sources("anything") == []
    assert db.query_verifications() == []
    assert db.query_checks() == []
    assert db.export_verified_formulas() == []
    assert db.export_benchmark_cases() == []
    assert db.search_similar_formulas([0.1, 0.2]) == []
    db.close()


def test_database_manager_is_singleton():
    from formulagate.database import get_database, close_database

    close_database()
    first = get_database()
    second = get_database()
    assert first is second
    close_database()


def test_label_verification_from_database():
    from formulagate.database import _label_verification

    # Z3-proven correct
    assert _label_verification(dimensions="consistent", equivalence="equivalent") == ("correct", 1.0)
    assert _label_verification(dimensions="consistent", equivalence=None) == ("correct", 1.0)

    # Z3-proven incorrect
    assert _label_verification(dimensions="inconsistent", equivalence=None) == ("incorrect", 1.0)
    assert _label_verification(dimensions="consistent", equivalence="different") == ("incorrect", 1.0)

    # Uncertain
    assert _label_verification(dimensions="unknown", equivalence=None) == ("uncertain", 0.0)
    assert _label_verification(dimensions="consistent", equivalence="unknown") == ("uncertain", 0.0)


def test_label_check_from_database():
    from formulagate.database import _label_check

    # Generate → correct (with Platt confidence)
    assert _label_check(action="generate", physics_ok=1.0, confidence=0.89) == ("correct", 0.89)
    assert _label_check(action="generate", physics_ok=None, confidence=None) == ("correct", 0.5)

    # Abstain with Z3 veto → incorrect
    assert _label_check(action="abstain", physics_ok=-1.0, confidence=0.2) == ("incorrect", 1.0)

    # Abstain without veto → uncertain
    assert _label_check(action="abstain", physics_ok=0.0, confidence=None) == ("uncertain", 0.0)
    assert _label_check(action="abstain", physics_ok=None, confidence=None) == ("uncertain", 0.0)


def test_api_returns_labels():
    """/verify and /check return ground_truth + label_confidence."""
    from formulagate.api import app
    from starlette.testclient import TestClient

    client = TestClient(app)

    # /verify with a proven-incorrect formula
    r = client.post("/verify", json={"formula": "E = m c^3"})
    assert r.status_code == 200
    body = r.json()
    assert "ground_truth" in body
    assert "label_confidence" in body
    # E = m c^3 is dimensionally inconsistent → Z3 labels it incorrect
    assert body["ground_truth"] == "incorrect"
    assert body["label_confidence"] == 1.0

    # /check
    r = client.post("/check", json={
        "brief": "What is F=ma?",
        "draft": "Newton's second law: $F = m a$",
        "sources": [{
            "id": "n1",
            "text": "Newton's second law",
            "formula": "F = m a",
        }],
    })
    assert r.status_code == 200
    body = r.json()
    assert "ground_truth" in body
    assert "label_confidence" in body


def test_export_endpoints_404_without_postgres():
    """Export endpoints (and search) return 404 when no DB configured."""
    from formulagate.api import app
    from starlette.testclient import TestClient

    client = TestClient(app)

    assert client.get("/db/export/verified").status_code == 404
    assert client.get("/db/export/benchmark").status_code == 404
    assert client.post("/db/search/similar", json={"formula": "E=mc^2"}).status_code == 404


def test_no_billing_code_remains():
    """The old billing subsystem is gone — replaced by the Dodo Payments
    freemium billing module with checkout, portal, and webhooks."""
    import importlib
    import sys

    # The new billing module imports fine.
    mod = importlib.import_module("formulagate.billing")
    assert hasattr(mod, "create_checkout_session")
    assert hasattr(mod, "create_portal_session")
    assert hasattr(mod, "handle_webhook")
    assert hasattr(mod, "is_billing_enabled")

    # No old billing remnants (the old module was just a single file).
    mods = [m for m in sys.modules if m == "billing"
            or m.endswith(".billing") and "formulagate" not in m]
    assert not mods, f"old billing module still loaded: {mods}"


def test_make_embedding_works_or_graceful():
    """Embedding returns None when sentence-transformers is absent."""
    from formulagate.database import _make_embedding

    result = _make_embedding("E = m c^2")
    # It either works (list of 384 floats) or returns None gracefully
    assert result is None or (isinstance(result, list) and len(result) == 384)
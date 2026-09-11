"""PostgreSQL persistence for Formulagate — enriched data collection.

Stores scientific data — verification history, gate-check history, and
source documents — in a PostgreSQL database with **auto-labeling** and
**vector embeddings** for semantic search. Every record carries a
``ground_truth`` label (correct / incorrect / uncertain) assigned by Z3
proof, not by a probabilistic model.

The database is fully **optional**: without ``FORMULAGATE_DATABASE_URL``
set, no connection is made and the API behaves exactly as before.

Schema:
    verification_log  — one row per ``/verify`` call, with:
        - ground_truth: Z3 label (correct | incorrect | uncertain)
        - label_confidence: 1.0 for Z3 proof, Platt confidence for /check
        - embedding: pgvector(384) for semantic formula search

    check_log         — one row per ``/check`` call, with:
        - ground_truth: Z3 + calibration label
        - embedding_brief / embedding_draft: pgvector(384)

    sources           — scientific source documents (corpus records)

	    users             — registered users with hashed API keys

	    subscriptions     — Payment provider subscription records

	    usage_daily       — per-user daily usage counters (verify + check)

	    api_key_audit     — audit trail of API key usage

Usage:
    from formulagate.database import DatabaseManager, get_database

    db = DatabaseManager(url="postgresql://user:pass@host/db")
    db.save_verification(
        formula="E = m c^3", dimensions="inconsistent",
        parsed=True, ok=False, refuted=True,
        ground_truth="incorrect", label_confidence=1.0,
        embedding=[0.01, -0.03, ...],  # optional
    )
    db.close()
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# ─── Schema ──────────────────────────────────────────────────────────────────
# pgvector extension needed on the PostgreSQL side:
#   CREATE EXTENSION IF NOT EXISTS vector;
#
# All new columns are added via ALTER TABLE … ADD COLUMN IF NOT EXISTS so
# existing databases are migrated automatically on first connect.

_DEFAULT_TABLES = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS verification_log (
    id                SERIAL PRIMARY KEY,
    formula           TEXT NOT NULL,
    against           TEXT,
    parsed            BOOLEAN,
    ok                BOOLEAN,
    refuted           BOOLEAN,
    dimensions        TEXT,
    equivalence       TEXT,
    symbols           TEXT[],
    context           TEXT,
    client_ip         TEXT,
    ground_truth      TEXT,
        -- 'correct' | 'incorrect' | 'uncertain'  — assigned by Z3 proof
    label_confidence  FLOAT DEFAULT 1.0,
        -- 1.0 for Z3-proven, Platt-calibrated probability for /check
    embedding         vector(384),
        -- sentence-transformers/all-MiniLM-L6-v2, 384-d, cosine-normalised
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS check_log (
    id                SERIAL PRIMARY KEY,
    brief             TEXT NOT NULL,
    draft             TEXT NOT NULL,
    action            TEXT,
    confidence        FLOAT,
    domain            TEXT,
    physics_ok        BOOLEAN,
    detail            TEXT,
    sources           JSONB DEFAULT '[]',
    client_ip         TEXT,
    ground_truth      TEXT,
    label_confidence  FLOAT DEFAULT 1.0,
    embedding_brief   vector(384),
    embedding_draft   vector(384),
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sources (
    id                TEXT PRIMARY KEY,
    english           TEXT,
    math_formula      TEXT,
    scientific_domain TEXT,
    meta              JSONB DEFAULT '{}',
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vlog_created
    ON verification_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_clog_created
    ON check_log (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vlog_ground_truth
    ON verification_log (ground_truth);
CREATE INDEX IF NOT EXISTS idx_clog_ground_truth
    ON check_log (ground_truth);

CREATE TABLE IF NOT EXISTS users (
    id               SERIAL PRIMARY KEY,
    email            TEXT UNIQUE NOT NULL,
    api_key_hash     TEXT NOT NULL,
    api_key_prefix   TEXT NOT NULL,
        -- first 8 chars of the key (for display / revocation lookups)
    plan             TEXT NOT NULL DEFAULT 'free',
    payment_customer_id TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS subscriptions (
    id                   SERIAL PRIMARY KEY,
    user_id              INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan                 TEXT NOT NULL,
    payment_subscription_id TEXT,
    status               TEXT NOT NULL DEFAULT 'active',
        -- active | canceled | past_due | unpaid | incomplete
    current_period_start TIMESTAMPTZ,
    current_period_end   TIMESTAMPTZ,
    cancel_at            TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS usage_daily (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date          DATE NOT NULL DEFAULT CURRENT_DATE,
    verify_count  INTEGER NOT NULL DEFAULT 0,
    check_count   INTEGER NOT NULL DEFAULT 0,
    UNIQUE (user_id, date)
);

CREATE TABLE IF NOT EXISTS api_key_audit (
    id            SERIAL PRIMARY KEY,
    user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
    api_key_hash  TEXT NOT NULL,
    endpoint      TEXT NOT NULL,
    client_ip     TEXT,
    user_agent    TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);
"""

# ─── Migration (add columns that may be missing on existing tables) ──────────

_MIGRATE_SQL = [
    "ALTER TABLE verification_log ADD COLUMN IF NOT EXISTS ground_truth TEXT",
    "ALTER TABLE verification_log ADD COLUMN IF NOT EXISTS label_confidence FLOAT DEFAULT 1.0",
    "ALTER TABLE verification_log ADD COLUMN IF NOT EXISTS embedding vector(384)",
    "ALTER TABLE check_log ADD COLUMN IF NOT EXISTS ground_truth TEXT",
    "ALTER TABLE check_log ADD COLUMN IF NOT EXISTS label_confidence FLOAT DEFAULT 1.0",
    "ALTER TABLE check_log ADD COLUMN IF NOT EXISTS embedding_brief vector(384)",
    "ALTER TABLE check_log ADD COLUMN IF NOT EXISTS embedding_draft vector(384)",
    "CREATE INDEX IF NOT EXISTS idx_vlog_ground_truth ON verification_log (ground_truth)",
    "CREATE INDEX IF NOT EXISTS idx_clog_ground_truth ON check_log (ground_truth)",
    "CREATE INDEX IF NOT EXISTS idx_vlog_embedding ON verification_log "
    "USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS idx_clog_embedding_brief ON check_log "
    "USING hnsw (embedding_brief vector_cosine_ops)",
    # ── billing / auth tables ──────────────────────────────────────────
    "CREATE TABLE IF NOT EXISTS users ("
    "id SERIAL PRIMARY KEY, email TEXT UNIQUE NOT NULL, api_key_hash TEXT NOT NULL, "
    "api_key_prefix TEXT NOT NULL, plan TEXT NOT NULL DEFAULT 'free', "
    "payment_customer_id TEXT, created_at TIMESTAMPTZ DEFAULT NOW())",
    "CREATE TABLE IF NOT EXISTS subscriptions ("
    "id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
    "plan TEXT NOT NULL, payment_subscription_id TEXT, status TEXT NOT NULL DEFAULT 'active', "
    "current_period_start TIMESTAMPTZ, current_period_end TIMESTAMPTZ, "
    "cancel_at TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT NOW(), "
    "updated_at TIMESTAMPTZ DEFAULT NOW())",
    "CREATE TABLE IF NOT EXISTS usage_daily ("
    "id SERIAL PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, "
    "date DATE NOT NULL DEFAULT CURRENT_DATE, verify_count INTEGER NOT NULL DEFAULT 0, "
    "check_count INTEGER NOT NULL DEFAULT 0, UNIQUE (user_id, date))",
    "CREATE TABLE IF NOT EXISTS api_key_audit ("
    "id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id) ON DELETE SET NULL, "
    "api_key_hash TEXT NOT NULL, endpoint TEXT NOT NULL, "
    "client_ip TEXT, user_agent TEXT, created_at TIMESTAMPTZ DEFAULT NOW())",
    "ALTER TABLE usage_daily ADD COLUMN IF NOT EXISTS verify_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE usage_daily ADD COLUMN IF NOT EXISTS check_count INTEGER NOT NULL DEFAULT 0",
]


# ─── Embedding model (lazy singleton) ────────────────────────────────────────

_EMBEDDER: Any = None


def _get_embedder():
    """Return a SentenceTransformer model for building embeddings.

    Returns None when sentence-transformers is not installed or the model
    cannot be loaded — callers must handle None gracefully.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer

            _EMBEDDER = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            _EMBEDDER.max_seq_length = 256
        except Exception as exc:
            logger.debug("embedding model unavailable: %s", exc)
            _EMBEDDER = False  # sentinel — don't retry
    return _EMBEDDER if _EMBEDDER is not False else None


def _make_embedding(text: str) -> list[float] | None:
    """Encode ``text`` into a 384-d list, or None if the model is absent."""
    model = _get_embedder()
    if model is None or not text:
        return None
    try:
        vec = model.encode([text], normalize_embeddings=True)[0]
        return [float(v) for v in vec]
    except Exception as exc:
        logger.debug("embedding failed: %s", exc)
        return None


# ─── Label helpers ───────────────────────────────────────────────────────────


def _label_verification(*, dimensions: str, equivalence: str | None) -> tuple[str, float]:
    """Assign a ground-truth label to a /verify result.

    Returns:
        (label, confidence) where confidence is 1.0 for Z3-proven, 0.0 for uncertain.
    """
    if dimensions == "inconsistent":
        return "incorrect", 1.0
    if equivalence == "different":
        return "incorrect", 1.0
    if dimensions == "consistent" and (equivalence is None or equivalence == "equivalent"):
        return "correct", 1.0
    return "uncertain", 0.0


def _label_check(*, action: str, physics_ok: float | None,
                 confidence: float | None) -> tuple[str, float]:
    """Assign a ground-truth label to a /check result.

    Uses the gate's own calibrated confidence when it says "generate",
    and the physics veto when it says "abstain".
    """
    if action == "generate":
        conf = confidence or 0.5
        return "correct", float(conf)
    # abstain
    if physics_ok is not None and physics_ok < 0:
        return "incorrect", 1.0  # Z3 veto = certain
    return "uncertain", 0.0


# ─── DatabaseManager ─────────────────────────────────────────────────────────


class DatabaseManager:
    """A thin PostgreSQL connection pool with enriched data collection.

    Every method is safe to call even when the database is unreachable —
    failures are logged and swallowed, never raised — so the gate stays
    available when Postgres is down.
    """

    def __init__(self, url: str | None = None, *, create_tables: bool = True):
        self._url = url or _DEFAULT_DB_URL()
        self._enabled = bool(self._url)
        self._pool: Any = None
        if self._enabled:
            try:
                from psycopg2 import pool as pg_pool

                self._pool = pg_pool.ThreadedConnectionPool(1, 10, self._url)
                if create_tables:
                    with self._conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(_DEFAULT_TABLES)
                            for stmt in _MIGRATE_SQL:
                                try:
                                    cur.execute(stmt)
                                except Exception:
                                    pass  # column may already exist
                        conn.commit()
                logger.info("PostgreSQL connected (%s)", self._url.rsplit("@", 1)[-1])
            except Exception as exc:
                self._pool = None
                self._enabled = False
                logger.warning("PostgreSQL unavailable, running without persistence: %s", exc)
        else:
            logger.info("No FORMULAGATE_DATABASE_URL set — running without persistence")

    # ─── lifecycle ────────────────────────────────────────────────────────

    def _conn(self):
        conn = self._pool.getconn()
        return _PooledConn(conn, self._pool)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.closeall()
            except Exception:
                pass
            self._pool = None
            self._enabled = False

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ─── verification log ─────────────────────────────────────────────────

    def save_verification(
        self,
        *,
        formula: str,
        parsed: bool,
        ok: bool,
        refuted: bool,
        dimensions: str,
        against: str | None = None,
        equivalence: str | None = None,
        symbols: Sequence[str] = (),
        context: str = "",
        client_ip: str = "",
        ground_truth: str | None = None,
        label_confidence: float = 1.0,
        embedding: list[float] | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO verification_log
                            (formula, against, parsed, ok, refuted, dimensions,
                             equivalence, symbols, context, client_ip,
                             ground_truth, label_confidence, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            formula, against, parsed, ok, refuted, dimensions,
                            equivalence, list(symbols), context, client_ip,
                            ground_truth, label_confidence,
                            embedding,  # pgvector accepts float[] directly
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("failed to save verification: %s", exc)

    def query_verifications(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        formula_filter: str = "",
        client_ip: str = "",
        ground_truth_filter: str = "",
    ) -> list[dict[str, Any]]:
        """Browse verification history (newest first)."""
        if not self._enabled:
            return []
        try:
            conditions: list[str] = []
            params: list[Any] = []
            if formula_filter:
                conditions.append("formula ILIKE %s")
                params.append(f"%{formula_filter}%")
            if client_ip:
                conditions.append("client_ip = %s")
                params.append(client_ip)
            if ground_truth_filter:
                conditions.append("ground_truth = %s")
                params.append(ground_truth_filter)
            where = " AND ".join(conditions) if conditions else "TRUE"
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, formula, against, parsed, ok, refuted,
                               dimensions, equivalence, symbols, context,
                               client_ip, ground_truth, label_confidence,
                               created_at
                        FROM verification_log
                        WHERE {where}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (*params, limit, offset),
                    )
                    return [
                        {
                            "id": r[0],
                            "formula": r[1],
                            "against": r[2],
                            "parsed": r[3],
                            "ok": r[4],
                            "refuted": r[5],
                            "dimensions": r[6],
                            "equivalence": r[7],
                            "symbols": r[8] or [],
                            "context": r[9],
                            "client_ip": r[10],
                            "ground_truth": r[11],
                            "label_confidence": r[12],
                            "created_at": r[13].isoformat() if r[13] else None,
                        }
                        for r in cur.fetchall()
                    ]
        except Exception as exc:
            logger.warning("failed to query verifications: %s", exc)
            return []

    # ─── check log ────────────────────────────────────────────────────────

    def save_check(
        self,
        *,
        brief: str,
        draft: str,
        action: str,
        confidence: float | None = None,
        domain: str = "",
        physics_ok: bool | None = None,
        detail: str = "",
        sources: Sequence[dict[str, Any]] = (),
        client_ip: str = "",
        ground_truth: str | None = None,
        label_confidence: float = 1.0,
        embedding_brief: list[float] | None = None,
        embedding_draft: list[float] | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO check_log
                            (brief, draft, action, confidence, domain,
                             physics_ok, detail, sources, client_ip,
                             ground_truth, label_confidence,
                             embedding_brief, embedding_draft)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            brief, draft, action, confidence, domain,
                            physics_ok, detail,
                            json.dumps(list(sources), default=str),
                            client_ip,
                            ground_truth, label_confidence,
                            embedding_brief, embedding_draft,
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("failed to save check: %s", exc)

    def query_checks(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        brief_filter: str = "",
        action_filter: str = "",
        client_ip: str = "",
        ground_truth_filter: str = "",
    ) -> list[dict[str, Any]]:
        """Browse gate-check history (newest first)."""
        if not self._enabled:
            return []
        try:
            conditions: list[str] = []
            params: list[Any] = []
            if brief_filter:
                conditions.append("brief ILIKE %s")
                params.append(f"%{brief_filter}%")
            if action_filter:
                conditions.append("action = %s")
                params.append(action_filter)
            if client_ip:
                conditions.append("client_ip = %s")
                params.append(client_ip)
            if ground_truth_filter:
                conditions.append("ground_truth = %s")
                params.append(ground_truth_filter)
            where = " AND ".join(conditions) if conditions else "TRUE"
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, brief, draft, action, confidence, domain,
                               physics_ok, detail, sources, client_ip,
                               ground_truth, label_confidence, created_at
                        FROM check_log
                        WHERE {where}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (*params, limit, offset),
                    )
                    return [
                        {
                            "id": r[0],
                            "brief": r[1],
                            "draft": r[2],
                            "action": r[3],
                            "confidence": r[4],
                            "domain": r[5],
                            "physics_ok": r[6],
                            "detail": r[7],
                            "sources": r[8] if isinstance(r[8], list) else json.loads(r[8] or "[]"),
                            "client_ip": r[9],
                            "ground_truth": r[10],
                            "label_confidence": r[11],
                            "created_at": r[12].isoformat() if r[12] else None,
                        }
                        for r in cur.fetchall()
                    ]
        except Exception as exc:
            logger.warning("failed to query checks: %s", exc)
            return []

    # ─── sources (corpus records) ──────────────────────────────────────────

    def save_source(
        self,
        *,
        id: str,
        english: str = "",
        math_formula: str = "",
        scientific_domain: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO sources
                            (id, english, math_formula, scientific_domain, meta)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            english = EXCLUDED.english,
                            math_formula = EXCLUDED.math_formula,
                            scientific_domain = EXCLUDED.scientific_domain,
                            meta = EXCLUDED.meta
                        """,
                        (
                            id, english, math_formula, scientific_domain,
                            json.dumps(meta or {}, default=str),
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("failed to save source: %s", exc)

    def search_sources(
        self, query: str = "", limit: int = 20
    ) -> list[dict[str, Any]]:
        if not self._enabled:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    if query:
                        cur.execute(
                            """
                            SELECT id, english, math_formula, scientific_domain, meta
                            FROM sources
                            WHERE english ILIKE %s
                               OR math_formula ILIKE %s
                               OR scientific_domain ILIKE %s
                            ORDER BY created_at DESC
                            LIMIT %s
                            """,
                            (f"%{query}%", f"%{query}%", f"%{query}%", limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, english, math_formula, scientific_domain, meta
                            FROM sources
                            ORDER BY created_at DESC
                            LIMIT %s
                            """,
                            (limit,),
                        )
                    rows = cur.fetchall()
                    return [
                        {
                            "id": r[0],
                            "english": r[1],
                            "math_formula": r[2],
                            "scientific_domain": r[3],
                            "meta": r[4] if isinstance(r[4], dict) else json.loads(r[4] or "{}"),
                        }
                        for r in rows
                    ]
        except Exception as exc:
            logger.warning("failed to search sources: %s", exc)
            return []

    # ─── exports (data monetisation) ───────────────────────────────────────

    def export_verified_formulas(
        self,
        *,
        limit: int = 1000,
        offset: int = 0,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Export formula-verification pairs with Z3 ground-truth labels.

        Returns only records where ``ground_truth`` is ``correct`` or
        ``incorrect`` (not ``uncertain``). These are the labelled training
        examples — the commercially valuable dataset.
        """
        if not self._enabled:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT formula, against, dimensions, equivalence,
                               ground_truth, label_confidence, symbols,
                               created_at
                        FROM verification_log
                        WHERE ground_truth IN ('correct', 'incorrect')
                          AND label_confidence >= %s
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (min_confidence, limit, offset),
                    )
                    return [
                        {
                            "formula": r[0],
                            "against": r[1],
                            "dimensions": r[2],
                            "equivalence": r[3],
                            "label": r[4],
                            "label_confidence": r[5],
                            "symbols": r[6] or [],
                            "created_at": r[7].isoformat() if r[7] else None,
                        }
                        for r in cur.fetchall()
                    ]
        except Exception as exc:
            logger.warning("failed to export verified formulas: %s", exc)
            return []

    def export_benchmark_cases(
        self,
        *,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Export gate-check cases for LLM benchmark usage.

        Each record is a (brief, draft, verdict) triplet usable as a
        benchmark example for physics-aware evaluation of language models.
        """
        if not self._enabled:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT brief, draft, action, confidence, domain,
                               ground_truth, label_confidence,
                               created_at
                        FROM check_log
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (limit, offset),
                    )
                    return [
                        {
                            "brief": r[0],
                            "draft": r[1],
                            "verdict": r[2],
                            "confidence": r[3],
                            "domain": r[4],
                            "label": r[5],
                            "label_confidence": r[6],
                            "created_at": r[7].isoformat() if r[7] else None,
                        }
                        for r in cur.fetchall()
                    ]
        except Exception as exc:
            logger.warning("failed to export benchmark cases: %s", exc)
            return []

    def search_similar_formulas(
        self,
        embedding: list[float],
        *,
        limit: int = 10,
        min_similarity: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find semantically similar formulas by cosine distance.

        Requires pgvector and an embedding index on verification_log.
        Returns None-ranked results when embedding is missing or the DB
        is unavailable.
        """
        if not self._enabled or not embedding:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT formula, dimensions, ground_truth,
                               label_confidence,
                               1 - (embedding <=> %s::vector) AS similarity,
                               created_at
                        FROM verification_log
                        WHERE embedding IS NOT NULL
                          AND ground_truth IN ('correct', 'incorrect')
                          AND 1 - (embedding <=> %s::vector) >= %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (embedding, embedding, min_similarity, embedding, limit),
                    )
                    return [
                        {
                            "formula": r[0],
                            "dimensions": r[1],
                            "label": r[2],
                            "label_confidence": r[3],
                            "similarity": round(float(r[4]), 4),
                            "created_at": r[5].isoformat() if r[5] else None,
                        }
                        for r in cur.fetchall()
                    ]
        except Exception as exc:
            logger.warning("failed to search similar formulas: %s", exc)
            return []

    # ─── user management ───────────────────────────────────────────────────

    def create_user(
        self,
        *,
        email: str,
        api_key_hash: str,
        api_key_prefix: str,
        plan: str = "free",
    ) -> dict[str, Any] | None:
        """Create a user and return their row as a dict, or None on failure."""
        if not self._enabled:
            return None
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO users (email, api_key_hash, api_key_prefix, plan)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (email) DO NOTHING
                        RETURNING id, email, api_key_prefix, plan, payment_customer_id, created_at
                        """,
                        (email, api_key_hash, api_key_prefix, plan),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    if row is None:
                        return None  # email already exists
                    return {
                        "id": row[0],
                        "email": row[1],
                        "api_key_prefix": row[2],
                        "plan": row[3],
                        "payment_customer_id": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                    }
        except Exception as exc:
            logger.warning("failed to create user: %s", exc)
            return None

    def get_user_by_api_key(self, api_key_hash: str) -> dict[str, Any] | None:
        """Look up a user by hashed API key."""
        if not self._enabled:
            return None
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, email, api_key_prefix, plan, payment_customer_id, created_at
                        FROM users WHERE api_key_hash = %s
                        """,
                        (api_key_hash,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return {
                        "id": row[0],
                        "email": row[1],
                        "api_key_prefix": row[2],
                        "plan": row[3],
                        "payment_customer_id": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                    }
        except Exception as exc:
            logger.warning("failed to look up user by api key: %s", exc)
            return None

    def get_user_by_id(self, user_id: int) -> dict[str, Any] | None:
        """Look up a user by primary key."""
        if not self._enabled:
            return None
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, email, api_key_prefix, plan, payment_customer_id, created_at
                        FROM users WHERE id = %s
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return {
                        "id": row[0],
                        "email": row[1],
                        "api_key_prefix": row[2],
                        "plan": row[3],
                        "payment_customer_id": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                    }
        except Exception as exc:
            logger.warning("failed to look up user by id: %s", exc)
            return None

    def get_user_by_email(self, email: str) -> dict[str, Any] | None:
        """Look up a user by email address."""
        if not self._enabled:
            return None
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, email, api_key_prefix, plan, payment_customer_id, created_at
                        FROM users WHERE email = %s
                        """,
                        (email,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return {
                        "id": row[0],
                        "email": row[1],
                        "api_key_prefix": row[2],
                        "plan": row[3],
                        "payment_customer_id": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                    }
        except Exception as exc:
            logger.warning("failed to look up user by email: %s", exc)
            return None

    def update_user_plan(self, user_id: int, plan: str) -> bool:
        """Update a user's plan. Returns True on success."""
        if not self._enabled:
            return False
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET plan = %s WHERE id = %s",
                        (plan, user_id),
                    )
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("failed to update user plan: %s", exc)
            return False

    def update_user_payment_customer(self, user_id: int, payment_customer_id: str) -> bool:
        """Store a payment provider customer id against a user."""
        if not self._enabled:
            return False
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET payment_customer_id = %s WHERE id = %s",
                        (payment_customer_id, user_id),
                    )
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("failed to update payment customer: %s", exc)
            return False

    def list_users(self, *, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List registered users (admin)."""
        if not self._enabled:
            return []
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, email, api_key_prefix, plan, payment_customer_id, created_at
                        FROM users ORDER BY created_at DESC LIMIT %s OFFSET %s
                        """,
                        (limit, offset),
                    )
                    return [
                        {
                            "id": r[0], "email": r[1], "api_key_prefix": r[2],
                            "plan": r[3], "payment_customer_id": r[4],
                            "created_at": r[5].isoformat() if r[5] else None,
                        }
                        for r in cur.fetchall()
                    ]
        except Exception as exc:
            logger.warning("failed to list users: %s", exc)
            return []

    def revoke_user(self, user_id: int) -> bool:
        """Delete a user and all associated records (cascade)."""
        if not self._enabled:
            return False
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("failed to revoke user: %s", exc)
            return False

    # ─── usage tracking ────────────────────────────────────────────────────

    def track_usage(self, user_id: int, endpoint: str) -> None:
        """Atomically increment the daily usage counter for a user.

        ``endpoint`` must be ``"verify"`` or ``"check"``.
        """
        if not self._enabled:
            return
        column = "check_count" if endpoint == "check" else "verify_count"
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        INSERT INTO usage_daily (user_id, date, {column})
                        VALUES (%s, CURRENT_DATE, 1)
                        ON CONFLICT (user_id, date)
                        DO UPDATE SET {column} = usage_daily.{column} + 1
                        """,
                        (user_id,),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("failed to track usage: %s", exc)

    def get_monthly_usage(self, user_id: int) -> dict[str, int]:
        """Return total verify + check counts for the current calendar month.

        Returns ``{"verify": N, "check": M}`` (both zero when the database
        is unavailable).
        """
        if not self._enabled:
            return {"verify": 0, "check": 0}
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            COALESCE(SUM(verify_count), 0) AS v,
                            COALESCE(SUM(check_count), 0)  AS c
                        FROM usage_daily
                        WHERE user_id = %s
                          AND date >= date_trunc('month', CURRENT_DATE)
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
                    return {"verify": int(row[0]), "check": int(row[1])}
        except Exception as exc:
            logger.warning("failed to read monthly usage: %s", exc)
            return {"verify": 0, "check": 0}

    def check_quota(self, user_id: int, plan_key: str) -> tuple[bool, int, int]:
        """Check whether *user_id* has remaining quota for their plan.

        Returns:
            ``(ok, used_total, limit)`` — ``ok`` is ``True`` when under quota
            or the plan has no limit. ``limit`` is ``-1`` for unlimited.
        """
        from formulagate.plans import PLANS as _PLANS

        plan = _PLANS.get(plan_key, _PLANS["free"])
        usage = self.get_monthly_usage(user_id)
        used = usage["verify"] + usage["check"]

        if plan.monthly_quota is None:
            return (True, used, -1)  # unlimited

        return (used < plan.monthly_quota, used, plan.monthly_quota)

    def get_usage_stats(self) -> dict[str, int]:
        """Aggregate usage across all users (admin)."""
        if not self._enabled:
            return {"total_users": 0, "total_verify": 0, "total_check": 0}
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM users")
                    total_users = cur.fetchone()[0]
                    cur.execute(
                        """
                        SELECT
                            COALESCE(SUM(verify_count), 0),
                            COALESCE(SUM(check_count), 0)
                        FROM usage_daily
                        """
                    )
                    v, c = cur.fetchone()
                    return {
                        "total_users": total_users,
                        "total_verify": int(v),
                        "total_check": int(c),
                    }
        except Exception as exc:
            logger.warning("failed to read usage stats: %s", exc)
            return {"total_users": 0, "total_verify": 0, "total_check": 0}

    # ─── subscriptions ─────────────────────────────────────────────────────

    def save_subscription(
        self,
        *,
        user_id: int,
        plan: str,
        payment_subscription_id: str = "",
        status: str = "active",
        current_period_start: str = "",
        current_period_end: str = "",
        cancel_at: str = "",
    ) -> None:
        """Insert or update a subscription record."""
        if not self._enabled:
            return
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO subscriptions
                            (user_id, plan, payment_subscription_id, status,
                             current_period_start, current_period_end, cancel_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            user_id, plan, payment_subscription_id, status,
                            current_period_start or None,
                            current_period_end or None,
                            cancel_at or None,
                        ),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("failed to save subscription: %s", exc)

    def get_active_subscription(self, user_id: int) -> dict[str, Any] | None:
        """Return the most recent subscription for *user_id*, or None."""
        if not self._enabled:
            return None
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, plan, payment_subscription_id, status,
                               current_period_start, current_period_end,
                               cancel_at, created_at, updated_at
                        FROM subscriptions
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (user_id,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return None
                    return {
                        "id": row[0], "plan": row[1],
                        "payment_subscription_id": row[2], "status": row[3],
                        "current_period_start": row[4].isoformat() if row[4] else None,
                        "current_period_end": row[5].isoformat() if row[5] else None,
                        "cancel_at": row[6].isoformat() if row[6] else None,
                        "created_at": row[7].isoformat() if row[7] else None,
                        "updated_at": row[8].isoformat() if row[8] else None,
                    }
        except Exception as exc:
            logger.warning("failed to get active subscription: %s", exc)
            return None

    def update_subscription_status(self, payment_sub_id: str, status: str) -> None:
        """Update the status of a subscription identified by payment provider id."""
        if not self._enabled:
            return
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE subscriptions
                        SET status = %s, updated_at = NOW()
                        WHERE payment_subscription_id = %s
                        """,
                        (status, payment_sub_id),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("failed to update subscription status: %s", exc)

    def get_user_id_by_payment_customer(self, payment_customer_id: str) -> int | None:
        """Resolve a user id from a payment provider customer id."""
        if not self._enabled:
            return None
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM users WHERE payment_customer_id = %s",
                        (payment_customer_id,),
                    )
                    row = cur.fetchone()
                    return row[0] if row else None
        except Exception as exc:
            logger.warning("failed to resolve payment customer: %s", exc)
            return None

    # ─── key audit ─────────────────────────────────────────────────────────

    def log_api_key_usage(
        self,
        *,
        user_id: int | None,
        api_key_hash: str,
        endpoint: str,
        client_ip: str = "",
        user_agent: str = "",
    ) -> None:
        """Write an audit record for API key usage."""
        if not self._enabled:
            return
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO api_key_audit
                            (user_id, api_key_hash, endpoint, client_ip, user_agent)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (user_id, api_key_hash, endpoint, client_ip, user_agent),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("failed to log api key audit: %s", exc)

    # ─── stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, int]:
        """Row counts plus ground_truth breakdowns."""
        out = {
            "verification_log": 0, "check_log": 0, "sources": 0,
            "verified_correct": 0, "verified_incorrect": 0,
            "check_correct": 0, "check_incorrect": 0,
        }
        if not self._enabled:
            return out
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    for t in ("verification_log", "check_log", "sources"):
                        cur.execute(f"SELECT COUNT(*) FROM {t}")
                        out[t] = cur.fetchone()[0]
                    cur.execute(
                        "SELECT ground_truth, COUNT(*) FROM verification_log "
                        "WHERE ground_truth IS NOT NULL GROUP BY ground_truth"
                    )
                    for gt, cnt in cur.fetchall():
                        if gt == "correct":
                            out["verified_correct"] = cnt
                        elif gt == "incorrect":
                            out["verified_incorrect"] = cnt
                    cur.execute(
                        "SELECT ground_truth, COUNT(*) FROM check_log "
                        "WHERE ground_truth IS NOT NULL GROUP BY ground_truth"
                    )
                    for gt, cnt in cur.fetchall():
                        if gt == "correct":
                            out["check_correct"] = cnt
                        elif gt == "incorrect":
                            out["check_incorrect"] = cnt
        except Exception as exc:
            logger.warning("failed to read stats: %s", exc)
        return out


class _PooledConn:
    """Context manager that returns a pooled connection back to the pool."""

    def __init__(self, conn: Any, pool: Any):
        self._conn = conn
        self._pool = pool

    def __enter__(self) -> Any:
        return self._conn

    def __exit__(self, *exc: Any) -> None:
        try:
            self._pool.putconn(self._conn)
        except Exception:
            pass


# ─── module-level default ────────────────────────────────────────────────────

_DEFAULT_DB: DatabaseManager | None = None


def _DEFAULT_DB_URL() -> str | None:
    import os

    return os.environ.get("FORMULAGATE_DATABASE_URL") or os.environ.get("DATABASE_URL")


def get_database() -> DatabaseManager:
    """Return the process-wide DatabaseManager (lazily created)."""
    global _DEFAULT_DB
    if _DEFAULT_DB is None:
        _DEFAULT_DB = DatabaseManager()
    return _DEFAULT_DB


def close_database() -> None:
    """Close and reset the process-wide DatabaseManager."""
    global _DEFAULT_DB
    if _DEFAULT_DB is not None:
        _DEFAULT_DB.close()
        _DEFAULT_DB = None
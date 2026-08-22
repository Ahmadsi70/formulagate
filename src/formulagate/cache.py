"""Rate limiting, caching, and monitoring for Formulagate API calls.

Gate 5 of the MVP roadmap.  Provides:

  ``FormulagateCache`` — Simple LRU cache that stores check results keyed by
      (brief, draft, sources_hash).  Reuses the same source records for
      identical queries.

  ``RateLimiter`` — Token bucket rate limiter that throttles requests to a
      specified QPS (queries per second) or burst size.

  ``UsageMonitor`` — Tracks API usage, abstention rates, and latency metrics
      for observability and billing.

  ``CachedGate`` — Decorator that wraps any Formulagate gate instance with
      caching and rate limiting (default enabled).

Usage:
    from formulagate.cache import CachedGate, RateLimiter, UsageMonitor

    limiter = RateLimiter(qps=10, burst=20)
    monitor = UsageMonitor()

    gate = Formulagate(sources=my_docs)
    cached_gate = CachedGate(gate, limiter=limiter, monitor=monitor)

    result = cached_gate.check(brief="...", draft="...")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─── Cache Implementation ──────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """Cache entry with expiry and metrics."""

    result: Any
    created_at: float
    access_count: int = 0
    last_accessed: float = 0.0


class FormulagateCache:
    """LRU cache for Formulagate check results.

    Caches the full ClaimResult for identical (brief, draft, sources) calls.
    Sources are normalized by their hash before caching, allowing cached
    results to be reused across slightly different source records.

    Args:
        ttl_seconds: Time-to-live for cache entries (default 3600s = 1h).
        max_size: Maximum number of entries (default 1000).
        persist_path: Path to persist cache to disk on shutdown.
    """

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        persist_path: str | Path | None = None,
    ) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._persist_path = Path(persist_path) if persist_path else None
        self._hits: int = 0
        self._misses: int = 0

    def _key(
        self,
        brief: str,
        draft: str,
        sources: list[dict[str, Any]] | None,
    ) -> str:
        """Generate cache key from inputs."""
        # Hash sources to normalize them
        sources_hash = ""
        if sources:
            # Sort by key to ensure consistent hashing
            sorted_sources = sorted(sources, key=lambda x: (x.get("id", ""), x.get("text", "")))
            sources_json = json.dumps(sorted_sources, sort_keys=True)
            sources_hash = hashlib.md5(sources_json.encode()).hexdigest()

        key_parts = [brief, draft, sources_hash]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def get(
        self, brief: str, draft: str, sources: list[dict[str, Any]] | None
    ) -> Any | None:
        """Retrieve cached result or None if not found/m expired."""
        key = self._key(brief, draft, sources)

        entry = self._cache.get(key)
        if not entry:
            self._misses += 1
            return None

        # Check TTL
        if time.monotonic() - entry.created_at > self._ttl_seconds:
            del self._cache[key]
            self._invalidate_misses()
            return None

        # Update access tracking
        entry.access_count += 1
        entry.last_accessed = time.monotonic()
        self._hits += 1

        # Update LRU position (move to end)
        del self._cache[key]
        self._cache[key] = entry

        logger.debug("Cache HIT for key %s", key[:8])
        return entry.result

    def _invalidate_misses(self) -> None:
        """Internal helper to increment misses when cache expires."""
        self._misses += 1

    def set(
        self,
        brief: str,
        draft: str,
        sources: list[dict[str, Any]] | None,
        result: Any,
    ) -> None:
        """Store result in cache."""
        # Check size limit
        if len(self._cache) >= self._max_size:
            # Evict oldest entry (first in, first out)
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]
            logger.debug("Cache eviction: %s", oldest_key[:8])

        key = self._key(brief, draft, sources)
        self._cache[key] = CacheEntry(
            result=result,
            created_at=time.monotonic(),
            access_count=0,
            last_accessed=time.monotonic(),
        )
        logger.debug("Cache MISS: stored key %s", key[:8])

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def persist(self) -> None:
        """Persist cache to disk (not yet implemented)."""
        if self._persist_path:
            logger.warning("Cache persistence to %s not yet implemented", self._persist_path)


# ─── Rate Limiter ───────────────────────────────────────────────────────────────


class RateLimiter:
    """Token bucket rate limiter for API calls.

    Allows a burst of requests followed by a steady rate.  Compatible with
    request throttling in production deployments.

    Args:
        qps: Queries per second (default 10).
        burst: Maximum burst size (default 20).
    """

    def __init__(self, qps: int = 10, burst: int = 20) -> None:
        self._qps = qps
        self._burst = burst
        self._tokens: float = float(burst)
        self._last_update: float = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        refill = elapsed * self._qps
        self._tokens = min(self._burst, self._tokens + refill)
        self._last_update = now

    def acquire(self) -> bool:
        """Try to acquire a token. Returns True if successful."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _get_tokens(self) -> float:
        """Internal helper to get current token count for testing."""
        return self._tokens


# ─── Usage Monitor ──────────────────────────────────────────────────────────────


@dataclass
class UsageMetrics:
    """Metrics collected from Formulagate calls."""

    total_checks: int = 0
    generate_actions: int = 0
    abstain_actions: int = 0
    avg_latency_ms: float = 0.0
    latency_samples: int = 0
    last_check_time: float = 0.0


class UsageMonitor:
    """Monitor usage statistics for Formulagate API calls.

    Tracks latency, action distribution, and can be used for billing or
    anomaly detection.

    Args:
        min_latency_ms: Minimum latency to record (default 1ms).
    """

    def __init__(self, min_latency_ms: float = 1.0) -> None:
        self._metrics = UsageMetrics()
        self._min_latency_ms = min_latency_ms
        self._lock: Any = None  # Would use real lock in multi-threaded env

    def record_check(
        self, latency_ms: float, action: str, ok: bool
    ) -> None:
        """Record a single check call."""
        self._metrics.total_checks += 1
        if action == "generate":
            self._metrics.generate_actions += 1
        else:
            self._metrics.abstain_actions += 1

        # Update latency (weighted average)
        if latency_ms >= self._min_latency_ms:
            if self._metrics.latency_samples == 0:
                self._metrics.avg_latency_ms = latency_ms
            else:
                self._metrics.avg_latency_ms = (
                    (self._metrics.avg_latency_ms * self._metrics.latency_samples +
                     latency_ms) /
                    (self._metrics.latency_samples + 1)
                )
            self._metrics.latency_samples += 1

        self._metrics.last_check_time = time.monotonic()

    def stats(self) -> dict[str, Any]:
        """Return usage statistics."""
        abstain_rate = (
            self._metrics.abstain_actions / self._metrics.total_checks
            if self._metrics.total_checks > 0
            else 0.0
        )
        return {
            "total_checks": self._metrics.total_checks,
            "generate_actions": self._metrics.generate_actions,
            "abstain_actions": self._metrics.abstain_actions,
            "abstain_rate": abstain_rate,
            "avg_latency_ms": self._metrics.avg_latency_ms,
            "latency_samples": self._metrics.latency_samples,
        }


# ─── Cached Gate Decorator ───────────────────────────────────────────────────────


class CachedGate:
    """Gate wrapper that adds caching and rate limiting.

    Wraps any Formulagate gate instance and applies:
      - Result caching for identical queries
      - Rate limiting (default: 10 QPS, 20 burst)
      - Usage monitoring (default enabled)

    Args:
        gate: The underlying Formulagate instance.
        cache: Optional custom cache instance.
        limiter: Optional custom rate limiter.
        monitor: Optional custom usage monitor.
    """

    def __init__(
        self,
        gate: Any,
        cache: FormulagateCache | None = None,
        limiter: RateLimiter | None = None,
        monitor: UsageMonitor | None = None,
    ) -> None:
        self._gate = gate
        self._cache = cache or FormulagateCache()
        self._limiter = limiter or RateLimiter(qps=10, burst=20)
        self._monitor = monitor or UsageMonitor()

    def check(
        self,
        brief: str,
        draft: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Run gate with caching, rate limiting, and monitoring.

        Returns:
            ClaimResult from the underlying gate.
        """
        # Check cache first
        cached = self._cache.get(brief, draft, sources)
        if cached is not None:
            return cached

        # Rate limit
        if not self._limiter.acquire():
            logger.warning("Rate limit exceeded")
            # Return abstain with low confidence
            return {
                "action": "abstain",
                "confidence": 0.0,
                "detail": "rate limit exceeded",
                "sources": [],
            }

        # Record start time
        t0 = time.monotonic()

        # Call the gate
        result = self._gate.check(brief=brief, draft=draft, sources=sources)

        # Record latency
        latency_ms = (time.monotonic() - t0) * 1000
        action = result.get("action", "abstain")
        ok = result.get("ok", False) if isinstance(result, dict) else False

        # Monitor
        self._monitor.record_check(latency_ms, action, ok)

        # Cache the result
        self._cache.set(brief, draft, sources, result)

        return result

"""Rate limiting, caching, and monitoring for Formulagate API.

Gate 5 of the MVP roadmap.  Provides:

  ``RateLimiter`` — sliding-window rate limiter (in-memory or Redis-backed).
  ``FormulaCache`` — LRU cache for formula verification results.
  ``PrometheusMetrics`` — Prometheus-compatible metrics exporter.
  ``setup_middleware`` — one-call setup for all three on a FastAPI app.

Usage:
    from formulagate.middleware import setup_middleware

    app = FastAPI()
    setup_middleware(app, redis_url="redis://localhost:6379")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ─── Rate Limiter ────────────────────────────────────────────────────────────


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Args:
        requests_per_minute: Max requests per minute per client.
        requests_per_hour: Max requests per hour per client.
        burst_size: Max burst size for sliding window.
    """

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10


class RateLimiter:
    """Sliding-window rate limiter (in-memory or Redis-backed).

    Args:
        config: Rate limit configuration.
        redis_url: Optional Redis URL for distributed rate limiting.
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        redis_url: str | None = None,
    ) -> None:
        self._config = config or RateLimitConfig()
        self._redis_url = redis_url
        self._redis = None
        self._in_memory: dict[str, list[float]] = defaultdict(list)

        if redis_url:
            try:
                import redis

                self._redis = redis.from_url(redis_url, decode_responses=True)
                logger.info("Rate limiter connected to Redis at %s", redis_url)
            except ImportError:
                logger.warning("redis package not installed, using in-memory limiter")
            except Exception as exc:
                logger.warning("Redis connection failed, using in-memory: %s", exc)

    def is_allowed(self, client_id: str) -> tuple[bool, dict[str, int]]:
        """Check if a request from client_id is allowed.

        Returns:
            (allowed, limits) where limits contains remaining quotas.
        """
        if self._redis:
            return self._redis_check(client_id)
        return self._inmemory_check(client_id)

    def _inmemory_check(self, client_id: str) -> tuple[bool, dict[str, int]]:
        now = time.time()
        window_1m = [t for t in self._in_memory[client_id] if now - t < 60]
        window_1h = [t for t in self._in_memory[client_id] if now - t < 3600]

        self._in_memory[client_id] = window_1m + window_1h

        remaining_min = max(0, self._config.requests_per_minute - len(window_1m))
        remaining_hr = max(0, self._config.requests_per_hour - len(window_1h))

        allowed = (
            len(window_1m) < self._config.requests_per_minute
            and len(window_1h) < self._config.requests_per_hour
        )

        if allowed:
            self._in_memory[client_id].append(now)

        return allowed, {
            "remaining_minute": remaining_min,
            "remaining_hour": remaining_hr,
            "limit_minute": self._config.requests_per_minute,
            "limit_hour": self._config.requests_per_hour,
        }

    def _redis_check(self, client_id: str) -> tuple[bool, dict[str, int]]:
        now = time.time()
        pipe = self._redis.pipeline()

        key_1m = f"rate:{client_id}:1m"
        key_1h = f"rate:{client_id}:1h"

        pipe.zremrangebyscore(key_1m, 0, now - 60)
        pipe.zremrangebyscore(key_1h, 0, now - 3600)
        pipe.zadd(key_1m, {str(now): now})
        pipe.zadd(key_1h, {str(now): now})
        pipe.expire(key_1m, 120)
        pipe.expire(key_1h, 7200)
        pipe.zcard(key_1m)
        pipe.zcard(key_1h)

        results = pipe.execute()
        count_1m = results[-2]
        count_1h = results[-1]

        remaining_min = max(0, self._config.requests_per_minute - count_1m)
        remaining_hr = max(0, self._config.requests_per_hour - count_1h)

        allowed = count_1m < self._config.requests_per_minute and count_1h < self._config.requests_per_hour

        return allowed, {
            "remaining_minute": remaining_min,
            "remaining_hour": remaining_hr,
            "limit_minute": self._config.requests_per_minute,
            "limit_hour": self._config.requests_per_hour,
        }


# ─── Formula Cache ───────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """Cached verification result."""

    key: str
    result: dict[str, Any]
    timestamp: float
    ttl: int = 3600  # 1 hour default

    @property
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class FormulaCache:
    """LRU cache for formula verification results.

    Args:
        max_size: Max number of entries in cache.
        default_ttl: Default TTL in seconds.
        redis_url: Optional Redis URL for distributed cache.
    """

    def __init__(
        self,
        max_size: int = 10000,
        default_ttl: int = 3600,
        redis_url: str | None = None,
    ) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._redis_url = redis_url
        self._redis = None
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []

        if redis_url:
            try:
                import redis

                self._redis = redis.from_url(redis_url, decode_responses=True)
                logger.info("Formula cache connected to Redis at %s", redis_url)
            except ImportError:
                logger.warning("redis package not installed, using in-memory cache")
            except Exception as exc:
                logger.warning("Redis connection failed, using in-memory: %s", exc)

    def get(self, key: str) -> dict[str, Any] | None:
        """Get a cached result by key."""
        if self._redis:
            return self._redis_get(key)

        entry = self._cache.get(key)
        if entry and not entry.is_expired:
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return entry.result

        if entry:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

        return None

    def set(self, key: str, result: dict[str, Any], ttl: int | None = None) -> None:
        """Cache a verification result."""
        ttl = ttl or self._default_ttl

        if self._redis:
            self._redis_set(key, result, ttl)
            return

        if len(self._cache) >= self._max_size and key not in self._cache:
            self._evict()

        self._cache[key] = CacheEntry(
            key=key,
            result=result,
            timestamp=time.time(),
            ttl=ttl,
        )
        self._access_order.append(key)

    def _evict(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
            logger.debug("Evicted cache entry: %s", oldest[:50])

    def _redis_get(self, key: str) -> dict[str, Any] | None:
        data = self._redis.get(f"cache:{key}")
        if data:
            return json.loads(data)
        return None

    def _redis_set(self, key: str, result: dict[str, Any], ttl: int) -> None:
        self._redis.setex(f"cache:{key}", ttl, json.dumps(result))

    def clear(self) -> None:
        """Clear all cached entries."""
        if self._redis:
            pattern = "cache:*"
            keys = self._redis.keys(pattern)
            if keys:
                self._redis.delete(*keys)
        else:
            self._cache.clear()
            self._access_order.clear()


# ─── Prometheus Metrics ──────────────────────────────────────────────────────


class PrometheusMetrics:
    """Prometheus-compatible metrics for Formulagate API.

    Tracks:
        - formulagate_requests_total: Total requests by endpoint and status
        - formulagate_request_duration_seconds: Request duration histogram
        - formulagate_verifications_total: Total formula verifications
        - formulagate_abstain_total: Total abstentions
        - formulagate_cache_hits_total: Cache hit/miss counts
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, int]] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize all metric counters to zero."""
        metrics = [
            "requests_total",
            "verifications_total",
            "abstains_total",
            "cache_hits_total",
            "cache_misses_total",
            "rate_limited_total",
        ]
        for metric in metrics:
            self._registry[metric] = defaultdict(int)

    def increment(self, metric: str, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        key = self._make_key(metric, labels)
        self._registry.setdefault(metric, defaultdict(int))[key] += 1

    def observe_duration(self, endpoint: str, duration_seconds: float) -> None:
        """Record a request duration."""
        self._histograms[endpoint].append(duration_seconds)
        # Keep only last 1000 observations to prevent memory leak
        if len(self._histograms[endpoint]) > 1000:
            self._histograms[endpoint] = self._histograms[endpoint][-1000:]

    def get_metrics(self) -> str:
        """Return metrics in Prometheus text format."""
        lines = ["# HELP formulagate_requests_total Total requests by endpoint"]
        lines.append("# TYPE formulagate_requests_total counter")

        for endpoint, count in self._registry.get("requests_total", {}).items():
            lines.append(f'formulagate_requests_total{{endpoint="{endpoint}"}} {count}')

        lines.append("")
        lines.append("# HELP formulagate_verifications_total Total formula verifications")
        lines.append("# TYPE formulagate_verifications_total counter")

        for label, count in self._registry.get("verifications_total", {}).items():
            lines.append(f'formulagate_verifications_total{{label="{label}"}} {count}')

        lines.append("")
        lines.append("# HELP formulagate_abstains_total Total abstentions")
        lines.append("# TYPE formulagate_abstains_total counter")

        for label, count in self._registry.get("abstains_total", {}).items():
            lines.append(f'formulagate_abstains_total{{label="{label}"}} {count}')

        lines.append("")
        lines.append("# HELP formulagate_cache_hits_total Cache hits")
        lines.append("# TYPE formulagate_cache_hits_total counter")

        hits = self._registry.get("cache_hits_total", {})
        lines.append(f'formulagate_cache_hits_total {hits.get("", 0)}')

        lines.append("")
        lines.append("# HELP formulagate_cache_misses_total Cache misses")
        lines.append("# TYPE formulagate_cache_misses_total counter")

        misses = self._registry.get("cache_misses_total", {})
        lines.append(f'formulagate_cache_misses_total {misses.get("", 0)}')

        lines.append("")
        lines.append("# HELP formulagate_request_duration_seconds Request duration histogram")
        lines.append("# TYPE formulagate_request_duration_seconds histogram")

        for endpoint, durations in self._histograms.items():
            if durations:
                avg = sum(durations) / len(durations)
                lines.append(f'formulagate_request_duration_seconds{{endpoint="{endpoint}"}} {avg:.6f}')

        return "\n".join(lines)

    def _make_key(self, metric: str, labels: dict[str, str] | None) -> str:
        if labels:
            return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return ""


# ─── Structured Logger ───────────────────────────────────────────────────────


class StructuredLogger:
    """Structured JSON logger for ELK stack compatibility.

    Usage:
        logger = StructuredLogger("formulagate")
        logger.info("request_complete", endpoint="/verify", duration_ms=123)
    """

    def __init__(self, name: str = "formulagate") -> None:
        self._logger = logging.getLogger(name)

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        entry = {
            "timestamp": time.time(),
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "level": level,
            "message": message,
            **kwargs,
        }
        log_method = getattr(self._logger, level, self._logger.info)
        log_method(json.dumps(entry, default=str))

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("error", message, **kwargs)

    def request(self, method: str, path: str, status: int, duration_ms: float, **kwargs: Any) -> None:
        """Log an HTTP request."""
        self._log(
            "info",
            "http_request",
            method=method,
            path=path,
            status=status,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )


# ─── Middleware Setup ────────────────────────────────────────────────────────


def setup_middleware(
    app: Any,
    *,
    redis_url: str | None = None,
    rate_limit: RateLimitConfig | None = None,
    cache_max_size: int = 10000,
    cache_ttl: int = 3600,
) -> dict[str, Any]:
    """One-call setup for rate limiting, caching, and monitoring.

    Args:
        app: FastAPI application instance.
        redis_url: Optional Redis URL for distributed rate limiting and caching.
        rate_limit: Rate limit configuration.
        cache_max_size: Max cache entries.
        cache_ttl: Default cache TTL in seconds.

    Returns:
        Dict with created components for manual access.
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    # Create components
    rate_limiter = RateLimiter(config=rate_limit, redis_url=redis_url)
    cache = FormulaCache(max_size=cache_max_size, default_ttl=cache_ttl, redis_url=redis_url)
    metrics = PrometheusMetrics()
    struct_logger = StructuredLogger("formulagate")

    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        """Main middleware that ties everything together."""
        client_id = request.client.host if request.client else "unknown"
        start_time = time.monotonic()

        # Rate limiting
        allowed, limits = rate_limiter.is_allowed(client_id)
        if not allowed:
            duration_ms = (time.monotonic() - start_time) * 1000
            metrics.increment("rate_limited_total")
            struct_logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                **limits,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "limits": limits,
                    "retry_after": 60,
                },
                headers={
                    "X-RateLimit-Limit-Minute": str(limits["limit_minute"]),
                    "X-RateLimit-Remaining-Minute": str(limits["remaining_minute"]),
                    "X-RateLimit-Limit-Hour": str(limits["limit_hour"]),
                    "X-RateLimit-Remaining-Hour": str(limits["remaining_hour"]),
                    "Retry-After": "60",
                },
            )

        # Process request
        response = await call_next(request)
        duration_ms = (time.monotonic() - start_time) * 1000

        # Record metrics
        metrics.increment("requests_total", {"endpoint": request.url.path})
        metrics.observe_duration(request.url.path, duration_ms / 1000)
        struct_logger.request(
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            client_id=client_id,
        )

        # Add rate limit headers
        response.headers["X-RateLimit-Limit-Minute"] = str(limits["limit_minute"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(limits["remaining_minute"])
        response.headers["X-RateLimit-Limit-Hour"] = str(limits["limit_hour"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(limits["remaining_hour"])

        return response

    # Add metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(content=metrics.get_metrics())

    # Add cache management endpoint
    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the formula cache (admin endpoint)."""
        cache.clear()
        return {"status": "ok", "message": "cache cleared"}

    return {
        "rate_limiter": rate_limiter,
        "cache": cache,
        "metrics": metrics,
        "logger": struct_logger,
    }

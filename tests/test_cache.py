"""Tests for Formulagate caching and rate limiting."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from formulagate.cache import (
    CacheEntry,
    FormulagateCache,
    RateLimiter,
    UsageMonitor,
)


# ─── Cache Tests ───────────────────────────────────────────────────────────────


def test_cache_init():
    """Test basic cache initialization."""
    cache = FormulagateCache()
    assert cache._ttl_seconds == 3600
    assert cache._max_size == 1000


def test_cache_key_generation():
    """Test that cache keys are deterministic."""
    cache = FormulagateCache()

    sources1 = [{"id": "1", "text": "E = mc^2"}]
    sources2 = [{"id": "1", "text": "E = mc^2"}]

    key1 = cache._key("What is E=mc^2?", "E = mc^2", sources1)
    key2 = cache._key("What is E=mc^2?", "E = mc^2", sources2)

    assert key1 == key2


def test_cache_get_miss():
    """Test cache miss."""
    cache = FormulagateCache()

    result = cache.get(
        brief="What is E=mc^2?",
        draft="E = mc^2",
        sources=[{"id": "1", "text": "E = mc^2"}],
    )

    assert result is None
    assert cache._misses == 1
    assert cache._hits == 0


def test_cache_set_and_get():
    """Test setting and retrieving cached results."""
    cache = FormulagateCache()
    mock_result = MagicMock()

    cache.set(
        brief="What is E=mc^2?",
        draft="E = mc^2",
        sources=[{"id": "1", "text": "E = mc^2"}],
        result=mock_result,
    )

    retrieved = cache.get(
        brief="What is E=mc^2?",
        draft="E = mc^2",
        sources=[{"id": "1", "text": "E = mc^2"}],
    )

    assert retrieved is mock_result
    assert cache._hits == 1
    assert cache._misses == 0  # Setting doesn't increment misses


def test_cache_eviction():
    """Test that cache evicts old entries when full."""
    cache = FormulagateCache(max_size=3)

    # Fill cache
    for i in range(3):
        cache.set(
            brief=f"Q{i}",
            draft=f"A{i}",
            sources=[{"id": f"{i}", "text": f"Text{i}"}],
            result=MagicMock(),
        )

    assert len(cache._cache) == 3

    # Add one more, should evict oldest
    cache.set(
        brief="Q4",
        draft="A4",
        sources=[{"id": "4", "text": "Text4"}],
        result=MagicMock(),
    )

    assert len(cache._cache) == 3


def test_cache_ttl():
    """Test cache entry expiration."""
    cache = FormulagateCache(ttl_seconds=0.1)  # 100ms TTL

    mock_result = MagicMock()
    cache.set(
        brief="What is E=mc^2?",
        draft="E = mc^2",
        sources=[{"id": "1", "text": "E = mc^2"}],
        result=mock_result,
    )

    # Should retrieve immediately
    assert cache.get(
        brief="What is E=mc^2?",
        draft="E = mc^2",
        sources=[{"id": "1", "text": "E = mc^2"}],
    )

    # Wait for TTL
    time.sleep(0.15)

    # Should miss now
    assert cache.get(
        brief="What is E=mc^2?",
        draft="E = mc^2",
        sources=[{"id": "1", "text": "E = mc^2"}],
    ) is None


def test_cache_stats():
    """Test cache statistics."""
    cache = FormulagateCache(max_size=2)

    cache.set(
        brief="Q1",
        draft="A1",
        sources=[{"id": "1", "text": "Text1"}],
        result=MagicMock(),
    )

    cache.set(
        brief="Q2",
        draft="A2",
        sources=[{"id": "2", "text": "Text2"}],
        result=MagicMock(),
    )

    cache.get(
        brief="Q1",
        draft="A1",
        sources=[{"id": "1", "text": "Text1"}],
    )

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0  # Only get operations increment misses
    assert stats["size"] == 2


def test_cache_clear():
    """Test clearing cache."""
    cache = FormulagateCache()

    cache.set(
        brief="Q1",
        draft="A1",
        sources=[{"id": "1", "text": "Text1"}],
        result=MagicMock(),
    )

    assert len(cache._cache) == 1

    cache.clear()
    assert len(cache._cache) == 0


# ─── Rate Limiter Tests ───────────────────────────────────────────────────────


def test_rate_limiter_init():
    """Test rate limiter initialization."""
    limiter = RateLimiter(qps=10, burst=20)
    assert limiter._qps == 10
    assert limiter._burst == 20
    assert limiter._tokens >= 20


def test_rate_limiter_acquisition():
    """Test rate limiter token acquisition."""
    limiter = RateLimiter(qps=1, burst=1)

    # Should acquire one token
    assert limiter.acquire() is True
    # After acquire, tokens should be close to 0 (but small positive due to refill)
    assert limiter._get_tokens() < 1.0

    # Should fail after one token used
    assert limiter.acquire() is False
    # After second acquire (which fails immediately), tokens should be near 0
    assert limiter._get_tokens() < 1.0


def test_rate_limiter_refill():
    """Test rate limiter token refill."""
    limiter = RateLimiter(qps=10, burst=20)

    # Consume all tokens
    for _ in range(20):
        limiter.acquire()

    # After refill, tokens should be near 0 (but not exactly 0 due to float)
    assert limiter._get_tokens() < 0.1

    # Wait for refill
    time.sleep(0.1)

    # Should have some tokens back (QPS=10, so 0.1s should refill at least 1 token)
    assert limiter._get_tokens() > 0


# ─── Usage Monitor Tests ───────────────────────────────────────────────────────


def test_usage_monitor_init():
    """Test usage monitor initialization."""
    monitor = UsageMonitor(min_latency_ms=1.0)
    assert monitor._metrics.total_checks == 0


def test_usage_monitor_record():
    """Test recording usage metrics."""
    monitor = UsageMonitor()

    monitor.record_check(latency_ms=100.0, action="generate", ok=True)
    monitor.record_check(latency_ms=200.0, action="abstain", ok=False)
    monitor.record_check(latency_ms=150.0, action="generate", ok=True)

    stats = monitor.stats()

    assert stats["total_checks"] == 3
    assert stats["generate_actions"] == 2
    assert stats["abstain_actions"] == 1
    assert stats["abstain_rate"] == 1.0 / 3.0
    assert stats["avg_latency_ms"] == 150.0
    assert stats["latency_samples"] == 3


def test_usage_monitor_average_latency():
    """Test weighted average latency calculation."""
    monitor = UsageMonitor()

    monitor.record_check(latency_ms=10.0, action="generate", ok=True)
    monitor.record_check(latency_ms=100.0, action="generate", ok=True)
    monitor.record_check(latency_ms=1000.0, action="generate", ok=True)

    stats = monitor.stats()
    expected = (10.0 + 100.0 + 1000.0) / 3.0
    assert stats["avg_latency_ms"] == expected


# ─── CacheEntry Tests ───────────────────────────────────────────────────────────


def test_cache_entry():
    """Test CacheEntry creation."""
    entry = CacheEntry(
        result="test_result",
        created_at=123456.0,
        access_count=5,
    )

    assert entry.result == "test_result"
    assert entry.created_at == 123456.0
    assert entry.access_count == 5

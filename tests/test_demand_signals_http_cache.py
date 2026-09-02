"""Offline tests for demand_signals.http_cache (no real network)."""

import json as json_lib

import pytest
import requests

from demand_signals.http_cache import CachedClient, DiskCache, RateLimiter, retry_with_backoff


# ── DiskCache ────────────────────────────────────────────────────────────────

def test_disk_cache_miss_then_hit(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("k1") is None
    cache.set("k1", "hello")
    assert cache.get("k1") == "hello"


def test_disk_cache_creates_its_directory(tmp_path):
    target = tmp_path / "nested" / "dir"
    DiskCache(target)
    assert target.is_dir()


# ── RateLimiter ──────────────────────────────────────────────────────────────

def test_rate_limiter_throttles_after_budget_exhausted(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    limiter = RateLimiter(rate=2, window=1.0)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()  # 3rd call within the window should trigger a sleep
    assert len(slept) == 1


def test_rate_limiter_does_not_throttle_under_budget(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    limiter = RateLimiter(rate=5, window=1.0)
    for _ in range(5):
        limiter.acquire()
    assert slept == []


# ── retry_with_backoff ───────────────────────────────────────────────────────

def test_retry_with_backoff_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("transient")
        return "ok"

    assert retry_with_backoff(flaky, max_attempts=3) == "ok"
    assert calls["n"] == 3


def test_retry_with_backoff_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise requests.ConnectionError("down")

    with pytest.raises(requests.ConnectionError):
        retry_with_backoff(always_fails, max_attempts=3)
    assert calls["n"] == 3


# ── CachedClient ─────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, body):
        self.text = body

    def raise_for_status(self):
        pass


def test_cached_client_caches_and_skips_network_on_hit(tmp_path, monkeypatch):
    client = CachedClient(user_agent="test-agent", cache_dir=tmp_path, rate=100)
    calls = {"n": 0}

    def fake_request(method, url, json=None, headers=None, timeout=30):
        calls["n"] += 1
        return _FakeResponse(json_lib.dumps({"call": calls["n"]}))

    monkeypatch.setattr(client.session, "request", fake_request)

    first = client.request("https://example.test/x", cache_key="k1")
    second = client.request("https://example.test/x", cache_key="k1")

    assert first == {"call": 1}
    assert second == first
    assert calls["n"] == 1


def test_cached_client_force_bypasses_cache(tmp_path, monkeypatch):
    client = CachedClient(user_agent="test-agent", cache_dir=tmp_path, rate=100)
    calls = {"n": 0}

    def fake_request(method, url, json=None, headers=None, timeout=30):
        calls["n"] += 1
        return _FakeResponse(json_lib.dumps({"call": calls["n"]}))

    monkeypatch.setattr(client.session, "request", fake_request)

    client.request("https://example.test/x", cache_key="k1")
    client.request("https://example.test/x", cache_key="k1", force=True)
    assert calls["n"] == 2


def test_cached_client_posts_json_body_and_headers(tmp_path, monkeypatch):
    client = CachedClient(user_agent="test-agent", cache_dir=tmp_path, rate=100)
    captured = {}

    def fake_request(method, url, json=None, headers=None, timeout=30):
        captured.update(method=method, url=url, json=json, headers=headers)
        return _FakeResponse("{}")

    monkeypatch.setattr(client.session, "request", fake_request)

    client.request(
        "https://example.test/query",
        method="POST",
        json_body={"a": 1},
        headers={"Authorization": "Bearer tok"},
    )
    assert captured["method"] == "POST"
    assert captured["json"] == {"a": 1}
    assert captured["headers"] == {"Authorization": "Bearer tok"}

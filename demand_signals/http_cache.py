"""
Shared HTTP-cache-plus-rate-limit plumbing for demand_signals' sources.

Same PATTERN as edgar/core.py's Layer 1 (rate limiter + on-disk cache keyed
by an opaque string), generalized so each source (FINRA, a future options
vendor, ...) gets its OWN RateLimiter + cache directory + requests.Session
instance rather than sharing one. Sharing a single rate-limit budget across
unrelated hosts (SEC vs FINRA vs Yahoo) would incorrectly throttle all three
together, and edgar/core.py itself is left untouched -- it already works and
is already covered by its own tests.

Adds one thing edgar/core.py doesn't have: retry_with_backoff(), a small
polite-retry helper for transient failures (429/5xx), since FINRA's Query
API is a POST with a JSON body (not just a GET like SEC's endpoints) and
this package's own docstring commits to "polite backoff" explicitly.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path

import requests


def cache_date() -> str:
    """Today's date as a cache-key suffix. A seam so tests can pin it."""
    return date.today().strftime("%Y%m%d")


class RateLimiter:
    """Sliding-window rate limiter. Same shape as edgar/core.py's
    _RateLimiter, but public and independently instantiable per source."""

    def __init__(self, rate: int, window: float = 1.0):
        self.rate, self.window = rate, window
        self.calls: deque = deque()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > self.window:
                self.calls.popleft()
            if len(self.calls) >= self.rate:
                time.sleep(max(self.window - (now - self.calls[0]), 0))
            self.calls.append(time.monotonic())


class DiskCache:
    """Flat on-disk cache keyed by an opaque string, one file per key."""

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> str | None:
        f = self.dir / key
        return f.read_text() if f.exists() else None

    def set(self, key: str, text: str) -> None:
        (self.dir / key).write_text(text)


def retry_with_backoff(fn, max_attempts: int = 3, base_delay: float = 1.0):
    """Call fn() with polite exponential backoff on requests.RequestException.

    Retries transient failures (network errors, 429, 5xx via raise_for_status)
    up to max_attempts times, sleeping base_delay * 2**attempt between tries.
    Re-raises the last exception if every attempt fails.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_exc


class CachedClient:
    """A requests.Session + RateLimiter + DiskCache bundle, one per source.

    Mirrors edgar/core.py's get(url, cache_key, force, is_json) shape and
    semantics, generalized to any HTTP method and an optional JSON body /
    extra headers (FINRA's Query API needs POST + a bearer token header;
    edgar/core.py never needed that).
    """

    def __init__(self, user_agent: str, cache_dir: Path, rate: int = 5, window: float = 1.0):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.limiter = RateLimiter(rate=rate, window=window)
        self.cache = DiskCache(cache_dir)

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        cache_key: str | None = None,
        force: bool = False,
        is_json: bool = True,
        json_body: dict | None = None,
        headers: dict | None = None,
    ):
        """Rate-limited, cached, politely-retried HTTP call.

        Returns parsed JSON (is_json=True) or raw text. A cache hit skips
        the network entirely -- no rate-limit or retry cost.
        """
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None and not force:
                return json.loads(cached) if is_json else cached

        def do_request():
            self.limiter.acquire()
            resp = self.session.request(method, url, json=json_body, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp

        resp = retry_with_backoff(do_request)
        text = resp.text
        if cache_key:
            self.cache.set(cache_key, text)
        return json.loads(text) if is_json else text

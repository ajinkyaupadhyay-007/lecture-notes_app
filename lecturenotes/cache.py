"""Persistent disk caching using diskcache.

Provides caching for expensive operations like sentence embeddings and Whisper transcriptions.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from lecturenotes.config import load_config

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_cache = None


def get_cache():
    """Get or initialize the singleton diskcache.Cache instance."""
    global _cache
    if _cache is None:
        try:
            import diskcache  # type: ignore[import-untyped]

            cfg = load_config().cache
            cache_dir = Path(cfg.directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            size_limit_bytes = cfg.size_limit_gb * 1024 * 1024 * 1024
            _cache = diskcache.Cache(str(cache_dir), size_limit=size_limit_bytes)
            logger.info("Initialized diskcache at: %s (limit %d GB)", cache_dir, cfg.size_limit_gb)
        except ImportError:
            logger.warning("diskcache not installed; using in-memory dictionary cache fallback.")
            _cache = {}
    return _cache


def make_cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """Generate a stable MD5 key from arguments."""
    payload = json.dumps({"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}}, sort_keys=True)
    hash_digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{hash_digest}"


def cached_step(prefix: str):
    """Decorator to cache function results using diskcache."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache()
            key = make_cache_key(prefix, *args, **kwargs)
            if key in cache:
                logger.debug("Cache hit for: %s", key)
                return cache[key]
            result = fn(*args, **kwargs)
            try:
                cache[key] = result
            except Exception as e:
                logger.warning("Failed to write to diskcache for %s: %e", key, e)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator

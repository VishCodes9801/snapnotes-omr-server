"""Tiny on-disk JSON cache keyed by a SHA-256 hex digest. Callers compute
the digest incrementally over the upload bytes (see main.py) so we never
allocate a single contiguous buffer the size of all pages combined."""

import json
import logging
import os
from pathlib import Path
from typing import Any

# Cache directory is relative by default ("cache/" under CWD), which lines
# up with Docker's WORKDIR=/app. Override via SNAPNOTES_CACHE_DIR for
# deployments that need a persistent / mounted volume, or for local
# uvicorn launches from a different CWD.
CACHE_DIR = Path(os.getenv("SNAPNOTES_CACHE_DIR", "cache"))

# Soft cap. Each entry is a few KB of JSON; 200 ≈ a few MB worst case but
# stops unbounded growth across months of use. Eviction is LRU-by-mtime
# triggered opportunistically on write.
MAX_ENTRIES = 200

# Truncated SHA-256 length used for filenames. 16 hex chars = 64 bits of
# collision space, plenty for a 200-entry cache.
_KEY_LEN = 16

log = logging.getLogger("snapnotes.cache")


def _path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key[:_KEY_LEN]}.json"


def get(key: str) -> dict[str, Any] | None:
    p = _path(key)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # Touch to refresh mtime so a frequently-read entry survives eviction.
    try:
        p.touch()
    except OSError:
        pass
    return payload


def put(key: str, payload: dict[str, Any]) -> None:
    _path(key).write_text(json.dumps(payload))
    _evict_if_full()


def _evict_if_full() -> None:
    try:
        files = list(CACHE_DIR.glob("*.json"))
    except OSError:
        return
    if len(files) <= MAX_ENTRIES:
        return
    files.sort(key=lambda f: f.stat().st_mtime)
    to_remove = files[: len(files) - MAX_ENTRIES]
    for f in to_remove:
        try:
            f.unlink()
        except OSError:
            pass
    log.info("cache eviction: removed %d entries", len(to_remove))

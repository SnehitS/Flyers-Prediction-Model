"""JSON disk cache for NHL API responses.

Historical payloads are immutable, so a hit is reused for the rest of the
rebuild and any later runs. Cache lives under data/raw/cache/ (gitignored).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("data/raw/cache")

hits = 0
misses = 0


def reset_stats() -> None:
    global hits, misses
    hits = 0
    misses = 0


def stats() -> dict:
    return {"hits": hits, "misses": misses, "root": str(CACHE_ROOT)}


def _path(namespace: str, key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(key))
    return CACHE_ROOT / namespace / f"{safe}.json"


def load(namespace: str, key: str) -> Optional[Any]:
    path = _path(namespace, key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Corrupt cache file %s, will refetch", path)
        return None


def save(namespace: str, key: str, payload: Any) -> None:
    path = _path(namespace, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(path)


def cached(namespace: str, key: str, fetcher: Callable[[], Any]) -> Any:
    """Return cached payload, or call fetcher and store the result."""
    global hits, misses
    hit = load(namespace, key)
    if hit is not None:
        hits += 1
        return hit
    misses += 1
    payload = fetcher()
    if payload is not None:
        save(namespace, key, payload)
    return payload

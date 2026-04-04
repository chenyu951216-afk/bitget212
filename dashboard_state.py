from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict


class TimedPayloadCache:
    def __init__(self, ttl_seconds: float = 15.0) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self._payload: Dict[str, Any] | None = None
        self._ts = 0.0
        self._lock = threading.RLock()

    def get_or_build(self, builder: Callable[[], Dict[str, Any]], force: bool = False) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            if not force and self._payload is not None and (now - self._ts) < self.ttl_seconds:
                return dict(self._payload)
            payload = dict(builder() or {})
            self._payload = payload
            self._ts = now
            return dict(payload)


state_lite_cache = TimedPayloadCache(ttl_seconds=8.0)
ai_panel_cache = TimedPayloadCache(ttl_seconds=15.0)
positions_cache = TimedPayloadCache(ttl_seconds=5.0)

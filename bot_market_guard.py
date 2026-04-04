import threading
import time
from typing import Dict, Tuple


class MarketDirectionGuard:
    def __init__(self, required_confirmations: int = 2, ttl_seconds: int = 4 * 3600):
        self.required_confirmations = max(1, int(required_confirmations))
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._state: Dict[str, object] = {
            'direction': None,
            'count': 0,
            'last_ts': 0.0,
        }
        self._lock = threading.RLock()

    def register(self, direction: str) -> Tuple[bool, int]:
        now = time.time()
        direction = str(direction or '').strip() or None
        with self._lock:
            if self._state['direction'] == direction and (now - float(self._state['last_ts'] or 0)) <= self.ttl_seconds:
                self._state['count'] = int(self._state['count'] or 0) + 1
            else:
                self._state['direction'] = direction
                self._state['count'] = 1
            self._state['last_ts'] = now
            return int(self._state['count']) >= self.required_confirmations, int(self._state['count'])

    def snapshot(self):
        with self._lock:
            return dict(self._state)

from __future__ import annotations

import queue
import threading
from typing import Callable, Any


class LearningTaskQueue:
    """Single-worker queue to serialize post-close learning.

    Keeps old logic intact while avoiding one-thread-per-close bursts.
    """

    def __init__(self, handler: Callable[[Any], None], name: str = 'learning-worker') -> None:
        self._handler = handler
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._name = name
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            worker = threading.Thread(target=self._run, daemon=True, name=self._name)
            worker.start()
            self._started = True

    def enqueue(self, item: Any) -> int:
        self.start()
        self._queue.put(item)
        return self._queue.qsize()

    def qsize(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                self._handler(item)
            finally:
                self._queue.task_done()

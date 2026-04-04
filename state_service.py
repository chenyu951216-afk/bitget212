from __future__ import annotations

import os
import threading
from copy import deepcopy
from typing import Any, Dict


def env_or_blank(name: str, fallback: str = '') -> str:
    return str(os.getenv(name, fallback) or '').strip()


def env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, '1' if default else '0') or '').strip().lower()
    return raw in ('1', 'true', 'yes', 'on')


class RuntimeState:
    def __init__(self):
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {
            'threshold': {},
            'ai_panel': {},
            'auto_backtest': {},
            'risk_status': {},
            'market_state': {},
            'session_state': {},
            'audit': {},
            'meta': {},
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def get(self, key: str, default=None):
        with self._lock:
            value = self._state.get(key, default)
            return deepcopy(value)

    def set(self, key: str, value: Any) -> Any:
        with self._lock:
            self._state[key] = deepcopy(value)
            return deepcopy(self._state[key])

    def update(self, **kwargs: Any) -> Dict[str, Any]:
        with self._lock:
            for key, value in kwargs.items():
                self._state[key] = deepcopy(value)
            return deepcopy(self._state)

    def push_audit(self, symbol: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            audit = self._state.setdefault('audit', {})
            audit[str(symbol or '')] = deepcopy(payload or {})
            return deepcopy(audit[str(symbol or '')])

    def set_threshold(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.set('threshold', payload or {})

    def set_ai_panel(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.set('ai_panel', payload or {})

    def set_auto_backtest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.set('auto_backtest', payload or {})


DEFAULT_RUNTIME_STATE = RuntimeState()


def build_learning_dataset_meta(reset_from: str = '') -> Dict[str, Any]:
    dataset_version = env_or_blank('LEARNING_DATASET_VERSION', 'v16_decision_funnel')
    learning_generation = env_or_blank('LEARNING_GENERATION', 'g2')
    activated_from = env_or_blank('LEARNING_ACTIVATED_FROM', reset_from or '')
    return {
        'dataset_version': dataset_version,
        'learning_generation': learning_generation,
        'activated_from': activated_from,
        'deactivated_at': env_or_blank('LEARNING_DEACTIVATED_AT', ''),
        'feature_schema_version': env_or_blank('FEATURE_SCHEMA_VERSION', 'features_v3'),
        'decision_model_version': env_or_blank('DECISION_MODEL_VERSION', 'decision_calibrator_v2'),
    }

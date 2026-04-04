from __future__ import annotations

from typing import Any, Dict, Iterable


def pick_keys(payload: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    src = dict(payload or {})
    return {k: src.get(k) for k in keys if k in src}


def build_state_lite_payload(base_payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        'last_update', 'scan_progress', 'equity', 'total_pnl', 'threshold_info',
        'risk_status', 'market_info', 'latest_news_title', 'learn_summary',
        'lt_info', 'fvg_orders', 'top_signals'
    ]
    return pick_keys(base_payload, keys)


def build_positions_payload(base_payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = ['active_positions', 'trailing_info', 'protection_state', 'trade_history']
    return pick_keys(base_payload, keys)


def build_ai_panel_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = ['ai_panel', 'auto_backtest', 'trend_dashboard', 'top_signals', 'learn_summary']
    return pick_keys(payload, keys)

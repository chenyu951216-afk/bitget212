from __future__ import annotations

from typing import Any, Dict

DATASET_POLICY: Dict[str, Any] = {
    'ai_min_sample_effect': 5,
    'trend_ai_semi_trades': 30,
    'trend_ai_full_trades': 50,
    'symbol_block_min_trades': 11,
    'symbol_block_min_winrate': 40.0,
    'strategy_capital_min_trades': 5,
    'strategy_block_min_trades': 11,
    'strategy_block_min_winrate': 45.0,
    'trusted_min_quality': 0.72,
    'usable_min_quality': 0.5,
    'review_min_quality': 0.32,
}

DECISION_POLICY: Dict[str, Any] = {
    'order_threshold': 60,
    'order_threshold_default': 60,
    'order_threshold_high': 80,
    'order_threshold_drop': 2,
    'order_threshold_floor': 55,
    'min_rr_hard_floor': 1.20,
    'neutral_regime_block': True,
    'decision_priority_order': ['regime', 'setup', 'risk', 'symbol', 'trigger'],
}

RISK_POLICY: Dict[str, Any] = {
    'risk_pct': 0.05,
    'atr_risk_pct': 0.01,
    'min_margin_pct': 0.01,
    'max_margin_pct': 0.08,
    'max_open_positions': 7,
    'max_same_direction': 5,
    'time_stop_bars_15m': 15,
}

EXECUTION_POLICY: Dict[str, Any] = {
    'entry_lock_sec': 300,
    'news_cache_ttl_sec': 300,
    'score_smooth_alpha': 0.35,
    'anti_chase_atr': 1.25,
    'breakout_lookback': 20,
    'pullback_buffer_atr': 0.35,
    'scale_in_min_ratio': 0.35,
    'scale_in_max_ratio': 0.45,
    'fake_breakout_penalty': 8,
}


def get_policy_snapshot() -> Dict[str, Dict[str, Any]]:
    return {
        'dataset_policy': dict(DATASET_POLICY),
        'decision_policy': dict(DECISION_POLICY),
        'risk_policy': dict(RISK_POLICY),
        'execution_policy': dict(EXECUTION_POLICY),
    }

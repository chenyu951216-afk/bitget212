from __future__ import annotations

from typing import Any, Dict


def apply_position_formula(base_margin_pct: float, signal_advantage: float, execution_quality_mult: float, market_state_discount: float, min_margin_pct: float, max_margin_pct: float) -> Dict[str, Any]:
    margin = float(base_margin_pct or 0.0)
    signal_advantage = max(0.45, min(1.6, float(signal_advantage or 1.0)))
    execution_quality_mult = max(0.4, min(1.25, float(execution_quality_mult or 1.0)))
    market_state_discount = max(0.45, min(1.1, float(market_state_discount or 1.0)))
    final_margin = margin * signal_advantage * execution_quality_mult * market_state_discount
    final_margin = max(float(min_margin_pct), min(float(max_margin_pct), final_margin))
    return {
        'margin_pct': round(final_margin, 4),
        'signal_advantage_mult': round(signal_advantage, 4),
        'execution_quality_mult': round(execution_quality_mult, 4),
        'market_state_discount': round(market_state_discount, 4),
    }

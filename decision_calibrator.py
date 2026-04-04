from __future__ import annotations

import math
from typing import Any, Dict


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _sigmoid(x: float) -> float:
    x = clamp(x, -20.0, 20.0)
    return 1.0 / (1.0 + math.exp(-x))


def calibrate_trade_decision(*,
    score: float,
    threshold: float,
    rr_ratio: float,
    entry_quality: float,
    regime_confidence: float,
    profile: Dict[str, Any] | None = None,
    execution_quality: Dict[str, Any] | None = None,
    market_consensus: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    profile = dict(profile or {})
    execution_quality = dict(execution_quality or {})
    market_consensus = dict(market_consensus or {})

    wr = safe_float(profile.get('win_rate', 0.0), 0.0)
    ev = safe_float(profile.get('ev_per_trade', 0.0), 0.0)
    local_samples = safe_float(profile.get('sample_count', 0.0), 0.0)
    profile_conf = safe_float(profile.get('confidence', 0.0), 0.0)
    exec_score = safe_float(execution_quality.get('execution_score', execution_quality.get('score', 0.7)), 0.7)
    spread_pct = abs(safe_float(execution_quality.get('spread_pct', 0.0), 0.0))
    depth_ratio = safe_float(execution_quality.get('top_depth_ratio', 0.18), 0.18)
    consensus_strength = safe_float(market_consensus.get('market_consensus_strength', 0.0), 0.0)
    consensus_bias = str(market_consensus.get('market_consensus_bias', 'mixed') or 'mixed')

    edge_score = (safe_float(score) - safe_float(threshold)) / 8.0
    rr_component = (safe_float(rr_ratio, 1.0) - 1.2) / 0.8
    eq_component = (safe_float(entry_quality, 0.0) - 2.2) / 1.4
    wr_component = (wr - 50.0) / 12.0
    ev_component = ev / 0.18
    conf_component = (safe_float(regime_confidence, 0.5) - 0.5) * 1.8 + (profile_conf - 0.35) * 1.2
    sample_component = min(local_samples / 18.0, 1.0) * 0.65
    exec_component = (exec_score - 0.55) * 1.8 - max(0.0, spread_pct - 0.18) * 2.2 + min(depth_ratio, 0.5)
    consensus_component = 0.0 if consensus_bias == 'mixed' else consensus_strength * 0.35

    raw = (
        edge_score * 1.15
        + rr_component * 0.85
        + eq_component * 0.7
        + wr_component * 0.65
        + ev_component * 0.55
        + conf_component * 0.55
        + sample_component
        + exec_component * 0.45
        + consensus_component
    )
    p_win_est = clamp(_sigmoid(raw), 0.03, 0.97)
    expected_value_est = (p_win_est * max(rr_ratio, 0.2)) - (1.0 - p_win_est)
    confidence_calibrated = clamp((p_win_est * 0.55) + (profile_conf * 0.25) + (min(local_samples, 30.0) / 30.0 * 0.1) + (max(exec_score, 0.0) * 0.1), 0.05, 0.98)

    return {
        'p_win_est': round(p_win_est, 4),
        'expected_value_est': round(expected_value_est, 4),
        'confidence_calibrated': round(confidence_calibrated, 4),
        'edge_score': round(edge_score, 4),
        'exec_score_used': round(exec_score, 4),
        'sample_component': round(sample_component, 4),
        'rr_component': round(rr_component, 4),
        'eq_component': round(eq_component, 4),
    }

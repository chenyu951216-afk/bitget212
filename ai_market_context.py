from typing import Any, Dict


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _trend_label(price: float, ma_fast: float, ma_slow: float) -> str:
    if price > ma_fast > ma_slow:
        return 'bull'
    if price < ma_fast < ma_slow:
        return 'bear'
    return 'mixed'


def build_market_consensus(btc_info: Dict[str, Any], eth_info: Dict[str, Any], market_meta: Dict[str, Any] = None) -> Dict[str, Any]:
    btc = dict(btc_info or {})
    eth = dict(eth_info or {})
    market_meta = dict(market_meta or {})
    btc_trend = _trend_label(safe_float(btc.get('price')), safe_float(btc.get('ma_fast')), safe_float(btc.get('ma_slow')))
    eth_trend = _trend_label(safe_float(eth.get('price')), safe_float(eth.get('ma_fast')), safe_float(eth.get('ma_slow')))
    vol_state = str(market_meta.get('volatility_state') or 'normal')
    breadth = safe_float(market_meta.get('breadth', 0.0), 0.0)
    if btc_trend == eth_trend and btc_trend in ('bull', 'bear'):
        bias = btc_trend
        strength = 1.0 if vol_state != 'chaos' else 0.7
    else:
        bias = 'mixed'
        strength = 0.45
    if breadth >= 0.6 and bias == 'bull':
        strength += 0.1
    elif breadth <= -0.6 and bias == 'bear':
        strength += 0.1
    return {
        'btc_trend': btc_trend,
        'eth_trend': eth_trend,
        'volatility_state': vol_state,
        'breadth': round(breadth, 4),
        'market_consensus_bias': bias,
        'market_consensus_strength': round(min(max(strength, 0.0), 1.2), 3),
    }

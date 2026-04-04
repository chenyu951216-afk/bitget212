import time
from typing import Any, Dict, List


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def exchange_quality_snapshot(exchange, symbol: str = '', side: str = '') -> Dict[str, Any]:
    snap = {
        'symbol': symbol,
        'side': side,
        'spread_pct': 0.0,
        'mark_last_deviation_pct': 0.0,
        'top_depth_ratio': 0.0,
        'api_error_streak': 0,
        'status': 'ok',
        'notes': [],
        'ts': int(time.time()),
    }
    try:
        ob = exchange.fetch_order_book(symbol, limit=10) if symbol else {'bids': [], 'asks': []}
        bids = ob.get('bids') or []
        asks = ob.get('asks') or []
        best_bid = safe_float(bids[0][0], 0.0) if bids else 0.0
        best_ask = safe_float(asks[0][0], 0.0) if asks else 0.0
        if best_bid > 0 and best_ask > 0:
            mid = (best_bid + best_ask) / 2.0
            snap['spread_pct'] = round((best_ask - best_bid) / max(mid, 1e-9) * 100.0, 6)
        bid_depth = sum(safe_float(x[1], 0.0) for x in bids[:5])
        ask_depth = sum(safe_float(x[1], 0.0) for x in asks[:5])
        depth = max(min(bid_depth, ask_depth), 0.0)
        total = max(bid_depth + ask_depth, 1e-9)
        snap['top_depth_ratio'] = round(depth / total, 6)
    except Exception as e:
        snap['notes'].append(f'order_book_error:{e}')
        snap['status'] = 'degraded'
    try:
        ticker = exchange.fetch_ticker(symbol) if symbol else {}
        mark = safe_float((ticker or {}).get('info', {}).get('markPrice', None), 0.0)
        last = safe_float((ticker or {}).get('last', None), 0.0)
        if mark > 0 and last > 0:
            snap['mark_last_deviation_pct'] = round(abs(mark - last) / max(last, 1e-9) * 100.0, 6)
    except Exception as e:
        snap['notes'].append(f'ticker_error:{e}')
        snap['status'] = 'degraded'
    return snap


def execution_gate(snapshot: Dict[str, Any], api_error_streak: int = 0) -> Dict[str, Any]:
    snap = dict(snapshot or {})
    action = 'ok'
    reasons: List[str] = []
    margin_mult = 1.0
    if safe_float(snap.get('spread_pct', 0.0)) >= 0.35:
        action = 'pause'
        reasons.append('spread過大')
    elif safe_float(snap.get('spread_pct', 0.0)) >= 0.18:
        action = 'reduce'
        margin_mult = min(margin_mult, 0.6)
        reasons.append('spread偏大')
    if safe_float(snap.get('mark_last_deviation_pct', 0.0)) >= 0.45:
        action = 'pause'
        reasons.append('mark/last偏差異常')
    elif safe_float(snap.get('mark_last_deviation_pct', 0.0)) >= 0.2:
        action = 'reduce' if action != 'pause' else action
        margin_mult = min(margin_mult, 0.7)
        reasons.append('mark/last偏差偏大')
    if safe_float(snap.get('top_depth_ratio', 0.0)) <= 0.08:
        action = 'pause'
        reasons.append('深度過薄')
    elif safe_float(snap.get('top_depth_ratio', 0.0)) <= 0.16:
        action = 'reduce' if action != 'pause' else action
        margin_mult = min(margin_mult, 0.65)
        reasons.append('深度偏薄')
    if int(api_error_streak or 0) >= 3:
        action = 'pause'
        reasons.append('API連續錯誤')
    return {
        'action': action,
        'reasons': reasons,
        'margin_mult': round(margin_mult, 4),
        'snapshot': snap,
    }


def protection_failure_action(symbol: str, verify_state: Dict[str, Any], *, missing_seconds: float = 0.0) -> Dict[str, Any]:
    state = dict(verify_state or {})
    sl_ok = bool(state.get('sl_ok'))
    tp_ok = bool(state.get('tp_ok'))
    action = 'ok'
    reason = ''
    if not sl_ok and missing_seconds >= 3.0:
        action = 'force_protect'
        reason = '止損保護單缺失'
    elif not tp_ok and missing_seconds >= 5.0:
        action = 'tighten_only'
        reason = '止盈保護單缺失'
    return {
        'symbol': symbol,
        'action': action,
        'reason': reason,
        'verify_state': state,
        'missing_seconds': round(float(missing_seconds or 0.0), 3),
    }

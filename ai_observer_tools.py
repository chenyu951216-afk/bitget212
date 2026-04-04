from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _parse_dt(value: Any):
    text = str(value or '').strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(text[:19], fmt)
        except Exception:
            continue
    return None


def sample_health_score(trade: Dict[str, Any], dedupe_weight: float = 1.0) -> Dict[str, Any]:
    t = dict(trade or {})
    score = 1.0
    reasons: List[str] = []
    source = str(t.get('source') or '')
    if 'live' not in source:
        score *= 0.65
        reasons.append('非live降權')
    bd = dict(t.get('breakdown') or {})
    if not bd:
        score *= 0.2
        reasons.append('缺breakdown')
    if abs(_safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', 0)))) > 20:
        score = 0.0
        reasons.append('異常損益')
    if abs(_safe_float(bd.get('NewsScore', 0))) >= 8 or str(bd.get('Regime', '')) == 'news':
        score *= 0.85
        reasons.append('news波動降權')
    if _safe_float(bd.get('進場品質', 0)) >= 7:
        score *= 1.05
    if dedupe_weight < 1.0:
        score *= dedupe_weight
        reasons.append('同波樣本降權')
    if not t.get('exit_time') and t.get('result') in ('win', 'loss'):
        score *= 0.7
        reasons.append('平倉資訊不完整')
    return {'score': round(max(0.0, min(score, 1.2)), 4), 'reasons': reasons}


def session_bucket(value: Any) -> str:
    dt = value if isinstance(value, datetime) else _parse_dt(value)
    if dt is None:
        return 'unknown'
    hour = dt.hour
    if 0 <= hour < 6:
        return 'asia_late'
    if 6 <= hour < 12:
        return 'asia_day'
    if 12 <= hour < 18:
        return 'europe_prep'
    if 18 <= hour < 22:
        return 'europe_us_open'
    return 'us_late'


def symbol_personality_from_rows(rows: Iterable[Dict[str, Any]], weight_map: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    weight_map = dict(weight_map or {})
    fake_break = trend = range_revert = spikes = total = 0.0
    for t in rows or []:
        w = float(weight_map.get(str((t or {}).get('id') or (t or {}).get('trade_id') or ''), 1.0) or 1.0)
        if w <= 0:
            continue
        total += w
        setup = str(((t or {}).get('breakdown') or {}).get('Setup') or (t or {}).get('setup_label') or '')
        if any(k in setup for k in ['假突破', '假跌破', '回收', '回落']):
            fake_break += w
        if any(k in setup for k in ['回踩', '續攻', '延續', '反彈續跌']):
            trend += w
        if any(k in setup for k in ['區間', '震盪', '箱體', '均值回歸']):
            range_revert += w
        if abs(_safe_float((t or {}).get('missed_move_pct', 0))) >= 2.0:
            spikes += w
    if total <= 0:
        return {'label': 'generic', 'fake_break_ratio': 0.0, 'trend_ratio': 0.0, 'range_ratio': 0.0, 'spike_ratio': 0.0}
    fp = fake_break / total
    tp = trend / total
    rp = range_revert / total
    sp = spikes / total
    if fp >= max(tp, rp) and fp >= 0.3:
        label = 'fake_break_prone'
    elif tp >= max(fp, rp) and tp >= 0.3:
        label = 'trend_friendly'
    elif rp >= max(fp, tp) and rp >= 0.3:
        label = 'range_revert_friendly'
    elif sp >= 0.25:
        label = 'spiky'
    else:
        label = 'balanced'
    return {'label': label, 'fake_break_ratio': round(fp, 4), 'trend_ratio': round(tp, 4), 'range_ratio': round(rp, 4), 'spike_ratio': round(sp, 4)}


def exchange_quality_snapshot(ticker: Optional[Dict[str, Any]], orderbook: Optional[Dict[str, Any]] = None, funding: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    reasons: List[str] = []
    score = 1.0
    bid = _safe_float((ticker or {}).get('bid', 0))
    ask = _safe_float((ticker or {}).get('ask', 0))
    last = _safe_float((ticker or {}).get('last', 0))
    mark = _safe_float((ticker or {}).get('markPrice', 0) or (ticker or {}).get('info', {}).get('markPrice', 0))
    if bid > 0 and ask > 0:
        spread_pct = (ask - bid) / max(((ask + bid) / 2.0), 1e-9) * 100.0
        if spread_pct > 0.15:
            score -= 0.25
            reasons.append('spread偏大')
    else:
        spread_pct = 0.0
    bids = list((orderbook or {}).get('bids') or [])[:5]
    asks = list((orderbook or {}).get('asks') or [])[:5]
    depth = sum(_safe_float(x[1], 0) for x in bids + asks if isinstance(x, (list, tuple)) and len(x) >= 2)
    if depth and depth < 500:
        score -= 0.15
        reasons.append('深度偏薄')
    if mark > 0 and last > 0:
        dev_pct = abs(mark - last) / max(last, 1e-9) * 100.0
        if dev_pct > 0.2:
            score -= 0.15
            reasons.append('mark/last偏離')
    else:
        dev_pct = 0.0
    fr = _safe_float((funding or {}).get('fundingRate', 0))
    if abs(fr) > 0.003:
        score -= 0.1
        reasons.append('funding異常')
    score = max(0.0, min(score, 1.0))
    if score >= 0.8:
        label = 'good'
        penalty = 0.0
        margin_mult = 1.0
    elif score >= 0.55:
        label = 'fair'
        penalty = 0.8
        margin_mult = 0.85
    else:
        label = 'poor'
        penalty = 1.8
        margin_mult = 0.7
    return {'score': round(score, 4), 'label': label, 'spread_pct': round(spread_pct, 4), 'depth5': round(depth, 4), 'mark_last_dev_pct': round(dev_pct, 4), 'penalty': penalty, 'margin_mult': margin_mult, 'reasons': reasons}


def drift_report(live_rows: List[Dict[str, Any]], backtest_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    live_count = len(live_rows or [])
    live_wins = sum(1 for x in (live_rows or []) if str((x or {}).get('result') or '') == 'win')
    live_wr = (live_wins / live_count * 100.0) if live_count else 0.0
    live_ev = sum(_safe_float((x or {}).get('learn_pnl_pct', (x or {}).get('account_pnl_pct', 0))) for x in (live_rows or [])) / max(live_count, 1)
    bt_count = len(backtest_rows or [])
    bt_wr_vals = [_safe_float(x.get('win_rate', 0)) for x in (backtest_rows or []) if x]
    bt_ev_vals = [_safe_float(x.get('avg_pnl', x.get('ev_per_trade', 0))) for x in (backtest_rows or []) if x]
    bt_wr = sum(bt_wr_vals) / max(len(bt_wr_vals), 1) if bt_wr_vals else 0.0
    bt_ev = sum(bt_ev_vals) / max(len(bt_ev_vals), 1) if bt_ev_vals else 0.0
    return {'live_count': live_count, 'live_win_rate': round(live_wr, 2), 'live_ev': round(live_ev, 4), 'backtest_count': bt_count, 'backtest_win_rate': round(bt_wr, 2), 'backtest_ev': round(bt_ev, 4), 'win_rate_gap': round(live_wr - bt_wr, 2), 'ev_gap': round(live_ev - bt_ev, 4)}


def learning_circuit_breaker(rows: List[Dict[str, Any]], weight_map: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    weight_map = dict(weight_map or {})
    total = weighted = missing = newsy = duplicates = 0.0
    seen = set()
    for t in rows or []:
        total += 1.0
        key = str((t or {}).get('id') or (t or {}).get('trade_id') or '')
        w = float(weight_map.get(key, 1.0) or 1.0)
        weighted += w
        if not (t or {}).get('breakdown'):
            missing += 1.0
        if str((((t or {}).get('breakdown') or {}).get('Regime') or '')) == 'news':
            newsy += 1.0
        sig = str((t or {}).get('symbol') or '') + '|' + str((((t or {}).get('breakdown') or {}).get('Setup') or '')) + '|' + str((t or {}).get('entry_time') or '')[:16]
        if sig in seen:
            duplicates += 1.0
        seen.add(sig)
    if total <= 0:
        return {'score': 0.0, 'status': 'ok', 'reasons': []}
    reasons = []
    score = 0.0
    if missing / total >= 0.1:
        score += 1.2
        reasons.append('缺欄位偏多')
    if newsy / total >= 0.35:
        score += 0.8
        reasons.append('極端news樣本偏多')
    if duplicates / total >= 0.12:
        score += 0.8
        reasons.append('同質樣本偏多')
    if weighted / max(total, 1.0) <= 0.55:
        score += 0.8
        reasons.append('有效權重偏低')
    status = 'ok' if score < 1.5 else 'observe' if score < 2.5 else 'halt'
    return {'score': round(score, 3), 'status': status, 'reasons': reasons, 'total_rows': int(total), 'effective_weight': round(weighted, 3)}


def tri_color_status(ai_mode: str, circuit_breaker: Dict[str, Any], anomalies: List[Dict[str, Any]], ready_ratio: float = 0.5) -> Dict[str, Any]:
    ai_mode = str(ai_mode or 'normal')
    cb_score = _safe_float((circuit_breaker or {}).get('score', 0))
    recent_bad = sum(1 for x in (anomalies or [])[-20:] if str((x or {}).get('level') or '') in ('warn', 'error'))
    if ai_mode == 'observe' or cb_score >= 2.5:
        return {'color': 'red', 'label': 'AI暫停介入', 'reason': 'observe模式或學習停損器啟動'}
    if ai_mode == 'conservative' or recent_bad >= 3 or ready_ratio < 0.5:
        return {'color': 'yellow', 'label': 'AI低權重觀察', 'reason': '保守模式/異常偏多/樣本不足'}
    return {'color': 'green', 'label': 'AI可接管', 'reason': '模式正常且學習健康'}


def neutral_failure_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        'neutral_fake_break': {'count': 0, 'wins': 0, 'avg_pnl': 0.0},
        'neutral_pullback_wait': {'count': 0, 'wins': 0, 'avg_pnl': 0.0},
        'neutral_compress': {'count': 0, 'wins': 0, 'avg_pnl': 0.0},
        'neutral_chaos': {'count': 0, 'wins': 0, 'avg_pnl': 0.0},
    }
    sums = {k: 0.0 for k in stats}
    for t in rows or []:
        bd = dict((t or {}).get('breakdown') or {})
        regime = str(bd.get('Regime') or 'neutral')
        subtype = regime if regime.startswith('neutral_') else str((t or {}).get('refined_regime') or regime)
        if subtype not in stats:
            continue
        pnl = _safe_float((t or {}).get('learn_pnl_pct', (t or {}).get('account_pnl_pct', 0)))
        stats[subtype]['count'] += 1
        if pnl > 0:
            stats[subtype]['wins'] += 1
        sums[subtype] += pnl
    for k, row in stats.items():
        c = max(row['count'], 1)
        row['win_rate'] = round(row['wins'] / c * 100.0, 2) if row['count'] else 0.0
        row['avg_pnl'] = round(sums[k] / c, 4) if row['count'] else 0.0
        if row['count'] >= 3 and row['win_rate'] < 45:
            row['failure_label'] = '易失敗'
        elif row['count'] >= 3 and row['avg_pnl'] > 0:
            row['failure_label'] = '可延續'
        else:
            row['failure_label'] = '樣本不足'
    return stats


def trigger_hit_leaderboard(rows: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    bucket: Dict[str, Dict[str, Any]] = {}
    seq = list(rows or [])[-max(limit, 1):]
    for t in seq:
        bd = dict((t or {}).get('breakdown') or {})
        trigger = str(bd.get('Trigger') or bd.get('Setup') or (t or {}).get('setup_label') or 'unknown')
        regime = str(bd.get('Regime') or 'neutral')
        pnl = _safe_float((t or {}).get('learn_pnl_pct', (t or {}).get('account_pnl_pct', 0)))
        key = f'{trigger}|{regime}'
        row = bucket.setdefault(key, {'trigger': trigger, 'regime': regime, 'count': 0, 'wins': 0, 'avg_pnl': 0.0})
        row['count'] += 1
        if pnl > 0:
            row['wins'] += 1
        row['avg_pnl'] += pnl
    out = []
    for row in bucket.values():
        c = max(row['count'], 1)
        row['win_rate'] = round(row['wins'] / c * 100.0, 2) if row['count'] else 0.0
        row['avg_pnl'] = round(row['avg_pnl'] / c, 4) if row['count'] else 0.0
        row['score'] = round(row['win_rate'] * 0.4 + row['avg_pnl'] * 30 + row['count'] * 0.5, 3)
        out.append(row)
    out.sort(key=lambda x: x['score'], reverse=True)
    return out[:10]

import math
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandas_ta as ta


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _parse_dt(text: Any):
    if not text:
        return None
    s = str(text).strip().replace('T', ' ')
    fmts = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']
    for fmt in fmts:
        try:
            return datetime.strptime(s[:len(fmt.replace('%Y','0000').replace('%m','00').replace('%d','00').replace('%H','00').replace('%M','00').replace('%S','00'))], fmt)
        except Exception:
            pass
    return None


def _normalize_setup_mode(setup: str = '') -> str:
    s = str(setup or '')
    sl = s.lower()
    if ('突破' in s) or ('爆發' in s) or ('news' in sl) or ('breakout' in sl):
        return 'breakout'
    if (
        ('區間' in s) or ('震盪' in s) or ('箱體' in s) or ('均值回歸' in s)
        or ('掃低回收' in s) or ('掃高回落' in s) or ('回補' in s)
        or ('range' in sl) or ('mean reversion' in sl)
    ):
        return 'range'
    if ('回踩' in s) or ('續攻' in s) or ('延續' in s) or ('反彈續跌' in s):
        return 'trend'
    return 'main'


def apply_decision_inertia(symbol: str, regime_info: Dict[str, Any], last_snapshot: Dict[str, Any] = None) -> Dict[str, Any]:
    info = dict(regime_info or {})
    last_snapshot = dict(last_snapshot or {})
    current_regime = str(info.get('regime') or 'neutral')
    prev_regime = str(last_snapshot.get('regime') or '')
    conf = _safe_float(info.get('confidence', 0.5), 0.5)
    note = str(info.get('note') or '')

    conf_delta = 0.0
    switched = bool(prev_regime and prev_regime != current_regime)
    if prev_regime:
        if prev_regime == current_regime:
            conf_delta += 0.10
        else:
            conf_delta -= 0.15
            prev_conf = _safe_float(last_snapshot.get('confidence', 0.5), 0.5)
            # 慣性保留：只有新 regime 優勢明顯，才真正切換
            if prev_regime != 'neutral' and conf + conf_delta < prev_conf + 0.08:
                current_regime = prev_regime
                note = f'{note}|慣性維持{prev_regime}' if note else f'慣性維持{prev_regime}'
                switched = False

    conf = max(0.18, min(0.97, conf + conf_delta))
    info['regime'] = current_regime
    info['confidence'] = round(conf, 3)
    info['decision_inertia_delta'] = round(conf_delta, 3)
    info['decision_inertia_prev'] = prev_regime or ''
    info['decision_inertia_switched'] = switched
    info['note'] = note
    return info



def detect_market_tempo(df15: pd.DataFrame) -> Dict[str, Any]:
    try:
        if df15 is None or len(df15) < 30:
            return {'tempo': 'normal', 'tempo_score': 0.0, 'atr_ratio': 0.0, 'vol_ratio': 1.0, 'candle_speed': 0.0}
        c = df15['c'].astype(float)
        h = df15['h'].astype(float)
        l = df15['l'].astype(float)
        o = df15['o'].astype(float)
        v = df15['v'].astype(float)
        curr = float(c.iloc[-1])
        atr_now = _safe_float(ta.atr(h, l, c, length=14).iloc[-1], curr * 0.004)
        atr_base = max(_safe_float(ta.atr(h, l, c, length=14).tail(24).mean(), atr_now), 1e-9)
        atr_ratio = atr_now / max(curr, 1e-9)
        vol_now = _safe_float(v.tail(3).mean(), _safe_float(v.iloc[-1], 1.0))
        vol_base = max(_safe_float(v.tail(24).mean(), vol_now), 1e-9)
        vol_ratio = vol_now / vol_base
        body_now = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
        body_base = max(_safe_float((c.tail(24) - o.tail(24)).abs().mean(), body_now), 1e-9)
        candle_speed = body_now / body_base
        tempo_score = ((atr_now / atr_base) - 1.0) * 0.9 + (vol_ratio - 1.0) * 0.7 + (candle_speed - 1.0) * 0.55
        if tempo_score >= 1.2 or (atr_ratio >= 0.012 and vol_ratio >= 1.5):
            tempo = 'fast'
        elif tempo_score <= -0.55 or (atr_ratio <= 0.005 and vol_ratio <= 0.8 and candle_speed <= 0.82):
            tempo = 'slow'
        else:
            tempo = 'normal'
        return {
            'tempo': tempo,
            'tempo_score': round(float(tempo_score), 3),
            'atr_ratio': round(float(atr_ratio), 5),
            'vol_ratio': round(float(vol_ratio), 3),
            'candle_speed': round(float(candle_speed), 3),
        }
    except Exception:
        return {'tempo': 'normal', 'tempo_score': 0.0, 'atr_ratio': 0.0, 'vol_ratio': 1.0, 'candle_speed': 0.0}



def classify_exit_type(trade: Dict[str, Any], post_profile: Dict[str, Any]) -> str:
    pnl = _safe_float(trade.get('learn_pnl_pct', trade.get('account_pnl_pct', 0)), 0.0)
    run_pct = _safe_float(post_profile.get('run_pct', trade.get('post_run_pct', 0)), 0.0)
    pullback_pct = _safe_float(post_profile.get('pullback_pct', trade.get('post_pullback_pct', 0)), 0.0)
    cont = bool(post_profile.get('continuation', trade.get('trend_continuation', False)))
    if pnl > 0:
        if cont and run_pct >= max(abs(pnl) * 0.75, 1.2) and pullback_pct <= max(run_pct * 0.55, 0.35):
            return 'too_early'
        if run_pct <= max(abs(pnl) * 0.45, 0.9):
            return 'correct_exit'
        if cont:
            return 'should_hold'
        return 'correct_exit'
    if cont and run_pct >= 1.0 and pullback_pct <= max(run_pct * 0.5, 0.35):
        return 'fake_hold'
    return 'correct_exit'



def _is_bad_learning_sample(trade: Dict[str, Any]) -> bool:
    t = dict(trade or {})
    try:
        result = str(t.get('result') or '')
        if result not in ('win', 'loss'):
            return True
        entry = _safe_float(t.get('entry_price', t.get('entry', 0)), 0.0)
        exit_p = _safe_float(t.get('exit_price', 0), 0.0)
        lev = _safe_float(t.get('leverage', 1), 1.0)
        pnl = _safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', t.get('pnl_pct', 0))), 0.0)
        rr = _safe_float(t.get('rr_ratio', (t.get('breakdown') or {}).get('RR', 0)), 0.0)
        exec_i = _safe_float(t.get('execution_integrity', 1.0), 1.0)
        label_c = _safe_float(t.get('label_confidence', 1.0), 1.0)
        edge_pct = abs(_safe_float(t.get('edge_pct', t.get('raw_pnl_pct', 0)), 0.0))
        if entry <= 0 or exit_p <= 0:
            return True
        if lev <= 0 or lev > 200:
            return True
        if abs(pnl) > 80 or edge_pct > 35:
            return True
        if rr and (rr < 0.35 or rr > 8.0):
            return True
        if exec_i < 0.18 or label_c < 0.18:
            return True
        return False
    except Exception:
        return True


def trade_learning_influence(trade: Dict[str, Any], reset_from: str = '') -> float:
    t = dict(trade or {})
    result = str(t.get('result') or '')
    if result not in ('win', 'loss') or _is_bad_learning_sample(t):
        return 0.0
    pnl = _safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', t.get('pnl_pct', 0))), 0.0)
    exit_type = str(t.get('exit_type') or '')
    regime = str((t.get('breakdown') or {}).get('Regime') or 'neutral')
    w = 1.0
    tr_dt = _parse_dt(t.get('exit_time') or t.get('created_at') or t.get('entry_time'))
    reset_dt = _parse_dt(reset_from)
    old_before_reset = bool(reset_dt and tr_dt and tr_dt < reset_dt)
    if old_before_reset:
        w *= 0.22
        if pnl > 0:
            w *= 1.65
        elif pnl < 0:
            w *= 0.45
        if abs(pnl) >= 2.5 and pnl < 0:
            w *= 0.55
    if regime == 'neutral':
        w *= 0.88
    elif regime.startswith('neutral_'):
        w *= 0.82
    if exit_type == 'correct_exit':
        w *= 1.05
    elif exit_type == 'too_early':
        w *= 0.92
    elif exit_type == 'fake_hold':
        w *= 0.72
    if bool(t.get('trend_continuation')) and pnl > 0:
        w *= 1.04
    return round(max(0.0, min(w, 1.25)), 4)



def weighted_trade_stats(rows: List[Dict[str, Any]], reset_from: str = '') -> Dict[str, Any]:
    filtered = []
    weights = []
    for row in rows or []:
        if _is_bad_learning_sample(row):
            continue
        w = trade_learning_influence(row, reset_from=reset_from)
        if w <= 0:
            continue
        filtered.append(row)
        weights.append(w)
    if not filtered:
        return {
            'count': 0, 'effective_count': 0.0, 'win_rate': 0.0, 'avg_pnl': 0.0,
            'ev_per_trade': 0.0, 'profit_factor': None, 'max_drawdown_pct': None,
            'std_pnl': None, 'weight_sum': 0.0,
        }
    pnls = [_safe_float(r.get('learn_pnl_pct', r.get('account_pnl_pct', r.get('pnl_pct', 0))), 0.0) for r in filtered]
    weight_sum = max(sum(weights), 1e-9)
    weighted_wins = sum(w for p, w in zip(pnls, weights) if p > 0)
    wr = weighted_wins / weight_sum * 100.0
    avg = sum(p * w for p, w in zip(pnls, weights)) / weight_sum
    gross_win = sum(max(p, 0.0) * w for p, w in zip(pnls, weights))
    gross_loss = sum(abs(min(p, 0.0)) * w for p, w in zip(pnls, weights))
    pf = (gross_win / max(gross_loss, 1e-9)) if gross_loss > 0 else (999.0 if gross_win > 0 else None)
    std = math.sqrt(sum(((p - avg) ** 2) * w for p, w in zip(pnls, weights)) / weight_sum)
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for p, w in zip(pnls, weights):
        step = max(0.01, 1.0 + ((p * max(w, 0.1)) / 100.0))
        equity *= step
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    return {
        'count': len(filtered),
        'effective_count': round(float(weight_sum), 3),
        'win_rate': round(float(wr), 2),
        'avg_pnl': round(float(avg), 4),
        'ev_per_trade': round(float(avg), 4),
        'profit_factor': None if pf is None else round(float(pf), 3),
        'max_drawdown_pct': round(float(min(max_dd, 100.0)), 3),
        'std_pnl': round(float(std), 4),
        'weight_sum': round(float(weight_sum), 3),
    }



def recent_setup_loss_streak(rows: List[Dict[str, Any]], symbol: str = '', regime: str = 'neutral', setup: str = '') -> Dict[str, Any]:
    symbol = str(symbol or '')
    regime = str(regime or 'neutral')
    mode = _normalize_setup_mode(setup)
    streak = 0
    matches = 0
    for t in reversed(list(rows or [])):
        bd = dict(t.get('breakdown') or {})
        t_regime = str(bd.get('Regime') or 'neutral')
        t_symbol = str(t.get('symbol') or '')
        t_mode = _normalize_setup_mode(t.get('setup_label') or bd.get('Setup') or t.get('setup') or '')
        if t_symbol != symbol or t_regime != regime or t_mode != mode:
            continue
        if str(t.get('result') or '') not in ('win', 'loss'):
            continue
        matches += 1
        if t.get('result') == 'loss':
            streak += 1
        else:
            break
        if matches >= 8:
            break
    suppress_mult = 0.5 if streak >= 3 else 1.0
    return {'loss_streak': streak, 'sample_checked': matches, 'suppress_mult': suppress_mult}



def confidence_position_multiplier(confidence: float, tempo: str = 'normal') -> Tuple[float, str]:
    conf = _safe_float(confidence, 0.0)
    mult = 1.0
    note = '一般倉位'
    if conf > 0.7:
        mult *= 1.5
        note = '高信心放大倉位'
    elif conf < 0.4:
        mult *= 0.5
        note = '低信心縮小倉位'
    elif conf >= 0.58:
        mult *= 1.18
        note = '中高信心加碼'
    if tempo == 'fast':
        mult *= 1.08
        note += '+快節奏'
    elif tempo == 'slow':
        mult *= 0.92
        note += '+慢節奏'
    return round(max(0.45, min(mult, 1.75)), 4), note



def apply_exit_learning_to_params(param_sets: Dict[str, Dict[str, Any]], recent_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    updated = {k: dict(v or {}) for k, v in (param_sets or {}).items()}
    bucket: Dict[str, Dict[str, int]] = {}
    for t in recent_rows or []:
        regime = str((t.get('breakdown') or {}).get('Regime') or 'neutral')
        exit_type = str(t.get('exit_type') or '')
        if exit_type not in ('too_early', 'correct_exit', 'should_hold', 'fake_hold'):
            continue
        rec = bucket.setdefault(regime, {'too_early': 0, 'correct_exit': 0, 'should_hold': 0, 'fake_hold': 0})
        rec[exit_type] += 1
    for regime, stat in bucket.items():
        p = updated.setdefault(regime, dict(updated.get('neutral') or {}))
        too_early = int(stat.get('too_early', 0) or 0)
        should_hold = int(stat.get('should_hold', 0) or 0)
        fake_hold = int(stat.get('fake_hold', 0) or 0)
        correct_exit = int(stat.get('correct_exit', 0) or 0)
        if too_early >= 3 or should_hold >= 4:
            p['trail_pct'] = round(min(_safe_float(p.get('trail_pct', 0.035), 0.035) * 1.08, 0.08), 4)
            p['tp_mult'] = round(min(_safe_float(p.get('tp_mult', 3.0), 3.0) * 1.04, 5.6), 2)
        if fake_hold >= 3:
            p['trail_pct'] = round(max(_safe_float(p.get('trail_pct', 0.035), 0.035) * 0.94, 0.018), 4)
            p['sl_mult'] = round(min(_safe_float(p.get('sl_mult', 2.0), 2.0) * 1.03, 3.2), 2)
        if correct_exit >= 4 and too_early == 0 and fake_hold == 0:
            p['trail_trigger_atr'] = round(max(_safe_float(p.get('trail_trigger_atr', 1.5), 1.5) * 0.99, 0.9), 2)
    return updated

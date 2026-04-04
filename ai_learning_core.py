import math
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def coarse_setup_mode(setup: str = '') -> str:
    s = str(setup or '')
    sl = s.lower()
    if any(k in s for k in ['假突破回收', '假跌破回收']):
        return 'range'
    if ('突破' in s) or ('爆發' in s) or ('news' in sl) or ('breakout' in sl):
        return 'breakout'
    if any(k in s for k in ['區間', '震盪', '箱體', '均值回歸', '掃低回收', '掃高回落', '回補']):
        return 'range'
    if any(k in s for k in ['回踩', '續攻', '延續', '反彈續跌']):
        return 'trend'
    return 'main'


def canonical_setup_key(setup: str = '', side: str = '', regime: str = '') -> str:
    s = str(setup or '')
    sl = s.lower()
    side = str(side or '').lower()
    regime = str(regime or '')
    is_long = side in ('buy', 'long', '多')
    is_short = side in ('sell', 'short', '空')
    if not (is_long or is_short):
        if '做空' in s or '空' in s:
            is_short = True
        elif '做多' in s or '多' in s:
            is_long = True
    if 'news' in sl or ('消息' in s and '突破' in s):
        return 'news_breakout_follow'
    if any(k in s for k in ['假突破回收', '掃低回收']) or 'reclaim' in sl:
        return 'fake_break_reclaim_long'
    if any(k in s for k in ['假跌破回收', '掃高回落']) or 'reject' in sl:
        return 'fake_break_reject_short'
    if coarse_setup_mode(s) == 'range':
        return 'range_revert_long' if is_long else 'range_revert_short'
    if coarse_setup_mode(s) == 'breakout':
        if regime == 'news':
            return 'news_breakout_follow'
        return 'breakout_follow_long' if is_long else 'breakout_follow_short'
    if coarse_setup_mode(s) == 'trend':
        return 'trend_pullback_long' if is_long else 'trend_pullback_short'
    return 'main_long' if is_long else 'main_short' if is_short else 'main'


def classify_neutral_subtype(regime_info: Dict[str, Any], breakdown: Dict[str, Any] = None) -> Tuple[str, str]:
    regime_info = dict(regime_info or {})
    breakdown = dict(breakdown or {})
    adx = safe_float(regime_info.get('adx', 0))
    atr_ratio = abs(safe_float(regime_info.get('atr_ratio', 0)))
    bb_width = abs(safe_float(regime_info.get('bb_width', 0)))
    vol_ratio = safe_float(regime_info.get('vol_ratio', 1))
    move = abs(safe_float(regime_info.get('move_3bars_pct', 0)))
    direction = str(regime_info.get('direction', '中性') or '中性')
    chase_risk = abs(safe_float(breakdown.get('追價風險', 0)))
    regime_bias = abs(safe_float(breakdown.get('RegimeBias', 0)))
    if adx <= 15 and bb_width <= 0.012 and atr_ratio <= 0.0055 and vol_ratio <= 0.95:
        return 'neutral_compress', '壓縮盤: 低ADX/低波動/量縮'
    if move >= 1.0 and vol_ratio >= 1.15 and chase_risk >= 4:
        return 'neutral_fake_break', '假突破盤: 短線有位移但追價風險偏高'
    if direction in ('多', '空') and regime_bias >= 2 and move <= 1.4 and 0.75 <= vol_ratio <= 1.5:
        return 'neutral_pullback_wait', '回踩待續盤: 結構偏方向但尚待確認'
    return 'neutral_chaos', '混沌雜訊盤: 特徵混合且不夠乾淨'


def normalize_regime_key(regime: str, regime_info: Dict[str, Any] = None, breakdown: Dict[str, Any] = None) -> Tuple[str, str]:
    regime = str(regime or 'neutral')
    if regime != 'neutral':
        return regime, str((regime_info or {}).get('note') or '')
    return classify_neutral_subtype(regime_info or {}, breakdown or {})


def parse_dt(value: Any):
    text = str(value or '').strip()
    if not text:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(text[:19], fmt)
        except Exception:
            continue
    return None


def learning_sample_weight(trade: Dict[str, Any], reset_from: str = '') -> Tuple[float, List[str]]:
    t = dict(trade or {})
    reasons: List[str] = []
    if str(t.get('result') or '') not in ('win', 'loss'):
        return 0.0, ['未平倉或結果無效']
    if not str(t.get('symbol') or ''):
        return 0.0, ['缺symbol']
    bd = dict(t.get('breakdown') or {})
    regime = str(bd.get('Regime', 'neutral') or 'neutral')
    eq = safe_float(bd.get('進場品質', 0))
    rr = safe_float(bd.get('RR', 0))
    learn_pnl = safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', 0)))
    if abs(learn_pnl) > 20:
        return 0.0, ['異常損益樣本']
    if not bd:
        return 0.0, ['缺breakdown']
    weight = 1.0
    entry_time = parse_dt(t.get('entry_time'))
    reset_dt = parse_dt(reset_from)
    if reset_dt is not None and entry_time is not None and entry_time < reset_dt:
        weight *= 0.35
        reasons.append('舊資料降權')
        if abs(learn_pnl) >= 0.05 and eq >= 5:
            weight = max(weight, 0.6)
            reasons.append('舊資料但品質可留')
    if regime == 'neutral_chaos':
        weight *= 0.15
        reasons.append('neutral_chaos強降權')
    elif regime == 'neutral':
        weight *= 0.45
        reasons.append('neutral舊桶降權')
    elif regime == 'neutral_compress':
        weight *= 0.5
        reasons.append('壓縮盤中性降權')
    elif regime == 'neutral_pullback_wait':
        weight *= 0.75
    if eq < 3:
        weight *= 0.55
        reasons.append('進場品質低')
    elif eq >= 7:
        weight *= 1.05
    if rr < 1.15:
        weight *= 0.75
        reasons.append('RR偏弱')
    if abs(learn_pnl) < 0.0001 and str(t.get('result')) == 'loss':
        weight *= 0.25
        reasons.append('零損失樣本弱化')
    return round(clamp(weight, 0.0, 1.2), 4), reasons


def trade_signature(trade: Dict[str, Any]) -> str:
    t = dict(trade or {})
    bd = dict(t.get('breakdown') or {})
    regime = str(bd.get('Regime') or 'neutral')
    setup = canonical_setup_key(t.get('setup_label') or bd.get('Setup') or '', t.get('side') or '', regime)
    entry = safe_float(t.get('entry_price', 0))
    score = safe_float(t.get('entry_score', 0))
    return '|'.join([
        str(t.get('symbol') or ''),
        regime,
        setup,
        str(t.get('side') or ''),
        f'{entry:.6f}',
        f'{score:.1f}',
    ])


def dedupe_learning_samples(trades: List[Dict[str, Any]], minutes: int = 10) -> Dict[str, float]:
    out: Dict[str, float] = {}
    seen: Dict[str, datetime] = {}
    window_sec = max(int(minutes), 1) * 60
    for t in sorted([dict(x or {}) for x in (trades or [])], key=lambda x: str(x.get('entry_time') or '')):
        sig = trade_signature(t)
        dt = parse_dt(t.get('entry_time'))
        key = str(t.get('id') or t.get('trade_id') or sig)
        weight = 1.0
        if dt is not None and sig in seen:
            gap = abs((dt - seen[sig]).total_seconds())
            if gap <= window_sec:
                weight = 0.2
        if dt is not None:
            seen[sig] = dt
        out[key] = weight
    return out


def weighted_stats(rows: List[Dict[str, Any]], weight_map: Dict[str, float] = None) -> Dict[str, Any]:
    weight_map = dict(weight_map or {})
    weighted_rows = []
    for t in rows or []:
        key = str(t.get('id') or t.get('trade_id') or trade_signature(t))
        w = safe_float(weight_map.get(key, 1.0), 1.0)
        if w <= 0:
            continue
        weighted_rows.append((dict(t or {}), w))
    if not weighted_rows:
        return {'count': 0, 'weight_sum': 0.0, 'win_rate': 0.0, 'avg_pnl': 0.0, 'ev_per_trade': 0.0, 'profit_factor': None, 'max_drawdown_pct': None, 'std_pnl': None, 'confidence': 0.0}
    pnls = []
    wins = 0.0
    gross_win = 0.0
    gross_loss = 0.0
    weight_sum = 0.0
    for t, w in weighted_rows:
        pnl = safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', t.get('edge_pct', t.get('pnl_pct', 0)))))
        pnls.append((pnl, w))
        weight_sum += w
        if pnl > 0:
            wins += w
            gross_win += pnl * w
        elif pnl < 0:
            gross_loss += abs(pnl) * w
    avg = sum(p * w for p, w in pnls) / max(weight_sum, 1e-9)
    wr = wins / max(weight_sum, 1e-9) * 100.0
    pf = (gross_win / max(gross_loss, 1e-9)) if gross_loss > 0 else (999.0 if gross_win > 0 else None)
    # weighted std
    var = sum(((p - avg) ** 2) * w for p, w in pnls) / max(weight_sum, 1e-9)
    std = math.sqrt(max(var, 0.0))
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for pnl, w in pnls:
        step = max(0.01, 1.0 + (pnl / 100.0) * max(min(w, 1.0), 0.2))
        equity *= step
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / max(peak, 1e-9) * 100.0)
    confidence = min(weight_sum / 50.0, 1.0) * max(0.0, 1.0 - min(std / 3.0, 1.0))
    return {
        'count': len(weighted_rows),
        'weight_sum': round(weight_sum, 3),
        'win_rate': round(wr, 2),
        'avg_pnl': round(avg, 4),
        'ev_per_trade': round(avg, 4),
        'profit_factor': None if pf is None else round(float(pf), 3),
        'max_drawdown_pct': round(float(min(max_dd, 100.0)), 3),
        'std_pnl': round(float(std), 4),
        'confidence': round(float(confidence), 3),
    }


def exit_reason_type(trade: Dict[str, Any]) -> str:
    t = dict(trade or {})
    run_pct = safe_float(t.get('post_run_pct', 0))
    pullback_pct = safe_float(t.get('post_pullback_pct', 0))
    missed_pct = safe_float(t.get('missed_move_pct', 0))
    trend_cont = bool(t.get('trend_continuation'))
    if trend_cont and pullback_pct <= max(run_pct * 0.35, 0.4):
        return 'hold_too_short'
    if missed_pct >= 1.8 and pullback_pct <= 0.45:
        return 'breakeven_too_early'
    if missed_pct >= 1.2 and pullback_pct <= 0.9:
        return 'trailing_too_tight'
    if run_pct < 0.8:
        return 'normal_noise'
    return 'normal_exit'


def score_priority(stats: Dict[str, Any]) -> Tuple[float, float, float, float]:
    count = safe_float(stats.get('weight_sum', stats.get('count', 0)))
    ev = safe_float(stats.get('ev_per_trade', 0))
    dd = safe_float(stats.get('max_drawdown_pct', 0))
    pf = safe_float(stats.get('profit_factor', 0))
    wr = safe_float(stats.get('win_rate', 0))
    conf = safe_float(stats.get('confidence', 0))
    score = ev * 35.0 - dd * 0.35 + min(pf, 3.0) * 12.0 + ((wr - 50.0) * 0.6) + min(count, 30) * 0.5 + conf * 8.0
    return round(score, 3), round(count, 3), round(pf, 3), round(wr, 2)


def ai_arbiter(base_score: float, base_threshold: float, rr: float, margin_pct: float, profile: Dict[str, Any], symbol_profile: Dict[str, Any], *, cold_start: bool = False) -> Dict[str, Any]:
    reasons: List[str] = []
    score_delta = 0.0
    threshold_delta = 0.0
    rr_mult = 1.0
    margin_mult = 1.0
    ev = safe_float(profile.get('ev_per_trade', 0))
    dd = safe_float(profile.get('max_drawdown_pct', 0))
    wr = safe_float(profile.get('win_rate', 0))
    conf = safe_float(profile.get('confidence', 0))
    source = str(profile.get('source') or 'live_only')
    source_weight = 1.0 if source.endswith('local') or source == 'local' else 0.7 if source.endswith('mid') or source == 'mid' else 0.4
    score_delta += ev * 20.0 * source_weight
    score_delta += (wr - 50.0) * 0.04 * source_weight
    if dd >= 12:
        threshold_delta += 2.0
        margin_mult *= 0.9
        reasons.append('DD偏高保守')
    if ev <= 0:
        threshold_delta += 2.5
        rr_mult *= 0.97
        margin_mult *= 0.88
        reasons.append('EV未轉正')
    elif ev >= 0.08:
        threshold_delta -= 1.2
        rr_mult *= 1.03
        reasons.append('EV正向')
    if wr < 45:
        threshold_delta += 1.5
        margin_mult *= 0.9
    elif wr >= 58:
        threshold_delta -= 0.8
    if cold_start or safe_float(profile.get('sample_count', 0)) < 5:
        rr_mult = min(rr_mult, 1.0)
        margin_mult = min(margin_mult, 0.8)
        threshold_delta = max(threshold_delta, 0.0)
        reasons.append('冷啟動保護')
    sym_wr = safe_float(symbol_profile.get('win_rate', 0))
    sym_ev = safe_float(symbol_profile.get('avg_pnl', 0))
    if symbol_profile.get('count', 0) >= 5:
        if sym_wr < 42 or sym_ev < 0:
            threshold_delta += 1.0
            margin_mult *= 0.92
            reasons.append('幣種偏弱')
        elif sym_wr >= 58 and sym_ev > 0:
            score_delta += 0.8
            reasons.append('幣種偏強')
    if conf < 0.45:
        margin_mult *= 0.85
        rr_mult = min(rr_mult, 1.0)
        reasons.append('信心不足降風險')
    return {
        'effective_score': round(base_score + score_delta, 4),
        'effective_threshold': round(base_threshold + threshold_delta, 4),
        'rr_target': round(max(rr * rr_mult, 1.0), 4),
        'margin_pct': round(max(margin_pct * margin_mult, 0.0001), 6),
        'score_delta': round(score_delta, 4),
        'threshold_delta': round(threshold_delta, 4),
        'rr_mult': round(rr_mult, 4),
        'margin_mult': round(margin_mult, 4),
        'reasons': reasons,
    }


def counterfactual_outcome(side: str, entry_price: float, candles: List[float]) -> Dict[str, Any]:
    side = str(side or '').lower()
    entry_price = safe_float(entry_price, 0)
    vals = [safe_float(x, 0) for x in (candles or []) if x is not None]
    if entry_price <= 0 or not vals:
        return {'future_move_pct': 0.0, 'missed_good_trade': False, 'avoided_loss': False}
    max_p = max(vals)
    min_p = min(vals)
    if side in ('buy', 'long'):
        future_move = (max_p - entry_price) / max(entry_price, 1e-9) * 100.0
        adverse = (entry_price - min_p) / max(entry_price, 1e-9) * 100.0
    else:
        future_move = (entry_price - min_p) / max(entry_price, 1e-9) * 100.0
        adverse = (max_p - entry_price) / max(entry_price, 1e-9) * 100.0
    return {
        'future_move_pct': round(future_move, 4),
        'adverse_move_pct': round(adverse, 4),
        'missed_good_trade': bool(future_move >= 1.0 and adverse <= 0.8),
        'avoided_loss': bool(adverse >= 1.0 and future_move <= 0.5),
    }

from datetime import datetime
from typing import Any, Dict, List
from ai_learning_core import learning_sample_weight, dedupe_learning_samples
from ai_observer_tools import sample_health_score


def _parse_dt(value: Any):
    s = str(value or '').strip()
    if not s:
        return None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(s[:19], fmt)
        except Exception:
            pass
    return None


def time_decay_weight(trade: Dict[str, Any], live_rank: int = 9999) -> float:
    t = dict(trade or {})
    src = str(t.get('source') or '')
    if 'live' in src and live_rank < 30:
        return 1.0
    if 'live' in src and live_rank < 100:
        return 0.72
    dt = _parse_dt(t.get('entry_time') or t.get('exit_time'))
    if dt is None:
        return 0.25 if 'live' not in src else 0.45
    age_hours = max((datetime.utcnow() - dt).total_seconds() / 3600.0, 0.0)
    if 'live' in src:
        if age_hours <= 72:
            return 0.85
        if age_hours <= 24 * 14:
            return 0.62
        if age_hours <= 24 * 45:
            return 0.45
        return 0.28
    # 舊回測更低
    if age_hours <= 24 * 14:
        return 0.22
    return 0.12


def suspicious_sample_review(trade: Dict[str, Any], dedupe_weight: float = 1.0) -> Dict[str, Any]:
    h = sample_health_score(trade, dedupe_weight=dedupe_weight)
    t = dict(trade or {})
    bd = dict(t.get('breakdown') or {})
    reasons = []
    gates = 0
    if h.get('score', 1.0) <= 0.18:
        gates += 1
        reasons.append('健康分數極低')
    if not bd or abs(float(t.get('learn_pnl_pct', t.get('account_pnl_pct', 0)) or 0)) > 20:
        gates += 1
        reasons.append('缺結構或損益異常')
    if dedupe_weight <= 0.2:
        gates += 1
        reasons.append('重複噪音樣本')
    should_quarantine = gates >= 2  # 雙重確認
    return {
        'should_quarantine': should_quarantine,
        'double_confirmed': should_quarantine,
        'reasons': reasons + list(h.get('reasons') or []),
        'health_score': h.get('score', 0.0),
        'gate_count': gates,
    }


def build_learning_weights(trades: List[Dict[str, Any]], reset_from: str = '') -> Dict[str, float]:
    weights = {}
    dedupe = dedupe_learning_samples(trades, minutes=10)
    live_rows = [dict(t or {}) for t in (trades or []) if 'live' in str((t or {}).get('source') or '')]
    live_rows.sort(key=lambda x: str(x.get('entry_time') or x.get('exit_time') or ''), reverse=True)
    live_rank_map = {str((t or {}).get('id') or (t or {}).get('trade_id') or ''): idx for idx, t in enumerate(live_rows)}
    for t in trades or []:
        key = str((t or {}).get('id') or (t or {}).get('trade_id') or '')
        if not key:
            continue
        base_weight, _ = learning_sample_weight(t, reset_from=reset_from)
        ddw = float(dedupe.get(key, 1.0) or 1.0)
        health = sample_health_score(t, dedupe_weight=ddw)
        decay = time_decay_weight(t, live_rank=live_rank_map.get(key, 9999))
        review = suspicious_sample_review(t, dedupe_weight=ddw)
        final_weight = base_weight * float(health.get('score', 1.0) or 1.0) * decay
        if review.get('should_quarantine'):
            final_weight *= 0.02
        weights[key] = round(final_weight, 4)
    return weights


def learning_weight_summary(trades: List[Dict[str, Any]], reset_from: str = '') -> Dict[str, Any]:
    wm = build_learning_weights(trades, reset_from=reset_from)
    vals = list(wm.values())
    return {
        'count': len(vals),
        'avg_weight': round(sum(vals) / max(len(vals), 1), 4) if vals else 0.0,
        'max_weight': round(max(vals), 4) if vals else 0.0,
        'min_weight': round(min(vals), 4) if vals else 0.0,
        'recent_live_high_weight': sum(1 for v in vals if v >= 0.9),
        'quarantined_like': sum(1 for v in vals if v <= 0.03),
    }

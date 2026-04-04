from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

from ai_learning_core import canonical_setup_key, learning_sample_weight, coarse_setup_mode
from ai_observer_tools import sample_health_score
from state_service import build_learning_dataset_meta


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


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


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v or '').strip().lower() in ('1', 'true', 'yes', 'y', 'ok')


def feature_completeness(trade: Dict[str, Any]) -> float:
    t = dict(trade or {})
    bd = dict(t.get('breakdown') or {})
    checks = [
        bool(bd.get('Regime') or t.get('regime')),
        bool(t.get('setup_label') or bd.get('Setup')),
        safe_float(bd.get('RR', t.get('rr_ratio', 0.0))) > 0,
        safe_float(bd.get('進場品質', bd.get('EntryGate', t.get('entry_quality', 0.0)))) > 0,
        bool(bd.get('MarketTempo') or t.get('market_tempo')),
        bool(t.get('session_bucket') or t.get('session') or _session_bucket(t) != 'unknown'),
        bool(t.get('execution_snapshot') or t.get('execution_quality')),
    ]
    score = sum(1 for v in checks if v) / max(len(checks), 1)
    return round(float(score), 4)


def label_confidence(trade: Dict[str, Any]) -> float:
    t = dict(trade or {})
    result = str(t.get('result') or '')
    exit_type = str(t.get('exit_type') or '').lower()
    close_reason = str(t.get('close_reason') or t.get('reason') or '').lower()
    score = 0.72 if result in ('win', 'loss') else 0.18
    if 'manual' in close_reason or _truthy(t.get('manual_close')):
        score -= 0.28
    if 'api' in close_reason or _truthy(t.get('api_recover_fill')):
        score -= 0.2
    if exit_type in ('correct_exit', 'should_hold'):
        score += 0.08
    elif exit_type in ('fake_hold', 'abnormal', 'unknown'):
        score -= 0.15
    if abs(safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', 0.0)))) >= 12:
        score -= 0.12
    return round(max(0.0, min(1.0, score)), 4)


def execution_integrity(trade: Dict[str, Any]) -> float:
    t = dict(trade or {})
    snap = dict(t.get('execution_snapshot') or t.get('execution_quality') or {})
    score = 0.84
    spread = abs(safe_float(snap.get('spread_pct', 0.0)))
    deviation = abs(safe_float(snap.get('mark_last_deviation_pct', 0.0)))
    depth = safe_float(snap.get('top_depth_ratio', 0.18), 0.18)
    if spread >= 0.35:
        score -= 0.25
    elif spread >= 0.18:
        score -= 0.12
    if deviation >= 0.45:
        score -= 0.2
    elif deviation >= 0.2:
        score -= 0.08
    if depth <= 0.08:
        score -= 0.22
    elif depth <= 0.16:
        score -= 0.1
    if list(snap.get('notes') or []):
        score -= 0.08
    if _truthy(t.get('protection_order_failed')):
        score -= 0.28
    if _truthy(t.get('api_recover_fill')):
        score -= 0.16
    return round(max(0.0, min(1.0, score)), 4)


def exit_integrity(trade: Dict[str, Any]) -> float:
    t = dict(trade or {})
    close_reason = str(t.get('close_reason') or t.get('reason') or '').lower()
    exit_type = str(t.get('exit_type') or '').lower()
    score = 0.8
    if 'manual' in close_reason or _truthy(t.get('manual_close')):
        score -= 0.28
    if 'panic' in close_reason:
        score -= 0.18
    if _truthy(t.get('protection_order_failed')):
        score -= 0.18
    if exit_type in ('correct_exit', 'should_hold'):
        score += 0.06
    elif exit_type in ('fake_hold', 'abnormal', 'unknown'):
        score -= 0.16
    return round(max(0.0, min(1.0, score)), 4)


def batch_quality_penalty(trade: Dict[str, Any]) -> Dict[str, Any]:
    t = dict(trade or {})
    penalty = 0.0
    reasons: List[str] = []
    execution_i = execution_integrity(t)
    exit_i = exit_integrity(t)
    label_i = label_confidence(t)
    if execution_i <= 0.42:
        penalty += 0.22
        reasons.append('執行完整性偏低')
    if exit_i <= 0.42:
        penalty += 0.16
        reasons.append('出場完整性偏低')
    if label_i <= 0.4:
        penalty += 0.16
        reasons.append('標籤可信度偏低')
    if _truthy(t.get('protection_order_failed')) and _truthy(t.get('api_recover_fill')):
        penalty += 0.2
        reasons.append('保護單/API 雙異常')
    return {'penalty': round(min(penalty, 0.55), 4), 'reasons': reasons}


def execution_quality_bucket(snapshot: Dict[str, Any] | None) -> str:
    snap = dict(snapshot or {})
    spread = abs(safe_float(snap.get('spread_pct', 0.0)))
    deviation = abs(safe_float(snap.get('mark_last_deviation_pct', 0.0)))
    depth_ratio = safe_float(snap.get('top_depth_ratio', 0.0))
    notes = list(snap.get('notes') or [])
    if spread >= 0.35 or deviation >= 0.45 or depth_ratio <= 0.08:
        return 'bad'
    if spread >= 0.18 or deviation >= 0.2 or depth_ratio <= 0.16 or notes:
        return 'mid'
    return 'good'


def _bucketize_distance(value: float) -> str:
    av = abs(safe_float(value, 0.0))
    if av >= 1.4:
        return 'far'
    if av >= 0.75:
        return 'mid'
    return 'near'


def _session_bucket(trade: Dict[str, Any]) -> str:
    raw = str((trade or {}).get('session_bucket') or (trade or {}).get('session') or '').strip()
    if raw:
        return raw
    dt = _parse_dt((trade or {}).get('entry_time') or (trade or {}).get('exit_time'))
    if not dt:
        return 'unknown'
    hour = dt.hour
    if 0 <= hour < 8:
        return 'asia'
    if 8 <= hour < 16:
        return 'eu'
    return 'us'


def infer_entry_zone_type(trade: Dict[str, Any]) -> str:
    t = dict(trade or {})
    bd = dict(t.get('breakdown') or {})
    setup_label = str(t.get('setup_label') or bd.get('Setup') or '')
    sl = setup_label.lower()
    chase = safe_float(bd.get('追價風險', bd.get('ChaseRisk', 0.0)))
    if '假突破回收' in setup_label or '掃低回收' in setup_label:
        return 'liquidity_sweep_reclaim'
    if '假跌破回收' in setup_label or '掃高回落' in setup_label:
        return 'liquidity_sweep_reclaim'
    if '區間' in setup_label or '震盪' in setup_label or '均值回歸' in setup_label:
        return 'range_revert'
    if '回踩' in setup_label or 'pullback' in sl or '延續' in setup_label:
        return 'pullback'
    if '突破' in setup_label or 'breakout' in sl or '爆發' in setup_label:
        return 'breakout_late' if chase >= 4 else 'breakout_early'
    mode = coarse_setup_mode(setup_label)
    if mode == 'range':
        return 'range_revert'
    if mode == 'trend':
        return 'pullback'
    return 'breakout_late' if chase >= 5 else 'pullback'


def build_decision_fingerprint(trade: Dict[str, Any]) -> str:
    t = dict(trade or {})
    bd = dict(t.get('breakdown') or {})
    regime = str(bd.get('Regime') or t.get('regime') or 'neutral')
    side = str(t.get('side') or '').lower()
    setup_key = canonical_setup_key(t.get('setup_label') or bd.get('Setup') or '', side, regime)
    eq_bucket = 'hq' if safe_float(bd.get('進場品質', bd.get('EntryGate', 0))) >= 7 else 'mq' if safe_float(bd.get('進場品質', bd.get('EntryGate', 0))) >= 4 else 'lq'
    rr_bucket = 'rr3' if safe_float(bd.get('RR', t.get('rr_ratio', 0))) >= 2.5 else 'rr2' if safe_float(bd.get('RR', t.get('rr_ratio', 0))) >= 1.5 else 'rr1'
    vwap_bucket = _bucketize_distance(bd.get('VWAPDistanceATR', bd.get('distance_from_vwap', 0.0)))
    ema_bucket = _bucketize_distance(bd.get('EMA20DistanceATR', bd.get('distance_from_ema20', 0.0)))
    sr_bucket = _bucketize_distance(bd.get('SRDistanceATR', bd.get('distance_from_4h_sr', 0.0)))
    chase_bucket = 'high' if safe_float(bd.get('追價風險', bd.get('ChaseRisk', 0.0))) >= 6 else 'mid' if safe_float(bd.get('追價風險', bd.get('ChaseRisk', 0.0))) >= 3 else 'low'
    vol_ratio = safe_float(bd.get('VolRatio', bd.get('volume_ratio', 1.0)), 1.0)
    vol_bucket = 'high' if vol_ratio >= 1.8 else 'mid' if vol_ratio >= 1.1 else 'low'
    exec_bucket = execution_quality_bucket(t.get('execution_snapshot') or t.get('execution_quality'))
    entry_zone = infer_entry_zone_type(t)
    session_bucket = _session_bucket(t)
    return '|'.join([
        regime,
        setup_key,
        entry_zone,
        rr_bucket,
        eq_bucket,
        f'vwap:{vwap_bucket}',
        f'ema20:{ema_bucket}',
        f'sr4h:{sr_bucket}',
        f'chase:{chase_bucket}',
        f'vol:{vol_bucket}',
        f'session:{session_bucket}',
        f'exec:{exec_bucket}',
    ])


def classify_learning_bucket(trade: Dict[str, Any], reset_from: str = '') -> Dict[str, Any]:
    t = dict(trade or {})
    source = str(t.get('source') or '').lower()
    is_live = source.startswith('live')
    weight, weight_reasons = learning_sample_weight(t, reset_from=reset_from)
    health = sample_health_score(t, dedupe_weight=1.0)
    health_score = safe_float(health.get('score', 0.0), 0.0)
    fingerprint = build_decision_fingerprint(t)
    eq_bucket = execution_quality_bucket(t.get('execution_snapshot') or t.get('execution_quality'))
    entry_dt = _parse_dt(t.get('entry_time') or t.get('exit_time'))
    reset_dt = _parse_dt(reset_from)
    old_live = bool(is_live and entry_dt and reset_dt and entry_dt < reset_dt)
    dataset_meta = build_learning_dataset_meta(reset_from=reset_from)
    feat_score = feature_completeness(t)
    label_i = label_confidence(t)
    exec_i = execution_integrity(t)
    exit_i = exit_integrity(t)
    batch_q = batch_quality_penalty(t)
    reasons: List[str] = []
    quarantine = False

    if not is_live:
        bucket = 'other'
        reasons.append('非實單資料')
    else:
        result = str(t.get('result') or '')
        bd = dict(t.get('breakdown') or {})
        if result not in ('open', 'win', 'loss'):
            quarantine = True
            reasons.append('結果欄位異常')
        if result in ('win', 'loss') and not bd:
            quarantine = True
            reasons.append('缺少breakdown')
        if abs(safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', 0.0)))) > 20:
            quarantine = True
            reasons.append('損益異常過大')
        if health_score <= 0.18:
            quarantine = True
            reasons.append('健康分數過低')
        if feat_score <= 0.42:
            quarantine = True
            reasons.append('特徵完整度不足')
        elif feat_score <= 0.7:
            reasons.append('特徵完整度偏低')
        if label_i <= 0.22:
            quarantine = True
            reasons.append('標籤可信度過低')
        elif label_i <= 0.48:
            reasons.append('標籤可信度偏低')
        if exec_i <= 0.22:
            quarantine = True
            reasons.append('執行完整性過低')
        elif exec_i <= 0.48:
            reasons.append('執行完整性偏低')
        if exit_i <= 0.22:
            quarantine = True
            reasons.append('出場完整性過低')
        elif exit_i <= 0.48:
            reasons.append('出場完整性偏低')
        if batch_q.get('penalty', 0.0) >= 0.32:
            quarantine = True
            reasons.extend(list(batch_q.get('reasons') or []))
        elif batch_q.get('reasons'):
            reasons.extend(list(batch_q.get('reasons') or []))
        if eq_bucket == 'bad' and result in ('win', 'loss'):
            reasons.append('成交品質差')
        if old_live:
            reasons.append('舊實單降級保存')
        if quarantine:
            bucket = 'quarantine'
        else:
            strong_live = (
                result in ('open', 'win', 'loss') and
                health_score >= 0.58 and
                weight >= 0.42 and
                eq_bucket != 'bad' and
                feat_score >= 0.72 and
                label_i >= 0.55 and
                exec_i >= 0.55 and
                exit_i >= 0.55 and
                not old_live
            )
            bucket = 'trusted_live' if strong_live else 'soft_live'

    quality_score = round(max(0.0, min(1.0, 0.28 * health_score + 0.22 * min(weight, 1.0) + 0.16 * feat_score + 0.14 * label_i + 0.1 * exec_i + 0.1 * exit_i - batch_q.get('penalty', 0.0))), 4)
    return {
        'learning_bucket': bucket,
        'source_pool': ('live_execution_pool' if is_live else ('paper_sim_pool' if source.startswith('paper') or source.startswith('sim') else 'backtest_pool')),
        'sample_tier': (
            'blocked' if bucket == 'quarantine' else
            'trusted' if bucket == 'trusted_live' else
            'review' if old_live else
            'usable'
        ),
        'can_influence_live_decision': bool(bucket == 'trusted_live'),
        'quarantine_flag': bool(bucket == 'quarantine'),
        'quality_score': quality_score,
        'health_score': round(health_score, 4),
        'sample_weight': round(weight, 4),
        'label_confidence': label_i,
        'execution_integrity': exec_i,
        'exit_integrity': exit_i,
        'feature_completeness': feat_score,
        'batch_quality_penalty': batch_q.get('penalty', 0.0),
        'weight_reasons': list(dict.fromkeys(weight_reasons + reasons + list(health.get('reasons') or []))),
        'legacy_retained': bool(old_live),
        'dataset_version': dataset_meta.get('dataset_version'),
        'learning_generation': dataset_meta.get('learning_generation'),
        'activated_from': dataset_meta.get('activated_from'),
        'deactivated_at': dataset_meta.get('deactivated_at'),
        'feature_schema_version': dataset_meta.get('feature_schema_version'),
        'decision_model_version': dataset_meta.get('decision_model_version'),
        'decision_fingerprint': fingerprint,
        'execution_quality_bucket': eq_bucket,
        'entry_zone_type': infer_entry_zone_type(t),
        'setup_mode': coarse_setup_mode(str(t.get('setup_label') or (t.get('breakdown') or {}).get('Setup') or '')),
        'session_bucket': _session_bucket(t),
    }


def enrich_learning_trade(trade: Dict[str, Any], reset_from: str = '') -> Dict[str, Any]:
    t = dict(trade or {})
    info = classify_learning_bucket(t, reset_from=reset_from)
    for k, v in info.items():
        t[k] = v
    return t


def filter_learning_pool(trades: List[Dict[str, Any]], pool: str = 'trusted_soft', closed_only: bool = False, reset_from: str = '') -> List[Dict[str, Any]]:
    out = []
    allowed = {
        'all': {'trusted_live', 'soft_live', 'quarantine', 'other'},
        'trusted_live': {'trusted_live'},
        'soft_live': {'soft_live'},
        'trusted_soft': {'trusted_live', 'soft_live'},
        'quarantine': {'quarantine'},
    }.get(pool, {'trusted_live', 'soft_live'})
    for row in trades or []:
        t = row if isinstance(row, dict) else dict(row or {})
        if 'learning_bucket' not in t or 'decision_fingerprint' not in t:
            t = enrich_learning_trade(t, reset_from=reset_from)
        if closed_only and str(t.get('result') or '') not in ('win', 'loss'):
            continue
        if str(t.get('learning_bucket') or 'other') in allowed:
            out.append(t)
    return out


def summarize_learning_pools(trades: List[Dict[str, Any]], reset_from: str = '') -> Dict[str, Any]:
    enriched = [enrich_learning_trade(dict(t or {}), reset_from=reset_from) for t in (trades or [])]
    live_closed = [t for t in enriched if str(t.get('source') or '').lower().startswith('live') and str(t.get('result') or '') in ('win', 'loss')]
    counts = Counter(str(t.get('learning_bucket') or 'other') for t in live_closed)
    symbol_counter = Counter(str(t.get('symbol') or '') for t in live_closed if str(t.get('learning_bucket') or '') in ('trusted_live', 'soft_live') and t.get('symbol'))
    avg_label = round(sum(safe_float(t.get('label_confidence', 0.0)) for t in live_closed) / max(len(live_closed), 1), 4) if live_closed else 0.0
    avg_exec = round(sum(safe_float(t.get('execution_integrity', 0.0)) for t in live_closed) / max(len(live_closed), 1), 4) if live_closed else 0.0
    avg_exit = round(sum(safe_float(t.get('exit_integrity', 0.0)) for t in live_closed) / max(len(live_closed), 1), 4) if live_closed else 0.0
    return {
        'trusted_live_count': int(counts.get('trusted_live', 0)),
        'soft_live_count': int(counts.get('soft_live', 0)),
        'quarantine_count': int(counts.get('quarantine', 0)),
        'other_count': int(counts.get('other', 0)),
        'closed_live_count': len(live_closed),
        'local_ready_symbols': sorted([sym for sym, cnt in symbol_counter.items() if cnt >= 5]),
        'avg_label_confidence': avg_label,
        'avg_execution_integrity': avg_exec,
        'avg_exit_integrity': avg_exit,
    }


def phase_from_counts(global_count: int, local_count: int, effective_count: float) -> str:
    if int(global_count or 0) < 30 or int(local_count or 0) < 5:
        return 'learning'
    if int(global_count or 0) < 50 or int(local_count or 0) < 12 or float(effective_count or 0.0) < 8.0:
        return 'semi'
    return 'full'


def filter_learning_samples_by_tier(trades: List[Dict[str, Any]], tiers=None, source_pools=None, reset_from: str = '') -> List[Dict[str, Any]]:
    tiers = set(tiers or ['trusted', 'usable'])
    source_pools = set(source_pools or ['live_execution_pool'])
    out: List[Dict[str, Any]] = []
    for row in trades or []:
        t = row if isinstance(row, dict) else dict(row or {})
        if 'sample_tier' not in t or 'source_pool' not in t:
            t = enrich_learning_trade(t, reset_from=reset_from)
        if str(t.get('sample_tier') or 'review') in tiers and str(t.get('source_pool') or 'backtest_pool') in source_pools:
            out.append(t)
    return out

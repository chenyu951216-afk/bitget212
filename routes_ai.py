from __future__ import annotations

import json
from typing import Any, Dict, List

from ai_learning_core import weighted_stats
from learning_engine import summarize_learning_pools, phase_from_counts, enrich_learning_trade, safe_float


def _recent_stats(rows: List[Dict[str, Any]], limit: int = 10) -> Dict[str, Any]:
    picked = list(rows or [])[-max(int(limit or 10), 1):]
    return weighted_stats(picked, {}) if picked else {'count': 0, 'ev_per_trade': 0.0, 'profit_factor': None}


def build_ai_db_stats_payload(*, live_open, live_closed, ai_panel, backtest_db, ai_db, reset_from: str = '') -> Dict[str, Any]:
    live_closed_enriched = [enrich_learning_trade(dict(t or {}), reset_from=reset_from) for t in (live_closed or [])]
    summary = summarize_learning_pools(live_closed_enriched, reset_from=reset_from)
    latest = None
    if live_closed_enriched:
        lt = live_closed_enriched[-1]
        latest = {
            'symbol': lt.get('symbol'),
            'result': lt.get('result'),
            'pnl_pct': lt.get('pnl_pct'),
            'learn_pnl_pct': lt.get('learn_pnl_pct'),
            'exit_time': lt.get('exit_time') or lt.get('time'),
            'source': lt.get('source'),
            'learning_bucket': lt.get('learning_bucket'),
        }
    trusted_soft = [t for t in live_closed_enriched if t.get('learning_bucket') in ('trusted_live', 'soft_live')]
    recent = _recent_stats(trusted_soft, limit=10)
    global_count = len(trusted_soft)
    phase = phase_from_counts(global_count=global_count, local_count=5 if global_count >= 5 else global_count, effective_count=global_count)
    return {
        'ok': True,
        'data_scope': 'live_only',
        'open_live_count': len([t for t in (live_open or []) if t.get('result') == 'open']),
        'closed_live_count': len(live_closed_enriched),
        'trusted_live_count': summary.get('trusted_live_count', 0),
        'soft_live_count': summary.get('soft_live_count', 0),
        'quarantine_count': summary.get('quarantine_count', 0),
        'last_learning': (ai_panel or {}).get('last_learning', '--'),
        'symbols': sorted(list({str(t.get('symbol') or '') for t in live_closed_enriched if t.get('symbol')})),
        'latest': latest,
        'backtest_run_count': len(((backtest_db or {}).get('runs') or [])),
        'ai_phase': phase,
        'ai_ready': phase == 'full',
        'mode': phase,
        'local_ready_symbols': summary.get('local_ready_symbols', []),
        'symbol_blocked_list': sorted(list((ai_db or {}).get('blocked_symbols', []) or [])),
        'recent_ev_10': round(float(recent.get('ev_per_trade', 0.0) or 0.0), 4),
        'recent_pf_10': None if recent.get('profit_factor') is None else round(float(recent.get('profit_factor') or 0.0), 3),
        'recent_miss_good_trade_count': int((ai_db or {}).get('recent_miss_good_trade_count', 0) or 0),
        'recent_fake_breakout_loss_count': int((ai_db or {}).get('recent_fake_breakout_loss_count', 0) or 0),
        'avg_label_confidence': summary.get('avg_label_confidence', 0.0),
        'avg_execution_integrity': summary.get('avg_execution_integrity', 0.0),
        'avg_exit_integrity': summary.get('avg_exit_integrity', 0.0),
    }


def build_ai_learning_recent_payload(*, sqlite_fetch_dicts, sqlite_order_clause, limit: int, sqlite_db_path: str, json_module=json) -> Dict[str, Any]:
    rows, error = [], None
    try:
        order_clause = sqlite_order_clause('learning_trades', ['updated_at', 'created_at', 'exit_time', 'entry_time'])
        rows = sqlite_fetch_dicts(
            f"""
            SELECT trade_id, symbol, result, source, entry_time, exit_time, created_at, updated_at, data_json
            FROM learning_trades
            ORDER BY {order_clause}
            LIMIT ?
            """,
            (limit,),
        )
        parsed = []
        for row in rows:
            payload = {}
            raw = row.get('data_json')
            try:
                payload = json_module.loads(raw) if raw else {}
            except Exception:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload = enrich_learning_trade(payload)
            parsed.append({
                'trade_id': row.get('trade_id'),
                'symbol': row.get('symbol'),
                'result': row.get('result'),
                'source': row.get('source'),
                'entry_time': row.get('entry_time'),
                'exit_time': row.get('exit_time'),
                'created_at': row.get('created_at'),
                'updated_at': row.get('updated_at'),
                'pnl_pct': payload.get('pnl_pct'),
                'learn_pnl_pct': payload.get('learn_pnl_pct'),
                'edge_pct': payload.get('edge_pct'),
                'side': payload.get('side'),
                'setup_label': payload.get('setup_label'),
                'regime': (payload.get('breakdown') or {}).get('Regime') if isinstance(payload.get('breakdown'), dict) else None,
                'weight': payload.get('sample_weight'),
                'health_score': payload.get('health_score'),
                'quality_score': payload.get('quality_score'),
                'learning_bucket': payload.get('learning_bucket'),
                'quarantine_flag': payload.get('quarantine_flag'),
                'setup_mode': payload.get('setup_mode'),
                'decision_fingerprint': payload.get('decision_fingerprint'),
                'exit_type': payload.get('exit_type'),
                'execution_quality_bucket': payload.get('execution_quality_bucket'),
                'label_confidence': payload.get('label_confidence'),
                'execution_integrity': payload.get('execution_integrity'),
                'exit_integrity': payload.get('exit_integrity'),
                'feature_completeness': payload.get('feature_completeness'),
                'dataset_version': payload.get('dataset_version'),
                'learning_generation': payload.get('learning_generation'),
            })
        rows = parsed
    except Exception as e:
        error = str(e)
    return {'ok': error is None, 'limit': limit, 'count': len(rows), 'data': rows, 'error': error}


def _derive_reject_stage(decision: Dict[str, Any]) -> str:
    reasons = list((decision or {}).get('reasons') or [])
    txt = ' | '.join(str(x) for x in reasons)
    if 'AI封鎖' in txt:
        return 'ai_block'
    if 'RR不足' in txt:
        return 'rr_gate'
    if '進場品質不足' in txt:
        return 'entry_quality_gate'
    if '分數未過門檻' in txt:
        return 'threshold_gate'
    if '方向衝突' in txt or '大盤方向不符' in txt:
        return 'regime_gate'
    if '已有持倉' in txt or '同向持倉已滿' in txt or '進場冷卻中' in txt:
        return 'risk_gate'
    return 'passed' if bool((decision or {}).get('will_order')) else 'mixed_gate'


def build_ai_debug_payload(*, audit_map, threshold_state, risk_status, market_state, session_state, now_text: str) -> Dict[str, Any]:
    decorated = {}
    for symbol, decision in dict(audit_map or {}).items():
        d = dict(decision or {})
        d['final_reject_stage'] = _derive_reject_stage(d)
        d['threshold_breakdown'] = {
            'effective_threshold': d.get('threshold'),
            'effective_score': d.get('effective_score'),
            'rotation_adj': d.get('rotation_adj'),
        }
        d['profile_source_level'] = str(d.get('ai_source', 'none')).split(':')[-1] if d.get('ai_source') else 'none'
        d['rr_floor_used'] = 1.2
        d['entry_quality_floor_used'] = 2 if float(d.get('effective_score', 0) or 0) >= float(d.get('threshold', 0) or 0) + 4 else 3
        d['execution_gate_reasons'] = [r for r in list(d.get('reasons') or []) if 'spread' in str(r) or 'depth' in str(r) or 'mark' in str(r)]
        d['block_type'] = 'symbol' if '幣種' in str(d.get('ai_note', '')) else 'strategy' if '策略' in str(d.get('ai_note', '')) else d['final_reject_stage']
        decorated[symbol] = d
    return {
        'ok': True,
        'threshold': {
            'current': threshold_state.get('current'),
            'default': 60,
            'high': 80,
            'floor': 55,
            'drop': 2,
            'state': dict(threshold_state or {}),
        },
        'risk_status': dict(risk_status or {}),
        'market_info': dict(market_state or {}),
        'session_info': dict(session_state or {}),
        'auto_order_audit': decorated,
        'symbols': sorted(list(decorated.keys())),
        'count': len(decorated),
        'updated_at': now_text,
    }


def build_ai_learning_health_payload(*, live_closed, reset_from: str = '') -> Dict[str, Any]:
    rows = [enrich_learning_trade(dict(t or {}), reset_from=reset_from) for t in (live_closed or [])]
    total = len(rows)
    bucket_counts = {}
    exit_type_bad = {}
    regime_bad = {}
    symbol_dirty = {}
    for t in rows:
        bucket = str(t.get('learning_bucket') or 'other')
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        dirty_like = (
            safe_float(t.get('label_confidence', 0.0)) < 0.5 or
            safe_float(t.get('execution_integrity', 0.0)) < 0.5 or
            safe_float(t.get('exit_integrity', 0.0)) < 0.5 or
            safe_float(t.get('feature_completeness', 0.0)) < 0.6 or
            bucket == 'quarantine'
        )
        if dirty_like:
            exit_key = str(t.get('exit_type') or 'unknown')
            regime_key = str((t.get('breakdown') or {}).get('Regime') or t.get('regime') or 'neutral')
            sym = str(t.get('symbol') or '')
            exit_type_bad[exit_key] = exit_type_bad.get(exit_key, 0) + 1
            regime_bad[regime_key] = regime_bad.get(regime_key, 0) + 1
            if sym:
                symbol_dirty[sym] = symbol_dirty.get(sym, 0) + 1
    return {
        'ok': True,
        'count': total,
        'bucket_counts': bucket_counts,
        'bucket_ratios': {k: round(v / max(total, 1), 4) for k, v in bucket_counts.items()},
        'avg_label_confidence': round(sum(safe_float(t.get('label_confidence', 0.0)) for t in rows) / max(total, 1), 4) if rows else 0.0,
        'avg_execution_integrity': round(sum(safe_float(t.get('execution_integrity', 0.0)) for t in rows) / max(total, 1), 4) if rows else 0.0,
        'avg_exit_integrity': round(sum(safe_float(t.get('exit_integrity', 0.0)) for t in rows) / max(total, 1), 4) if rows else 0.0,
        'avg_feature_completeness': round(sum(safe_float(t.get('feature_completeness', 0.0)) for t in rows) / max(total, 1), 4) if rows else 0.0,
        'dirtiest_exit_types': sorted(([{'exit_type': k, 'count': v} for k, v in exit_type_bad.items()]), key=lambda x: (-x['count'], x['exit_type']))[:10],
        'dirtiest_regimes': sorted(([{'regime': k, 'count': v} for k, v in regime_bad.items()]), key=lambda x: (-x['count'], x['regime']))[:10],
        'dirtiest_symbols': sorted(([{'symbol': k, 'count': v} for k, v in symbol_dirty.items()]), key=lambda x: (-x['count'], x['symbol']))[:15],
    }


def build_ai_strategy_matrix_payload(*, live_closed, reset_from: str = '') -> Dict[str, Any]:
    rows = [enrich_learning_trade(dict(t or {}), reset_from=reset_from) for t in (live_closed or [])]
    matrix = {}
    for t in rows:
        if str(t.get('result') or '') not in ('win', 'loss'):
            continue
        regime = str((t.get('breakdown') or {}).get('Regime') or t.get('regime') or 'neutral')
        setup = str(t.get('setup_mode') or 'main')
        symbol = str(t.get('symbol') or '')
        key = (symbol, regime, setup)
        slot = matrix.setdefault(key, {'symbol': symbol, 'regime': regime, 'setup_mode': setup, 'count': 0, 'wins': 0, 'losses': 0, 'sum_pnl': 0.0, 'max_drawdown_proxy': 0.0, 'blocked_hint': False, 'weight_sum': 0.0})
        slot['count'] += 1
        pnl = safe_float(t.get('learn_pnl_pct', t.get('account_pnl_pct', t.get('pnl_pct', 0.0))))
        slot['sum_pnl'] += pnl
        slot['weight_sum'] += safe_float(t.get('sample_weight', 1.0), 1.0)
        slot['max_drawdown_proxy'] = max(slot['max_drawdown_proxy'], abs(min(pnl, 0.0)))
        if str(t.get('result')) == 'win':
            slot['wins'] += 1
        else:
            slot['losses'] += 1
    items = []
    for _, slot in matrix.items():
        count = slot['count']
        win_rate = slot['wins'] / max(count, 1) * 100.0
        avg_pnl = slot['sum_pnl'] / max(count, 1)
        blocked_hint = count >= 11 and win_rate < 45.0
        items.append({
            **slot,
            'win_rate': round(win_rate, 2),
            'ev_per_trade': round(avg_pnl, 4),
            'avg_pnl': round(avg_pnl, 4),
            'blocked_hint': blocked_hint,
            'worth_weighting': bool(count >= 5 and win_rate >= 55.0 and avg_pnl > 0),
        })
    items.sort(key=lambda x: (-x['count'], -x['win_rate'], x['symbol'], x['regime'], x['setup_mode']))
    return {'ok': True, 'count': len(items), 'items': items[:300]}


def build_ai_decision_explain_payload(*, symbol: str, audit_map, replay_items) -> Dict[str, Any]:
    decision = dict((audit_map or {}).get(symbol) or {})
    replay = None
    for item in list(replay_items or []):
        if str((item.get('_meta') or {}).get('symbol') or item.get('symbol') or '') == symbol:
            replay = dict(item or {})
            break
    gates = dict((replay or {}).get('gating') or {})
    calibrator = dict((replay or {}).get('decision_calibrator') or {})
    return {
        'ok': bool(decision or replay),
        'symbol': symbol,
        'decision': decision,
        'current_setup': (replay or {}).get('setup_key') or decision.get('setup_key') or '',
        'regime': ((replay or {}).get('regime_snapshot') or {}).get('regime') or decision.get('regime') or 'neutral',
        'score_breakdown': (replay or {}).get('signal_snapshot') or {},
        'threshold': decision.get('threshold'),
        'gates': {
            'passed': [k for k, v in gates.items() if bool(v)],
            'failed': [k for k, v in gates.items() if not bool(v)],
            'raw': gates,
        },
        'learning_profile': (replay or {}).get('sample_weight_summary') or {},
        'allow_now': bool(decision.get('will_order') or ((replay or {}).get('decision') or {}).get('will_order')),
        'reject_reasons_ranked': list(dict.fromkeys(list(decision.get('reasons') or []) + list(((replay or {}).get('decision') or {}).get('reasons') or []))),
        'decision_calibrator': calibrator,
        'execution_snapshot': (replay or {}).get('execution_quality') or {},
        'market_consensus': (replay or {}).get('market_consensus') or {},
        'replay_meta': (replay or {}).get('_meta') or {},
    }


def build_learning_sample_review_payload(*, live_closed, limit: int = 50, reset_from: str = '') -> Dict[str, Any]:
    rows = [enrich_learning_trade(dict(t or {}), reset_from=reset_from) for t in (live_closed or [])]
    rows.sort(key=lambda x: str(x.get('exit_time') or x.get('entry_time') or x.get('created_at') or ''), reverse=True)
    rows = rows[:limit]
    items = []
    for t in rows:
        items.append({
            'symbol': t.get('symbol'),
            'result': t.get('result'),
            'source': t.get('source'),
            'source_pool': t.get('source_pool'),
            'sample_tier': t.get('sample_tier'),
            'learning_bucket': t.get('learning_bucket'),
            'can_influence_live_decision': bool(t.get('can_influence_live_decision')),
            'weight': t.get('sample_weight'),
            'health_score': t.get('health_score'),
            'quality_score': t.get('quality_score'),
            'quarantine_flag': t.get('quarantine_flag'),
            'label_confidence': t.get('label_confidence'),
            'execution_integrity': t.get('execution_integrity'),
            'exit_integrity': t.get('exit_integrity'),
            'feature_completeness': t.get('feature_completeness'),
            'decision_fingerprint': t.get('decision_fingerprint'),
            'weight_reasons': list(t.get('weight_reasons') or []),
            'entry_time': t.get('entry_time'),
            'exit_time': t.get('exit_time'),
        })
    tier_counts = {}
    for t in rows:
        tier = str(t.get('sample_tier') or 'review')
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    return {
        'ok': True,
        'count': len(items),
        'limit': limit,
        'tier_counts': tier_counts,
        'trusted_live_count': len(filter_learning_samples_by_tier(rows, tiers=['trusted'], source_pools=['live_execution_pool'], reset_from=reset_from)),
        'items': items,
    }

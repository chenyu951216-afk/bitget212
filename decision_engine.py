from __future__ import annotations

from typing import Any, Dict, List


def merge_decision_explain(*, gating: Dict[str, Any], calibrator: Dict[str, Any], profile: Dict[str, Any], reasons=None) -> Dict[str, Any]:
    return {
        'gating': dict(gating or {}),
        'calibrator': dict(calibrator or {}),
        'profile': dict(profile or {}),
        'reasons': list(reasons or []),
    }


def derive_final_stage(*, gating: Dict[str, Any], ai_ok: bool, allow_now: bool) -> str:
    if allow_now:
        return 'final_decision'
    ordered = [
        ('signal_generate', True),
        ('signal_validate', bool(gating.get('setup', True) and gating.get('regime', True))),
        ('ai_profile_adjust', bool(ai_ok)),
        ('risk_validate', bool(gating.get('risk', True))),
        ('execution_validate', bool(gating.get('symbol', True) and gating.get('calibrated_winrate', True) and gating.get('positive_ev', True))),
        ('final_decision', bool(gating.get('trigger', True))),
    ]
    for stage, ok in ordered:
        if not ok:
            return stage
    return 'final_decision'


def derive_reject_reason(*, gating: Dict[str, Any], reasons: List[Any], ai_ok: bool) -> str:
    if not ai_ok:
        return 'ai_block'
    if not gating.get('regime', True):
        return 'regime_gate'
    if not gating.get('setup', True):
        txt = ' | '.join(str(r) for r in reasons)
        if 'RR不足' in txt:
            return 'rr_gate'
        if '進場品質不足' in txt:
            return 'entry_quality_gate'
        return 'setup_gate'
    if not gating.get('risk', True):
        return 'risk_gate'
    if not gating.get('symbol', True):
        return 'symbol_gate'
    if not gating.get('calibrated_winrate', True):
        return 'calibrated_winrate_gate'
    if not gating.get('positive_ev', True):
        return 'positive_ev_gate'
    if not gating.get('trigger', True):
        return 'threshold_gate'
    return 'passed'


def normalize_decision_summary(*, allow_now: bool, gating: Dict[str, Any], reasons: List[Any], profile: Dict[str, Any],
                               effective_score: float, effective_threshold: float, decision_calibrator: Dict[str, Any],
                               signal_snapshot: Dict[str, Any] | None = None) -> Dict[str, Any]:
    signal_snapshot = dict(signal_snapshot or {})
    ai_ok = bool(profile.get('allow_profile', True))
    stage = derive_final_stage(gating=gating, ai_ok=ai_ok, allow_now=allow_now)
    reject_reason = derive_reject_reason(gating=gating, reasons=reasons, ai_ok=ai_ok)
    return {
        'can_trade': bool(allow_now),
        'stage': stage,
        'reject_reason': reject_reason,
        'score_raw': round(float(signal_snapshot.get('score', 0.0) or 0.0), 4),
        'score_adjusted': round(float(effective_score or 0.0), 4),
        'threshold_raw': round(float(signal_snapshot.get('threshold_raw', effective_threshold) or effective_threshold), 4),
        'threshold_final': round(float(effective_threshold or 0.0), 4),
        'profile_name': str(profile.get('profile_name') or profile.get('source_level') or profile.get('source') or 'unknown'),
        'profile_samples': int(profile.get('sample_count', 0) or 0),
        'execution_score': round(float((signal_snapshot.get('execution_quality') or {}).get('execution_score', decision_calibrator.get('exec_score_used', 0.0)) or 0.0), 4),
    }


def build_decision_funnel_payload(items: List[Dict[str, Any]], limit: int) -> Dict[str, Any]:
    rows = []
    for item in list(items or [])[:limit]:
        row = dict(item or {})
        rows.append({
            'symbol': row.get('symbol'),
            'can_trade': bool(row.get('can_trade', row.get('allow_now', False))),
            'stage': row.get('stage') or row.get('final_reject_stage') or 'unknown',
            'reject_reason': row.get('reject_reason') or row.get('block_type') or 'unknown',
            'score_raw': row.get('score_raw', row.get('score')),
            'score_adjusted': row.get('score_adjusted', row.get('effective_score')),
            'threshold_raw': row.get('threshold_raw', row.get('threshold')),
            'threshold_final': row.get('threshold_final', row.get('effective_threshold', row.get('threshold'))),
            'profile_name': row.get('profile_name', row.get('profile_source_level', 'unknown')),
            'profile_samples': row.get('profile_samples', 0),
            'execution_score': row.get('execution_score', 0),
            'reasons': list(row.get('reasons') or []),
        })
    return {'ok': True, 'count': len(rows), 'limit': limit, 'items': rows}

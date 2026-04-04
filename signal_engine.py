from __future__ import annotations

from typing import Any, Dict


def build_signal_quality_snapshot(sig: Dict[str, Any] | None) -> Dict[str, Any]:
    sig = dict(sig or {})
    bd = dict(sig.get('breakdown') or {})
    return {
        'symbol': sig.get('symbol'),
        'score': sig.get('score'),
        'rr_ratio': sig.get('rr_ratio'),
        'entry_quality': sig.get('entry_quality'),
        'regime': bd.get('Regime') or sig.get('regime') or 'neutral',
        'setup': sig.get('setup_label') or bd.get('Setup') or '',
        'market_tempo': bd.get('MarketTempo') or sig.get('market_tempo') or 'normal',
        'execution_quality': dict(sig.get('execution_quality') or {}),
    }

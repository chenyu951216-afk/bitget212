from __future__ import annotations

from typing import Any, Dict


def execution_score_from_snapshot(snapshot: Dict[str, Any] | None) -> float:
    snap = dict(snapshot or {})
    try:
        spread = abs(float(snap.get('spread_pct', 0.0) or 0.0))
        deviation = abs(float(snap.get('mark_last_deviation_pct', 0.0) or 0.0))
        depth = float(snap.get('top_depth_ratio', 0.18) or 0.18)
        notes = list(snap.get('notes') or [])
        score = 0.86
        score -= min(spread / 0.45, 0.45)
        score -= min(deviation / 0.6, 0.25)
        if depth < 0.18:
            score -= min((0.18 - depth) * 1.5, 0.22)
        if notes:
            score -= min(len(notes) * 0.08, 0.2)
        return round(max(0.05, min(1.0, score)), 4)
    except Exception:
        return 0.45

from typing import Any, Dict


def session_bucket_from_hour(hour_tw: int) -> str:
    h = int(hour_tw)
    if 0 <= h < 7:
        return 'asia_late'
    if 7 <= h < 15:
        return 'asia_day'
    if 15 <= h < 20:
        return 'eu_prep'
    if 20 <= h < 23:
        return 'us_open'
    return 'us_late'


def build_session_bias(bucket: str, session_stats: Dict[str, Any] = None) -> Dict[str, Any]:
    stats = dict(session_stats or {})
    row = dict(stats.get(bucket) or {})
    wr = float(row.get('win_rate', 0.0) or 0.0)
    ev = float(row.get('ev_per_trade', 0.0) or 0.0)
    bias = 0.0
    reasons = []
    if ev > 0.03:
        bias += 1.0
        reasons.append('時段EV偏強')
    elif ev < -0.03:
        bias -= 1.0
        reasons.append('時段EV偏弱')
    if wr >= 58:
        bias += 0.5
    elif 0 < wr < 45:
        bias -= 0.5
    return {'bucket': bucket, 'bias': round(bias, 3), 'reasons': reasons}

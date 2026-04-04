from typing import Any, Dict


def derive_auto_mode(api_error_streak: int = 0, protection_fail_streak: int = 0, learning_stale_minutes: float = 0.0, schema_ok: bool = True) -> Dict[str, Any]:
    mode = 'normal'
    color = 'green'
    reasons = []
    if learning_stale_minutes >= 45 or not schema_ok:
        mode = 'observe'
        color = 'red'
        reasons.append('學習或資料庫異常')
    if protection_fail_streak >= 2:
        mode = 'observe'
        color = 'red'
        reasons.append('保護單異常')
    elif api_error_streak >= 3:
        mode = 'conservative'
        color = 'yellow'
        reasons.append('API連續失敗')
    return {'mode': mode, 'status_light': color, 'reasons': reasons}

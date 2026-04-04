import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List


def _now() -> str:
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def ensure_replay_tables(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS decision_replay_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT,
                side TEXT,
                regime TEXT,
                setup_key TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_replay_inputs_sym ON decision_replay_inputs(symbol, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_replay_inputs_regime ON decision_replay_inputs(regime, created_at DESC);
            '''
        )
        conn.commit()
    finally:
        conn.close()


def save_decision_input_snapshot(db_path: str, payload: Dict[str, Any]) -> None:
    ensure_replay_tables(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute(
            'INSERT INTO decision_replay_inputs (created_at, symbol, side, regime, setup_key, payload_json) VALUES (?, ?, ?, ?, ?, ?)',
            (
                _now(),
                str((payload or {}).get('symbol') or ''),
                str((payload or {}).get('side') or ''),
                str((payload or {}).get('regime_snapshot', {}).get('regime') or (payload or {}).get('regime') or ''),
                str((payload or {}).get('setup_key') or ''),
                json.dumps(payload or {}, ensure_ascii=False),
            )
        )
        conn.execute('DELETE FROM decision_replay_inputs WHERE id NOT IN (SELECT id FROM decision_replay_inputs ORDER BY id DESC LIMIT 1500)')
        conn.commit()
    finally:
        conn.close()


def load_decision_input_snapshots(db_path: str, limit: int = 50) -> List[Dict[str, Any]]:
    ensure_replay_tables(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('SELECT * FROM decision_replay_inputs ORDER BY id DESC LIMIT ?', (max(1, int(limit or 50)),)).fetchall()
    finally:
        conn.close()
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row['payload_json'])
        except Exception:
            payload = {}
        payload['_meta'] = {'id': row['id'], 'created_at': row['created_at'], 'symbol': row['symbol'], 'side': row['side'], 'regime': row['regime'], 'setup_key': row['setup_key']}
        out.append(payload)
    return out

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from learning_engine import enrich_learning_trade


def _utc_now_str() -> str:
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


class BotStorage:
    def __init__(self, db_path: str, *, legacy_learn_json: Optional[str] = None, legacy_backtest_json: Optional[str] = None):
        self.db_path = db_path
        self.legacy_learn_json = legacy_learn_json
        self.legacy_backtest_json = legacy_backtest_json
        self._lock = threading.RLock()
        self._init_db()
        self._ensure_learning_trade_columns()
        self._migrate_legacy_if_needed()

    def _connect(self):
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.executescript(
                '''
                CREATE TABLE IF NOT EXISTS learning_trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    result TEXT,
                    source TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    learning_bucket TEXT DEFAULT '',
                    quarantine_flag INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0,
                    fingerprint TEXT DEFAULT '',
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_learning_symbol ON learning_trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_learning_result ON learning_trades(result);
                CREATE INDEX IF NOT EXISTS idx_learning_source ON learning_trades(source);
                CREATE INDEX IF NOT EXISTS idx_learning_bucket ON learning_trades(learning_bucket);

                CREATE TABLE IF NOT EXISTS learning_meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    time_text TEXT,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_trade_history_created ON trade_history(created_at DESC);

                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS backtest_meta (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_risk_events_type ON risk_events(event_type, created_at DESC);

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_logs_cat ON audit_logs(category, created_at DESC);
                '''
            )


    def _ensure_learning_trade_columns(self):
        required = {
            'learning_bucket': "TEXT DEFAULT ''",
            'quarantine_flag': 'INTEGER DEFAULT 0',
            'quality_score': 'REAL DEFAULT 0',
            'fingerprint': "TEXT DEFAULT ''",
        }
        with self._lock, self._connect() as conn:
            cols = {str(r['name']) for r in conn.execute('PRAGMA table_info(learning_trades)').fetchall()}
            for name, spec in required.items():
                if name not in cols:
                    conn.execute(f'ALTER TABLE learning_trades ADD COLUMN {name} {spec}')
            conn.commit()

    def _table_count(self, table: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(f'SELECT COUNT(1) AS c FROM {table}').fetchone()
            return int(row['c'] if row else 0)

    def _read_json_file(self, path: Optional[str], default: Any) -> Any:
        if not path or not os.path.exists(path):
            return default
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except Exception:
            return default

    def _migrate_legacy_if_needed(self):
        try:
            if self._table_count('learning_trades') == 0 and self._table_count('learning_meta') == 0:
                legacy = self._read_json_file(self.legacy_learn_json, None)
                if isinstance(legacy, dict):
                    self.save_learning_state(legacy)
                    self.append_audit_log('migration', 'legacy learn_db json imported', {
                        'path': self.legacy_learn_json,
                        'trade_count': len(legacy.get('trades', []) or []),
                    })
            if self._table_count('backtest_meta') == 0 and self._table_count('backtest_runs') == 0:
                legacy_bt = self._read_json_file(self.legacy_backtest_json, None)
                if isinstance(legacy_bt, dict):
                    self.save_backtest_state(legacy_bt)
                    self.append_audit_log('migration', 'legacy backtest json imported', {
                        'path': self.legacy_backtest_json,
                        'run_count': len(legacy_bt.get('runs', []) or []),
                    })
        except Exception:
            pass

    def load_learning_state(self, default: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(default or {})
        with self._lock, self._connect() as conn:
            trades = []
            for row in conn.execute('SELECT data_json FROM learning_trades ORDER BY COALESCE(entry_time, created_at) ASC, created_at ASC'):
                try:
                    trades.append(json.loads(row['data_json']))
                except Exception:
                    continue
            data['trades'] = trades
            for row in conn.execute('SELECT key, value_json FROM learning_meta'):
                try:
                    data[row['key']] = json.loads(row['value_json'])
                except Exception:
                    pass
        return data

    def save_learning_state(self, db: Dict[str, Any]) -> None:
        payload = dict(db or {})
        trades = [dict(t or {}) for t in (payload.pop('trades', []) or [])]
        now = _utc_now_str()
        with self._lock, self._connect() as conn:
            conn.execute('BEGIN')
            for idx, raw_trade in enumerate(trades):
                trade = enrich_learning_trade(raw_trade)
                trade_id = str(trade.get('id') or trade.get('trade_id') or '')
                if not trade_id:
                    trade_id = f"{trade.get('symbol','unknown')}_{trade.get('entry_time','')}_{idx}"
                    trade['id'] = trade_id
                row = conn.execute('SELECT created_at FROM learning_trades WHERE trade_id=?', (trade_id,)).fetchone()
                created_at = str((row['created_at'] if row and row['created_at'] else now))
                conn.execute(
                    '''
                    INSERT INTO learning_trades (
                        trade_id, symbol, result, source, entry_time, exit_time,
                        learning_bucket, quarantine_flag, quality_score, fingerprint,
                        data_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trade_id) DO UPDATE SET
                        symbol=excluded.symbol,
                        result=excluded.result,
                        source=excluded.source,
                        entry_time=excluded.entry_time,
                        exit_time=excluded.exit_time,
                        learning_bucket=excluded.learning_bucket,
                        quarantine_flag=excluded.quarantine_flag,
                        quality_score=excluded.quality_score,
                        fingerprint=excluded.fingerprint,
                        data_json=excluded.data_json,
                        updated_at=excluded.updated_at
                    ''',
                    (
                        trade_id,
                        str(trade.get('symbol') or ''),
                        str(trade.get('result') or ''),
                        str(trade.get('source') or ''),
                        str(trade.get('entry_time') or ''),
                        str(trade.get('exit_time') or ''),
                        str(trade.get('learning_bucket') or ''),
                        1 if bool(trade.get('quarantine_flag')) else 0,
                        float(trade.get('quality_score', 0) or 0),
                        str(trade.get('decision_fingerprint') or ''),
                        json.dumps(trade, ensure_ascii=False),
                        created_at,
                        now,
                    ),
                )
            for key, value in payload.items():
                conn.execute(
                    '''
                    INSERT INTO learning_meta (key, value_json, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                    ''',
                    (str(key), json.dumps(value, ensure_ascii=False), now),
                )
            conn.commit()

    def append_trade_history_record(self, record: Dict[str, Any]) -> None:
        now = _utc_now_str()
        with self._lock, self._connect() as conn:
            conn.execute(
                'INSERT INTO trade_history (symbol, side, time_text, created_at, data_json) VALUES (?, ?, ?, ?, ?)',
                (
                    str((record or {}).get('symbol') or ''),
                    str((record or {}).get('side') or ''),
                    str((record or {}).get('time') or ''),
                    now,
                    json.dumps(record or {}, ensure_ascii=False),
                ),
            )
            conn.execute(
                '''
                DELETE FROM trade_history
                WHERE id NOT IN (
                    SELECT id FROM trade_history ORDER BY id DESC LIMIT 500
                )
                '''
            )
            conn.commit()

    def load_recent_trade_history(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                'SELECT data_json FROM trade_history ORDER BY id DESC LIMIT ?',
                (max(1, int(limit or 30)),),
            ).fetchall()
        out = []
        for row in rows:
            try:
                out.append(json.loads(row['data_json']))
            except Exception:
                pass
        return out

    def load_backtest_state(self, default: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(default or {})
        runs: List[Dict[str, Any]] = []
        with self._lock, self._connect() as conn:
            for row in conn.execute('SELECT data_json FROM backtest_runs ORDER BY id ASC'):
                try:
                    runs.append(json.loads(row['data_json']))
                except Exception:
                    continue
            data['runs'] = runs
            for row in conn.execute('SELECT key, value_json FROM backtest_meta'):
                try:
                    data[row['key']] = json.loads(row['value_json'])
                except Exception:
                    pass
        if 'runs' not in data:
            data['runs'] = []
        return data

    def save_backtest_state(self, db: Dict[str, Any]) -> None:
        payload = dict(db or {})
        runs = [dict(r or {}) for r in (payload.pop('runs', []) or [])]
        now = _utc_now_str()
        with self._lock, self._connect() as conn:
            conn.execute('BEGIN')
            conn.execute('DELETE FROM backtest_runs')
            for row in runs:
                conn.execute(
                    'INSERT INTO backtest_runs (created_at, data_json) VALUES (?, ?)',
                    (now, json.dumps(row, ensure_ascii=False)),
                )
            conn.execute('DELETE FROM backtest_meta')
            for key, value in payload.items():
                conn.execute(
                    'INSERT INTO backtest_meta (key, value_json, updated_at) VALUES (?, ?, ?)',
                    (str(key), json.dumps(value, ensure_ascii=False), now),
                )
            conn.commit()

    def append_risk_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                'INSERT INTO risk_events (event_type, created_at, payload_json) VALUES (?, ?, ?)',
                (str(event_type or ''), _utc_now_str(), json.dumps(payload or {}, ensure_ascii=False)),
            )
            conn.execute(
                '''
                DELETE FROM risk_events
                WHERE id NOT IN (
                    SELECT id FROM risk_events ORDER BY id DESC LIMIT 5000
                )
                '''
            )
            conn.commit()

    def append_audit_log(self, category: str, message: str, payload: Dict[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                'INSERT INTO audit_logs (category, message, created_at, payload_json) VALUES (?, ?, ?, ?)',
                (str(category or ''), str(message or ''), _utc_now_str(), json.dumps(payload or {}, ensure_ascii=False)),
            )
            conn.execute(
                '''
                DELETE FROM audit_logs
                WHERE id NOT IN (
                    SELECT id FROM audit_logs ORDER BY id DESC LIMIT 10000
                )
                '''
            )
            conn.commit()

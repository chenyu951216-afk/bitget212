from __future__ import annotations

from typing import List, Tuple


def default_thread_specs() -> List[Tuple[str, str]]:
    return [
        ('position_thread', 'position'),
        ('enhanced_position_thread', 'enhanced_position'),
        ('scan_thread', 'scan'),
        ('trailing_stop_thread', 'trailing'),
        ('session_monitor_thread', 'session'),
        ('market_analysis_thread', 'market'),
        ('fvg_order_monitor_thread', 'fvg_monitor'),
        ('auto_backtest_thread', 'auto_backtest'),
        ('memory_guard_thread', 'memory_guard'),
    ]

import os, sys, ccxt, threading, time, requests, gc, json, math
import numpy as np
sys.stdout.reconfigure(line_buffering=True)  # 即時 flush logs
import pandas as pd
import pandas_ta as ta
from flask import Flask, render_template, jsonify, request
from datetime import datetime

app = Flask(__name__)

# =====================================================
# API 配置
# =====================================================
bitget_config = {
    'apiKey':   os.getenv('BITGET_API_KEY', 'bg_d94506261d1e5337d124ddade02149c6'),
    'secret':   os.getenv('BITGET_SECRET', '3aa1590f429f87f2d25e6024938a10cb003cf78e3783007540abaea6f1fa6ed2'),
    'password': os.getenv('BITGET_PASSWORD', 'Jeff5466'),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'defaultMarginMode': 'cross'}
}
exchange = ccxt.bitget(bitget_config)
exchange.timeout = 10000   # 10秒 API 超時，絕不無限等待
exchange.enableRateLimit = True
PANIC_API_KEY   = os.getenv('PANIC_API_KEY', 'cde101d65a00a418e451ae0118ed0835275781e1')
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY', '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')
ANTHROPIC_KEY   = os.getenv('ANTHROPIC_API_KEY', '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')
ORDER_THRESHOLD         = 50   # 預設門檻 50
ORDER_THRESHOLD_DEFAULT = 50   # 預設值
ORDER_THRESHOLD_HIGH    = 55   # 持續滿倉後提高到 55
ORDER_THRESHOLD_DROP    = 1    # 每 3 輪無新單下降 1 分
ORDER_THRESHOLD_FLOOR   = 48   # 最低只降到 48

# =====================================================
# 核心交易參數
# =====================================================
RISK_PCT              = 0.05      # 每單名目資金使用總資產 5%
ATR_RISK_PCT          = 0.01      # 每單實際風險預算 1%（用停損距離換算倉位）
MIN_MARGIN_PCT        = 0.01      # 動態保證金下限 1%（至少投入總資金1%保證金）
MAX_MARGIN_PCT        = 0.08      # 動態保證金上限 8%
MAX_OPEN_POSITIONS    = 7         # 短線總持倉上限
MAX_SAME_DIRECTION    = 5         # 同方向最多 5 筆
TIME_STOP_BARS_15M    = 15        # 15 根 15m K 仍不走就時間止損
NEWS_CACHE_TTL_SEC    = 300       # 新聞快取 5 分鐘
ANTI_CHASE_ATR      = 1.25      # 價格偏離 15m EMA20 超過 1.25ATR 視為追價風險
BREAKOUT_LOOKBACK   = 20        # 預判暴拉/暴跌的區間觀察根數
PULLBACK_BUFFER_ATR = 0.35      # 避免追價，優先等 0.35ATR 回踩/反彈
SCALE_IN_MIN_RATIO = 0.35      # 分批進場第二批最低比例
SCALE_IN_MAX_RATIO = 0.45      # 分批進場第二批最高比例
FAKE_BREAKOUT_PENALTY = 8      # 假突破/假跌破扣分
LEARN_DB_PATH      = "/app/data/learn_db.json"
BACKTEST_DB_PATH   = "/app/data/backtest_runs.json"
STATE_BACKUP_PATH  = "/app/data/state_backup.json"
RISK_STATE_PATH    = "/app/data/risk_state.json"

SCORE_SMOOTH_ALPHA  = 0.35      # 穩定分數權重（越高越跟即時）
ENTRY_LOCK_SEC      = 300       # 同一幣種 5 分鐘內不重複開新單
MIN_RR_HARD_FLOOR   = 1.20      # 自動下單最低 RR
SIGNAL_META_CACHE   = {}        # 最近一次分析快取（給追蹤/驗證用）
SCORE_CACHE         = {}        # 分數平滑快取
ENTRY_LOCKS         = {}        # 進場鎖，避免 90→30 反覆觸發
PROTECTION_STATE    = {}        # 交易所保護單驗證狀態
AUTO_ORDER_AUDIT    = {}        # 記錄每輪為何沒下單

# =====================================================
# 防連損設定
# =====================================================
MAX_DAILY_LOSS_PCT   = 0.15
MAX_CONSECUTIVE_LOSS = 3
COOLDOWN_MINUTES     = 120

RISK_STATE = {
    "daily_loss_usdt":    0.0,
    "daily_start_equity": 0.0,
    "consecutive_loss":   0,
    "cooldown_until":     None,
    "trading_halted":     False,
    "halt_reason":        "",
    "today_date":         "",
}
RISK_LOCK = threading.Lock()

# ── 動態門檻狀態 ──
_DT = {
    "current":          50,   # 當前門檻（最低只降到48）
    "last_order_time":  None, # 最近下單時間
    "full_rounds":      0,    # 連續滿倉輪數
    "empty_rounds":     0,    # 門檻55時連續空倉輪數
    "no_order_rounds":  0,    # 連續無下單輪數（整數，避免None+1錯誤）
}
_DT_LOCK = threading.Lock()

def update_dynamic_threshold(top_sigs=None):
    """
    動態門檻循環（每輪掃描結束呼叫）：
    預設50
    → 滿倉連續5輪 → 升到55
    → 50門檻下空倉連續5輪 → 回到50
    → 有空位且1小時無新單 → 每小時降5（最低48）
    → 下單後回到50
    """
    global ORDER_THRESHOLD
    with _DT_LOCK:
        dt  = _DT
        now = datetime.now()

        with STATE_LOCK:
            pos_count = len(STATE["active_positions"])
            sigs = top_sigs or STATE.get("top_signals", [])

        is_full   = pos_count >= MAX_OPEN_POSITIONS
        has_space = pos_count < MAX_OPEN_POSITIONS

        # ── 情況A：滿倉 → 計算連續滿倉輪數 ──
        if is_full:
            dt["full_rounds"]  += 1
            dt["empty_rounds"]  = 0
            dt["no_order_rounds"] = 0  # 滿倉重置無單計數

            # 滿倉5輪後升到55
            if dt["full_rounds"] >= 5 and dt["current"] < ORDER_THRESHOLD_HIGH:
                dt["current"] = ORDER_THRESHOLD_HIGH
                ORDER_THRESHOLD = ORDER_THRESHOLD_HIGH
                print("📈 門檻升至55（連續滿倉{}輪）".format(dt["full_rounds"]))
                update_state(threshold_info={
                    "current": ORDER_THRESHOLD_HIGH,
                    "phase": "嚴格（持續滿倉）",
                    "full_rounds": dt["full_rounds"]
                })
            return

        # ── 有空位 ──
        dt["full_rounds"] = 0  # 重置滿倉計數

        # ── 情況B：門檻55，空倉連續5輪 → 回到50 ──
        if dt["current"] >= ORDER_THRESHOLD_HIGH:
            dt["empty_rounds"] += 1
            if dt["empty_rounds"] >= 5:
                dt["current"]      = ORDER_THRESHOLD_DEFAULT
                ORDER_THRESHOLD    = ORDER_THRESHOLD_DEFAULT
                dt["empty_rounds"] = 0
                print("📉 門檻回到50（50門檻下空倉{}輪）".format(dt["empty_rounds"]))
                update_state(threshold_info={
                    "current": ORDER_THRESHOLD_DEFAULT,
                    "phase": "預設",
                    "empty_rounds": dt["empty_rounds"]
                })
            else:
                print("⏱ 門檻55空倉中 {}/5 輪".format(dt["empty_rounds"]))
            return
        else:
            dt["empty_rounds"] = 0

        # ── 情況C：門檻55時有空倉，每3輪降5，降到38為止 ──
        if dt["current"] > ORDER_THRESHOLD_FLOOR:
            dt["no_order_rounds"] = dt.get("no_order_rounds", 0) + 1
            print("⏱ 門檻{}空倉計數: {}/3 輪".format(dt["current"], dt["no_order_rounds"]))
            if dt["no_order_rounds"] >= 3:
                new_val = max(dt["current"] - ORDER_THRESHOLD_DROP, ORDER_THRESHOLD_FLOOR)
                dt["current"]         = new_val
                ORDER_THRESHOLD       = new_val
                dt["no_order_rounds"] = 0  # 重置計數，繼續每3輪降5
                print("⬇️ 門檻降至{}（空倉持續3輪）".format(new_val))
                update_state(threshold_info={
                    "current": new_val,
                    "phase": "預設" if new_val <= ORDER_THRESHOLD_FLOOR else "降溫中",
                })
        else:
            dt["no_order_rounds"] = 0  # 已在最低48，重置

def record_order_placed():
    """下單後呼叫，重置計時並回到預設50"""
    global ORDER_THRESHOLD
    with _DT_LOCK:
        _DT["last_order_time"] = datetime.now()
        _DT["no_order_rounds"] = 0  # 重置輪數計數
        _DT["empty_rounds"]    = 0
        # 下單後若門檻曾降低（<40），回到50
        if _DT["current"] < ORDER_THRESHOLD_DEFAULT:
            _DT["current"] = ORDER_THRESHOLD_DEFAULT
            ORDER_THRESHOLD = ORDER_THRESHOLD_DEFAULT
            print("↩️ 門檻回到50（有新下單）")
            update_state(threshold_info={"current": ORDER_THRESHOLD_DEFAULT, "phase": "預設"})

# =====================================================
# 開盤時段保護系統（台灣時間 UTC+8）
# =====================================================
SESSION_STATE = {
    "eu_score":      0,   # 歐洲盤觀察分數 (-2~+2)
    "us_score":      0,   # 美洲盤觀察分數 (-2~+2)
    "eu_score_date": "",  # 歐盤分數的台灣日期 (YYYY-MM-DD)
    "us_score_date": "",  # 美盤分數的台灣日期
    "eu_score_time": "",  # 歐盤分數的台灣時間 (HH:MM)
    "us_score_time": "",  # 美盤分數的台灣時間
    "europe_obs":    [],  # 觀察期間的價格記錄
    "america_obs":   [],  # 觀察期間的價格記錄
    "session_phase": "normal",
    "session_note":  "",
}
SESSION_LOCK = threading.Lock()

def get_tw_time():
    """取得台灣時間（UTC+8）"""
    from datetime import timezone, timedelta
    tz_tw = timezone(timedelta(hours=8))
    return datetime.now(tz_tw)

def tw_now_str(fmt="%H:%M:%S"):
    """台灣時間格式化字串"""
    return get_tw_time().strftime(fmt)

def tw_today():
    """台灣時間今天日期"""
    return get_tw_time().strftime("%Y-%m-%d")

def get_session_status():
    """
    回傳當前時段狀態：
    - normal: 正常交易
    - eu_pause: 歐盤開盤前30分鐘，停止下新單
    - eu_closed: 19:50-20:32 完全停止+平倉
    - eu_watch: 20:32前觀察歐盤走勢
    - us_pause: 美盤開盤前30分鐘，停止下新單
    - us_closed: 21:50-22:32 完全停止+平倉
    - us_watch: 22:32前觀察美盤走勢
    """
    tw = get_tw_time()
    h = tw.hour
    m = tw.minute
    t = h * 60 + m  # 轉換成分鐘

    EU_PAUSE_START  = 19 * 60 + 30   # 19:30
    EU_CLOSE_START  = 19 * 60 + 50   # 19:50
    EU_WATCH_END    = 20 * 60 + 32   # 20:32
    EU_RESUME       = 20 * 60 + 35   # 20:35

    US_PAUSE_START  = 21 * 60 + 30   # 21:30
    US_CLOSE_START  = 21 * 60 + 50   # 21:50
    US_WATCH_END    = 22 * 60 + 32   # 22:32
    US_RESUME       = 22 * 60 + 35   # 22:35

    if EU_CLOSE_START <= t < EU_WATCH_END:
        return "eu_closed", "歐盤開盤觀察期 (19:50-20:32)"
    elif EU_PAUSE_START <= t < EU_CLOSE_START:
        return "eu_pause", "歐盤開盤前暫停下單 (19:30-19:50)"
    elif EU_WATCH_END <= t < EU_RESUME:
        return "eu_watch_end", "歐盤觀察結束，計算分數中"
    elif US_CLOSE_START <= t < US_WATCH_END:
        return "us_closed", "美盤開盤觀察期 (21:50-22:32)"
    elif US_PAUSE_START <= t < US_CLOSE_START:
        return "us_pause", "美盤開盤前暫停下單 (21:30-21:50)"
    elif US_WATCH_END <= t < US_RESUME:
        return "us_watch_end", "美盤觀察結束，計算分數中"
    return "normal", ""

def observe_session_market(session="eu"):
    """
    觀察開盤走勢，計算額外評分 (-2 ~ +2)
    邏輯：看 BTC 在觀察期間的漲跌幅
    """
    try:
        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        price  = float(ticker['last'])
        pct    = float(ticker.get('percentage', 0) or 0)

        with SESSION_LOCK:
            obs_key_map = {"eu": "europe_obs", "us": "america_obs", "europe": "europe_obs", "america": "america_obs"}
            score_key_map = {"eu": "eu_score", "us": "us_score", "europe": "eu_score", "america": "us_score"}
            date_key_map = {"eu": "eu_score_date", "us": "us_score_date", "europe": "eu_score_date", "america": "us_score_date"}
            time_key_map = {"eu": "eu_score_time", "us": "us_score_time", "europe": "eu_score_time", "america": "us_score_time"}
            key = obs_key_map.get(session, "{}_obs".format(session))
            SESSION_STATE.setdefault(key, [])
            SESSION_STATE[key].append(price)
            # 只保留最近20筆
            if len(SESSION_STATE[key]) > 20:
                SESSION_STATE[key] = SESSION_STATE[key][-20:]

            prices = SESSION_STATE[key]
            if len(prices) < 2:
                return

            # 計算觀察期間漲跌
            first_price = prices[0]
            last_price  = prices[-1]
            change_pct  = (last_price - first_price) / first_price * 100

            # 評分邏輯
            if change_pct > 1.5:
                score = 2; note = "{}盤強勢上漲{:.1f}% +2分".format(
                    "歐洲" if session=="eu" else "美洲", change_pct)
            elif change_pct > 0.5:
                score = 1; note = "{}盤小幅上漲{:.1f}% +1分".format(
                    "歐洲" if session=="eu" else "美洲", change_pct)
            elif change_pct < -1.5:
                score = -2; note = "{}盤強勢下跌{:.1f}% -2分".format(
                    "歐洲" if session=="eu" else "美洲", abs(change_pct))
            elif change_pct < -0.5:
                score = -1; note = "{}盤小幅下跌{:.1f}% -1分".format(
                    "歐洲" if session=="eu" else "美洲", abs(change_pct))
            else:
                score = 0; note = "{}盤橫盤 0分".format(
                    "歐洲" if session=="eu" else "美洲")

            score_key = score_key_map.get(session, "{}_score".format(session))
            date_key  = date_key_map.get(session, "{}_score_date".format(session))
            time_key  = time_key_map.get(session, "{}_score_time".format(session))
            SESSION_STATE[score_key] = score
            SESSION_STATE[date_key]  = tw_today()        # 記錄台灣日期
            SESSION_STATE[time_key]  = tw_now_str("%H:%M")  # 記錄台灣時間
            SESSION_STATE["session_note"] = note
            print("📊 {}盤觀察: {} | BTC {:.2f}% | 分數有效至明日2點".format(
                "歐洲" if session=="eu" else "美洲", note, change_pct))

            # 同步到 STATE 給 UI 顯示
            update_state(session_info={
                "phase":    SESSION_STATE["session_phase"],
                "note":     note,
                "eu_score": SESSION_STATE["eu_score"],
                "us_score": SESSION_STATE["us_score"],
                "eu_time":  SESSION_STATE.get("eu_score_time",""),
                "us_time":  SESSION_STATE.get("us_score_time",""),
            })
    except Exception as e:
        print("觀察市場失敗: {}".format(e))

def get_session_score():
    """取得當前時段的額外評分，隔天2點後自動清零"""
    with SESSION_LOCK:
        now_tw   = get_tw_time()
        today    = tw_today()
        now_h    = now_tw.hour

        # 每天凌晨2點後清零（隔天重置）
        for sess in ["eu", "us"]:
            score_date = SESSION_STATE.get("{}_score_date".format(sess), "")
            score_val  = SESSION_STATE.get("{}_score".format(sess), 0)
            if score_val != 0 and score_date:
                # 如果是昨天的分數且現在超過2點 → 清零
                if score_date < today and now_h >= 2:
                    SESSION_STATE["{}_score".format(sess)]      = 0
                    SESSION_STATE["{}_score_date".format(sess)] = ""
                    SESSION_STATE["{}_score_time".format(sess)] = ""
                    print("🔄 {}盤分數已過期（{}），清零".format(
                        "歐洲" if sess=="eu" else "美洲", score_date))

        return SESSION_STATE.get("eu_score", 0) + SESSION_STATE.get("us_score", 0)

def session_monitor_thread():
    """時段監控執行緒"""
    prev_status = "normal"
    closed_positions = False

    while True:
        try:
            status, note = get_session_status()
            tw = get_tw_time()

            with SESSION_LOCK:
                SESSION_STATE["session_phase"] = status

            # 歐盤觀察期：每2分鐘記錄一次價格
            if status == "eu_closed":
                observe_session_market("eu")
                closed_positions = False

            # 美盤觀察期：每2分鐘記錄一次價格
            elif status == "us_closed":
                observe_session_market("us")
                closed_positions = False

            # 歐盤觀察結束，重置歐盤記錄
            elif status == "eu_watch_end" and prev_status == "eu_closed":
                with SESSION_LOCK:
                    SESSION_STATE["europe_obs"] = []
                print("✅ 歐盤觀察結束，分數:{:+d}，恢復交易".format(
                    SESSION_STATE["eu_score"]))

            # 美盤觀察結束，重置美盤記錄
            elif status == "us_watch_end" and prev_status == "us_closed":
                with SESSION_LOCK:
                    SESSION_STATE["america_obs"] = []
                print("✅ 美盤觀察結束，分數:{:+d}，恢復交易".format(
                    SESSION_STATE["us_score"]))

            # 恢復正常時重置分數
            elif status == "normal" and prev_status not in ("normal", "eu_pause", "us_pause"):
                with SESSION_LOCK:
                    SESSION_STATE["eu_score"]  = 0
                    SESSION_STATE["us_score"]  = 0
                    SESSION_STATE["europe_obs"] = []
                    SESSION_STATE["america_obs"] = []
                print("🔄 時段重置，分數歸零")

            # 平倉邏輯（19:50 和 21:50）
            if status in ("eu_closed", "us_closed") and not closed_positions:
                tw_min = tw.hour * 60 + tw.minute
                if tw_min in range(19*60+50, 19*60+53) or tw_min in range(21*60+50, 21*60+53):
                    print("🔔 開盤保護：縮倉/平倉處理現有持倉")
                    try:
                        positions = exchange.fetch_positions()
                        for p in positions:
                            contracts = float(p.get('contracts', 0) or 0)
                            if abs(contracts) <= 0:
                                continue
                            tighten_position_for_session(
                                p['symbol'],
                                contracts,
                                (p.get('side') or '').lower(),
                                float(p.get('entryPrice', 0) or 0),
                                float(p.get('markPrice', 0) or 0),
                            )
                    except Exception as se:
                        print("開盤保護倉位處理失敗: {}".format(se))
                    closed_positions = True

            prev_status = status
            update_state(session_info={
                "phase": status,
                "note":  note,
                "eu_score": SESSION_STATE.get("eu_score", 0),
                "us_score": SESSION_STATE.get("us_score", 0),
            })

        except Exception as e:
            print("時段監控失敗: {}".format(e))
        time.sleep(120)  # 每2分鐘檢查一次

# =====================================================
# 大盤走勢分析系統（BTC 日線 + 歷史型態對比）
# =====================================================
MARKET_STATE = {
    "pattern":      "初始化中",
    "direction":    "中性",
    "score":        0,
    "strength":     0.0,
    "detail":       "",
    "history_match": "",   # 歷史相似型態
    "prediction":   "",    # 預測走勢
    "last_update":  "",
    "btc_price":    0.0,
    "btc_change":   0.0,
    "long_term_pos": None, # 長期倉位狀態
}
MARKET_LOCK = threading.Lock()

def find_similar_history(df, current_window=30, top_n=3):
    """
    在 BTC 歷史日線中找最相似的K線型態
    用標準化後的收盤價序列做相似度比對（歐幾里得距離）
    """
    try:
        closes = df['c'].values.astype(float)
        n = len(closes)
        if n < current_window + 30:
            return []

        # 取最近 current_window 根K棒作為當前型態
        current = closes[-(current_window):]
        # 標準化（0-1 範圍）
        c_min, c_max = current.min(), current.max()
        if c_max == c_min:
            return []
        current_norm = (current - c_min) / (c_max - c_min)

        similarities = []
        # 從歷史中滑動比對（至少保留100根後續K棒用來看結果）
        search_end = n - current_window - 30
        for i in range(0, search_end - current_window, 5):  # 每5根跳一格
            window = closes[i:i+current_window]
            w_min, w_max = window.min(), window.max()
            if w_max == w_min:
                continue
            window_norm = (window - w_min) / (w_max - w_min)

            # 計算相似度（1 - 標準化距離）
            dist = np.sqrt(np.mean((current_norm - window_norm)**2))
            similarity = max(0, 1 - dist * 2)  # 0~1，越高越相似

            if similarity > 0.55:  # 放寬到55%（Bitget日線有限）
                # 看這個時間點之後30根的漲跌
                future = closes[i+current_window:i+current_window+30]
                if len(future) >= 10:
                    future_ret = (future[-1] - future[0]) / future[0] * 100
                    # 取得日期（用索引反推）
                    similarities.append({
                        'idx': i,
                        'similarity': round(similarity * 100, 1),
                        'future_ret': round(future_ret, 1),
                        'entry_price': round(closes[i+current_window-1], 0),
                    })

        # 按相似度排序取前N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_n]
    except Exception as e:
        print("歷史比對失敗:", e)
        return []

def analyze_btc_market_trend():
    """
    分析 BTC 日線走勢，識別當前型態並對比歷史
    回傳詳細分析結果
    """
    try:

        # 抓 BTC 日線 - 抓最多500根做歷史比對（約1.5年）
        ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", "1d", limit=1000)  # 盡量多抓
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])

        # 均線
        df['ma7']  = df['c'].rolling(7).mean()
        df['ma25'] = df['c'].rolling(25).mean()
        df['ma50'] = df['c'].rolling(50).mean()
        df['ma99'] = df['c'].rolling(min(99,len(df))).mean()

        curr  = float(df['c'].iloc[-1])
        ma7   = float(df['ma7'].iloc[-1])
        ma25  = float(df['ma25'].iloc[-1])
        ma50  = float(df['ma50'].iloc[-1])
        ma99  = float(df['ma99'].iloc[-1])
        prev  = float(df['c'].iloc[-2])
        change_pct = (curr - prev) / prev * 100

        # 近期高低點
        high_30 = float(df['h'].tail(30).max())
        low_30  = float(df['l'].tail(30).min())
        high_7  = float(df['h'].tail(7).max())
        low_7   = float(df['l'].tail(7).min())
        range_30 = high_30 - low_30

        # ATR
        atr_s = ta.atr(df['h'], df['l'], df['c'], length=14)
        atr   = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else curr*0.02

        # 趨勢斜率（線性回歸）
        c7  = df['c'].tail(7).values
        c14 = df['c'].tail(14).values
        c30 = df['c'].tail(30).values
        x7  = np.arange(len(c7));  slope_7  = np.polyfit(x7,  c7,  1)[0]/curr*100
        x14 = np.arange(len(c14)); slope_14 = np.polyfit(x14, c14, 1)[0]/curr*100
        x30 = np.arange(len(c30)); slope_30 = np.polyfit(x30, c30, 1)[0]/curr*100

        # 成交量趨勢
        vol_7  = float(df['v'].tail(7).mean())
        vol_30 = float(df['v'].tail(30).mean())
        vol_ratio = vol_7 / max(vol_30, 1)

        # ══════════════════════════
        # 型態識別
        # ══════════════════════════
        pattern = ""; direction = "中性"; score = 0; strength = 0.5
        detail = ""; history_match = ""; prediction = ""

        # 1. 均線完全多頭排列
        if curr > ma7 > ma25 > ma50 > ma99:
            pattern = "均線完全多頭排列"
            direction = "強多"
            score = 5; strength = 0.9
            detail = "BTC 日線4條均線完全多頭排列，大盤處於強勢牛市結構"
            history_match = "歷史對比：2020年10月、2021年2月、2024年2月出現相同結構"
            prediction = "短期：維持多頭，回調買入 | 中期：若量配合可創新高 | 建議：做多為主"

        # 2. 均線完全空頭排列
        elif curr < ma7 < ma25 < ma50 < ma99:
            pattern = "均線完全空頭排列"
            direction = "強空"
            score = -5; strength = 0.9
            detail = "BTC 日線4條均線完全空頭排列，大盤處於弱勢熊市結構"
            history_match = "歷史對比：2018年下半年、2022年5-11月出現相同結構"
            prediction = "短期：反彈做空，不宜追多 | 中期：底部不明謹慎 | 建議：做空反彈嚴控損"

        # 3. 突破近期高點（牛市突破）
        elif curr > high_30 * 0.99 and slope_7 > 0.3:
            pattern = "突破近期30日高點"
            direction = "多"
            score = 4; strength = 0.8
            detail = "BTC 正突破近30日高點 {:.0f}，突破型態看多".format(high_30)
            history_match = "歷史對比：突破型態後續上漲概率約65-70%"
            prediction = "短期：多頭動能強，觀察能否站穩高點 | 建議：做多，止損高點下方ATR×1.5"

        # 4. 跌破近期低點（熊市跌破）
        elif curr < low_30 * 1.01 and slope_7 < -0.3:
            pattern = "跌破近期30日低點"
            direction = "空"
            score = -4; strength = 0.8
            detail = "BTC 跌破近30日低點 {:.0f}，跌破型態看空".format(low_30)
            history_match = "歷史對比：跌破低點後續下跌概率約60-65%"
            prediction = "短期：空頭動能強，避免抄底 | 建議：輕倉做空或觀望"

        # 5. 均線糾纏（盤整）
        elif abs(curr - ma25) / curr < 0.02 and abs(slope_14) < 0.15:
            pattern = "均線糾纏盤整"
            direction = "中性"
            score = 0; strength = 0.3
            detail = "BTC 均線糾纏，市場方向不明，處於盤整階段"
            history_match = "歷史對比：盤整後突破方向決定下一波趨勢"
            prediction = "等待方向選擇，突破做多/跌破做空 | 建議：降低倉位等待突破"

        # 6. 多頭回調（主升浪中回調）
        elif curr > ma50 and curr < ma25 and slope_30 > 0.2:
            pattern = "多頭主升浪中回調"
            direction = "多"
            score = 3; strength = 0.7
            detail = "BTC 在長期上升趨勢中回調至MA25附近，健康回調"
            history_match = "歷史對比：主升浪回調通常是買入機會（回調幅度10-20%）"
            prediction = "回調結束信號：日線收復MA25 | 建議：分批做多，止損MA50下方"

        # 7. 死貓彈（熊市中反彈）
        elif curr < ma50 and slope_7 > 0.5 and slope_30 < -0.1:
            pattern = "熊市中死貓彈反彈"
            direction = "空"
            score = -3; strength = 0.7
            detail = "BTC 在長期下降趨勢中出現反彈，小心死貓彈"
            history_match = "歷史對比：熊市反彈通常在MA50下方夭折"
            prediction = "反彈目標：MA25-MA50之間 | 建議：反彈做空，不追高"

        # 8. 成交量萎縮橫盤
        elif abs(slope_7) < 0.1 and vol_ratio < 0.7:
            pattern = "縮量橫盤"
            direction = "中性"
            score = 0; strength = 0.2
            detail = "BTC 成交量萎縮、價格橫盤，市場觀望情緒濃厚"
            history_match = "歷史對比：縮量橫盤後通常有一波較大行情"
            prediction = "蓄勢待發方向未定 | 建議：等待放量突破後跟進"

        # 預設：弱多/弱空
        else:
            if slope_14 > 0:
                pattern = "弱多趨勢"
                direction = "多"
                score = 2; strength = 0.4
                detail = "BTC 日線斜率向上，弱多格局"
                prediction = "趨勢偏多但力道不強，謹慎做多"
            else:
                pattern = "弱空趨勢"
                direction = "空"
                score = -2; strength = 0.4
                detail = "BTC 日線斜率向下，弱空格局"
                prediction = "趨勢偏空但力道不強，謹慎操作"

        # 真實歷史相似度比對
        similar_cases = find_similar_history(df, current_window=20, top_n=3)
        if similar_cases:
            hist_lines = []
            bull_count = sum(1 for s in similar_cases if s['future_ret'] > 2)
            bear_count = sum(1 for s in similar_cases if s['future_ret'] < -2)
            for s in similar_cases:
                trend = "📈+{:.1f}%".format(s['future_ret']) if s['future_ret'] > 0 else "📉{:.1f}%".format(s['future_ret'])
                hist_lines.append("相似度{}% → 後續30日{}".format(s['similarity'], trend))
            hist_conclusion = "歷史{}次相似型態：{}看多 / {}看空".format(
                len(similar_cases), bull_count, bear_count)
            history_match = hist_conclusion + " | " + " | ".join(hist_lines)
        else:
            # 數據不足時用型態文字描述
            history_match = history_match or "相似度不足（近期走勢較特殊，無高度相似歷史）"

        return {
            "pattern": pattern,
            "direction": direction,
            "score": score,
            "strength": strength,
            "detail": detail,
            "history_match": history_match,
            "prediction": prediction,
            "btc_price": round(curr, 2),
            "btc_change": round(change_pct, 2),
            "ma7": round(ma7, 2),
            "ma25": round(ma25, 2),
            "ma50": round(ma50, 2),
            "slope_7": round(slope_7, 3),
            "slope_30": round(slope_30, 3),
            "vol_ratio": round(vol_ratio, 2),
            "last_update": tw_now_str(),
        }
    except Exception as e:
        print("大盤分析失敗: {}".format(e))
        return None

def market_analysis_thread():
    """每小時更新一次大盤分析"""
    print("大盤分析執行緒啟動")
    time.sleep(20)  # 等掃描執行緒先啟動
    while True:
        try:
            result = analyze_btc_market_trend()
            if result:
                with MARKET_LOCK:
                    MARKET_STATE.update(result)
                update_state(market_info=result)
                print("📊 大盤分析: {} | {} | BTC {:.0f} ({:+.1f}%)".format(
                    result["pattern"], result["direction"],
                    result["btc_price"], result["btc_change"]))
                # 自動管理長期倉位
                check_long_term_position()
        except Exception as e:
            print("大盤分析執行緒錯誤: {}".format(e))
        time.sleep(3600)  # 每小時更新

# =====================================================
# 長期倉位系統（獨立於短線7個倉位之外）
# =====================================================
LT_STATE = {
    "position": None,   # None / "long" / "short"
    "entry_price": 0.0,
    "entry_time": "",
    "symbol": "BTC/USDT:USDT",
    "contracts": 0.0,
    "unrealized_pnl": 0.0,
    "leverage": 5,      # 長期倉位用低槓桿
    "note": "",
}
LT_LOCK = threading.Lock()

# =====================================================
# FVG 限價掛單追蹤系統
# =====================================================
FVG_ORDERS = {}   # { symbol: { order_id, side, price, score, sl, tp, placed_time, support, resist } }
FVG_LOCK   = threading.Lock()

def register_fvg_order(symbol, order_id, side, price, score, sl, tp, support, resist):
    """登記一筆 FVG 限價掛單"""
    with FVG_LOCK:
        if symbol in FVG_ORDERS:
            print("⚠️ FVG防重複：{} 已有掛單，跳過".format(symbol))
            return False
        FVG_ORDERS[symbol] = {
            "order_id":    order_id,
            "side":        side,
            "price":       price,
            "score":       score,
            "sl":          sl,
            "tp":          tp,
            "support":     support,
            "resist":      resist,
            "placed_time": tw_now_str("%H:%M:%S"),
            "created_ts":  time.time(),
            "curr_price":   price,
            "curr_score":   score,
            "status":      "掛單中",
        }
        print("📌 FVG掛單登記: {} {} @{:.6f}".format(symbol, side, price))
        update_state(fvg_orders=dict(FVG_ORDERS))
        return True

def cancel_fvg_order(symbol, reason=""):
    """取消並登出一筆 FVG 掛單"""
    with FVG_LOCK:
        if symbol not in FVG_ORDERS:
            return
        order = FVG_ORDERS.pop(symbol)
    try:
        exchange.cancel_order(order["order_id"], symbol)
        print("🗑 FVG掛單取消: {} | 原因: {}".format(symbol, reason))
    except Exception as e:
        print("FVG取消失敗(可能已成交): {}".format(e))
    update_state(fvg_orders=dict(FVG_ORDERS))

def fvg_order_monitor_thread():
    """
    FVG 限價掛單追蹤執行緒（每30秒檢查一次）
    - 掛單已成交 → 登出，正常追蹤止盈止損
    - 分數反轉或跌破支撐/突破壓力 → 取消掛單
    - 掛單超過4小時未成交 → 取消
    """
    print("FVG掛單追蹤執行緒啟動")
    while True:
        try:
            with FVG_LOCK:
                syms = list(FVG_ORDERS.keys())

            for symbol in syms:
                try:
                    with FVG_LOCK:
                        if symbol not in FVG_ORDERS:
                            continue
                        order = FVG_ORDERS[symbol]

                    # 查詢掛單狀態
                    try:
                        od = exchange.fetch_order(order["order_id"], symbol)
                        status = od.get("status", "")
                    except:
                        status = "unknown"

                    # 已成交 → 登出
                    if status in ("closed", "filled"):
                        with FVG_LOCK:
                            FVG_ORDERS.pop(symbol, None)
                        print("✅ FVG掛單成交: {} @{}".format(symbol, order["price"]))
                        update_state(fvg_orders=dict(FVG_ORDERS))
                        continue

                    # 已取消 → 清除
                    if status == "canceled":
                        with FVG_LOCK:
                            FVG_ORDERS.pop(symbol, None)
                        update_state(fvg_orders=dict(FVG_ORDERS))
                        continue

                    # 抓當前價格做判斷
                    ticker = exchange.fetch_ticker(symbol)
                    curr   = float(ticker['last'])

                    # 重新分析分數（看是否反轉）
                    sc = extract_analysis_score(analyze(symbol))

                    cancel_reason = None

                    # 分數低於30或反轉 → 取消
                    if order["side"] == "long" and sc < 30:
                        cancel_reason = "做多分數不足{}（<30），取消掛單".format(round(sc,1))
                    elif order["side"] == "short" and sc > -30:
                        cancel_reason = "做空分數不足{}（>-30），取消掛單".format(round(sc,1))

                    # 跌破支撐（做多掛單）→ 取消
                    elif order["side"] == "long" and order["support"] > 0:
                        if curr < order["support"] * 0.998:
                            cancel_reason = "跌破支撐{:.4f}，取消做多掛單".format(order["support"])

                    # 突破壓力（做空掛單）→ 取消
                    elif order["side"] == "short" and order["resist"] > 0:
                        if curr > order["resist"] * 1.002:
                            cancel_reason = "突破壓力{:.4f}，取消做空掛單".format(order["resist"])

                    # 超過4小時 → 取消
                    placed = order.get("placed_time", "")
                    if placed:
                        try:
                            # 簡單判斷：掛單超過240分鐘
                            h_now = int(tw_now_str("%H")) * 60 + int(tw_now_str("%M"))
                            h_placed = int(placed[:2]) * 60 + int(placed[3:5])
                            diff = (h_now - h_placed) % (24*60)
                            if diff > 240:
                                cancel_reason = "掛單超過4小時，自動取消"
                        except:
                            pass

                    if cancel_reason:
                        cancel_fvg_order(symbol, cancel_reason)
                    else:
                        # 更新狀態顯示
                        with FVG_LOCK:
                            if symbol in FVG_ORDERS:
                                FVG_ORDERS[symbol]["curr_price"] = round(curr, 6)
                                FVG_ORDERS[symbol]["curr_score"] = round(sc, 1)
                                FVG_ORDERS[symbol]["status"] = "掛單中 | 現價{:.4f} | 分數{}".format(curr, round(sc,1))
                        update_state(fvg_orders=dict(FVG_ORDERS))

                except Exception as e:
                    print("FVG追蹤{}錯誤: {}".format(symbol, e))

        except Exception as e:
            print("FVG追蹤執行緒錯誤: {}".format(e))
        time.sleep(30)  # 每30秒檢查一次

def open_long_term_position(direction, reason=""):
    """開長期倉位（BTC，低槓桿5x，5%資產）"""
    try:
        with LT_LOCK:
            if LT_STATE["position"] is not None:
                print("長期倉位已存在，跳過")
                return False

        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        price  = float(ticker['last'])
        with STATE_LOCK:
            equity = STATE.get("equity", 100)
        usdt   = equity * 0.05          # 用5%資產
        lev    = LT_STATE["leverage"]

        # 設定槓桿
        try:
            exchange.set_leverage(lev, "BTC/USDT:USDT")
        except:
            pass

        contracts = round(usdt * lev / price, 4)
        side = "buy" if direction == "long" else "sell"

        order = exchange.create_order(
            "BTC/USDT:USDT", "market", side, contracts,
            params={"tdMode": "cross"}
        )

        with LT_LOCK:
            LT_STATE["position"]    = direction
            LT_STATE["entry_price"] = price
            LT_STATE["entry_time"]  = tw_now_str("%Y-%m-%d %H:%M")
            LT_STATE["contracts"]   = contracts
            LT_STATE["note"]        = reason

        print("✅ 長期{}倉開倉 BTC {:.2f} | {}張 | 原因:{}".format(
            "多" if direction=="long" else "空",
            price, contracts, reason))
        return True
    except Exception as e:
        print("長期倉位開倉失敗: {}".format(e))
        return False

def close_long_term_position(reason=""):
    """平長期倉位"""
    try:
        with LT_LOCK:
            if LT_STATE["position"] is None:
                return False
            side      = LT_STATE["position"]
            contracts = LT_STATE["contracts"]

        close_side = "sell" if side == "long" else "buy"
        exchange.create_order(
            "BTC/USDT:USDT", "market", close_side, abs(contracts),
            params={"tdMode": "cross", "reduceOnly": True}
        )

        with LT_LOCK:
            entry = LT_STATE["entry_price"]
            LT_STATE["position"]  = None
            LT_STATE["contracts"] = 0.0

        ticker = exchange.fetch_ticker("BTC/USDT:USDT")
        curr   = float(ticker['last'])
        pnl    = (curr - entry) / entry * 100 if side=="long" else (entry - curr) / entry * 100
        print("📤 長期倉位平倉 | 損益:{:+.2f}% | 原因:{}".format(pnl, reason))
        return True
    except Exception as e:
        print("長期倉位平倉失敗: {}".format(e))
        return False

def check_long_term_position():
    """每小時由大盤分析執行緒呼叫，根據大盤方向管理長期倉位"""
    with MARKET_LOCK:
        direction  = MARKET_STATE.get("direction", "中性")
        strength   = MARKET_STATE.get("strength", 0)
        pattern    = MARKET_STATE.get("pattern", "")
        prediction = MARKET_STATE.get("prediction", "")

    with LT_LOCK:
        curr_pos = LT_STATE["position"]

    # 強度不夠不動
    if strength < 0.6:
        print("⏸ 大盤強度不足({:.1f})，長期倉位維持現狀".format(strength))
        return

    # 強多 → 開多/維持多
    if direction in ("強多", "多") and curr_pos != "long":
        if curr_pos == "short":
            close_long_term_position("方向轉多，平空倉")
        open_long_term_position("long", "{} | {}".format(pattern, prediction[:30]))

    # 強空 → 開空/維持空
    elif direction in ("強空", "空") and curr_pos != "short":
        if curr_pos == "long":
            close_long_term_position("方向轉空，平多倉")
        open_long_term_position("short", "{} | {}".format(pattern, prediction[:30]))

    # 中性 → 平倉觀望
    elif direction == "中性" and curr_pos is not None:
        close_long_term_position("大盤中性，觀望")

def check_risk_ok():
    """回傳 (可否下單, 原因)（不用鎖，避免死鎖）"""
    try:
        rs = RISK_STATE  # 直接讀
        today = tw_today()

        # 新的一天重置日虧損
        if rs["today_date"] != today:
            rs["today_date"]         = today
            rs["daily_loss_usdt"]    = 0.0
            rs["daily_start_equity"] = STATE.get("equity", 0)
            rs["trading_halted"]     = False
            rs["halt_reason"]        = ""
            print("新的一天，重置風控狀態")

        # 冷靜期檢查
        if rs["cooldown_until"] and datetime.now() < rs["cooldown_until"]:
            remaining = int((rs["cooldown_until"] - datetime.now()).total_seconds() / 60)
            return False, "連損冷靜期，剩餘 {} 分鐘".format(remaining)

        # 已人工停止
        if rs["trading_halted"]:
            return False, rs["halt_reason"]

        # 日虧損上限
        equity = STATE.get("equity", 0)
        if equity > 0 and rs["daily_start_equity"] > 0:
            # 用即時總資產計算虧損%（不是單筆累加）
            equity_loss_pct = (rs["daily_start_equity"] - equity) / rs["daily_start_equity"]
            if equity_loss_pct >= MAX_DAILY_LOSS_PCT:
                rs["trading_halted"] = True
                rs["halt_reason"] = "總資產虧損已達 {:.1f}%，停止交易".format(equity_loss_pct*100)
                return False, rs["halt_reason"]

        return True, "正常"
    except Exception as e:
        print("check_risk_ok 錯誤: {}".format(e))
        return True, "正常"

def record_trade_result(pnl_usdt):
    """每筆平倉後呼叫，更新風控狀態"""
    with RISK_LOCK:
        rs = RISK_STATE
        if pnl_usdt < 0:
            rs["daily_loss_usdt"]  += abs(pnl_usdt)
            rs["consecutive_loss"] += 1
            if rs["consecutive_loss"] >= MAX_CONSECUTIVE_LOSS:
                from datetime import timedelta
                rs["cooldown_until"] = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
                rs["consecutive_loss"] = 0
                print("連續虧損 {} 單，進入冷靜期 {} 分鐘".format(
                    MAX_CONSECUTIVE_LOSS, COOLDOWN_MINUTES))
        else:
            rs["consecutive_loss"] = 0  # 勝利重置連損計數

def get_risk_status():
    """給 UI 顯示用（不用鎖，避免死鎖）"""
    try:
        rs = RISK_STATE  # 直接讀，不加鎖
        ok = not rs.get("trading_halted", False)
        if rs.get("cooldown_until") and datetime.now() < rs["cooldown_until"]:
            ok = False
        equity = STATE.get("equity", 1)
        start_eq = rs.get("daily_start_equity", equity) or equity
        return {
            "trading_ok":        ok,
            "halt_reason":       rs.get("halt_reason", ""),
            "consecutive_loss":  rs.get("consecutive_loss", 0),
            "daily_loss_usdt":   round(rs.get("daily_loss_usdt", 0), 2),
            "daily_loss_pct":    round((start_eq - equity) / max(start_eq, 1) * 100, 1) if equity > 0 else 0,
            "max_daily_loss_pct": int(MAX_DAILY_LOSS_PCT * 100),
            "cooldown_until":    rs["cooldown_until"].strftime("%H:%M") if rs.get("cooldown_until") and datetime.now() < rs["cooldown_until"] else None,
            "current_threshold": _DT.get("current", 50),
        }
    except Exception as e:
        return {"trading_ok": True, "halt_reason": "", "consecutive_loss": 0,
                "daily_loss_usdt": 0, "daily_loss_pct": 0, "max_daily_loss_pct": 15}



def _position_drawdown_pct(pos):
    try:
        entry = float(pos.get('entryPrice',0) or 0)
        mark = float(pos.get('markPrice',0) or 0)
        side = str(pos.get('side','') or '').lower()
        if entry <= 0 or mark <= 0:
            return 0.0
        if side == 'long':
            return round(max((entry - mark) / entry * 100.0, 0.0), 2)
        return round(max((mark - entry) / entry * 100.0, 0.0), 2)
    except Exception:
        return 0.0


def _position_leveraged_pnl_pct(pos):
    try:
        p = pos.get('percentage', None)
        if p is not None:
            return round(float(p or 0), 2)
    except Exception:
        pass
    try:
        dd = _position_drawdown_pct(pos)
        lev = float(pos.get('leverage',1) or 1)
        side = str(pos.get('side','') or '').lower()
        entry = float(pos.get('entryPrice',0) or 0)
        mark = float(pos.get('markPrice',0) or 0)
        favorable = (mark >= entry) if side == 'long' else (mark <= entry)
        signed = dd * lev
        return round(signed if favorable else -signed, 2)
    except Exception:
        return 0.0


def _manual_release_risk_state():
    with RISK_LOCK:
        RISK_STATE['trading_halted'] = False
        RISK_STATE['halt_reason'] = ''
        RISK_STATE['cooldown_until'] = None
        RISK_STATE['consecutive_loss'] = 0
    update_state(risk_status=get_risk_status())
    return {'ok': True, 'message': '已手動解除風控暫停'}

# =====================================================
# 評分權重（滿分100）
# =====================================================
# =====================================================
# 指標權重（根據2025量化研究最佳化，滿分100）
# 類別分配：
#   趨勢確認(37): EMA+MACD+ADX+趨勢線
#   價格結構(29): 壓力支撐+OB機構
#   流量/動能(18): 流動性+成交量+VWAP
#   動能振盪(8):  RSI+KD
#   情境濾網(8):  K棒+圖形+新聞
# =====================================================


def _position_drawdown_pct(pos):
    try:
        entry = float(pos.get('entryPrice',0) or 0)
        mark = float(pos.get('markPrice',0) or 0)
        side = str(pos.get('side','') or '').lower()
        if entry <= 0 or mark <= 0:
            return 0.0
        if side == 'long':
            return round(max((entry - mark) / entry * 100.0, 0.0), 2)
        return round(max((mark - entry) / entry * 100.0, 0.0), 2)
    except Exception:
        return 0.0


def _position_leveraged_pnl_pct(pos):
    try:
        p = pos.get('percentage', None)
        if p is not None:
            return round(float(p or 0), 2)
    except Exception:
        pass
    try:
        dd = _position_drawdown_pct(pos)
        lev = float(pos.get('leverage',1) or 1)
        side = str(pos.get('side','') or '').lower()
        entry = float(pos.get('entryPrice',0) or 0)
        mark = float(pos.get('markPrice',0) or 0)
        favorable = (mark >= entry) if side == 'long' else (mark <= entry)
        signed = dd * lev
        return round(signed if favorable else -signed, 2)
    except Exception:
        return 0.0


def _manual_release_risk_state():
    with RISK_LOCK:
        RISK_STATE['trading_halted'] = False
        RISK_STATE['halt_reason'] = ''
        RISK_STATE['cooldown_until'] = None
        RISK_STATE['consecutive_loss'] = 0
    update_state(risk_status=get_risk_status())
    return {'ok': True, 'message': '已手動解除風控暫停'}

# =====================================================
# 評分權重 - 同類指標共享總分預算，有幾個就除幾個
# =====================================================
# 類別預算分配（研究依據：ICT/SMC + 量化研究）
# 趨勢類  22分：最重要，避免逆勢交易
# 結構類  22分：OB/壓力支撐是機構進出核心
# ICT類   20分：BOS/CHoCH/掃單是現代量化核心
# 動量類  14分：確認動能，非主導
# 量能類  12分：資金流向驗證
# 新聞類  10分：宏觀情緒濾網

# 各類別 → 指標清單（重組後）
# 改動說明：
#   KD 移除（與RSI高度重複，浪費5分）
#   新聞 10→2分（API不穩，不應主導評分）
#   chart_pat 降為獨立3分（低觸發率）
#   多時框確認 新增14分（15m+4H+日線一致，最重要的勝率來源）
_W_CAT = {
    "trend":     (22, ["ema_trend", "trendline", "adx"]),   # 22/3=7分each
    "structure": (19, ["support_res", "order_block"]),       # 移除chart_pat共享，OB+SR各9/10
    "ict":       (20, ["bos_choch", "liq_sweep", "candle", "fvg"]), # 20/4=5分each
    "mtf":       (14, ["mtf_confirm"]),                      # ★新增：多時框方向一致14分
    "momentum":  (10, ["macd", "rsi"]),                      # 移除KD，各5分
    "volume":    (12, ["vwap", "whale"]),                    # 各6分
    "chart":     (3,  ["chart_pat"]),                        # 降到3分（低觸發率）
    "news_cat":  (2,  ["news"]),                             # 降到2分（不穩定）
}

W = {}
for cat, (budget, inds) in _W_CAT.items():
    per = round(budget / len(inds))
    for ind in inds:
        W[ind] = per

# 微調讓總分剛好100
_total = sum(W.values())
if _total != 100:
    W["support_res"] += (100 - _total)

assert sum(W.values()) == 100, "權重總和{}不等於100".format(sum(W.values()))

# 新聞單獨計分
NEWS_WEIGHT = W.get("news", 2)


# =====================================================
# 學習資料庫
# =====================================================
def load_learn_db():
    try:
        with open(LEARN_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            "trades": [],
            "pattern_stats": {},
            "symbol_stats": {},     # 每個幣的勝率統計
            "atr_params": {"default_sl": 2.0, "default_tp": 3.5},
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
        }

def save_learn_db(db):
    try:
        dir1 = os.path.dirname(LEARN_DB_PATH)
        if dir1: os.makedirs(dir1, exist_ok=True)
        with open(LEARN_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("學習DB儲存失敗: {}".format(e))


def load_backtest_db():
    try:
        with open(BACKTEST_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"runs": [], "summary": {}, "latest": {}}

def save_backtest_db(db):
    try:
        dir1 = os.path.dirname(BACKTEST_DB_PATH)
        if dir1: os.makedirs(dir1, exist_ok=True)
        with open(BACKTEST_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("回測DB儲存失敗: {}".format(e))


def _is_live_source(src):
    s = str(src or '').lower()
    return s.startswith('live')

def get_live_trades(closed_only=False):
    with LEARN_LOCK:
        trades = list(LEARN_DB.get("trades", []) or [])
    rows = [t for t in trades if _is_live_source(t.get("source"))]
    if closed_only:
        rows = [t for t in rows if t.get("result") in ("win", "loss")]
    return rows

def _trade_learn_metric(trade):
    """AI learning metric: use account impact when available, otherwise fallback safely.
    - learn_pnl_pct: preferred persistent field
    - account_pnl_pct: leverage * margin normalized account impact
    - pnl_pct: legacy fallback
    """
    if not isinstance(trade, dict):
        return 0.0
    for k in ("learn_pnl_pct", "account_pnl_pct", "pnl_pct"):
        v = trade.get(k, None)
        if v is not None:
            try:
                return float(v or 0.0)
            except Exception:
                pass
    return 0.0

def _trade_edge_pct(trade):
    """Pure market edge without leverage. Useful for debug and strategy analysis."""
    if not isinstance(trade, dict):
        return 0.0
    for k in ("edge_pct", "raw_pnl_pct", "pnl_pct"):
        v = trade.get(k, None)
        if v is not None:
            try:
                return float(v or 0.0)
            except Exception:
                pass
    return 0.0

def _live_trade_stats(symbol=None, regime=None):
    rows = get_live_trades(closed_only=True)
    if symbol:
        rows = [t for t in rows if str(t.get("symbol")) == str(symbol)]
    if regime:
        rows = [t for t in rows if str((t.get("breakdown") or {}).get("Regime", "neutral")) == str(regime)]
    if not rows:
        return {"count": 0, "win_rate": 0.0, "avg_pnl": 0.0, "ev_per_trade": 0.0, "profit_factor": None, "max_drawdown_pct": None, "std_pnl": None}
    pnls = [_trade_learn_metric(t) for t in rows]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    cnt = len(pnls)
    wr = (len(wins) / max(cnt, 1)) * 100.0
    avg = sum(pnls) / max(cnt, 1)
    pf = (sum(wins) / max(sum(losses), 1e-9)) if losses else (999.0 if wins else None)

    # v38：用 100 基準淨值曲線計算回撤，避免小樣本時出現 3000%+ 假回撤
    equity = 100.0
    peak = 100.0
    max_dd = 0.0
    for p in pnls:
        step = max(0.01, 1.0 + (float(p) / 100.0))
        equity *= step
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100.0)
    max_dd = min(max_dd, 100.0)

    mean = avg
    std = (sum((p - mean) ** 2 for p in pnls) / max(cnt, 1)) ** 0.5
    return {
        "count": cnt,
        "win_rate": round(wr, 2),
        "avg_pnl": round(avg, 4),
        "ev_per_trade": round(avg, 4),
        "profit_factor": None if pf is None else round(float(pf), 3),
        "max_drawdown_pct": round(float(max_dd), 3),
        "std_pnl": round(float(std), 4),
    }

def _ai_confidence_from_live(stats):
    cnt = int(stats.get("count", 0) or 0)
    std = float(stats.get("std_pnl", 0) or 0)
    base = min(cnt / 50.0, 1.0)
    stability = max(0.0, 1.0 - min(std / 3.0, 1.0))
    return round(base * stability, 3)

def _ai_status_from_live(stats):
    cnt = int(stats.get("count", 0) or 0)
    pf = stats.get("profit_factor", None)
    ev = float(stats.get("ev_per_trade", 0) or 0)
    dd = float(stats.get("max_drawdown_pct", 0) or 0)
    wr = float(stats.get("win_rate", 0) or 0)
    avg = float(stats.get("avg_pnl", 0) or 0)
    # v38：三階段，不讓 AI 太早接管
    if cnt < 20:
        return "warmup"
    if cnt < 50:
        return "observe"
    if (pf is not None and float(pf) < 0.95) or ev <= 0 or dd > 35 or wr < 42 or avg <= -0.25:
        return "reject"
    return "valid"

def _normalize_setup_mode(setup=''):
    s = str(setup or '')
    if ('突破' in s) or ('爆發' in s) or ('news' in s.lower()) or ('breakout' in s.lower()):
        return 'breakout'
    if ('回踩' in s) or ('續攻' in s) or ('延續' in s) or ('反彈續跌' in s):
        return 'trend'
    if ('區間' in s) or ('震盪' in s) or ('range' in s.lower()):
        return 'range'
    return 'main'

def _regime_setup_fit(regime='neutral', setup=''):
    mode = _normalize_setup_mode(setup)
    regime = str(regime or 'neutral')
    if regime == 'range':
        if mode in ('trend', 'breakout'):
            return False, '震盪市場不追趨勢/突破'
        return True, '震盪市場配區間打法'
    if regime in ('news', 'breakout'):
        if mode == 'range':
            return False, '爆發盤不做區間反向'
        return True, '爆發盤允許突破/趨勢'
    # neutral / trend-like
    if mode == 'range':
        return False, '趨勢/中性盤不優先做區間逆勢'
    return True, '市場與策略相容'

def _symbol_hard_block(symbol=''):
    rows = [t for t in get_live_trades(closed_only=True) if str(t.get('symbol')) == str(symbol)]
    cnt = len(rows)
    if cnt < 12:
        return False, ''
    wins = sum(1 for t in rows if t.get('result') == 'win')
    wr = wins / max(cnt, 1)
    avg = sum(_trade_learn_metric(t) for t in rows) / max(cnt, 1)
    if wr < 0.35 and avg < 0:
        return True, '實單長期虧損，封鎖幣種'
    return False, ''

def _ai_warmup_mode():
    return len(get_live_trades(closed_only=True)) < 20

def save_full_state():
    """備份移動止盈狀態到硬碟"""
    try:
        dir2 = os.path.dirname(STATE_BACKUP_PATH)
        if dir2: os.makedirs(dir2, exist_ok=True)
        with TRAILING_LOCK:
            trail_copy = {k: {
                "side": v.get("side"),
                "entry_price": v.get("entry_price"),
                "highest_price": v.get("highest_price"),
                "lowest_price": v.get("lowest_price"),
                "trail_pct": v.get("trail_pct"),
                "initial_sl": v.get("initial_sl"),
                "atr": v.get("atr"),
            } for k, v in TRAILING_STATE.items()}
        backup = {
            "trailing_state": trail_copy,
            "threshold": {"current": _DT.get("current", 40)},
            "timestamp": datetime.now().isoformat()
        }
        with open(STATE_BACKUP_PATH, 'w', encoding='utf-8') as f:
            json.dump(backup, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("狀態備份失敗: {}".format(e))

def load_full_state():
    """從硬碟恢復移動止盈狀態"""
    global ORDER_THRESHOLD
    try:
        if not os.path.exists(STATE_BACKUP_PATH):
            print("⚠️ 無狀態備份，從頭開始")
            return
        with open(STATE_BACKUP_PATH, 'r', encoding='utf-8') as f:
            backup = json.load(f)
        with TRAILING_LOCK:
            for sym, ts in backup.get("trailing_state", {}).items():
                TRAILING_STATE[sym] = ts
        thresh = backup.get("threshold", {}).get("current", 38)
        # 重新部署後門檻重置為預設38，不沿用上次的高門檻
        thresh = min(thresh, ORDER_THRESHOLD_DEFAULT)
        with _DT_LOCK:
            _DT["current"] = thresh
        ORDER_THRESHOLD = thresh
        print("✅ 狀態已從備份恢復，門檻:{}".format(thresh))
    except FileNotFoundError:
        print("⚠️ 無狀態備份，從頭開始")
    except Exception as e:
        print("狀態恢復失敗: {}".format(e))

def save_risk_state():
    """備份風控狀態"""
    try:
        dir3 = os.path.dirname(RISK_STATE_PATH)
        if dir3: os.makedirs(dir3, exist_ok=True)
        with open(RISK_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump({
                "today_date": RISK_STATE.get("today_date", ""),
                "daily_loss_usdt": RISK_STATE.get("daily_loss_usdt", 0),
                "consecutive_loss": RISK_STATE.get("consecutive_loss", 0),
                "trading_halted": RISK_STATE.get("trading_halted", False),
                "halt_reason": RISK_STATE.get("halt_reason", ""),
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("風控備份失敗: {}".format(e))

def load_risk_state():
    """從硬碟恢復風控狀態"""
    try:
        if not os.path.exists(RISK_STATE_PATH):
            print("⚠️ 無風控備份，從頭開始")
            return
        with open(RISK_STATE_PATH, 'r', encoding='utf-8') as f:
            backup = json.load(f)
        # 只恢復今天的資料
        today = tw_today()
        if backup.get("today_date") == today:
            with RISK_LOCK:
                RISK_STATE["today_date"]      = today
                RISK_STATE["daily_loss_usdt"] = backup.get("daily_loss_usdt", 0)
                RISK_STATE["consecutive_loss"]= backup.get("consecutive_loss", 0)
                RISK_STATE["trading_halted"]  = backup.get("trading_halted", False)
                RISK_STATE["halt_reason"]     = backup.get("halt_reason", "")
            print("✅ 風控狀態已恢復（今日虧損:{:.2f}U）".format(
                backup.get("daily_loss_usdt", 0)))
        else:
            print("⚠️ 風控備份是昨天的，重置")
    except FileNotFoundError:
        print("⚠️ 無風控備份，從頭開始")
    except Exception as e:
        print("風控恢復失敗: {}".format(e))

LEARN_DB   = load_learn_db()
BACKTEST_DB = load_backtest_db()
LEARN_LOCK = threading.Lock()

# =====================================================
# 全域狀態
# =====================================================
STATE = {
    "news_score":        0,
    "latest_news_title": "等待新聞...",
    "news_sentiment":    "初始化中",
    "top_signals":       [],
    "active_positions":  [],
    "scan_progress":     "啟動中，約需 2 分鐘完成首輪掃描...",
    "trade_history":     [],
    "total_pnl":         0.0,
    "equity":            0.0,   # 帳戶總資產（即時）
    "last_update":       "--",
    "scan_count":        0,
    "halt_reason":       "",     # 風控停止原因
    "risk_status":       {},     # 風控狀態摘要
    "trailing_info":     {},     # 移動止盈追蹤狀態（給UI顯示）
    "session_info":      {"phase":"normal","note":"","eu_score":0,"us_score":0},
    "market_info":       {"pattern":"初始化中","direction":"中性","btc_price":0,"prediction":""},
    "lt_info":           {"position":None,"entry_price":0,"pnl":0,"pattern":"","prediction":""},
    "fvg_orders":        {},
    "threshold_info":    {"current": 48, "phase": "預設"},  # 動態門檻資訊
    "auto_order_audit":  {},
    "protection_state":  {},
    "learn_summary": {
        "total_trades":    0,
        "win_rate":        0.0,
        "avg_pnl":         0.0,
        "current_sl_mult": 2.0,
        "current_tp_mult": 3.0,
        "top_patterns":    [],
        "worst_patterns":  [],
        "blocked_symbols": [],  # 勝率 < 40% 的幣，觀察中
    }
}
STATE_LOCK = threading.Lock()

def update_state(**kwargs):
    with STATE_LOCK:
        STATE.update(kwargs)

def smooth_signal_score(symbol, raw_score):
    prev = SCORE_CACHE.get(symbol, raw_score)
    stable = prev * (1 - SCORE_SMOOTH_ALPHA) + raw_score * SCORE_SMOOTH_ALPHA
    if abs(raw_score) >= 70 and abs(prev) < 45:
        stable = prev * 0.55 + raw_score * 0.45
    SCORE_CACHE[symbol] = stable
    return round(stable, 2)

def score_jump_alert(symbol, raw_score, stable_score):
    prev = SCORE_CACHE.get(symbol, stable_score)
    delta = raw_score - prev
    if abs(delta) >= 25:
        return '分數快變 {:.1f}'.format(delta)
    return ''

def can_reenter_symbol(symbol):
    ts = ENTRY_LOCKS.get(symbol, 0)
    return (time.time() - ts) >= ENTRY_LOCK_SEC

def touch_entry_lock(symbol):
    ENTRY_LOCKS[symbol] = time.time()

def fetch_real_atr(symbol, timeframe='15m', limit=60):
    try:
        d = pd.DataFrame(exchange.fetch_ohlcv(symbol, timeframe, limit=limit), columns=['t','o','h','l','c','v'])
        atr = ta.atr(d['h'], d['l'], d['c'], length=14)
        val = safe_last(atr, 0)
        if val > 0:
            return float(val)
    except Exception as e:
        print('fetch_real_atr失敗 {}: {}'.format(symbol, e))
    return 0.0

def verify_protection_orders(symbol, side, sl_price, tp_price):
    side = (side or '').lower()
    try:
        orders = exchange.fetch_open_orders(symbol)
    except Exception as e:
        print('查詢保護單失敗 {}: {}'.format(symbol, e))
        orders = []
    sl_ok = False
    tp_ok = False
    sl_keys = ['stop', 'stoploss', 'loss', 'sl']
    tp_keys = ['takeprofit', 'profit', 'tp']
    for o in orders:
        text = json.dumps(o, ensure_ascii=False).lower()
        if not sl_ok and any(k in text for k in sl_keys):
            sl_ok = True
        if not tp_ok and any(k in text for k in tp_keys):
            tp_ok = True
    PROTECTION_STATE[symbol] = {
        'sl_ok': sl_ok,
        'tp_ok': tp_ok,
        'sl': round(float(sl_price or 0), 8),
        'tp': round(float(tp_price or 0), 8),
        'side': side,
        'updated_at': tw_now_str(),
    }
    update_state(protection_state=dict(PROTECTION_STATE))
    return sl_ok, tp_ok


def ensure_exchange_protection(sym, side, pos_side, qty, sl_price, tp_price, verify_wait_sec=1.0):
    """下主單後立即補掛交易所保護單；若止損驗證失敗，回傳 sl_ok=False。"""
    sl_side = 'sell' if side == 'buy' else 'buy'
    qty = float(qty or 0)
    sl_ok = False
    tp_ok = False

    if qty <= 0:
        PROTECTION_STATE[sym] = {
            'sl_ok': False, 'tp_ok': False, 'sl': round(float(sl_price or 0), 8),
            'tp': round(float(tp_price or 0), 8), 'side': (side or '').lower(),
            'updated_at': tw_now_str(), 'note': 'qty<=0，未掛保護單'
        }
        update_state(protection_state=dict(PROTECTION_STATE))
        return False, False

    # 止損單（三種格式依序嘗試）
    try:
        exchange.create_order(sym, 'market', sl_side, qty, params={
            'reduceOnly':  True,
            'stopPrice':   str(sl_price),
            'orderType':   'stop',
            'posSide':     pos_side,
            'tdMode':      'cross',
        })
        print("止損單成功(格式1): {} @{}".format(sym, sl_price))
        sl_ok = True
    except Exception:
        pass

    if not sl_ok:
        try:
            exchange.create_order(sym, 'market', sl_side, qty, params={
                'reduceOnly':    True,
                'stopLossPrice': str(sl_price),
                'posSide':       pos_side,
                'tdMode':        'cross',
            })
            print("止損單成功(格式2): {} @{}".format(sym, sl_price))
            sl_ok = True
        except Exception:
            pass

    if not sl_ok:
        try:
            exchange.create_order(sym, 'market', sl_side, qty, params={
                'reduceOnly':   True,
                'triggerPrice': str(sl_price),
                'triggerType':  'mark_price',
                'posSide':      pos_side,
            })
            print("止損單成功(格式3): {} @{}".format(sym, sl_price))
            sl_ok = True
        except Exception as e3:
            print("止損三種格式都失敗: {}".format(e3))

    # 止盈單（兩種格式依序嘗試）
    try:
        exchange.create_order(sym, 'market', sl_side, qty, params={
            'reduceOnly':  True,
            'stopPrice':   str(tp_price),
            'orderType':   'takeProfit',
            'posSide':     pos_side,
            'tdMode':      'cross',
        })
        print("止盈單成功(格式1): {} @{}".format(sym, tp_price))
        tp_ok = True
    except Exception:
        pass

    if not tp_ok:
        try:
            exchange.create_order(sym, 'market', sl_side, qty, params={
                'reduceOnly':      True,
                'takeProfitPrice': str(tp_price),
                'posSide':         pos_side,
                'tdMode':          'cross',
            })
            print("止盈單成功(格式2): {} @{}".format(sym, tp_price))
            tp_ok = True
        except Exception as tp_err:
            print("止盈掛單失敗，依賴移動止盈系統: {}".format(tp_err))

    # 以交易所開放掛單再次驗證，避免 create_order 成功但其實沒掛上
    try:
        time.sleep(max(float(verify_wait_sec), 0.2))
    except Exception:
        pass
    v_sl_ok, v_tp_ok = verify_protection_orders(sym, side, sl_price, tp_price)
    sl_ok = bool(sl_ok or v_sl_ok)
    tp_ok = bool(tp_ok or v_tp_ok)
    PROTECTION_STATE[sym] = {
        'sl_ok': sl_ok,
        'tp_ok': tp_ok,
        'sl': round(float(sl_price or 0), 8),
        'tp': round(float(tp_price or 0), 8),
        'side': (side or '').lower(),
        'updated_at': tw_now_str(),
        'note': '交易所止損已驗證' if sl_ok else '交易所止損驗證失敗',
    }
    update_state(protection_state=dict(PROTECTION_STATE))
    return sl_ok, tp_ok


PENDING_LEARN_IDS = set()

def _parse_time_to_ms(s):
    try:
        return int(pd.Timestamp(str(s)).timestamp() * 1000)
    except Exception:
        return None

def resolve_exchange_exit_fill(symbol, entry_side=None, entry_time=None):
    """
    嘗試從交易所最近成交中還原真正平倉價，避免 TP/SL 由交易所觸發時學習沒有記錄。
    回傳: {exit_price, realized_pnl_usdt, fill_side, info}
    """
    result = {
        'exit_price': None,
        'realized_pnl_usdt': None,
        'fill_side': None,
        'info': '',
    }
    try:
        close_side = 'sell' if str(entry_side or '').lower() in ('buy', 'long') else 'buy'
        since_ms = _parse_time_to_ms(entry_time)
        candidates = []

        try:
            trades = exchange.fetch_my_trades(symbol, since=since_ms, limit=30)
        except Exception:
            trades = []

        for tr in trades or []:
            raw = json.dumps(tr, ensure_ascii=False).lower()
            side = str(tr.get('side') or '').lower()
            if side and side != close_side:
                continue
            if 'open' in raw and 'close' not in raw and 'reduce' not in raw:
                continue
            ts = tr.get('timestamp') or 0
            price = tr.get('price')
            if price is None:
                continue
            pnl = tr.get('realizedPnl')
            if pnl is None:
                info = tr.get('info') or {}
                for key in ('realizedPnl', 'achievedProfits', 'profit', 'closeProfit'):
                    if isinstance(info, dict) and info.get(key) is not None:
                        pnl = info.get(key)
                        break
            candidates.append((ts, float(price), float(pnl or 0), side, 'my_trades'))

        try:
            orders = exchange.fetch_closed_orders(symbol, since=since_ms, limit=20)
        except Exception:
            orders = []

        for od in orders or []:
            raw = json.dumps(od, ensure_ascii=False).lower()
            side = str(od.get('side') or '').lower()
            if side and side != close_side:
                continue
            if not any(k in raw for k in ('reduce', 'close', 'stop', 'tp', 'sl', 'profit', 'loss')):
                continue
            price = od.get('average') or od.get('price') or od.get('stopPrice')
            if price is None:
                continue
            ts = od.get('lastTradeTimestamp') or od.get('timestamp') or 0
            candidates.append((ts, float(price), None, side, 'closed_orders'))

        if candidates:
            candidates.sort(key=lambda x: x[0] or 0, reverse=True)
            ts, px, pnl, side, src = candidates[0]
            result.update({
                'exit_price': px,
                'realized_pnl_usdt': pnl,
                'fill_side': side,
                'info': src,
            })
    except Exception as e:
        print('resolve_exchange_exit_fill失敗 {}: {}'.format(symbol, e))
    return result


def queue_learn_for_closed_symbol(sym, active_syms=None):
    """
    補強：不管是機器人手動平倉，還是交易所 TP/SL 觸發，只要倉位已消失就補記學習。
    """
    try:
        if active_syms and sym in active_syms:
            return False

        with LEARN_LOCK:
            open_trade = None
            for t in reversed(LEARN_DB.get('trades', [])):
                if t.get('symbol') == sym and t.get('result') == 'open':
                    open_trade = t
                    break
            if not open_trade:
                return False
            trade_id = open_trade.get('id')
            if trade_id in PENDING_LEARN_IDS:
                return False

        fill = resolve_exchange_exit_fill(sym, open_trade.get('side'), open_trade.get('entry_time'))
        exit_price = fill.get('exit_price')
        realized_pnl_usdt = fill.get('realized_pnl_usdt')

        if exit_price is None:
            try:
                ticker = exchange.fetch_ticker(sym)
                exit_price = float(ticker.get('last') or 0)
            except Exception:
                exit_price = 0

        with LEARN_LOCK:
            for t in LEARN_DB.get('trades', []):
                if t.get('id') == trade_id and t.get('result') == 'open':
                    if exit_price:
                        t['exit_price'] = exit_price
                    if realized_pnl_usdt is not None:
                        t['realized_pnl_usdt'] = realized_pnl_usdt
                    break
            save_learn_db(LEARN_DB)
            PENDING_LEARN_IDS.add(trade_id)

        print('偵測到平倉: {}，開始學習分析... exit_price={} source={}'.format(sym, exit_price, fill.get('info') or 'ticker'))
        threading.Thread(target=learn_from_closed_trade, args=(trade_id,), daemon=True).start()
        return True
    except Exception as e:
        print('queue_learn_for_closed_symbol失敗 {}: {}'.format(sym, e))
        return False



def _resolve_backtest_symbol(symbol=None):
    s = str(symbol or '').strip()
    if s and s.lower() not in ('auto', 'best', 'ai'):
        return s
    try:
        with AI_LOCK:
            rows = list((AUTO_BACKTEST_STATE.get('results') or []))
        if rows:
            return rows[0].get('symbol') or 'BTC/USDT:USDT'
    except Exception:
        pass
    try:
        with STATE_LOCK:
            sigs = list((STATE.get('top_signals') or []))
        if sigs:
            return sigs[0].get('symbol') or 'BTC/USDT:USDT'
    except Exception:
        pass
    return 'BTC/USDT:USDT'


def _normalize_wr_percent(value):
    try:
        x = float(value or 0)
    except Exception:
        return 0.0
    return x * 100.0 if 0 <= x <= 1 else x




def _ai_strategy_profile(symbol, regime='neutral', setup=''):
    """v34: 真 AI 接管。回測只做候選，接管只吃實單。"""
    strategy_key = f'{regime}|{setup}|{symbol}'
    setup_mode = _normalize_setup_mode(setup)
    profile = {
        'ready': False,
        'source': 'live_only',
        'sample_count': 0,
        'win_rate': 0.0,
        'avg_pnl': 0.0,
        'ev_per_trade': 0.0,
        'profit_factor': None,
        'max_drawdown_pct': None,
        'threshold_adjust': 0.0,
        'hard_block': False,
        'strategy': strategy_key,
        'strategy_mode': setup_mode,
        'note': 'AI樣本不足，沿用主策略（回測僅供候選）',
        'confidence': 0.0,
        'status': 'warmup',
        'symbol_blocked': False,
    }
    try:
        stats = _live_trade_stats(symbol=symbol, regime=regime)
        symbol_stats = _live_trade_stats(symbol=symbol, regime=None)
        cnt = int(stats.get('count', 0) or 0)
        wr = float(stats.get('win_rate', 0) or 0)
        avg = float(stats.get('avg_pnl', 0) or 0)
        ev = float(stats.get('ev_per_trade', 0) or 0)
        pf = stats.get('profit_factor', None)
        dd = stats.get('max_drawdown_pct', None)
        conf = _ai_confidence_from_live(stats)
        status = _ai_status_from_live(stats)
        fit_ok, fit_note = _regime_setup_fit(regime, setup)
        sym_block, sym_note = _symbol_hard_block(symbol)

        profile.update({
            'sample_count': cnt,
            'win_rate': wr,
            'avg_pnl': avg,
            'ev_per_trade': ev,
            'profit_factor': pf,
            'max_drawdown_pct': dd,
            'confidence': conf,
            'status': status,
            'ready': (
                not sym_block and (
                    (status == 'valid' and conf >= 0.25 and cnt >= 50) or
                    (status == 'observe' and cnt >= 35 and conf >= 0.18)
                )
            ),
            'symbol_blocked': sym_block,
        })

        notes = []
        if sym_block:
            profile['hard_block'] = True
            profile['threshold_adjust'] = 999.0
            notes.append(sym_note or '幣種長期虧損封鎖')
        elif status == 'reject':
            profile['hard_block'] = False
            profile['threshold_adjust'] = 4.5
            notes.append('AI弱勢策略，升高門檻觀察')
            if pf is not None:
                notes.append(f'PF偏弱 {float(pf):.2f}')
            notes.append(f'EV偏弱 {ev:+.4f}')
        elif status == 'warmup':
            profile['threshold_adjust'] = -6.0
            notes.append('探索模式')
            notes.append(f'樣本 {cnt}/20')
            notes.append('前20單優先累積實單')
        elif status == 'observe':
            profile['threshold_adjust'] = -1.5 if fit_ok else 1.0
            notes.append('半接管模式')
            notes.append(f'樣本 {cnt}/50')
            if not fit_ok:
                notes.append(fit_note)
        else:
            th_adj = 0.0
            if pf is not None and pf >= 1.35:
                th_adj -= 2.0
            elif pf is not None and pf < 1.08:
                th_adj += 3.0
            if ev >= 0.12:
                th_adj -= 1.5
            elif ev <= 0:
                th_adj += 2.5
            if wr >= 58:
                th_adj -= 1.0
            elif wr < 42:
                th_adj += 1.8
            if dd is not None and dd >= 10:
                th_adj += 2.0
            elif dd is not None and dd <= 3:
                th_adj -= 0.5
            if not fit_ok:
                th_adj += 2.5
                notes.append(fit_note)
            else:
                notes.append(fit_note)
            if setup_mode == 'breakout':
                th_adj += 1.0
                notes.append('爆發策略保守過濾')
            elif setup_mode == 'range':
                th_adj += 0.5 if regime != 'range' else -0.5
                notes.append('區間策略')
            else:
                notes.append('趨勢主策略')
            scnt = int(symbol_stats.get('count', 0) or 0)
            swr = float(symbol_stats.get('win_rate', 0) or 0)
            if scnt >= 8 and swr < 45:
                th_adj += 1.5
                notes.append('幣種實單偏弱')
            profile['threshold_adjust'] = round(th_adj, 2)

        notes.append(f'EV/筆 {ev:+.4f}')
        if pf is not None:
            notes.append(f'PF {float(pf):.2f}')
        if dd is not None:
            notes.append(f'DD {float(dd):.2f}%')
        notes.append(f'信心 {conf*100:.0f}%')
        profile['note'] = '｜'.join(notes)
    except Exception as e:
        profile['note'] = f'AI策略讀取失敗:{str(e)[:40]}'
    return profile


def ai_decide_trade(sig, eff_threshold, mkt_ok, side_ok, same_dir_cnt, pos_syms, already_closing):
    symbol = str(sig.get('symbol') or '')
    score = abs(float(sig.get('score', 0) or 0))
    rr = float(sig.get('rr_ratio', 0) or 0)
    eq = float(sig.get('entry_quality', 0) or 0)
    bd = dict(sig.get('breakdown') or {})
    regime = str(bd.get('Regime', 'neutral') or 'neutral')
    setup = str(sig.get('setup_label') or bd.get('Setup', '') or '')
    profile = _ai_strategy_profile(symbol, regime=regime, setup=setup)

    global_live_count = len(get_live_trades(closed_only=True))
    if global_live_count < 20:
        phase = 'learning'
    elif global_live_count < 50:
        phase = 'semi'
    else:
        phase = 'full'

    base_threshold = float(eff_threshold)
    fit_ok, fit_note = _regime_setup_fit(regime, setup)
    mode = str(profile.get('strategy_mode') or 'main')

    if phase == 'learning':
        ai_threshold = max(40.0, base_threshold - 6.0)
        rr_floor = max(1.05, MIN_RR_HARD_FLOOR - 0.15)
        min_entry_quality = 1.5 if score >= ai_threshold else 2.0
    elif phase == 'semi':
        ai_threshold = max(46.0, base_threshold + float(profile.get('threshold_adjust', 0) or 0))
        rr_floor = max(1.20, MIN_RR_HARD_FLOOR)
        min_entry_quality = 2.2
        if mode == 'breakout':
            min_entry_quality = max(min_entry_quality, 2.6)
    else:
        ai_threshold = max(48.0, base_threshold + float(profile.get('threshold_adjust', 0) or 0))
        rr_floor = max(1.35, MIN_RR_HARD_FLOOR)
        min_entry_quality = 2.6
        if mode == 'breakout':
            min_entry_quality = max(min_entry_quality, 3.0)
        if regime == 'range':
            min_entry_quality = max(min_entry_quality, 2.8)

    base_ok = (
        score >= ai_threshold
        and eq >= min_entry_quality
        and rr >= rr_floor
        and side_ok
        and symbol not in pos_syms
        and symbol not in already_closing
        and symbol not in SHORT_TERM_EXCLUDED
        and sig.get('allowed', True)
        and mkt_ok
        and same_dir_cnt < MAX_SAME_DIRECTION
        and can_reenter_symbol(symbol)
    )

    ai_ok = True
    reasons = []
    reasons.append({'learning': '探索模式', 'semi': '半接管', 'full': 'AI真接管'}[phase])
    reasons.append('回測只供候選排序')
    reasons.append(f'全域樣本 {global_live_count}')

    if phase == 'learning':
        reasons.append('前20單優先累積實單')
        if profile.get('sample_count') is not None:
            reasons.append('AI樣本{}'.format(profile.get('sample_count', 0)))
        if profile.get('note'):
            reasons.append(str(profile.get('note')))
    elif phase == 'semi':
        if bool(profile.get('symbol_blocked')):
            ai_ok = False
            reasons.append('幣種長期虧損封鎖')
        elif int(profile.get('sample_count', 0) or 0) >= 20 and float(profile.get('avg_pnl', 0) or 0) <= 0 and float(profile.get('win_rate', 0) or 0) < 45:
            ai_ok = False
            reasons.append('半接管封鎖虧損策略')
        elif not fit_ok and mode in ('breakout', 'trend'):
            ai_ok = False
            reasons.append(fit_note)
        if profile.get('sample_count') is not None:
            reasons.append('AI樣本{}'.format(profile.get('sample_count', 0)))
        if profile.get('note'):
            reasons.append(str(profile.get('note')))
    else:
        hard_block = bool(profile.get('hard_block'))
        if not fit_ok:
            hard_block = True
            reasons.append(fit_note)
        if int(profile.get('sample_count', 0) or 0) < 20 and score < max(ai_threshold + 5.0, 58.0):
            hard_block = True
            reasons.append('局部樣本不足')
        ai_ok = not hard_block
        if profile.get('ready'):
            reasons.append('AI策略已就緒')
        else:
            reasons.append('AI未完全就緒，維持保守')
        if profile.get('sample_count') is not None:
            reasons.append('AI樣本{}'.format(profile.get('sample_count', 0)))
        if profile.get('note'):
            reasons.append(str(profile.get('note')))

    if not base_ok:
        if score < ai_threshold:
            reasons.append('分數未過AI門檻')
        if eq < min_entry_quality:
            reasons.append('進場品質不足')
        if rr < rr_floor:
            reasons.append('RR不足')
    return {
        'allow_now': bool(base_ok and ai_ok),
        'effective_threshold': round(ai_threshold, 2),
        'reasons': list(dict.fromkeys(reasons)),
        'profile': dict(profile, phase=phase),
    }

def build_auto_order_reason(sig, eff_threshold, mkt_ok, side_ok, same_dir_cnt, pos_syms, already_closing, ai_decision=None):
    reasons = []
    eq = float(sig.get('entry_quality', 0) or 0)
    rr = float(sig.get('rr_ratio', 0) or 0)
    score = abs(float(sig.get('score', 0) or 0))
    min_entry_quality = 2 if score >= max(eff_threshold + 4, 54) else 3
    if score < eff_threshold:
        reasons.append('分數未過門檻')
    if eq < min_entry_quality:
        reasons.append('進場品質不足')
    if rr < MIN_RR_HARD_FLOOR:
        reasons.append('RR不足')
    if not side_ok:
        reasons.append('方向衝突')
    if sig['symbol'] in pos_syms:
        reasons.append('已有持倉')
    if sig['symbol'] in already_closing:
        reasons.append('反向平倉中')
    if sig['symbol'] in SHORT_TERM_EXCLUDED:
        reasons.append('短線排除名單')
    if not sig.get('allowed', True):
        reasons.append('歷史勝率封鎖')
    if not mkt_ok:
        reasons.append('大盤方向不符')
    if same_dir_cnt >= MAX_SAME_DIRECTION:
        reasons.append('同向持倉已滿')
    if not can_reenter_symbol(sig['symbol']):
        reasons.append('進場冷卻中')
    if ai_decision:
        profile = dict(ai_decision.get('profile') or {})
        if profile.get('ready'):
            reasons.append('AI門檻 {}'.format(ai_decision.get('effective_threshold')))
            reasons.append('AI樣本 {}'.format(profile.get('sample_count', 0)))
            reasons.append('AI勝率 {:.1f}%'.format(float(profile.get('win_rate', 0) or 0)))
            if profile.get('hard_block'):
                reasons.append('AI封鎖此策略')
            note = profile.get('note')
            if note:
                reasons.append(str(note))
    return list(dict.fromkeys(reasons))

def safe_last(series, default=0):
    try:
        v = series.iloc[-1]
        return float(v) if v == v else default
    except:
        return default

# =====================================================
# 預判暴拉 / 暴跌前的蓄勢結構 + 追價風險偵測
# =====================================================
def _linreg_slope(values):
    try:
        arr = np.array(list(values), dtype=float)
        if len(arr) < 3:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        slope = np.polyfit(x, arr, 1)[0]
        return float(slope)
    except:
        return 0.0

def analyze_pre_breakout_setup(d15, d4h):
    """
    找「還沒噴、但已經在蓄勢」的結構：
    - 波動收斂（BB寬度/ATR縮小）
    - 靠近區間高/低點
    - 高低點逐步抬高/壓低
    - 4H 主趨勢同向
    """
    try:
        if len(d15) < 60 or len(d4h) < 30:
            return 0, "蓄勢數據不足"

        c = d15['c'].astype(float)
        h = d15['h'].astype(float)
        l = d15['l'].astype(float)
        v = d15['v'].astype(float)
        curr = float(c.iloc[-1])
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)

        bb = ta.bbands(c, length=20, std=2)
        if bb is None or bb.empty:
            return 0, "蓄勢無BB"
        bb_up = safe_last(bb.iloc[:, 0], curr)
        bb_mid = safe_last(bb.iloc[:, 1], curr)
        bb_low = safe_last(bb.iloc[:, 2], curr)
        bb_width_now = max((bb_up - bb_low) / max(bb_mid, 1e-9), 0)
        width_hist = ((bb.iloc[:, 0] - bb.iloc[:, 2]) / bb.iloc[:, 1].replace(0, np.nan)).dropna().tail(40)
        width_med = float(width_hist.median()) if len(width_hist) else bb_width_now

        atr_series = ta.atr(h, l, c, length=14)
        atr_now = safe_last(atr_series, atr)
        atr_recent = float(pd.Series(atr_series).tail(8).mean()) if atr_series is not None else atr_now
        atr_prev = float(pd.Series(atr_series).tail(32).head(16).mean()) if atr_series is not None else atr_now
        atr_prev = atr_prev if atr_prev and atr_prev == atr_prev else atr_now

        range_high = float(h.tail(BREAKOUT_LOOKBACK).iloc[:-1].max())
        range_low = float(l.tail(BREAKOUT_LOOKBACK).iloc[:-1].min())
        near_high = (range_high - curr) / max(atr_now, 1e-9) <= 0.45 and curr <= range_high * 1.003
        near_low = (curr - range_low) / max(atr_now, 1e-9) <= 0.45 and curr >= range_low * 0.997

        lows_slope = _linreg_slope(l.tail(6).tolist())
        highs_slope = _linreg_slope(h.tail(6).tolist())

        vol_recent = float(v.tail(4).mean())
        vol_prev = float(v.tail(24).head(12).mean()) if len(v) >= 24 else vol_recent
        vol_expand = vol_recent > vol_prev * 1.08 if vol_prev > 0 else False

        ema21_4h = safe_last(ta.ema(d4h['c'], length=21), curr)
        ema55_4h = safe_last(ta.ema(d4h['c'], length=55), curr)
        trend_up = curr > ema21_4h > ema55_4h
        trend_dn = curr < ema21_4h < ema55_4h

        squeeze = bb_width_now < width_med * 0.88 and atr_recent < atr_prev * 0.9
        score = 0
        tags = []

        if squeeze and near_high and lows_slope > 0 and trend_up:
            score += 6
            tags.append("收斂逼近前高")
            if vol_expand:
                score += 2
                tags.append("量能悄悄放大")
        elif squeeze and near_low and highs_slope < 0 and trend_dn:
            score -= 6
            tags.append("收斂逼近前低")
            if vol_expand:
                score -= 2
                tags.append("量能悄悄放大")

        # 假突破前的吸收：價格很接近前高/前低，但尚未大幅穿越
        last_body = abs(float(c.iloc[-1]) - float(d15['o'].iloc[-1]))
        if near_high and trend_up and last_body < atr_now * 0.75 and curr <= range_high * 1.0015:
            score += 1
            tags.append("上沿吸收中")
        elif near_low and trend_dn and last_body < atr_now * 0.75 and curr >= range_low * 0.9985:
            score -= 1
            tags.append("下沿吸收中")

        score = max(min(score, 8), -8)
        return score, "|".join(tags) if tags else "無明顯蓄勢"
    except Exception:
        return 0, "蓄勢分析失敗"

def analyze_extension_risk(d15, direction_hint=0):
    """避免追漲殺跌：已離均線太遠 + 連續單邊衝刺時直接降權。"""
    try:
        c = d15['c'].astype(float)
        o = d15['o'].astype(float)
        h = d15['h'].astype(float)
        l = d15['l'].astype(float)
        curr = float(c.iloc[-1])
        ema20 = safe_last(ta.ema(c, length=20), curr)
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        bb = ta.bbands(c, length=20, std=2)
        bb_up = safe_last(bb.iloc[:, 0], curr) if bb is not None and not bb.empty else curr
        bb_low = safe_last(bb.iloc[:, 2], curr) if bb is not None and not bb.empty else curr
        ext = (curr - ema20) / max(atr, 1e-9)
        bull3 = all(c.iloc[-i] > o.iloc[-i] for i in [1,2,3])
        bear3 = all(c.iloc[-i] < o.iloc[-i] for i in [1,2,3])

        if direction_hint >= 0 and ext > ANTI_CHASE_ATR and curr >= bb_up * 0.995 and bull3:
            penalty = -10 if ext > 1.9 else -7
            return penalty, "多頭過度延伸，避免追高"
        if direction_hint <= 0 and ext < -ANTI_CHASE_ATR and curr <= bb_low * 1.005 and bear3:
            penalty = 10 if ext < -1.9 else 7
            return penalty, "空頭過度延伸，避免追空"
        return 0, "延伸正常"
    except Exception:
        return 0, "延伸分析失敗"

def get_breakout_pullback_entry(symbol, side, current_price, atr):
    """
    追價保護：已經離均線/區間太遠時，不直接市價，改等回踩/反彈。
    """
    try:
        df = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=80), columns=['t','o','h','l','c','v'])
        if df.empty or len(df) < 30:
            return None, "pullback資料不足"
        c = df['c'].astype(float)
        o = df['o'].astype(float)
        h = df['h'].astype(float)
        l = df['l'].astype(float)
        curr = float(current_price or c.iloc[-1])
        atr = max(float(atr or 0), curr * 0.003)
        ema20 = safe_last(ta.ema(c, length=20), curr)
        hh = float(h.tail(BREAKOUT_LOOKBACK).iloc[:-1].max())
        ll = float(l.tail(BREAKOUT_LOOKBACK).iloc[:-1].min())
        last_body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
        ext = (curr - ema20) / max(atr, 1e-9)

        if side == 'long':
            breakout_now = curr >= hh * 0.999 and last_body > atr * 0.7
            if ext > ANTI_CHASE_ATR or breakout_now:
                limit_price = max(ema20, hh - atr * PULLBACK_BUFFER_ATR)
                if limit_price < curr * 0.999:
                    return round(limit_price, 6), "追高保護：改等回踩再多"
        else:
            breakout_now = curr <= ll * 1.001 and last_body > atr * 0.7
            if ext < -ANTI_CHASE_ATR or breakout_now:
                limit_price = min(ema20, ll + atr * PULLBACK_BUFFER_ATR)
                if limit_price > curr * 1.001:
                    return round(limit_price, 6), "追空保護：改等反彈再空"
        return None, "延伸正常"
    except Exception:
        return None, "pullback計算失敗"

# =====================================================
# 統計濾網：該幣勝率是否達標
# =====================================================
def is_symbol_allowed(symbol):
    """若該幣有 >=7 筆紀錄且勝率 <40%，封鎖下單，改為觀察"""
    with LEARN_LOCK:
        st = LEARN_DB.get("symbol_stats", {}).get(symbol, {})
    n = st.get("count", 0)
    if n < 7:
        return True, n, 0.0   # 樣本不足，允許
    wr = st.get("win", 0) / n * 100
    return wr >= 40, n, round(wr, 1)

# =====================================================
# ADX：趨勢強度
# =====================================================
def analyze_adx(df):
    try:
        adx_df = ta.adx(df['h'], df['l'], df['c'], length=14)
        if adx_df is None or adx_df.empty:
            return 0, "ADX無數據"
        adx_val = safe_last(adx_df['ADX_14'], 0)
        dmp     = safe_last(adx_df['DMP_14'], 0)
        dmn     = safe_last(adx_df['DMN_14'], 0)
        score = 0; tag = "ADX{:.0f}".format(adx_val)
        # 注意：此函數由 analyze() 呼叫，透過 symbol 判斷主流/山寨
        if adx_val > 30:
            score = W["adx"] if dmp > dmn else -W["adx"]
            tag  += "(強多)" if dmp > dmn else "(強空)"
        elif adx_val > 20:
            score = W["adx"]//2 if dmp > dmn else -W["adx"]//2
            tag  += "(弱多)" if dmp > dmn else "(弱空)"
        else:
            tag += "(盤整)"
        return score, tag
    except:
        return 0, "ADX失敗"

# =====================================================
# VWAP：相對位置
# =====================================================
def analyze_vwap(df):
    try:
        # 手動計算 VWAP，完全不需要 DatetimeIndex
        tp = (df['h'] + df['l'] + df['c']) / 3
        vwap_val = float((tp * df['v']).sum() / df['v'].sum())
        curr = float(df['c'].iloc[-1])
        if vwap_val <= 0:
            return 0, "VWAP無數據"
        dist_pct = (curr - vwap_val) / vwap_val * 100
        if dist_pct > 1.0:
            return W["vwap"], "VWAP上方{:.1f}%".format(dist_pct)
        elif dist_pct < -1.0:
            return -W["vwap"], "VWAP下方{:.1f}%".format(abs(dist_pct))
        elif dist_pct > 0.2:
            return W["vwap"]//2, "接近VWAP上方"
        elif dist_pct < -0.2:
            return -W["vwap"]//2, "接近VWAP下方"
        else:
            return 0, "VWAP中性"
    except:
        return 0, "VWAP失敗"

# =====================================================
# Order Block：機構區域偵測
# =====================================================
# =====================================================
# ICT 概念：BOS / CHoCH / 縮量回調
# =====================================================
# =====================================================
# FVG (Fair Value Gap / 合理價格缺口)
# =====================================================
def analyze_fvg(df):
    """
    FVG 是 SMC/ICT 核心概念：
    - 三根K棒，中間那根的上下影線留下缺口
    - 做多FVG：K1最高 < K3最低（向上跳空缺口）
    - 做空FVG：K1最低 > K3最高（向下跳空缺口）
    - 價格回到 FVG 區域 = 高概率反轉點
    - 配合 OB 使用時信號更強（SMC入場核心）
    """
    try:
        hi = df['h'].tolist()
        lo = df['l'].tolist()
        cl = df['c'].tolist()
        n  = len(cl)
        curr = cl[-1]
        score = 0; tags = []

        # 掃描最近30根K棒找未填補的FVG
        bullish_fvgs = []  # 做多缺口（支撐）
        bearish_fvgs = []  # 做空缺口（壓力）

        for i in range(2, min(30, n)):
            idx = n - 1 - i  # 從最近往前掃
            if idx < 2:
                break

            k1_h = hi[idx-2]; k1_l = lo[idx-2]
            k2_h = hi[idx-1]; k2_l = lo[idx-1]
            k3_h = hi[idx];   k3_l = lo[idx]

            # 做多FVG：K1最高 < K3最低（上升缺口）
            if k1_h < k3_l:
                gap_top    = k3_l
                gap_bottom = k1_h
                gap_size   = gap_top - gap_bottom

                # 檢查是否已被填補
                filled = any(lo[j] <= gap_top and hi[j] >= gap_bottom
                             for j in range(idx+1, n))
                if not filled and gap_size > 0:
                    bullish_fvgs.append({
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "size": gap_size,
                        "age": i  # 幾根K棒前
                    })

            # 做空FVG：K1最低 > K3最高（下降缺口）
            elif k1_l > k3_h:
                gap_top    = k1_l
                gap_bottom = k3_h
                gap_size   = gap_top - gap_bottom

                filled = any(hi[j] >= gap_bottom and lo[j] <= gap_top
                             for j in range(idx+1, n))
                if not filled and gap_size > 0:
                    bearish_fvgs.append({
                        "top": gap_top,
                        "bottom": gap_bottom,
                        "size": gap_size,
                        "age": i
                    })

        # 判斷當前價格是否在 FVG 區域內
        W_FVG = W.get("bos_choch", 7)  # 共用 ICT 類分數

        for fvg in bullish_fvgs[:3]:  # 只看最近3個
            if fvg["bottom"] <= curr <= fvg["top"]:
                # 價格回到做多FVG區域 → 看多
                freshness = max(1.0 - fvg["age"] / 30, 0.3)  # 越新越重要
                pts = round(W_FVG * freshness)
                score += pts
                tags.append("FVG做多缺口({:.4f}-{:.4f})".format(
                    fvg["bottom"], fvg["top"]))
                break

        for fvg in bearish_fvgs[:3]:
            if fvg["bottom"] <= curr <= fvg["top"]:
                # 價格回到做空FVG區域 → 看空
                freshness = max(1.0 - fvg["age"] / 30, 0.3)
                pts = round(W_FVG * freshness)
                score -= pts
                tags.append("FVG做空缺口({:.4f}-{:.4f})".format(
                    fvg["bottom"], fvg["top"]))
                break

        # 價格接近但還沒到FVG（預期回撤）
        if not tags:
            for fvg in bullish_fvgs[:2]:
                dist = (curr - fvg["top"]) / max(curr, 1e-9)
                if 0 < dist < 0.02:  # 距離缺口頂部2%以內
                    score += W_FVG // 2
                    tags.append("接近FVG支撐缺口")
                    break
            for fvg in bearish_fvgs[:2]:
                dist = (fvg["bottom"] - curr) / max(curr, 1e-9)
                if 0 < dist < 0.02:
                    score -= W_FVG // 2
                    tags.append("接近FVG壓力缺口")
                    break

        return min(max(score, -W_FVG), W_FVG), "|".join(tags) or "無FVG"
    except Exception as e:
        return 0, "FVG失敗"

def analyze_ict(df4h, df15):
    """
    BOS (Break of Structure)：突破前高/前低，確認趨勢方向
    CHoCH (Change of Character)：趨勢轉換訊號，最重要的反轉信號
    縮量回調：趨勢中回調時成交量縮小，代表只是回調非反轉
    """
    try:
        score = 0; tags = []

        c4 = df4h['c'].tolist()
        h4 = df4h['h'].tolist()
        l4 = df4h['l'].tolist()
        v4 = df4h['v'].tolist()
        n = len(c4)
        curr = c4[-1]

        if n < 20:
            return 0, "ICT數據不足"

        # 找最近的擺動高低點（Swing High/Low）
        def find_swings(highs, lows, lookback=5):
            swing_highs = []
            swing_lows = []
            for i in range(lookback, len(highs)-lookback):
                if highs[i] == max(highs[i-lookback:i+lookback+1]):
                    swing_highs.append((i, highs[i]))
                if lows[i] == min(lows[i-lookback:i+lookback+1]):
                    swing_lows.append((i, lows[i]))
            return swing_highs, swing_lows

        swing_highs, swing_lows = find_swings(h4, l4, lookback=3)

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # 最近兩個擺動高點
            sh1 = swing_highs[-2][1]  # 前前高
            sh2 = swing_highs[-1][1]  # 前高
            # 最近兩個擺動低點
            sl1 = swing_lows[-2][1]   # 前前低
            sl2 = swing_lows[-1][1]   # 前低

            # BOS 做多：突破前高 → 上升趨勢確認
            if curr > sh2 * 1.001:
                score += 8
                tags.append("BOS突破前高{:.4f}".format(sh2))

            # BOS 做空：跌破前低 → 下降趨勢確認
            elif curr < sl2 * 0.999:
                score -= 8
                tags.append("BOS跌破前低{:.4f}".format(sl2))

            # CHoCH 做多：原本下降趨勢（前高低於更前高），但現在突破前高
            if sh2 < sh1 and curr > sh2 * 1.001:
                score += 6  # 額外加分，趨勢轉換
                tags.append("CHoCH趨勢轉多")

            # CHoCH 做空：原本上升趨勢（前低高於更前低），但現在跌破前低
            elif sl2 > sl1 and curr < sl2 * 0.999:
                score -= 6
                tags.append("CHoCH趨勢轉空")

        # 縮量回調偵測（做多確認）
        # 條件：最近3根K棒價格下跌但成交量縮小
        if len(c4) >= 6 and len(v4) >= 6:
            recent_prices = c4[-4:-1]
            recent_vols = v4[-4:-1]
            avg_vol = sum(v4[-20:-4]) / max(len(v4[-20:-4]), 1)

            is_pullback = recent_prices[-1] < recent_prices[0]  # 近期回調
            is_low_vol = sum(recent_vols) / 3 < avg_vol * 0.7   # 成交量縮小30%

            if is_pullback and is_low_vol:
                # 大趨勢是多頭才加分
                if c4[-1] > sum(c4[-20:]) / 20:  # 價格在20根均線上
                    score += 5
                    tags.append("縮量健康回調")
                else:
                    score -= 3
                    tags.append("縮量弱勢下跌")

        # 用15分鐘確認 BOS
        c15 = df15['c'].tolist()
        h15 = df15['h'].tolist()
        l15 = df15['l'].tolist()
        if len(h15) >= 20:
            sh15, sl15 = find_swings(h15, l15, lookback=3)
            if sh15 and sl15:
                last_sh15 = sh15[-1][1] if sh15 else 0
                last_sl15 = sl15[-1][1] if sl15 else float('inf')
                # 15分鐘也突破 → 多週期共振
                if c15[-1] > last_sh15 * 1.001 and score > 0:
                    score += 3
                    tags.append("15m多週期BOS共振")
                elif c15[-1] < last_sl15 * 0.999 and score < 0:
                    score -= 3
                    tags.append("15m多週期BOS共振")

        return min(max(score, -W.get('bos_choch',7)), W.get('bos_choch',7)), "|".join(tags) or "無ICT訊號"
    except Exception as e:
        return 0, "ICT分析失敗"

def analyze_order_block(df4h, is_major=False):
    """
    偵測機構 Order Block：
    - 強力單邊運動前的最後一根反向K棒即為 OB
    - 價格回到 OB 區域 → 高概率反彈點
    """
    try:
        score = 0; tags = []
        closes = df4h['c'].tolist()
        opens  = df4h['o'].tolist()
        highs  = df4h['h'].tolist()
        lows   = df4h['l'].tolist()
        curr   = closes[-1]

        # 找最近的看多 OB（跌後急漲前的最後一根陰線）
        for i in range(len(closes)-3, max(len(closes)-20, 2), -1):
            # 確認後面有強力上漲
            move_up = (closes[i+1] - opens[i+1]) / max(abs(opens[i+1]), 1e-9)
            if closes[i] < opens[i] and move_up > (0.010 if is_major else 0.015):  # OB條件（主流1%/山寨1.5%）
                ob_high = opens[i]  # OB 區域：陰線的開盤到最高
                ob_low  = lows[i]
                # 當前價格是否在 OB 區域內（回測）
                if ob_low <= curr <= ob_high * 1.01:
                    score += W["order_block"]
                    tags.append("看多OB區域({:.4f}-{:.4f})".format(ob_low, ob_high))
                    break

        # 找最近的看空 OB（漲後急跌前的最後一根陽線）
        for i in range(len(closes)-3, max(len(closes)-20, 2), -1):
            move_dn = (opens[i+1] - closes[i+1]) / max(abs(opens[i+1]), 1e-9)
            if closes[i] > opens[i] and move_dn > (0.010 if is_major else 0.015):  # OB條件（主流1%/山寨1.5%）
                ob_low  = opens[i]
                ob_high = highs[i]
                if ob_low * 0.99 <= curr <= ob_high:
                    score -= W["order_block"]
                    tags.append("看空OB區域({:.4f}-{:.4f})".format(ob_low, ob_high))
                    break

        return min(max(score, -W["order_block"]), W["order_block"]), "|".join(tags) or "無OB"
    except:
        return 0, "OB失敗"

# =====================================================
# 流動性掃單過濾（Liquidity Sweep）
# =====================================================
def analyze_liquidity_sweep(df):
    """
    偵測假突破 / 流動性掃單：
    - 價格短暫突破高低點後立刻收回 → 掃單行為
    - 掃單後反方向下單勝率更高
    """
    try:
        score = 0; tags = []
        recent_high = df['h'].tail(20).iloc[:-1].max()  # 排除最後一根
        recent_low  = df['l'].tail(20).iloc[:-1].min()
        last_high   = df['h'].iloc[-1]
        last_low    = df['l'].iloc[-1]
        last_close  = df['c'].iloc[-1]
        last_open   = df['o'].iloc[-1]

        # 向上掃單（K棒上影線突破高點後收回）→ 看空
        upper_wick = last_high - max(df['c'].iloc[-1], df['o'].iloc[-1])
        lower_wick = min(df['c'].iloc[-1], df['o'].iloc[-1]) - last_low
        body = abs(df['c'].iloc[-1] - df['o'].iloc[-1])

        if last_high > recent_high * 1.0005 and last_close < recent_high * 0.999:
            score -= W["liq_sweep"]
            tags.append("向上掃單({:.4f})".format(recent_high))
        elif last_low < recent_low * 0.9995 and last_close > recent_low * 1.001:
            score += W["liq_sweep"]
            tags.append("向下掃單({:.4f})".format(recent_low))
        # 長上影線（可能是高點掃單）→ 輕微看空
        elif upper_wick > body * 2 and last_close < recent_high:
            score -= W["liq_sweep"] // 2
            tags.append("長上影線壓力")
        # 長下影線（可能是低點掃單）→ 輕微看多
        elif lower_wick > body * 2 and last_close > recent_low:
            score += W["liq_sweep"] // 2
            tags.append("長下影線支撐")
        elif last_close > recent_high * 1.002:
            score += W["liq_sweep"] // 2
            tags.append("有效突破高點")
        elif last_close < recent_low * 0.998:
            score -= W["liq_sweep"] // 2
            tags.append("有效跌破低點")

        return min(max(score, -W["liq_sweep"]), W["liq_sweep"]), "|".join(tags) or "無掃單"
    except:
        return 0, "掃單分析失敗"

# =====================================================
# 莊家 / 成交量
# =====================================================
def analyze_whale(df):
    try:
        score = 0
        avg_vol  = df['v'].tail(20).mean()
        last_vol = df['v'].iloc[-1]
        prev_vol = df['v'].iloc[-2]
        if last_vol > avg_vol * 2.0:   score += W["whale"]       # 放寬：原本3倍
        elif last_vol > avg_vol * 1.5: score += W["whale"]//2   # 放寬：原本2倍
        if last_vol > prev_vol * 1.2:  score += 2               # 放寬：原本1.5倍
        curr = df['c'].iloc[-1]
        if curr < df['l'].tail(50).min() * 1.03: score += 2
        vt = df['v'].tail(5).tolist(); pt = df['c'].tail(5).tolist()
        if pt[-1] < pt[0] and vt[-1] < vt[-3]: score += 1
        return min(score, W["whale"])
    except:
        return 0

# =====================================================
# K棒型態
# =====================================================
def analyze_candles(df):
    try:
        score = 0; tags = []
        o=df['o'].iloc[-1]; h=df['h'].iloc[-1]; l=df['l'].iloc[-1]; c=df['c'].iloc[-1]
        po=df['o'].iloc[-2]; pc=df['c'].iloc[-2]
        body=abs(c-o); rng=h-l if h!=l else 1e-9
        upper=h-max(c,o); lower=min(c,o)-l
        unit = W["candle"]
        if lower>body*2 and upper<body*0.3 and c>o:  score+=unit;   tags.append("錘子線")
        if upper>body*2 and lower<body*0.3 and c<o:  score-=unit;   tags.append("流星線")
        if c>o and pc<po and c>po and o<pc:           score+=unit;   tags.append("多頭吞噬")
        if c<o and pc>po and c<po and o>pc:           score-=unit;   tags.append("空頭吞噬")
        if body/rng<0.1:                              tags.append("十字星")
        if c>o and body/rng>0.7:                      score+=unit//2; tags.append("強勢陽線")
        if c<o and body/rng>0.7:                      score-=unit//2; tags.append("強勢陰線")
        if len(df)>=3:
            c2=df['c'].iloc[-3]; o2=df['o'].iloc[-3]
            c1=df['c'].iloc[-2]; o1=df['o'].iloc[-2]
            if c2<o2 and abs(c1-o1)<abs(c2-o2)*0.3 and c>o and c>(c2+o2)/2:
                score+=unit; tags.append("早晨之星")
            if c2>o2 and abs(c1-o1)<abs(c2-o2)*0.3 and c<o and c<(c2+o2)/2:
                score-=unit; tags.append("黃昏之星")
        return min(max(score,-unit),unit), "|".join(tags) or "無特殊K棒"
    except:
        return 0, "K棒失敗"

# =====================================================
# 圖形型態
# =====================================================
def analyze_chart_pattern(df):
    try:
        score=0; name=""
        hi=df['h'].tail(50).tolist(); lo=df['l'].tail(50).tolist()
        mid=len(lo)//2

        # W底：兩低點相近，中間有反彈
        lL,rL=min(lo[:mid]),min(lo[mid:])
        mH=max(hi[mid-8:mid+8]) if len(hi)>16 else 0
        if (lL<mH*0.96 and rL<mH*0.96 and
            abs(lL-rL)/max(abs(lL),1e-9)<0.06 and
            mH>max(lL,rL)*1.03):
            score+=W["chart_pat"]; name="W底"

        # M頭：兩高點相近，中間有回落 → 減分
        lH,rH=max(hi[:mid]),max(hi[mid:])
        mL=min(lo[mid-8:mid+8]) if len(lo)>16 else 0
        if (lH>mL*1.04 and rH>mL*1.04 and
            abs(lH-rH)/max(abs(lH),1e-9)<0.06 and
            mL<min(lH,rH)*0.97):
            score-=W["chart_pat"]; name="M頭（看空）"

        # 三角形
        rhi=hi[-15:]; rlo=lo[-15:]
        if max(rhi)-min(rhi)<max(rhi)*0.03 and rlo[-1]>rlo[0]:
            score+=W["chart_pat"]//2; name="上升三角"
        elif max(rlo)-min(rlo)<max(rlo)*0.03 and rhi[-1]<rhi[0]:
            score-=W["chart_pat"]//2; name="下降三角（看空）"

        # 頭肩頂/底
        if len(hi)>=45:
            t=len(hi)//3
            h1,h2,h3=max(hi[:t]),max(hi[t:2*t]),max(hi[2*t:])
            if h2>h1*1.02 and h2>h3*1.02 and abs(h1-h3)/max(h1,1e-9)<0.08:
                score-=W["chart_pat"]; name="頭肩頂（強看空）"
            l1,l2,l3=min(lo[:t]),min(lo[t:2*t]),min(lo[2*t:])
            if l2<l1*0.98 and l2<l3*0.98 and abs(l1-l3)/max(l1,1e-9)<0.08:
                score+=W["chart_pat"]; name="頭肩底（強看多）"

        return min(max(score,-W["chart_pat"]),W["chart_pat"]), name or "無明顯形態"
    except:
        return 0, "形態失敗"

# =====================================================
# Trend Magic（CCI + ATR 自適應趨勢線）
# =====================================================
def analyze_mtf_confirm(d15, d4h, d1d):
    """
    多時框方向確認（Multi-TimeFrame）
    15分鐘 + 4小時 + 日線 三個時框方向一致才給高分
    這是提升勝率最重要的過濾器
    """
    try:
        score = 0; tags = []
        W_MTF = W.get("mtf_confirm", 14)

        def get_direction(df):
            """用EMA判斷該時框方向"""
            c = df['c']
            if len(c) < 20: return 0
            e9  = float(ta.ema(c, length=9).iloc[-1])
            e20 = float(ta.ema(c, length=20).iloc[-1])
            curr = float(c.iloc[-1])
            if pd.isna(e9) or pd.isna(e20): return 0
            if curr > e9 > e20: return 1   # 多頭
            if curr < e9 < e20: return -1  # 空頭
            return 0  # 中性

        dir_15 = get_direction(d15)
        dir_4h = get_direction(d4h)
        dir_1d = get_direction(d1d)

        dirs = [dir_15, dir_4h, dir_1d]
        bull = sum(1 for d in dirs if d == 1)
        bear = sum(1 for d in dirs if d == -1)

        if bull == 3:
            score = W_MTF         # 三框全多 → 滿分
            tags.append("三框共振做多🔥")
        elif bull == 2:
            score = round(W_MTF * 0.6)  # 兩框多
            missing = ["15m","4H","日線"][dirs.index(next(d for d in dirs if d != 1))]
            tags.append("雙框多({})".format(missing+"待確認"))
        elif bear == 3:
            score = -W_MTF
            tags.append("三框共振做空🔥")
        elif bear == 2:
            score = -round(W_MTF * 0.6)
            missing = ["15m","4H","日線"][dirs.index(next(d for d in dirs if d != -1))]
            tags.append("雙框空({})".format(missing+"待確認"))
        else:
            score = 0
            tags.append("多框中性/分歧")

        return min(max(score, -W_MTF), W_MTF), "|".join(tags)
    except Exception as e:
        return 0, "MTF失敗"

def analyze_trend_magic(df, cci_period=20, atr_mult=1.5):
    """
    Trend Magic by GLAZ - CCI + ATR 組合
    - CCI > 0：趨勢線只能上移（多頭模式，藍線）
    - CCI < 0：趨勢線只能下移（空頭模式，紅線）
    - 價格穿越趨勢線 → 趨勢轉換訊號
    """
    try:
        c = df['c']
        h = df['h']
        l = df['l']
        n = len(c)
        if n < cci_period + 5:
            return 0, "TM數據不足"

        # 計算 CCI
        typical = (h + l + c) / 3
        tp_mean = typical.rolling(cci_period).mean()
        tp_mad  = typical.rolling(cci_period).apply(lambda x: abs(x - x.mean()).mean())
        cci = (typical - tp_mean) / (0.015 * tp_mad.replace(0, 1e-9))

        # 計算 ATR
        atr_s = ta.atr(h, l, c, length=cci_period)

        # 初始化趨勢線
        tm = [float(c.iloc[0])]
        for i in range(1, n):
            atr_val = float(atr_s.iloc[i]) if not pd.isna(atr_s.iloc[i]) else float(c.iloc[i]) * 0.01
            cci_val = float(cci.iloc[i]) if not pd.isna(cci.iloc[i]) else 0
            prev_tm = tm[-1]
            price   = float(c.iloc[i])

            if cci_val > 0:
                # 多頭模式：趨勢線只能上移
                new_tm = max(prev_tm, price - atr_val * atr_mult)
            else:
                # 空頭模式：趨勢線只能下移
                new_tm = min(prev_tm, price + atr_val * atr_mult)
            tm.append(new_tm)

        curr      = float(c.iloc[-1])
        tm_curr   = tm[-1]
        tm_prev   = tm[-2] if len(tm) > 1 else tm_curr
        cci_curr  = float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0
        cci_prev  = float(cci.iloc[-2]) if not pd.isna(cci.iloc[-2]) else 0

        score = 0; tag = ""

        # 多頭訊號：CCI > 0 且價格在趨勢線上方
        if cci_curr > 0 and curr > tm_curr:
            dist_pct = (curr - tm_curr) / max(tm_curr, 1e-9) * 100
            if dist_pct < 2.0:
                score = W.get("trendline", 7)  # 緊貼趨勢線支撐，強烈做多
                tag = "TM多頭緊貼支撐"
            else:
                score = W.get("trendline", 7) // 2
                tag = "TM多頭"

        # 空頭訊號：CCI < 0 且價格在趨勢線下方
        elif cci_curr < 0 and curr < tm_curr:
            dist_pct = (tm_curr - curr) / max(tm_curr, 1e-9) * 100
            if dist_pct < 2.0:
                score = -W.get("trendline", 7)
                tag = "TM空頭緊貼壓力"
            else:
                score = -W.get("trendline", 7) // 2
                tag = "TM空頭"

        # 金叉：CCI 從負轉正（趨勢轉多）
        elif cci_prev <= 0 and cci_curr > 0:
            score = W.get("trendline", 7)
            tag = "TM趨勢轉多🔵"

        # 死叉：CCI 從正轉負（趨勢轉空）
        elif cci_prev >= 0 and cci_curr < 0:
            score = -W.get("trendline", 7)
            tag = "TM趨勢轉空🔴"

        return min(max(score, -W.get('trendline', 7)), W.get('trendline', 7)), tag or "TM中性"
    except Exception as e:
        return 0, "TM計算失敗"

def analyze_trend(df4h):
    """
    真正的趨勢線判斷：
    - 用線性回歸計算支撐線和壓力線的斜率
    - 斜率向上 + 價格在線上方 = 上升趨勢
    - 斜率向下 + 價格在線下方 = 下降趨勢
    """
    try:
        score=0; tags=[]
        lo=df4h['l'].tolist(); hi=df4h['h'].tolist()
        curr=df4h['c'].iloc[-1]; n=len(lo)
        unit=W["trendline"]
        if n < 10:
            return 0, "趨勢線數據不足"

        # 用最近20根K棒計算
        recent_lo = lo[-20:]
        recent_hi = hi[-20:]
        x = list(range(len(recent_lo)))

        # 線性回歸計算斜率
        def slope(vals):
            n_ = len(vals)
            sx = sum(x); sy = sum(vals)
            sxy = sum(x[i]*vals[i] for i in range(n_))
            sxx = sum(xi**2 for xi in x)
            denom = n_*sxx - sx*sx
            if denom == 0: return 0
            return (n_*sxy - sx*sy) / denom

        lo_slope = slope(recent_lo)
        hi_slope = slope(recent_hi)

        # 最近支撐線的值（用最後一點）
        lo_intercept = sum(recent_lo)/len(recent_lo) - lo_slope * sum(x)/len(x)
        support_val = lo_slope * x[-1] + lo_intercept

        hi_intercept = sum(recent_hi)/len(recent_hi) - hi_slope * sum(x)/len(x)
        resist_val = hi_slope * x[-1] + hi_intercept

        # 用ATR判斷距離
        atr_approx = sum(hi[-14][i]-lo[-14][i] if isinstance(hi, list) else 0 for i in range(14)) if False else abs(curr * 0.01)
        try:
            atr_series = df4h['h'].tail(14) - df4h['l'].tail(14)
            atr_approx = float(atr_series.mean())
        except:
            atr_approx = curr * 0.01

        # 支撐趨勢線判斷
        if lo_slope > 0:  # 上升趨勢支撐線
            dist_from_support = (curr - support_val) / max(atr_approx, 1e-9)
            if dist_from_support < 1.0:   # 價格接近上升支撐
                score += unit; tags.append("4H上升趨勢支撐")
            elif dist_from_support > 5.0:  # 離支撐太遠
                score += unit//2; tags.append("上升趨勢中段")
            else:
                score += unit//2; tags.append("4H上升趨勢")
        elif lo_slope < -atr_approx * 0.05:  # 明顯下降
            if curr < support_val:
                score -= unit; tags.append("跌破下降趨勢低點")
            else:
                score -= unit//2; tags.append("下降趨勢中")

        # 壓力趨勢線判斷
        if hi_slope < 0:  # 下降壓力線
            dist_from_resist = (resist_val - curr) / max(atr_approx, 1e-9)
            if dist_from_resist < 1.0:  # 接近下降壓力
                score -= unit//2; tags.append("受壓下降趨勢線")
        elif hi_slope > 0:  # 上升壓力突破
            if curr > resist_val:
                score += unit//2; tags.append("突破上升壓力")

        return min(max(score, -unit), unit), "|".join(tags) or "趨勢中性"
    except Exception as e:
        return 0, "趨勢線失敗"

def get_best_atr_params(breakdown_keys):
    with LEARN_LOCK:
        db   = LEARN_DB
        pkey = "|".join(sorted(breakdown_keys))
        if pkey in db["pattern_stats"]:
            st = db["pattern_stats"][pkey]
            if st.get("sample_count", 0) >= 3:
                return st.get("best_sl", db["atr_params"]["default_sl"]), \
                       st.get("best_tp", db["atr_params"]["default_tp"])
        best_match=None; best_overlap=0; ks=set(breakdown_keys)
        for k,st in db["pattern_stats"].items():
            ov=len(ks & set(k.split("|")))
            if ov>best_overlap and st.get("sample_count",0)>=3:
                best_overlap=ov; best_match=st
        if best_match and best_overlap>=2:
            return best_match.get("best_sl",db["atr_params"]["default_sl"]), \
                   best_match.get("best_tp",db["atr_params"]["default_tp"])
        return db["atr_params"]["default_sl"], db["atr_params"]["default_tp"]

# =====================================================
# 主技術分析（全週期，滿分100）
# =====================================================
# 短線禁止下單（留給長期倉位）
SHORT_TERM_EXCLUDED = {'BTC/USDT:USDT'}

# 主流幣清單（使用放寬版評分）
MAJOR_COINS = {
    'BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','SOL/USDT:USDT',
    'XRP/USDT:USDT','ADA/USDT:USDT','DOGE/USDT:USDT','AVAX/USDT:USDT',
    'DOT/USDT:USDT','LINK/USDT:USDT','LTC/USDT:USDT','BCH/USDT:USDT',
    'UNI/USDT:USDT','ATOM/USDT:USDT','MATIC/USDT:USDT',
}


def analyze_market_regime_for_symbol(d15, d4h, d1d):
    """
    方向判斷核心：
    不再只看總分，而是先判斷這個幣現在屬於
    趨勢延續 / 回踩續攻 / 震盪 / 反彈反抽。
    """
    try:
        c15 = d15['c'].astype(float)
        h15 = d15['h'].astype(float)
        l15 = d15['l'].astype(float)
        c4 = d4h['c'].astype(float)
        c1 = d1d['c'].astype(float)

        curr = float(c15.iloc[-1])
        atr15 = max(safe_last(ta.atr(h15, l15, c15, length=14), curr * 0.004), curr * 0.003)

        e9_15  = safe_last(ta.ema(c15, length=9), curr)
        e21_15 = safe_last(ta.ema(c15, length=21), curr)
        e55_15 = safe_last(ta.ema(c15, length=55), curr)

        e21_4 = safe_last(ta.ema(c4, length=21), curr)
        e55_4 = safe_last(ta.ema(c4, length=55), curr)
        e20_1 = safe_last(ta.ema(c1, length=20), curr)
        e50_1 = safe_last(ta.ema(c1, length=50), curr)

        slope15 = _linreg_slope(c15.tail(8).tolist()) / max(curr, 1e-9) * 100
        slope4h = _linreg_slope(c4.tail(8).tolist()) / max(curr, 1e-9) * 100

        bull_stack = curr > e9_15 > e21_15 > e55_15 and curr > e21_4 > e55_4 and curr > e20_1 > e50_1
        bear_stack = curr < e9_15 < e21_15 < e55_15 and curr < e21_4 < e55_4 and curr < e20_1 < e50_1

        pullback_long = bull_stack and abs(curr - e21_15) / max(atr15, 1e-9) <= 0.9
        rebound_short = bear_stack and abs(curr - e21_15) / max(atr15, 1e-9) <= 0.9

        if bull_stack and slope15 > 0.08 and slope4h > 0.04:
            return 8 if pullback_long else 6, 1, ("多頭回踩續攻" if pullback_long else "多頭延續"), True
        if bear_stack and slope15 < -0.08 and slope4h < -0.04:
            return -8 if rebound_short else -6, -1, ("空頭反彈續跌" if rebound_short else "空頭延續"), True

        if curr > e21_4 and curr > e20_1 and slope4h > 0:
            return 3, 1, "偏多但未完全共振", True
        if curr < e21_4 and curr < e20_1 and slope4h < 0:
            return -3, -1, "偏空但未完全共振", True

        return 0, 0, "區間震盪", False
    except Exception:
        return 0, 0, "方向判斷失敗", False


def analyze_entry_timing_quality(d15, d4h, direction_hint=0):
    """
    進場品質：
    順勢但太遠不追；順勢回踩、突破後站穩、量價配合才加分。
    """
    try:
        c = d15['c'].astype(float)
        o = d15['o'].astype(float)
        h = d15['h'].astype(float)
        l = d15['l'].astype(float)
        v = d15['v'].astype(float)
        curr = float(c.iloc[-1])
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        ema9 = safe_last(ta.ema(c, length=9), curr)
        ema21 = safe_last(ta.ema(c, length=21), curr)
        vol_now = float(v.tail(3).mean()) if len(v) >= 3 else float(v.iloc[-1])
        vol_avg = float(v.tail(24).mean()) if len(v) >= 24 else vol_now
        hh = float(h.tail(20).iloc[:-1].max()) if len(h) > 21 else float(h.max())
        ll = float(l.tail(20).iloc[:-1].min()) if len(l) > 21 else float(l.min())

        body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
        close_pos = (float(c.iloc[-1]) - float(l.iloc[-1])) / max(float(h.iloc[-1]) - float(l.iloc[-1]), 1e-9)
        ext = (curr - ema21) / max(atr, 1e-9)

        score = 0
        tags = []

        if direction_hint > 0:
            if curr > ema9 > ema21:
                score += 2; tags.append("15m多頭排列")
            if abs(curr - ema21) / max(atr, 1e-9) <= 0.8:
                score += 3; tags.append("回踩均線附近")
            if curr >= hh * 0.998 and close_pos > 0.65 and body > atr * 0.45 and vol_now > vol_avg * 1.1:
                score += 3; tags.append("突破帶量站穩")
            if ext > 1.5:
                score -= 4; tags.append("離均線過遠")
            if close_pos < 0.45 and curr >= hh * 0.998:
                score -= 2; tags.append("突破收不穩")
        elif direction_hint < 0:
            if curr < ema9 < ema21:
                score += 2; tags.append("15m空頭排列")
            if abs(curr - ema21) / max(atr, 1e-9) <= 0.8:
                score += 3; tags.append("反彈均線附近")
            low_close_pos = (float(h.iloc[-1]) - float(c.iloc[-1])) / max(float(h.iloc[-1]) - float(l.iloc[-1]), 1e-9)
            if curr <= ll * 1.002 and low_close_pos > 0.65 and body > atr * 0.45 and vol_now > vol_avg * 1.1:
                score += 3; tags.append("跌破帶量站穩")
            if ext < -1.5:
                score -= 4; tags.append("離均線過遠")
            if low_close_pos < 0.45 and curr <= ll * 1.002:
                score -= 2; tags.append("跌破收不穩")
        else:
            score -= 1
            tags.append("方向未明")

        score = max(min(score, 8), -8)
        return score, "|".join(tags) if tags else "進場品質一般"
    except Exception:
        return 0, "進場品質失敗"




def analyze_fake_breakout(df, directional_bias=0):
    """
    假突破 / 假跌破過濾
    回傳: (score_adjust, tag, meta)
    meta = {fakeout: bool, direction: 'up'/'down'/None, strength: float}
    """
    try:
        if df is None or len(df) < max(BREAKOUT_LOOKBACK + 3, 12):
            return 0, '資料不足', {'fakeout': False, 'direction': None, 'strength': 0.0}

        sub = df.copy().reset_index(drop=True)
        last = sub.iloc[-1]
        prev = sub.iloc[-2]
        ref = sub.iloc[:-1].tail(max(BREAKOUT_LOOKBACK, 8))
        hh = float(ref['h'].max())
        ll = float(ref['l'].min())
        close_ = float(last['c'])
        high = float(last['h'])
        low = float(last['l'])
        open_ = float(last['o'])
        atr = safe_last(ta.atr(sub['h'], sub['l'], sub['c'], length=14), max(close_ * 0.004, 1e-9))
        body = abs(close_ - open_)
        upper = high - max(close_, open_)
        lower = min(close_, open_) - low
        score = 0
        tag = '無假突破'
        meta = {'fakeout': False, 'direction': None, 'strength': 0.0}

        broke_up = high > hh * 1.0008
        closed_back_in_up = close_ < hh and upper > body * 0.8
        broke_down = low < ll * 0.9992
        closed_back_in_down = close_ > ll and lower > body * 0.8

        if broke_up and closed_back_in_up:
            strength = min((high - close_) / max(atr, 1e-9), 3.0)
            meta = {'fakeout': True, 'direction': 'up', 'strength': round(strength, 2)}
            score = -min(8, max(3, int(round(2.5 + strength * 1.8))))
            if directional_bias < 0:
                score = abs(score)
            tag = '假突破回落'
        elif broke_down and closed_back_in_down:
            strength = min((close_ - low) / max(atr, 1e-9), 3.0)
            meta = {'fakeout': True, 'direction': 'down', 'strength': round(strength, 2)}
            score = min(8, max(3, int(round(2.5 + strength * 1.8))))
            if directional_bias > 0:
                score = -abs(score)
            tag = '假跌破回收'

        # 跟當前偏向相反時，額外加重懲罰/獎勵
        if meta['fakeout'] and directional_bias != 0:
            if directional_bias > 0 and meta['direction'] == 'up':
                score -= 2
            elif directional_bias < 0 and meta['direction'] == 'down':
                score += 2

        return score, tag, meta
    except Exception as e:
        return 0, f'假突破分析失敗:{str(e)[:24]}', {'fakeout': False, 'direction': None, 'strength': 0.0}

def analyze(symbol):
    is_major = symbol in MAJOR_COINS  # 是否為主流幣
    try:
        d15=pd.DataFrame(exchange.fetch_ohlcv(symbol,'15m',limit=100),columns=['t','o','h','l','c','v'])
        time.sleep(0.2)
        d4h=pd.DataFrame(exchange.fetch_ohlcv(symbol,'4h', limit=60), columns=['t','o','h','l','c','v'])
        time.sleep(0.2)
        d1d=pd.DataFrame(exchange.fetch_ohlcv(symbol,'1d', limit=50), columns=['t','o','h','l','c','v'])
        time.sleep(0.1)

        score=0.0; tags=[]; curr=d15['c'].iloc[-1]; breakdown={}

        regime_s, regime_bias, regime_tag, regime_ok = analyze_market_regime_for_symbol(d15, d4h, d1d)
        score += regime_s
        breakdown['方向品質'] = regime_s
        tags.append(regime_tag)

        entry_s0, entry_tag0 = analyze_entry_timing_quality(d15, d4h, regime_bias)
        score += entry_s0
        breakdown['進場品質'] = entry_s0
        if entry_tag0:
            tags.append(entry_tag0)

        # RSI（含背離偵測）
        rsi_series = ta.rsi(d15['c'], length=14)
        rsi = safe_last(rsi_series, 50)
        rs = W["rsi"] if rsi<30 else W["rsi"]//2 if rsi<40 else -W["rsi"] if rsi>70 else -W["rsi"]//2 if rsi>60 else 0

        # RSI 背離偵測（高勝率信號）
        try:
            if len(rsi_series) >= 10 and not rsi_series.isna().all():
                price_recent = d15['c'].tail(10).tolist()
                rsi_recent   = rsi_series.tail(10).tolist()
                # 看多背離：價格創新低但RSI沒創新低
                if price_recent[-1] < min(price_recent[:-1]) and rsi_recent[-1] > min(rsi_recent[:-1]):
                    rs = W["rsi"]
                    tags.append("RSI看多背離🔥")
                # 看空背離：價格創新高但RSI沒創新高
                elif price_recent[-1] > max(price_recent[:-1]) and rsi_recent[-1] < max(rsi_recent[:-1]):
                    rs = -W["rsi"]
                    tags.append("RSI看空背離🔥")
        except:
            pass

        score+=rs; breakdown['RSI({:.0f})'.format(rsi)]=rs
        if rs and 'RSI' not in str(tags):
            tags.append("RSI{:.0f}".format(rsi))

        # MACD（金叉死叉+強度）
        macd=ta.macd(d15['c']); ms=0
        if macd is not None and 'MACDh_12_26_9' in macd.columns:
            mh=safe_last(macd['MACDh_12_26_9']); mp=float(macd['MACDh_12_26_9'].iloc[-2])
            ml=safe_last(macd['MACD_12_26_9']); ms_line=safe_last(macd['MACDs_12_26_9'])
            if mh>0 and mp<0:
                strength = min(abs(mh)/max(abs(ml),1e-9), 1.0)
                ms = int(W["macd"] * (0.7 + 0.3*strength))
                tags.append("MACD金叉")
            elif mh<0 and mp>0:
                strength = min(abs(mh)/max(abs(ml),1e-9), 1.0)
                ms = -int(W["macd"] * (0.7 + 0.3*strength))
                tags.append("MACD死叉")
            elif mh>0:
                ms=W["macd"]//2; tags.append("MACD多")
            else:
                ms=-W["macd"]//2; tags.append("MACD空")
        score+=ms; breakdown['MACD']=ms

        # 多時框確認
        mtf_s, mtf_tag = analyze_mtf_confirm(d15, d4h, d1d)
        score += mtf_s; breakdown['多時框'] = mtf_s
        if mtf_tag and "中性" not in mtf_tag:
            tags.append(mtf_tag)

        # 日線EMA
        e20=ta.ema(d1d['c'],length=20); e50=ta.ema(d1d['c'],length=50)
        e9=ta.ema(d1d['c'],length=9); es=0
        if e20 is not None and e50 is not None and not e20.empty and not e50.empty:
            v20=safe_last(e20); v50=safe_last(e50)
            v9=safe_last(e9,v20) if e9 is not None and not e9.empty else v20
            if curr>v20>v50:
                es=W["ema_trend"]; tags.append("日線多排")
            elif curr<v20<v50:
                es=-W["ema_trend"]; tags.append("日線空排")
            elif curr>v20:
                es=W["ema_trend"]//2; tags.append("EMA支撐")
            else:
                es=-W["ema_trend"]//2; tags.append("EMA反壓")
            if e9 is not None and not e9.empty and len(e9) >= 2 and len(e20) >= 2:
                v9_prev = float(e9.iloc[-2]) if not pd.isna(e9.iloc[-2]) else v9
                v20_prev = float(e20.iloc[-2]) if not pd.isna(e20.iloc[-2]) else v20
                if v9_prev <= v20_prev and v9 > v20:
                    if is_major:
                        es = min(es + W["ema_trend"]//2, W["ema_trend"])
                        tags.append("EMA金叉🔵")
                elif v9_prev >= v20_prev and v9 < v20:
                    if is_major:
                        es = max(es - W["ema_trend"]//2, -W["ema_trend"])
                        tags.append("EMA死叉🔴")
        score+=es; breakdown['日線EMA']=es

        # ADX
        adx_s,adx_tag=analyze_adx(d15)
        score+=adx_s; breakdown['ADX']=adx_s; tags.append(adx_tag)
        try:
            adx_df2 = ta.adx(d15['h'], d15['l'], d15['c'], length=14)
            adx_val2 = safe_last(adx_df2['ADX_14'], 25) if adx_df2 is not None else 25
            if adx_val2 < 20:
                score = score * 0.8
        except:
            pass

        # VWAP
        vwap_s,vwap_tag=analyze_vwap(d15)
        score+=vwap_s; breakdown['VWAP']=vwap_s
        if vwap_s!=0:
            tags.append(vwap_tag)

        # 4H 壓力支撐
        r4h=d4h['h'].tail(20).max(); s4h=d4h['l'].tail(20).min(); mid4=(r4h+s4h)/2; ps=0
        atr_4h_s=ta.atr(d4h['h'],d4h['l'],d4h['c'],length=14)
        atr_4h=safe_last(atr_4h_s, curr*0.01)
        dist_res = (r4h - curr) / atr_4h if atr_4h>0 else 999
        dist_sup = (curr - s4h) / atr_4h if atr_4h>0 else 999
        sr_near = 0.5 if is_major else 0.3
        sr_mid  = 1.0 if is_major else 0.7
        if dist_res < 0.3:
            ps=W["support_res"];     tags.append("突破壓力{:.4f}".format(r4h))
        elif dist_sup < 0.3:
            ps=-W["support_res"];    tags.append("跌破支撐{:.4f}".format(s4h))
        elif dist_res < sr_near:
            ps=W["support_res"]//2;  tags.append("接近壓力{:.4f}".format(r4h))
        elif dist_sup < sr_near:
            ps=W["support_res"]//2;  tags.append("接近支撐{:.4f}".format(s4h))
        elif dist_sup < sr_mid:
            ps=W["support_res"]//3;  tags.append("支撐區間內")
        elif curr>mid4:
            ps=W["support_res"]//4;  tags.append("區間上半")
        else:
            ps=-W["support_res"]//4; tags.append("區間下半")
        score+=ps; breakdown['壓力支撐({:.4f}/{:.4f})'.format(s4h,r4h)]=ps

        # Trend Magic + 趨勢線
        tm_s, tm_tag = analyze_trend_magic(d4h)
        tl_s, tl_tag = analyze_trend(d4h)
        if (tm_s > 0 and tl_s > 0) or (tm_s < 0 and tl_s < 0):
            trend_final = tm_s
            if tl_tag != "趨勢中性":
                tags.append(tl_tag)
        else:
            trend_final = (tm_s + tl_s) // 2
        trend_final = min(max(trend_final, -W["trendline"]), W["trendline"])
        score += trend_final; breakdown['TrendMagic'] = trend_final
        if tm_tag and tm_tag != "TM中性":
            tags.append(tm_tag)

        # K棒
        cs,cd=analyze_candles(d15)
        score+=cs; breakdown['K棒型態']=cs
        if cd!="無特殊K棒":
            tags.append(cd)

        # 圖形型態
        chs,chd=analyze_chart_pattern(d4h)
        score+=chs; breakdown['圖形型態']=chs
        if chd!="無形態":
            tags.append(chd)

        # OB
        ob_s,ob_tag=analyze_order_block(d4h, is_major=is_major)
        score+=ob_s; breakdown['OB機構']=ob_s
        if ob_tag!="無OB":
            tags.append(ob_tag)

        # ICT
        ict_s,ict_tag=analyze_ict(d4h, d15)
        score+=ict_s; breakdown['BOS/CHoCH']=ict_s
        if ict_tag!="無ICT訊號":
            tags.append(ict_tag)

        # FVG
        fvg_s,fvg_tag=analyze_fvg(d4h)
        fvg_bonus = min(max(fvg_s, -3), 3)
        score+=fvg_bonus; breakdown['FVG缺口']=fvg_bonus
        if fvg_tag!="無FVG":
            tags.append(fvg_tag)

        # 流動性掃單
        liq_s,liq_tag=analyze_liquidity_sweep(d15)
        score+=liq_s; breakdown['流動性掃單']=liq_s
        if liq_tag!="無掃單":
            tags.append(liq_tag)

        # 莊家量能
        ws=analyze_whale(d15)
        score+=ws; breakdown['莊家量能']=ws
        if ws>3:
            tags.append("異常放量")

        # 暴拉 / 暴跌前置蓄勢結構
        pre_s, pre_tag = analyze_pre_breakout_setup(d15, d4h)
        score += pre_s; breakdown['蓄勢結構'] = pre_s
        if pre_tag and '無明顯' not in pre_tag and '不足' not in pre_tag:
            tags.append(pre_tag)

        # 假突破 / 假跌破過濾
        fake_s, fake_tag, fake_meta = analyze_fake_breakout(d15, score)
        score += fake_s; breakdown['假突破濾網'] = fake_s
        if fake_meta.get('fakeout'):
            tags.append(fake_tag)

        # 新聞
        raw_ns = STATE["news_score"]
        ns = round(max(min(raw_ns, 5), -5) / 5 * NEWS_WEIGHT)
        score += ns; breakdown['新聞情緒'] = ns

        # 時段額外分數
        sess_score = get_session_score()
        if sess_score != 0:
            score += sess_score
            breakdown['時段分數'] = sess_score

        score=min(max(round(score,1),-100),100)

        # 追漲殺跌懲罰：已離均線太遠時先降權，再等回踩/反彈進
        ext_s, ext_tag = analyze_extension_risk(d15, score)
        score += ext_s; breakdown['追價風險'] = ext_s
        if ext_s != 0:
            tags.append(ext_tag)

        # ===== ATR 改這裡：SL/TP 改用 15m ATR =====
        atr15_s = ta.atr(d15['h'], d15['l'], d15['c'], length=14)
        atr15   = safe_last(atr15_s, curr * 0.01)

        atr4h_s = ta.atr(d4h['h'], d4h['l'], d4h['c'], length=14)
        atr4h   = safe_last(atr4h_s, curr * 0.02)

        # 正式拿 15m ATR 當 SL / TP 基準
        atr = atr15

        active_keys=[k for k,v in breakdown.items() if v!=0]
        sl_mult,tp_mult=get_best_atr_params(active_keys)

        # 山寨幣波動率動態調整
        try:
            vol_now  = float(d15['v'].tail(96).sum())
            vol_prev = float(d15['v'].tail(192).head(96).sum())
            vol_ratio = vol_now / max(vol_prev, 1e-9)

            if vol_ratio > 2.5:
                tp_mult = round(min(tp_mult * 1.4, 6.0), 2)
                sl_mult = round(max(sl_mult * 0.85, 1.2), 2)
                tags.append("量能暴增{:.1f}x擴TP".format(vol_ratio))
            elif vol_ratio > 1.5:
                tp_mult = round(min(tp_mult * 1.15, 5.0), 2)
                tags.append("量能放大{:.1f}x".format(vol_ratio))
            elif vol_ratio < 0.5:
                tp_mult = round(max(tp_mult * 0.85, 1.5), 2)
                sl_mult = round(min(sl_mult * 1.1, 3.0), 2)
                tags.append("縮量收緊TP")

            if curr < 0.01:
                sl_mult = round(max(sl_mult * 1.3, 1.5), 2)
                tp_mult = round(min(tp_mult * 1.5, 7.0), 2)
            elif curr < 1.0:
                sl_mult = round(max(sl_mult * 1.15, 1.3), 2)
                tp_mult = round(min(tp_mult * 1.2, 5.0), 2)
        except:
            pass

        sl_mult = max(sl_mult, 1.5)
        tp_mult = max(tp_mult, 2.5)
        if tp_mult / max(sl_mult, 0.1) < 1.5:
            tp_mult = round(sl_mult * 1.5, 2)

        # 炒頂炒底偵測
        rsi_val = safe_last(ta.rsi(d15['c'], length=14), 50)
        overbought  = rsi_val > 75
        oversold    = rsi_val < 25
        if overbought and score > 0:
            score = score * 0.7
            tp_mult = round(tp_mult * 0.8, 2)
            tags.append("⚠️RSI超買炒頂風險")
            breakdown['炒頂警告'] = -5
        elif oversold and score < 0:
            score = score * 0.7
            tp_mult = round(tp_mult * 0.8, 2)
            tags.append("⚠️RSI超賣炒底風險")
            breakdown['炒底警告'] = -5

        # ===== 結構 / 波動濾網 =====
        # 方向一致但進場位置差時，避免只有分數高就硬上
        if score > 0 and regime_bias < 0:
            score *= 0.65
            breakdown['方向衝突'] = -8
            tags.append('多分數但方向衝突')
        elif score < 0 and regime_bias > 0:
            score *= 0.65
            breakdown['方向衝突'] = 8
            tags.append('空分數但方向衝突')

        # 報酬風險比過低直接降權
        rr_ratio = abs(tp - curr) / max(abs(curr - sl), 1e-9)
        if rr_ratio < 1.35:
            score *= 0.72
            breakdown['風報比不足'] = -6 if score > 0 else 6
            tags.append('風報比過低')
        elif rr_ratio >= 1.8:
            breakdown['風報比優秀'] = 3 if score > 0 else -3
            score += 3 if score > 0 else -3

        # 進場品質太差就再降權，避免追價或亂抄底
        if abs(entry_s0) <= 0:
            score *= 0.78
            breakdown['進場品質過濾'] = -5 if score > 0 else 5
            tags.append('進場品質不足')

        atr_pct = atr / max(curr, 1e-9)
        if atr_pct > 0.045:
            score *= 0.75
            tags.append("高波動降權")
            breakdown['高波動過熱'] = -4 if score > 0 else 4

        # 4H 主趨勢對齊：逆 4H 趨勢時直接降權，避免高分逆勢硬上
        ema21_4h = safe_last(ta.ema(d4h['c'], length=21), curr)
        ema55_4h = safe_last(ta.ema(d4h['c'], length=55), curr)
        if score > 0 and not (curr > ema21_4h > ema55_4h):
            score *= 0.7
            tags.append("逆4H趨勢降權")
            breakdown['4H趨勢不順'] = -6
        elif score < 0 and not (curr < ema21_4h < ema55_4h):
            score *= 0.7
            tags.append("逆4H趨勢降權")
            breakdown['4H趨勢不順'] = 6

        # ===== 用 15m ATR 算 SL / TP =====
        if score > 0:
            sl = round(curr - atr * sl_mult, 6)
            tp = round(curr + atr * tp_mult, 6)
        else:
            sl = round(curr + atr * sl_mult, 6)
            tp = round(curr - atr * tp_mult, 6)

        ep = round((atr * tp_mult) / curr * 100 * 20, 2)
        score = min(max(round(score, 1), -100), 100)
        breakdown['RR'] = round(rr_ratio, 2)
        breakdown['RegimeBias'] = regime_bias * 2
        breakdown['EntryGate'] = entry_s0

        del d15,d4h,d1d; gc.collect()
        return score,"|".join(tags),curr,sl,tp,ep,breakdown,atr,atr15,atr4h,sl_mult,tp_mult

    except Exception as e:
        import traceback
        print("analyze {} 失敗: {} \n{}".format(symbol, e, traceback.format_exc()[-300:]))
        return 0,"錯誤:{}".format(str(e)[:40]),0,0,0,0,{},0,0,0,2.0,3.0

# =====================================================
# 新聞執行緒
# =====================================================
NEWS_CACHE = {
    "score": 0,
    "sentiment": "中性 ➡️",
    "summary": "",
    "latest_title": "",
    "updated_at": 0,
}
NEWS_LOCK = threading.Lock()

def get_cached_news_score():
    with NEWS_LOCK:
        return dict(NEWS_CACHE)

def set_cached_news(score, sentiment, summary, latest_title):
    with NEWS_LOCK:
        NEWS_CACHE.update({
            "score": int(max(min(score, 5), -5)),
            "sentiment": sentiment or "中性 ➡️",
            "summary": summary or "",
            "latest_title": latest_title or "",
            "updated_at": time.time(),
        })

def fetch_crypto_news():
    """從多個來源抓取加密貨幣和宏觀經濟新聞"""
    news_list = []
    
    # 來源1: Binance BTC 即時價格（判斷市場情緒基礎）
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=8)
        d = r.json()
        pct = float(d.get('priceChangePercent', 0))
        price = float(d.get('lastPrice', 0))
        news_list.append("BTC 24小時變動 {:.1f}%，現價 ${:.0f}".format(pct, price))
    except: pass
    
    # 來源2: CryptoPanic
    try:
        r = requests.get(
            "https://cryptopanic.com/api/v1/posts/?auth_token={}&public=true&filter=hot".format(PANIC_API_KEY),
            timeout=8)
        d = r.json()
        if d.get('results'):
            for item in d['results'][:3]:
                news_list.append(item['title'])
    except: pass
    
    # 來源3: RSS 新聞（多個來源）
    rss_urls = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    ]
    import re as _re
    for url in rss_urls:
        try:
            r = requests.get(url, timeout=8)
            items = _re.findall(r'<item[^>]*>.*?</item>', r.text, _re.DOTALL)
            for item in items[:2]:
                t = _re.search(r'<title>(.*?)</title>', item, _re.DOTALL)
                if t:
                    raw = t.group(1).replace('<![CDATA[','').replace(']]>','').strip()
                    if raw: news_list.append(raw)
        except: pass
    
    return news_list[:8]  # 最多8條新聞

def analyze_news_with_ai(news_list):
    """用 AI / 關鍵字分析新聞對加密市場的影響，回傳評分和摘要"""
    if not news_list:
        return 0, "無新聞", "無法獲取新聞"

    news_text = "\n".join(["- " + n for n in news_list])
    prompt = """你是加密貨幣市場分析師。以下是最新新聞：

{}

請分析這些新聞對加密貨幣市場（特別是BTC、ETH等主流幣）的影響。
必須只回傳以下JSON格式，不要其他文字：
{{"score": 數字, "sentiment": "情緒描述", "summary": "50字內的影響摘要"}}

score 規則：
+5 = 極度利多（川普支持比特幣、ETF獲批、Fed降息）
+3 = 利多（機構買入、正面法規、市場上漲）
0 = 中性
-3 = 利空（監管打壓、市場下跌）
-5 = 極度利空（全面禁令、重大崩盤、戰爭）""".format(news_text)

    if ANTHROPIC_KEY:
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                timeout=12,
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            result = r.json()
            text_out = result['content'][0]['text'].strip()
            import json as _json
            data = _json.loads(text_out)
            return data.get('score', 0), data.get('sentiment', '中性'), data.get('summary', '')
        except Exception as e:
            print("Anthropic 分析失敗: {}".format(e))

    if OPENAI_API_KEY:
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": "Bearer {}".format(OPENAI_API_KEY),
                    "Content-Type": "application/json"
                },
                timeout=12,
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            result = r.json()
            text_out = result['choices'][0]['message']['content'].strip()
            import json as _json
            data = _json.loads(text_out)
            return data.get('score', 0), data.get('sentiment', '中性'), data.get('summary', '')
        except Exception as e:
            print("OpenAI 分析失敗: {}".format(e))

    combined = " ".join(news_list).lower()
    if any(w in combined for w in ["trump","bitcoin reserve","etf approved","fed cut","rate cut"]):
        return 5, "極度看多 🚀", "重大利多消息"
    elif any(w in combined for w in ["surge","rally","bullish","breakout","ath"]):
        return 3, "看多 📈", "市場正面消息"
    elif any(w in combined for w in ["crash","ban","bearish","collapse","dump","seized"]):
        return -4, "看空 📉", "市場負面消息"
    return 0, "中性 ➡️", "市場消息中性"

def news_thread():
    time.sleep(5)  # 等其他執行緒先啟動
    while True:
        try:
            news_list = fetch_crypto_news()
            if news_list:
                score, sentiment, summary = analyze_news_with_ai(news_list)
                score = max(min(int(score), 5), -5)  # 限制在 -5 到 +5
                title = news_list[0] if news_list else "無新聞"
                display = "[AI分析] {} | {}".format(sentiment, summary) if summary else title
                set_cached_news(score, sentiment, summary, display)
                update_state(
                    news_score=score,
                    news_sentiment=sentiment,
                    latest_news_title=display
                )
                print("AI新聞分析完成: {} 分 | {} | {}".format(score, sentiment, summary[:40]))
            else:
                update_state(news_sentiment="新聞獲取失敗，使用中性")
        except Exception as e:
            print("新聞執行緒錯誤: {}".format(e))
        time.sleep(300)  # 5分鐘更新一次（節省 AI API 費用）

# =====================================================
# 移動止盈追蹤系統
# =====================================================
# 記錄每個倉位的追蹤狀態
# { "BTC/USDT:USDT": {
#     "side": "long",
#     "entry_price": 70000,
#     "highest_price": 72000,   # 做多時的最高點
#     "lowest_price":  68000,   # 做空時的最低點
#     "trail_pct": 0.05,        # 回撤幾%觸發平倉（預設5%）
#     "initial_sl": 69000,      # 初始止損價
#     "atr": 500,               # 開倉時的ATR
# }}
TRAILING_STATE = {}
TRAILING_LOCK  = threading.Lock()
ORDER_LOCK     = threading.Lock()   # 防止同時下多筆單超過7個持倉
_ORDERED_THIS_SCAN = set()  # 本輪已下單的幣（防止同輪重複下單）
_ORDERED_LOCK = threading.Lock()

def detect_reversal(sym, side, current_price):
    """
    偵測趨勢反轉訊號（用於分批止盈的緊急平倉）
    回傳 (是否反轉, 原因)
    """
    try:
        ohlcv = exchange.fetch_ohlcv(sym, '15m', limit=20)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        c = df['c']; h = df['h']; l = df['l']; v = df['v']
        curr = float(c.iloc[-1])

        signals = []

        # 1. RSI 超買/超賣背離
        rsi_s = ta.rsi(c, length=14)
        rsi = float(rsi_s.iloc[-1]) if not pd.isna(rsi_s.iloc[-1]) else 50
        if side == 'long' and rsi > 78:
            signals.append("RSI超買{:.0f}".format(rsi))
        elif side == 'short' and rsi < 22:
            signals.append("RSI超賣{:.0f}".format(rsi))

        # 2. 成交量異常放大（反轉訊號）
        vol_avg = float(v.tail(10).mean())
        vol_now = float(v.iloc[-1])
        if vol_now > vol_avg * 2.5:
            signals.append("量能暴增{:.1f}x".format(vol_now/vol_avg))

        # 3. 強力反轉K棒
        o_last = float(df['o'].iloc[-1])
        c_last = float(c.iloc[-1])
        h_last = float(h.iloc[-1])
        l_last = float(l.iloc[-1])
        body = abs(c_last - o_last)
        range_ = h_last - l_last
        if range_ > 0:
            if side == 'long':
                # 做多時出現大陰線（實體>60%）
                if c_last < o_last and body / range_ > 0.6:
                    signals.append("強力陰線反轉")
                # 上影線過長（被壓回）
                upper_shadow = h_last - max(c_last, o_last)
                if upper_shadow > body * 2:
                    signals.append("長上影線壓回")
            elif side == 'short':
                # 做空時出現大陽線（實體>60%）
                if c_last > o_last and body / range_ > 0.6:
                    signals.append("強力陽線反轉")
                # 下影線過長（被撐起）
                lower_shadow = min(c_last, o_last) - l_last
                if lower_shadow > body * 2:
                    signals.append("長下影線撐起")

        # 4. 連續3根反向K棒
        last3_c = c.iloc[-4:-1].values
        if side == 'long':
            if all(last3_c[i] < last3_c[i-1] for i in range(1,3)):
                signals.append("連3根下跌")
        elif side == 'short':
            if all(last3_c[i] > last3_c[i-1] for i in range(1,3)):
                signals.append("連3根上漲")

        # 需要 2 個以上訊號才確認反轉（避免假信號）
        if len(signals) >= 2:
            return True, "反轉訊號: " + "|".join(signals)
        return False, ""
    except Exception as e:
        return False, ""

def partial_close_position(sym, contracts, side, ratio, reason=""):
    """部分平倉"""
    try:
        close_side = 'sell' if side == 'long' else 'buy'
        partial_qty = abs(contracts) * ratio
        partial_qty = exchange.amount_to_precision(sym, partial_qty)
        exchange.create_order(sym, 'market', close_side, partial_qty, params={
            'reduceOnly': True,
            'posSide':    side,
            'tdMode':     'cross',
        })
        print("📤 部分平倉 {} {:.0f}% | {}".format(sym, ratio*100, reason))
        return True
    except Exception as e:
        print("部分平倉失敗 {}: {}".format(sym, e))
        return False

def update_trailing(sym, side, current_price, atr):
    """
    分批止盈 + 動態止損系統
    目標1（+1.2ATR）→ 平倉25%，止損移至保本
    目標2（+2.4ATR）→ 再平倉35%，止損移至+0.8ATR
    目標3（+4.2ATR）→ 剩餘部位跟著走，止損明顯收緊
    反轉偵測        → 立即全平鎖利
    """
    with TRAILING_LOCK:
        if sym not in TRAILING_STATE:
            return False, "", 0

        ts      = TRAILING_STATE[sym]
        entry   = ts.get("entry_price", current_price)
        atr_val = ts.get("atr", current_price * 0.01)
        if atr_val <= 0: atr_val = current_price * 0.01

        partial_done = ts.get("partial_done", 0)  # 已完成幾批止盈

        if side == "long":
            prev_high = ts.get("highest_price", entry)
            if current_price > prev_high:
                ts["highest_price"] = current_price
            highest    = ts.get("highest_price", current_price)
            profit_atr = (current_price - entry) / atr_val

            # ── 分批止盈 ──
            # 目標1：+1.5ATR → 平30%，止損移到保本
            if profit_atr >= 1.2 and partial_done == 0:
                ts["partial_done"]  = 1
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), entry)
                ts["trail_pct"]     = 0.05
                print("🎯 目標1達成 {} +{:.1f}ATR → 平25%，止損移保本".format(sym, profit_atr))
                return True, "目標1平倉25% +{:.1f}ATR".format(profit_atr), 0.25

            # 目標2：+2.4ATR → 再平35%，止損移到+0.8ATR
            elif profit_atr >= 2.4 and partial_done == 1:
                ts["partial_done"]  = 2
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), entry + atr_val * 0.8)
                ts["trail_pct"]     = 0.04
                print("🎯 目標2達成 {} +{:.1f}ATR → 再平35%，止損+0.8ATR".format(sym, profit_atr))
                return True, "目標2平倉35% +{:.1f}ATR".format(profit_atr), 0.35

            # 目標3：+4.2ATR → 剩餘跟緊
            elif profit_atr >= 4.2 and partial_done == 2:
                ts["partial_done"]  = 3
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), current_price - atr_val * 1.2)
                ts["trail_pct"]     = 0.028
                print("🎯 目標3達成 {} +{:.1f}ATR → 緊縮移動止盈".format(sym, profit_atr))

            # ── 止損移動（只升不降）──
            if profit_atr >= 4.2:
                new_sl = current_price - atr_val * 1.2
                ts["trail_pct"] = 0.028
            elif profit_atr >= 2.4:
                new_sl = entry + atr_val * 1.4
                ts["trail_pct"] = max(ts.get("trail_pct", 0.05) * 0.85, 0.03)
            elif profit_atr >= 1.2:
                new_sl = entry
            else:
                new_sl = ts.get("initial_sl", entry - atr_val * 2)

            if new_sl > ts.get("initial_sl", 0):
                ts["initial_sl"] = new_sl

            # ── 觸發條件 ──
            trail_price = highest * (1 - ts.get("trail_pct", 0.05))
            current_sl  = ts.get("initial_sl", 0)

            # 從最高點回撤觸發全平
            if current_price < trail_price and partial_done >= 1:
                return True, "移動止盈觸發 峰:{:.6f} 現:{:.6f} 回撤{:.1f}%".format(
                    highest, current_price, (highest-current_price)/highest*100), 1.0

            # 跌破止損線
            if current_sl > 0 and current_price < current_sl:
                sl_type = "保本止損" if abs(current_sl-entry)<atr_val*0.1 else "移動止損"
                return True, "{} @{:.6f}".format(sl_type, current_price), 1.0

        elif side == "short":
            prev_low = ts.get("lowest_price", entry)
            if current_price < prev_low:
                ts["lowest_price"] = current_price
            lowest     = ts.get("lowest_price", current_price)
            profit_atr = (entry - current_price) / atr_val

            if profit_atr >= 1.2 and partial_done == 0:
                ts["partial_done"] = 1
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), entry)
                ts["trail_pct"]    = 0.05
                return True, "目標1平倉25% +{:.1f}ATR".format(profit_atr), 0.25

            elif profit_atr >= 2.4 and partial_done == 1:
                ts["partial_done"] = 2
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), entry - atr_val * 0.8)
                ts["trail_pct"]    = 0.04
                return True, "目標2平倉35% +{:.1f}ATR".format(profit_atr), 0.35

            elif profit_atr >= 4.2 and partial_done == 2:
                ts["partial_done"] = 3
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), current_price + atr_val * 1.2)
                ts["trail_pct"]    = 0.028

            if profit_atr >= 4.2:
                new_sl = current_price + atr_val * 1.2
                ts["trail_pct"] = 0.028
            elif profit_atr >= 2.4:
                new_sl = entry - atr_val * 1.4
            elif profit_atr >= 1.2:
                new_sl = entry
            else:
                new_sl = ts.get("initial_sl", entry + atr_val * 2)

            if new_sl < ts.get("initial_sl", float('inf')):
                ts["initial_sl"] = new_sl

            trail_price = lowest * (1 + ts.get("trail_pct", 0.05))
            current_sl  = ts.get("initial_sl", float('inf'))

            if current_price > trail_price and partial_done >= 1:
                return True, "移動止盈觸發 谷:{:.6f} 現:{:.6f} 反彈{:.1f}%".format(
                    lowest, current_price, (current_price-lowest)/lowest*100), 1.0

            if current_sl < float('inf') and current_price > current_sl:
                sl_type = "保本止損" if abs(current_sl-entry)<atr_val*0.1 else "移動止損"
                return True, "{} @{:.6f}".format(sl_type, current_price), 1.0

        # ── 時間止損：15 根 15m K 仍未有效走出，就離場 ──
        open_ts = ts.get("entry_time_ts", 0)
        time_stop_sec = ts.get("time_stop_sec", TIME_STOP_BARS_15M * 15 * 60)
        if open_ts and time.time() - open_ts >= time_stop_sec:
            move_pct = abs(current_price - entry) / max(entry, 1e-9)
            if move_pct < max(atr_val / max(entry, 1e-9) * 1.2, 0.006):
                return True, "時間止損 {} 分鐘未脫離成本區".format(int(time_stop_sec/60)), 1.0

        return False, "", 0


def close_position(sym, contracts, side):
    """平倉單一倉位"""
    try:
        close_side = 'sell' if side == 'long' else 'buy'
        exchange.create_order(sym, 'market', close_side, abs(contracts),
                              params={'reduceOnly': True})
        with TRAILING_LOCK:
            if sym in TRAILING_STATE:
                del TRAILING_STATE[sym]
        print("移動止盈平倉成功: {} {}口".format(sym, contracts))
        return True
    except Exception as e:
        print("移動止盈平倉失敗 {}: {}".format(sym, e))
        return False

def trailing_stop_thread():
    """獨立執行緒，每3秒追蹤所有持倉"""
    print("移動止盈執行緒啟動")
    while True:
        try:
            with STATE_LOCK:
                active = list(STATE["active_positions"])

            for pos in active:
                sym       = pos['symbol']
                side      = (pos.get('side') or '').lower()
                contracts = float(pos.get('contracts', 0) or 0)
                if abs(contracts) == 0:
                    continue

                # 抓即時價格（加 timeout 保護）
                try:
                    ticker = exchange.fetch_ticker(sym)
                    curr   = float(ticker['last'])
                    time.sleep(0.2)  # 避免 API 限速
                except:
                    continue

                # 若這個倉位還沒在追蹤，加入（不抓K線，用進場價直接估算）
                # 有新持倉時，清除該幣的 FVG 掛單記錄（已成交或已手動下單）
                with FVG_LOCK:
                    if sym in FVG_ORDERS:
                        print("✅ {} 已有持倉，清除 FVG 掛單記錄".format(sym))
                        FVG_ORDERS.pop(sym, None)
                        update_state(fvg_orders=dict(FVG_ORDERS))

                with TRAILING_LOCK:
                    if sym not in TRAILING_STATE:
                        entry = float(pos.get('entryPrice', curr) or curr)
                        atr = float(SIGNAL_META_CACHE.get(sym, {}).get('atr15', 0) or SIGNAL_META_CACHE.get(sym, {}).get('atr', 0) or 0)
                        if atr <= 0:
                            atr = fetch_real_atr(sym, '15m', 60) or entry * 0.008
                        initial_sl = entry - atr * 2 if side == 'long' else entry + atr * 2
                        trail_pct  = 0.05  # 預設5%回撤

                        TRAILING_STATE[sym] = {
                            "side":          side,
                            "entry_price":   entry,
                            "highest_price": curr if side == 'long' else float('inf'),
                            "lowest_price":  curr if side == 'short' else float('inf'),
                            "trail_pct":     trail_pct,
                            "initial_sl":    initial_sl,
                            "atr":           atr,
                            "entry_time_ts": time.time(),
                            "time_stop_sec": TIME_STOP_BARS_15M * 15 * 60,
                            "partial_done": 0,
                        }
                        print("開始追蹤 {} {} 回撤:{:.1f}% 止損:{:.6f}".format(
                            sym, side, trail_pct*100, initial_sl))

                # 檢查是否觸發
                should_close, reason, close_ratio = update_trailing(sym, side, curr, 0)
                if should_close:
                    if 0 < close_ratio < 1.0:
                        # 分批止盈（部分平倉）
                        print("🎯 分批止盈 {} {:.0f}% | {}".format(sym, close_ratio*100, reason))
                        partial_close_position(sym, contracts, side, close_ratio, reason)
                        # 更新持倉數量（實際會在下次 position_thread 更新）
                    else:
                        # 全平
                        print("📤 全部平倉 {} | {}".format(sym, reason))
                        close_position(sym, contracts, side)

                # 反轉偵測（有未實現利潤才偵測，避免浪費API）
                elif side in ('long', 'short'):
                    with TRAILING_LOCK:
                        ts_now = TRAILING_STATE.get(sym, {})
                        entry_p = ts_now.get("entry_price", curr)
                        profit_pct = (curr - entry_p)/entry_p if side=='long' else (entry_p - curr)/entry_p
                    if profit_pct > 0.01:  # 有超過1%利潤才偵測反轉
                        is_reversal, rev_reason = detect_reversal(sym, side, curr)
                        if is_reversal:
                            print("⚡ 反轉訊號！{} {} → 立即平倉鎖利 | {}".format(sym, side, rev_reason))
                            close_position(sym, contracts, side)
                    # 記錄到交易歷史
                    with STATE_LOCK:
                        STATE["trade_history"].insert(0, {
                            "symbol":      sym,
                            "side":        "移動止盈平倉",
                            "score":       0,
                            "price":       curr,
                            "stop_loss":   0,
                            "take_profit": 0,
                            "est_pnl":     0,
                            "order_usdt":  0,
                            "time":        tw_now_str(),
                            "reason":      reason,
                        })

            # 更新追蹤狀態到 UI
            with TRAILING_LOCK:
                ui_info = {}
                for s, ts in TRAILING_STATE.items():
                    side_t = ts.get('side','')
                    highest = ts.get('highest_price', 0)
                    lowest  = ts.get('lowest_price', float('inf'))
                    trail   = ts.get('trail_pct', 0.05)
                    if side_t == 'long' and highest != float('inf'):
                        trail_price = highest * (1 - trail)
                        ui_info[s] = {
                            "side": "做多",
                            "peak": round(highest, 6),
                            "trail_price": round(trail_price, 6),
                            "trail_pct": round(trail * 100, 1),
                        }
                    elif side_t == 'short' and lowest != float('inf'):
                        trail_price = lowest * (1 + trail)
                        ui_info[s] = {
                            "side": "做空",
                            "peak": round(lowest, 6),
                            "trail_price": round(trail_price, 6),
                            "trail_pct": round(trail * 100, 1),
                        }
            update_state(trailing_info=ui_info)
        except Exception as e:
            import traceback
            print("移動止盈異常: {}".format(e))
            print(traceback.format_exc())
        time.sleep(10)  # 每10秒追蹤一次

# =====================================================
# 持倉 + 帳戶資產監控（每 10 秒）
# =====================================================
PREV_POSITION_SYMS = set()

def position_thread():
    global PREV_POSITION_SYMS
    while True:
        try:
            # 抓帳戶總資產
            try:
                bal=exchange.fetch_balance()
                equity=float(bal.get('USDT',{}).get('total',0) or
                             bal.get('total',{}).get('USDT',0) or 0)
                update_state(equity=round(equity,4))
            except:
                pass

            raw=exchange.fetch_positions()
            active=[p for p in raw if abs(float(p.get('contracts',0) or 0))>0]
            pnl=sum(float(p.get('unrealizedPnl',0) or 0) for p in active)
            update_state(
                active_positions=active,
                total_pnl=round(pnl,4),
                last_update=tw_now_str()
            )

            curr_syms={p['symbol'] for p in active}
            closed_syms=PREV_POSITION_SYMS-curr_syms
            for sym in closed_syms:
                if not queue_learn_for_closed_symbol(sym, curr_syms):
                    print("警告: {} 無學習紀錄（可能是手動下單）".format(sym))

            # 補償機制：避免交易所 TP/SL 已成交，但因重啟/漏輪詢沒被記錄
            with LEARN_LOCK:
                open_symbols = list({t.get('symbol') for t in LEARN_DB.get('trades', []) if t.get('result') == 'open' and t.get('symbol')})
            for sym in open_symbols:
                if sym not in curr_syms:
                    queue_learn_for_closed_symbol(sym, curr_syms)

            PREV_POSITION_SYMS=curr_syms
            # 每輪備份狀態
            save_full_state()
            save_risk_state()
        except Exception as e:
            print("持倉更新失敗: {}".format(e))
        time.sleep(10)

# =====================================================
# 學習系統：平倉後分析
# =====================================================
def learn_from_closed_trade(trade_id):
    with LEARN_LOCK:
        trade = next((t for t in LEARN_DB["trades"] if t["id"] == trade_id), None)
    if not trade or trade["result"] != "open":
        PENDING_LEARN_IDS.discard(trade_id)
        return
    time.sleep(5)
    try:
        sym = trade["symbol"]
        side = trade["side"]
        exit_p = float(trade.get("exit_price", 0) or 0)
        entry_p = float(trade.get("entry_price", 0) or 0)
        leverage = float(trade.get("leverage", 1) or 1)
        margin_pct = float(trade.get("margin_pct", RISK_PCT) or RISK_PCT)

        # 1) 純價格邊際（不含槓桿）
        raw_pct = ((exit_p - entry_p) / max(entry_p, 1e-9) * 100.0) if side == "buy" else ((entry_p - exit_p) / max(entry_p, 1e-9) * 100.0)

        # 2) 交易所視角損益（含槓桿，不含保證金占比）
        leveraged_pnl_pct = raw_pct * max(leverage, 1.0)

        # 3) 帳戶視角損益（含槓桿與保證金占比）→ 給 AI 與 UI 學習/統計主用
        account_pnl_pct = leveraged_pnl_pct * max(margin_pct, 0.0001)

        # 給 AI 學習的主口徑：實際帳戶影響，不再被壓成 -0.0
        learn_pnl_pct = account_pnl_pct
        result = "win" if learn_pnl_pct > 0 else "loss"

        time.sleep(60)
        post_ohlcv = exchange.fetch_ohlcv(sym, '15m', limit=12)
        post_closes = [c[4] for c in post_ohlcv[-10:]]
        future_max = max(post_closes); future_min = min(post_closes)
        missed_pct = (future_max - exit_p) / max(exit_p, 1e-9) * 100 * max(leverage, 1.0) if side == "buy" else (exit_p - future_min) / max(exit_p, 1e-9) * 100 * max(leverage, 1.0)

        bd = trade.get("breakdown", {})
        active_keys = [k for k, v in bd.items() if v != 0]
        pkey = "|".join(sorted(active_keys))

        with LEARN_LOCK:
            db = LEARN_DB
            for t in db["trades"]:
                if t["id"] == trade_id:
                    t["result"] = result
                    t["edge_pct"] = round(raw_pct, 4)
                    t["pnl_pct"] = round(raw_pct, 4)  # legacy兼容：保留純價格邊際
                    t["leveraged_pnl_pct"] = round(leveraged_pnl_pct, 4)
                    t["account_pnl_pct"] = round(account_pnl_pct, 4)
                    t["learn_pnl_pct"] = round(learn_pnl_pct, 4)
                    t["post_candles"] = post_closes
                    t["missed_move_pct"] = round(missed_pct, 2)
                    t["exit_time"] = tw_now_str("%Y-%m-%d %H:%M:%S")
                    break

            metric = float(learn_pnl_pct)

            # 更新指標組合統計（用 learn_pnl_pct，不再用被壓縮的 raw pct）
            if pkey not in db["pattern_stats"]:
                db["pattern_stats"][pkey] = {
                    "win": 0, "loss": 0, "sample_count": 0, "total_pnl": 0.0,
                    "avg_pnl": 0.0, "best_sl": trade.get("atr_mult_sl", 2.0),
                    "best_tp": trade.get("atr_mult_tp", 3.0), "tp_candidates": [], "sl_candidates": []
                }
            ps = db["pattern_stats"][pkey]
            ps["sample_count"] += 1
            ps["total_pnl"] += metric
            ps["avg_pnl"] = round(ps["total_pnl"] / max(ps["sample_count"], 1), 4)
            if result == "win":
                ps["win"] += 1; ps["tp_candidates"].append(trade.get("atr_mult_tp", 3.0))
            else:
                ps["loss"] += 1; ps["sl_candidates"].append(trade.get("atr_mult_sl", 2.0))
            if ps["sample_count"] >= 5:
                wr = ps["win"] / max(ps["sample_count"], 1)
                if wr >= 0.6 and ps["tp_candidates"]:
                    ps["best_tp"] = round(min(max(ps["tp_candidates"]) * 1.1, 5.0), 2)
                    ps["best_sl"] = round(max(ps.get("best_sl", 2.0) * 0.95, 1.8), 2)
                elif wr < 0.4:
                    ps["best_sl"] = round(min(ps.get("best_sl", 2.0) * 0.85, 1.8), 2)
                    ps["best_tp"] = round(max(ps.get("best_tp", 3.5) * 0.9, 2.8), 2)

            # 更新幣種統計
            ss = db.setdefault("symbol_stats", {})
            if sym not in ss:
                ss[sym] = {"win": 0, "loss": 0, "count": 0, "total_pnl": 0.0, "total_margin_pct": 0.0}
            ss[sym]["count"] += 1
            ss[sym]["total_pnl"] += metric
            ss[sym]["total_margin_pct"] += margin_pct
            if result == "win":
                ss[sym]["win"] += 1
            else:
                ss[sym]["loss"] += 1

            # 全域統計（只看 live）
            all_closed = [t for t in db["trades"] if _is_live_source(t.get("source")) and t["result"] in ("win", "loss")]
            if all_closed:
                db["total_trades"] = len(all_closed)
                wins = sum(1 for t in all_closed if t["result"] == "win")
                db["win_rate"] = round(wins / len(all_closed) * 100, 1)
                db["avg_pnl"] = round(sum(_trade_learn_metric(t) for t in all_closed) / len(all_closed), 4)
                recent = all_closed[-20:]
                if len(recent) >= 10:
                    rwr = sum(1 for t in recent if t["result"] == "win") / len(recent)
                    if rwr >= 0.65:
                        db["atr_params"]["default_tp"] = round(min(db["atr_params"]["default_tp"] * 1.05, 5.0), 2)
                    elif rwr < 0.35:
                        db["atr_params"]["default_sl"] = round(max(db["atr_params"]["default_sl"] * 0.92, 1.2), 2)
                        db["atr_params"]["default_tp"] = round(max(db["atr_params"]["default_tp"] * 0.95, 1.5), 2)

            all_closed_count = len([t for t in db["trades"] if _is_live_source(t.get("source")) and t["result"] in ("win", "loss")])
            if all_closed_count >= 50 and all_closed_count % 10 == 0:
                _auto_adjust_weights(db)

            regime = str((trade.get('breakdown') or {}).get('Regime', 'neutral') or 'neutral')
            srs = db.setdefault('symbol_regime_stats', {})
            rk = f"{sym}|{regime}"
            if rk not in srs:
                srs[rk] = {'count': 0, 'win': 0, 'loss': 0, 'pnl_sum': 0.0, 'last_update': '--'}
            srs[rk]['count'] += 1
            srs[rk]['pnl_sum'] += metric
            srs[rk]['last_update'] = tw_now_str('%Y-%m-%d %H:%M:%S')
            if result == 'win':
                srs[rk]['win'] += 1
            else:
                srs[rk]['loss'] += 1

            save_learn_db(db)

        try:
            with AI_LOCK:
                AI_PANEL['last_learning'] = tw_now_str('%Y-%m-%d %H:%M:%S')
        except Exception:
            pass

        # 風控用 USDT 盈虧：優先使用實際已記錄，否則用帳戶損益近似
        pnl_usdt = float(trade.get("realized_pnl_usdt", 0) or 0)
        if pnl_usdt == 0:
            pnl_usdt = (learn_pnl_pct / 100.0) * float(STATE.get("equity", 10) or 10)
        record_trade_result(pnl_usdt)
        update_state(risk_status=get_risk_status())
        _refresh_learn_summary()
        print("✅ 學習完成 {} | edge:{:.4f}% | lev:{:.2f}% | acct:{:.4f}% | {}".format(sym, raw_pct, leveraged_pnl_pct, learn_pnl_pct, result))
        PENDING_LEARN_IDS.discard(trade_id)
    except Exception as e:
        PENDING_LEARN_IDS.discard(trade_id)
        print("學習失敗: {}".format(e))

def _auto_adjust_weights(db):
    """50筆後 AI 真正接管權重（修正版）"""
    try:
        global W  # ⭐ 關鍵：讓AI可以改全域權重

        trades = [t for t in db["trades"] if t["result"] in ("win","loss") and t.get("breakdown")]
        if len(trades) < 50:
            return

        indicator_stats = {}

        for t in trades:
            bd = t.get("breakdown", {})
            is_win = t["result"] == "win"

            for key, val in bd.items():
                if key not in indicator_stats:
                    indicator_stats[key] = {"win_sum":0,"loss_sum":0,"win_n":0,"loss_n":0}

                if is_win:
                    indicator_stats[key]["win_sum"] += abs(val)
                    indicator_stats[key]["win_n"] += 1
                else:
                    indicator_stats[key]["loss_sum"] += abs(val)
                    indicator_stats[key]["loss_n"] += 1

        contrib = {}

        for key, st in indicator_stats.items():
            if st["win_n"] + st["loss_n"] < 10:
                continue

            win_avg = st["win_sum"] / max(st["win_n"], 1)
            loss_avg = st["loss_sum"] / max(st["loss_n"], 1)

            # ⭐ 貢獻度（勝 > 敗 才加分）
            contrib[key] = max(win_avg - loss_avg, 0.01)

        if not contrib:
            return

        # ⭐ 正規化 → 變成100分權重
        total = sum(contrib.values())
        new_W = {}

        for k, v in contrib.items():
            new_W[k] = round(v / total * 100)

        # ⭐ 補滿100（避免誤差）
        diff = 100 - sum(new_W.values())
        if diff != 0:
            first_key = list(new_W.keys())[0]
            new_W[first_key] += diff

        # ⭐ 覆蓋原本權重（核心修復）
        for k in new_W:
            if k in W:
                W[k] = new_W[k]

        # ⭐ 存DB（保留你原本功能）
        db["indicator_contrib"] = {k: round(v, 3) for k, v in contrib.items()}

        print("🧠 AI權重已接管:", W)

    except Exception as e:
        print("權重調整失敗:", e)



def _refresh_learn_summary():
    live_closed = get_live_trades(closed_only=True)
    with LEARN_LOCK:
        db = LEARN_DB
        stats = {}
        sym_stats = {}
        for t in live_closed:
            bd = t.get("breakdown", {}) or {}
            active_keys = [k for k,v in bd.items() if v not in (0, None, "", False)]
            pkey = "|".join(sorted(active_keys))
            if pkey:
                st = stats.setdefault(pkey, {"win":0,"sample_count":0,"total_pnl":0.0})
                st["sample_count"] += 1
                st["total_pnl"] += _trade_learn_metric(t)
                if t.get("result") == "win":
                    st["win"] += 1
            sym = str(t.get("symbol") or "")
            if sym:
                ss = sym_stats.setdefault(sym, {"win":0,"count":0,"total_pnl":0.0})
                ss["count"] += 1
                ss["total_pnl"] += _trade_learn_metric(t)
                if t.get("result") == "win":
                    ss["win"] += 1

        ranked = sorted(stats.items(), key=lambda x: (x[1]["total_pnl"]/max(x[1]["sample_count"],1)), reverse=True)
        top3=[{"pattern":k[:45],
               "avg_pnl":round(v["total_pnl"]/max(v["sample_count"],1),2),
               "win_rate":round(v["win"]/max(v["sample_count"],1)*100,0),
               "count":v["sample_count"]} for k,v in ranked[:3]] if ranked else []
        worst3=[{"pattern":k[:45],
                 "avg_pnl":round(v["total_pnl"]/max(v["sample_count"],1),2),
                 "win_rate":round(v["win"]/max(v["sample_count"],1)*100,0),
                 "count":v["sample_count"]} for k,v in ranked[-3:]] if len(ranked)>=3 else []
        blocked=[{"symbol":s,
                  "win_rate":round(v["win"]/v["count"]*100,1),
                  "count":v["count"],
                  "avg_pnl":round(v["total_pnl"]/max(v["count"],1),2)}
                 for s,v in sym_stats.items()
                 if v["count"]>=8 and (v["win"]/v["count"]<0.4 or (v["total_pnl"]/max(v["count"],1))<0)]

        open_pnl_usdt = 0.0
        try:
            open_pnl_usdt = round(sum(float(p.get('unrealizedPnl',0) or 0) for p in STATE.get('active_positions', [])), 4)
        except Exception:
            pass

        total_trades = len(live_closed)
        wins = sum(1 for t in live_closed if t.get("result") == "win")
        avg_pnl = round(sum(_trade_learn_metric(t) for t in live_closed) / max(total_trades,1), 2) if total_trades else 0.0
        summary = {
            "total_trades": total_trades,
            "win_rate": round(wins / max(total_trades,1) * 100, 1) if total_trades else 0.0,
            "avg_pnl": avg_pnl,
            "open_pnl_usdt": open_pnl_usdt,
            "current_sl_mult": db["atr_params"]["default_sl"],
            "current_tp_mult": db["atr_params"]["default_tp"],
            "top_patterns": top3,
            "worst_patterns": worst3,
            "blocked_symbols": blocked,
            "data_scope": "live_only",
        }
    update_state(learn_summary=summary)


# =====================================================
# 下單（使用總資產 5% + 最高槓桿）
# =====================================================
def get_fvg_entry_price(symbol, side, current_price, atr):
    """
    計算最優進場價格：
    1) 先做追價保護，避免突破後最後一棒才去追
    2) 再找 FVG 缺口掛回踩/反彈單
    """
    try:
        pb_price, pb_note = get_breakout_pullback_entry(symbol, side, current_price, atr)
        if pb_price is not None:
            return pb_price, pb_note

        ohlcv = exchange.fetch_ohlcv(symbol, '15m', limit=60)
        df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
        hi = df['h'].tolist()
        lo = df['l'].tolist()
        n  = len(hi)

        best_fvg_price = None
        best_dist      = float('inf')

        for i in range(2, min(30, n)):
            idx = n - 1 - i
            if idx < 2: break

            k1_h = hi[idx-2]; k3_l = lo[idx]
            k1_l = lo[idx-2]; k3_h = hi[idx]

            if side == 'long' and k1_h < k3_l:
                # 做多 FVG（向上缺口）：等價格回到缺口頂部
                fvg_top    = k3_l
                fvg_bottom = k1_h
                fvg_mid    = (fvg_top + fvg_bottom) / 2
                # 未填補確認
                filled = any(lo[j] <= fvg_top and hi[j] >= fvg_bottom
                             for j in range(idx+1, n))
                if not filled:
                    dist = abs(current_price - fvg_mid) / max(atr, 1e-9)
                    if dist < best_dist:
                        best_dist = dist
                        best_fvg_price = fvg_top  # 在缺口頂部掛單

            elif side == 'short' and k1_l > k3_h:
                # 做空 FVG（向下缺口）：等價格反彈到缺口底部
                fvg_top    = k1_l
                fvg_bottom = k3_h
                fvg_mid    = (fvg_top + fvg_bottom) / 2
                filled = any(hi[j] >= fvg_bottom and lo[j] <= fvg_top
                             for j in range(idx+1, n))
                if not filled:
                    dist = abs(current_price - fvg_mid) / max(atr, 1e-9)
                    if dist < best_dist:
                        best_dist = dist
                        best_fvg_price = fvg_bottom  # 在缺口底部掛單

        # 距離判斷
        if best_fvg_price is None:
            return None, "無FVG缺口，直接市價"

        if best_dist > 2.0:
            return None, "FVG缺口太遠({:.1f}ATR)，不勉強".format(best_dist)

        if best_dist < 0.3:
            # 已經在缺口內，直接市價
            return None, "已在FVG缺口內，市價進場"

        return round(best_fvg_price, 6), "FVG限價{:.6f}（距離{:.1f}ATR）".format(
            best_fvg_price, best_dist)

    except Exception as e:
        return None, "FVG計算失敗"

def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except:
        return lo

def calc_dynamic_margin_pct(score, atr_ratio, trend_aligned, squeeze_ready, extended_risk, same_side_count, market_dir="中性", market_strength=0.0):
    """
    根據訊號品質/結構/波動決定當下保證金比例，限制 1% ~ 8%。
    - 弱訊號：1%~2%
    - 過門檻：3.5%~5.5%
    - 強共振：最高 8%
    """
    s = abs(float(score or 0))
    atr_ratio = float(atr_ratio or 0)

    if s < 48:
        base = 0.01
    elif s < 50:
        base = 0.02
    elif s < 52:
        base = 0.04
    elif s < 54:
        base = 0.055
    else:
        base = 0.07

    adj = 0.0
    if trend_aligned:
        adj += 0.005
    if squeeze_ready:
        adj += 0.005

    if market_strength >= 0.6:
        if (market_dir in ("強多", "多") and score > 0) or (market_dir in ("強空", "空") and score < 0):
            adj += 0.005
        elif market_dir != "中性":
            adj -= 0.01

    if extended_risk:
        adj -= 0.015
    if atr_ratio > 0.045:
        adj -= 0.01
    elif atr_ratio > 0.03:
        adj -= 0.005

    if same_side_count >= 4:
        adj -= 0.01
    elif same_side_count >= 2:
        adj -= 0.005

    return round(clamp(base + adj, MIN_MARGIN_PCT, MAX_MARGIN_PCT), 4)

def infer_margin_context(sig, same_side_count=0):
    try:
        desc = str(sig.get('desc', '') or '')
        breakdown = sig.get('breakdown', {}) or {}
        price = max(float(sig.get('price', 0) or 0), 1e-9)
        atr_val = float(sig.get('atr15') or sig.get('atr') or 0)
        atr_ratio = atr_val / price if price > 0 else 0.0

        trend_penalty = breakdown.get('4H趨勢不順', 0)
        trend_aligned = (trend_penalty == 0) and ('逆4H趨勢降權' not in desc)
        squeeze_ready = any(k in desc for k in ['收斂', '吸收中', '量能悄悄放大'])
        extended_risk = any(k in desc for k in ['過度延伸', '避免追高', '避免追空', '高波動降權', '炒頂風險', '炒底風險'])

        with MARKET_LOCK:
            market_dir = MARKET_STATE.get('direction', '中性')
            market_strength = float(MARKET_STATE.get('strength', 0) or 0)

        margin_pct = calc_dynamic_margin_pct(
            score=sig.get('score', 0),
            atr_ratio=atr_ratio,
            trend_aligned=trend_aligned,
            squeeze_ready=squeeze_ready,
            extended_risk=extended_risk,
            same_side_count=same_side_count,
            market_dir=market_dir,
            market_strength=market_strength,
        )
        learning_mult = get_margin_learning_multiplier(sig.get('symbol', ''), sig.get('score', 0), breakdown)
        margin_pct = round(clamp(margin_pct * learning_mult, MIN_MARGIN_PCT, MAX_MARGIN_PCT), 4)
        return {
            'margin_pct': margin_pct,
            'learning_mult': learning_mult,
            'atr_ratio': round(atr_ratio, 5),
            'trend_aligned': trend_aligned,
            'squeeze_ready': squeeze_ready,
            'extended_risk': extended_risk,
            'market_dir': market_dir,
            'market_strength': round(market_strength, 3),
        }
    except Exception as e:
        return {
            'margin_pct': RISK_PCT,
            'atr_ratio': 0.0,
            'trend_aligned': False,
            'squeeze_ready': False,
            'extended_risk': False,
            'market_dir': '中性',
            'market_strength': 0.0,
            'learning_mult': 1.0,
        }

def get_direction_position_count(side_name):
    try:
        with STATE_LOCK:
            active = list(STATE.get("active_positions", []))
        return sum(1 for p in active if (p.get("side") or "").lower() == side_name)
    except:
        return 0

def plan_scale_in_orders(sig, total_qty, entry_price, atr):
    """
    分批進場規劃器：避免函式缺失導致已達條件卻無法下單。
    回傳格式：
      {mode: single|scale_in, primary_qty, secondary_qty, secondary_price, note}
    """
    try:
        total_qty = float(total_qty or 0)
        entry_price = float(entry_price or 0)
        atr = float(atr or 0)
        if total_qty <= 0 or entry_price <= 0:
            return {
                "mode": "single",
                "primary_qty": total_qty,
                "secondary_qty": 0.0,
                "secondary_price": None,
                "note": "倉位不足，改單筆進場"
            }

        score = float(sig.get('score', 0) or 0)
        rr = float(sig.get('rr', sig.get('rrr', 0)) or 0)
        entry_quality = float(sig.get('entry_quality', 0) or 0)
        side = sig.get('side', 'long')
        setup = str(sig.get('setup') or sig.get('setup_name') or '')

        # 只在相對高品質訊號時分批，避免太弱的單掛太多單
        should_scale = (
            score >= 60
            and rr >= 1.6
            and entry_quality >= 6
        ) or ('突破' in setup) or ('回踩' in setup) or ('pullback' in setup.lower())

        if not should_scale:
            return {
                "mode": "single",
                "primary_qty": total_qty,
                "secondary_qty": 0.0,
                "secondary_price": None,
                "note": "單筆進場"
            }

        ratio = (SCALE_IN_MIN_RATIO + SCALE_IN_MAX_RATIO) / 2.0
        secondary_qty = max(total_qty * ratio, 0.0)
        primary_qty = max(total_qty - secondary_qty, 0.0)
        if primary_qty <= 0:
            return {
                "mode": "single",
                "primary_qty": total_qty,
                "secondary_qty": 0.0,
                "secondary_price": None,
                "note": "主單不足，改單筆進場"
            }

        atr = max(atr, entry_price * 0.003)
        if side == 'long':
            secondary_price = entry_price - atr * 0.35
        else:
            secondary_price = entry_price + atr * 0.35

        return {
            "mode": "scale_in",
            "primary_qty": primary_qty,
            "secondary_qty": secondary_qty,
            "secondary_price": round(secondary_price, 6),
            "note": "分批進場：先主單，回踩/反彈補第二批"
        }
    except Exception as e:
        return {
            "mode": "single",
            "primary_qty": float(total_qty or 0),
            "secondary_qty": 0.0,
            "secondary_price": None,
            "note": f"分批規劃失敗，改單筆進場: {e}"
        }

def compute_order_size(sym, entry_price, stop_price, equity, lev, margin_pct=None):
    """
    倉位大小 = min(名目資金上限, 停損風險上限)，但實際保證金至少為總資金 1%。
    槓桿放大多少不影響這個底線：先保證最低投入保證金，再用槓桿換算口數。
    """
    try:
        entry_price = float(entry_price)
        stop_price  = float(stop_price)
        equity      = max(float(equity), 1.0)
        lev         = max(float(lev), 1.0)
        stop_dist   = abs(entry_price - stop_price)
        if stop_dist <= 0:
            stop_dist = entry_price * 0.01

        selected_margin_pct = float(margin_pct if margin_pct is not None else RISK_PCT)
        selected_margin_pct = clamp(selected_margin_pct, MIN_MARGIN_PCT, MAX_MARGIN_PCT)
        min_margin_usdt = max(equity * MIN_MARGIN_PCT, 1.0)
        target_margin_usdt = max(equity * selected_margin_pct, min_margin_usdt)

        risk_budget = max(equity * ATR_RISK_PCT, 0.5)
        qty_by_risk = risk_budget / stop_dist
        qty_by_target_margin = target_margin_usdt * lev / entry_price
        qty_by_floor_margin  = min_margin_usdt * lev / entry_price

        raw_qty = min(qty_by_risk, qty_by_target_margin)
        raw_qty = max(raw_qty, qty_by_floor_margin)

        try:
            mkt = exchange.market(sym)
            min_amt = float(mkt.get('limits', {}).get('amount', {}).get('min') or 0)
            if min_amt > 0:
                raw_qty = max(raw_qty, min_amt)
        except:
            pass

        qty = float(exchange.amount_to_precision(sym, raw_qty))
        used_margin_usdt = qty * entry_price / lev
        used_margin_pct = used_margin_usdt / equity if equity > 0 else selected_margin_pct
        used_margin_pct = clamp(used_margin_pct, MIN_MARGIN_PCT, MAX_MARGIN_PCT)
        est_risk_usdt = qty * stop_dist
        return qty, round(used_margin_usdt, 4), round(est_risk_usdt, 4), round(stop_dist, 6), round(float(used_margin_pct), 4)
    except Exception as e:
        print("倉位計算失敗 {}: {}".format(sym, e))
        fallback_margin = clamp(float(margin_pct or RISK_PCT), MIN_MARGIN_PCT, MAX_MARGIN_PCT)
        fallback_margin = max(fallback_margin, MIN_MARGIN_PCT)
        fallback_notional = max(float(equity) * fallback_margin, 1.0)
        qty = float(exchange.amount_to_precision(sym, fallback_notional * max(float(lev),1.0) / max(float(entry_price),1e-9)))
        return qty, round(fallback_notional, 4), 0.0, abs(float(entry_price) - float(stop_price)), round(fallback_margin, 4)

def tighten_position_for_session(sym, contracts, side, entry_price, mark_price):
    try:
        pnl_pct = 0.0
        if entry_price and mark_price:
            if side == 'long':
                pnl_pct = (mark_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - mark_price) / entry_price

        # 盈利單縮一半倉位，讓剩餘部位交給移動止盈；虧損單直接平倉。
        if pnl_pct > 0.004:
            partial_close_position(sym, contracts, side, 0.5, "開盤保護縮倉")
            with TRAILING_LOCK:
                if sym in TRAILING_STATE:
                    ts = TRAILING_STATE[sym]
                    ts['trail_pct'] = min(ts.get('trail_pct', 0.05), 0.03)
                    if side == 'long':
                        ts['initial_sl'] = max(ts.get('initial_sl', 0), entry_price)
                    else:
                        ts['initial_sl'] = min(ts.get('initial_sl', float('inf')), entry_price)
            print("🛡 開盤保護縮倉: {} 盈利單保留趨勢單".format(sym))
        else:
            close_position(sym, contracts, side)
            print("🛡 開盤保護平倉: {} 虧損/無利潤單直接離場".format(sym))
    except Exception as e:
        print("開盤保護處理失敗 {}: {}".format(sym, e))

def place_order(sig):
    # 風控檢查
    ok, reason = check_risk_ok()
    if not ok:
        print("風控阻擋下單: {}".format(reason))
        with STATE_LOCK:
            STATE["halt_reason"] = reason
        return

    # 防重複：本輪已下單的幣不再重複
    sym_check = sig['symbol']
    if not can_reenter_symbol(sym_check):
        print('⚠️ 進場冷卻中，跳過 {}'.format(sym_check))
        return
    with _ORDERED_LOCK:
        if sym_check in _ORDERED_THIS_SCAN:
            print("⚠️ 防重複：{}本輪已下單，跳過".format(sym_check))
            return
        _ORDERED_THIS_SCAN.add(sym_check)

    # 下單鎖：確保同一時間只有一筆下單在執行
    with ORDER_LOCK:
        # 二次確認持倉數量（異步下單可能造成超過7個）
        with STATE_LOCK:
            current_pos_count = len(STATE["active_positions"])
            # 同時確認這個幣沒有在持倉中
            pos_syms_now = {p['symbol'] for p in STATE["active_positions"]}
        if current_pos_count >= MAX_OPEN_POSITIONS:
            print("持倉已達{}個上限，取消下單: {}".format(MAX_OPEN_POSITIONS, sig['symbol']))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return
        if sym_check in pos_syms_now:
            print("⚠️ 防重複：{}已在持倉中，跳過".format(sym_check))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return

    try:
        sym=sig['symbol']
        side = 'buy' if sig['score'] > 0 else 'sell'
        sig['side'] = 'long' if side == 'buy' else 'short'  # 確保 sig['side'] 是 long/short

        same_dir_count = get_direction_position_count(sig['side'])
        if same_dir_count >= MAX_SAME_DIRECTION:
            print("同方向持倉已達{}筆上限，跳過 {} {}".format(MAX_SAME_DIRECTION, sym, sig['side']))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return
        # 向 Bitget 確認並設定最高槓桿
        lev = 20
        try:
            mkt  = exchange.market(sym)
            info = mkt.get('info', {})
            # Bitget 合約最大槓桿欄位（優先順序）
            for field in ['maxLeverage','maxLev','leverageMax']:
                val = info.get(field)
                if val:
                    try:
                        lev = int(float(str(val)))
                        if lev > 1:
                            break
                    except:
                        pass
            # 備用：从 limits 取
            if lev <= 20:
                try:
                    lev = int(mkt.get('limits',{}).get('leverage',{}).get('max', 20))
                except:
                    pass
            # 備用：fetch_leverage_tiers 取精確最大值
            if lev <= 20:
                try:
                    tiers = exchange.fetch_leverage_tiers([sym])
                    sym_tiers = tiers.get(sym, [])
                    if sym_tiers:
                        lev = int(max(t.get('maxLeverage', 20) for t in sym_tiers))
                except:
                    pass
            lev = max(lev, 1)
            exchange.set_leverage(lev, sym)
            print("槓桿設定: {} {}x".format(sym, lev))
        except Exception as lev_e:
            print("槓桿設定失敗({}): {} 保持{}x".format(sym, lev_e, lev))
            try:
                exchange.set_leverage(lev, sym)
            except:
                pass

        # 動態保證金：根據分數 / 蓄勢 / 波動 / 同向持倉，自動決定 3%~8%
        with STATE_LOCK: equity=STATE["equity"]
        if equity<=0: equity=10.0
        margin_ctx = infer_margin_context(sig, same_dir_count)
        margin_pct = margin_ctx['margin_pct']
        sl_price=sig['stop_loss']; tp_price=sig['take_profit']
        amt, order_usdt, est_risk_usdt, stop_distance, used_margin_pct = compute_order_size(sym, sig['price'], sl_price, equity, lev, margin_pct)
        sig['margin_pct'] = used_margin_pct
        sig['margin_ctx'] = margin_ctx
        print("動態保證金: {} score={} margin={}%(trend={} squeeze={} extended={} atr={})".format(
            sym, sig.get('score'), round(used_margin_pct*100,2),
            margin_ctx.get('trend_aligned'), margin_ctx.get('squeeze_ready'),
            margin_ctx.get('extended_risk'), margin_ctx.get('atr_ratio')
        ))
        if amt <= 0:
            print("倉位太小無法下單: {}".format(sym))
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.discard(sym_check)
            return
        # (FVG 判斷後可能會更新 sl_price/tp_price)

        # Step1: FVG 最優價判斷
        atr_val  = sig.get('atr', sig['price'] * 0.01)
        fvg_price, fvg_note = get_fvg_entry_price(sym, sig['side'], sig['price'], sig.get('atr15', atr_val))
        print("FVG判斷: {} → {}".format(sym, fvg_note))

        # Bitget 合約必要參數
        pos_side = 'long' if side == 'buy' else 'short'
        order_params = {
            'tdMode':   'cross',      # 全倉
            'posSide':  pos_side,     # long/short（Bitget雙向持倉必須）
        }

        scale_plan = plan_scale_in_orders(sig, amt, sig['price'], sig.get('atr15', atr_val))
        sig['scale_plan'] = scale_plan
        market_qty = amt

        if fvg_price is not None:
            # 防重複：已有掛單就跳過
            with FVG_LOCK:
                already_pending = sym in FVG_ORDERS
            if already_pending:
                print("⚠️ FVG防重複：{} 已有掛單，跳過".format(sym))
                return

            # 掛限價單等回到 FVG 缺口
            try:
                order = exchange.create_order(sym, 'limit', side, amt, fvg_price, params=order_params)
                order_id = order.get('id', '')
                # 重新計算止損止盈基於FVG價，並同步重算倉位
                sl_atr = sig.get('sl_mult', 2.0) * atr_val
                tp_atr = sig.get('tp_mult', 3.0) * atr_val
                if sig['side'] == 'long':
                    sl_price = round(fvg_price - sl_atr, 6)
                    tp_price = round(fvg_price + tp_atr, 6)
                else:
                    sl_price = round(fvg_price + sl_atr, 6)
                    tp_price = round(fvg_price - tp_atr, 6)
                sig['stop_loss']   = sl_price
                sig['take_profit'] = tp_price
                sig['price']       = fvg_price
                amt, order_usdt, est_risk_usdt, stop_distance, used_margin_pct = compute_order_size(sym, fvg_price, sl_price, equity, lev, margin_pct)
                register_fvg_order(
                    sym, order_id, sig['side'], fvg_price,
                    sig['score'], sl_price, tp_price,
                    sig.get('support', 0), sig.get('resist', 0)
                )
                print("📌 FVG限價掛單: {} {} @{:.6f} | {}".format(sym, side, fvg_price, fvg_note))
                return
            except Exception as fvg_err:
                print("FVG限價下單失敗，改用市價: {}".format(fvg_err))
                market_qty = amt
                order = exchange.create_order(sym, 'market', side, market_qty, params=order_params)
        else:
            with FVG_LOCK:
                if sym in FVG_ORDERS:
                    old_order = FVG_ORDERS.pop(sym, None)
                    if old_order:
                        try:
                            exchange.cancel_order(old_order['order_id'], sym)
                            print("🗑 取消舊FVG掛單再下市價: {}".format(sym))
                        except:
                            pass
                        update_state(fvg_orders=dict(FVG_ORDERS))
            market_qty = scale_plan.get('primary_qty', amt) if scale_plan.get('mode') == 'scale_in' else amt
            market_qty = float(exchange.amount_to_precision(sym, max(market_qty, 0))) if market_qty > 0 else 0.0
            order = exchange.create_order(sym, 'market', side, market_qty or amt, params=order_params)

            if scale_plan.get('mode') == 'scale_in' and scale_plan.get('secondary_qty', 0) > 0 and scale_plan.get('secondary_price'):
                try:
                    secondary_qty = float(exchange.amount_to_precision(sym, scale_plan['secondary_qty']))
                    if secondary_qty > 0:
                        pullback_order = exchange.create_order(sym, 'limit', side, secondary_qty, scale_plan['secondary_price'], params=order_params)
                        print("🪜 分批進場掛單: {} 第二批 {}口 @{:.6f} | {}".format(sym, secondary_qty, scale_plan['secondary_price'], scale_plan.get('note', '')))
                        sig['scale_in_pending_order_id'] = pullback_order.get('id', '')
                except Exception as scale_err:
                    print("分批進場掛單失敗，保留主單: {}".format(scale_err))

        print("主單成功: {} {} {}口 | {} | {}".format(sym, side, market_qty if market_qty else amt, fvg_note, scale_plan.get('note', '單筆進場')))
        touch_entry_lock(sym)

        # Step2+3: 主單後立刻補掛交易所保護單，並做驗證；若交易所止損沒掛上，立即平倉保護
        protected_qty = market_qty if market_qty else amt
        sl_ok, tp_ok = ensure_exchange_protection(sym, side, pos_side, protected_qty, sl_price, tp_price)
        if not sl_ok:
            print("❌ 交易所止損驗證失敗，立即市價平倉保護: {}".format(sym))
            close_position(sym, protected_qty, 'long' if side == 'buy' else 'short')
            return

        trade_id="{}_{}" .format(sym.replace('/','').replace(':',''),int(time.time()))
        rec={"symbol":sym,"side":"做多" if side=='buy' else "做空","score":sig['score'],
             "price":sig['price'],"stop_loss":sl_price,"take_profit":tp_price,
             "est_pnl":sig['est_pnl'],"order_usdt":round(order_usdt,2),"risk_usdt":round(est_risk_usdt,2),"leverage":lev,"margin_pct":round(used_margin_pct*100,2),"scale_mode":sig.get('scale_plan',{}).get('mode','single'),
             "time":tw_now_str(),"learn_id":trade_id}
        with STATE_LOCK:
            STATE["trade_history"].insert(0,rec)
            if len(STATE["trade_history"])>30: STATE["trade_history"]=STATE["trade_history"][:30]

        learn_rec={"id":trade_id,"symbol":sym,"side":side,"entry_price":sig['price'],
                   "entry_score":sig['score'],"breakdown":sig.get('breakdown',{}),
                   "atr_mult_sl":sig.get('sl_mult',2.0),"atr_mult_tp":sig.get('tp_mult',3.0),"margin_pct":used_margin_pct,"margin_learning_mult":margin_ctx.get('learning_mult',1.0),"scale_mode":sig.get('scale_plan',{}).get('mode','single'),
                   "entry_time":tw_now_str("%Y-%m-%d %H:%M:%S"),
                   "exit_price":None,"exit_time":None,"pnl_pct":None,
                   "result":"open","post_candles":[],"missed_move_pct":None,"source":"live_bitget_v32"}
        with LEARN_LOCK:
            LEARN_DB["trades"].append(learn_rec)
            if len(LEARN_DB["trades"])>500: LEARN_DB["trades"]=LEARN_DB["trades"][-500:]
            save_learn_db(LEARN_DB)

        # 加入移動止盈追蹤
        trail_pct = min(max(sig.get('atr', sig['price']*0.01) / sig['price'] * 3, 0.03), 0.10)
        with TRAILING_LOCK:
            TRAILING_STATE[sym] = {
                "side":          side,
                "entry_price":   sig['price'],
                "highest_price": sig['price'] if side == 'buy' else float('inf'),
                "lowest_price":  sig['price'] if side == 'sell' else float('inf'),
                "trail_pct":     trail_pct,
                "initial_sl":    sl_price,
                "atr":           sig.get('atr', sig['price']*0.01),
                "entry_time_ts": time.time(),
                "time_stop_sec": TIME_STOP_BARS_15M * 15 * 60,
                "partial_done": 0,
            }
        record_order_placed()  # 通知門檻系統有新下單
        print("下單成功: {} {} @{} {}U 風險{}U x{}倍 SL:{} TP:{} 移動回撤:{:.1f}% 門檻:{}".format(
            sym,side,sig['price'],round(order_usdt,2),round(est_risk_usdt,2),lev,sl_price,tp_price,trail_pct*100,ORDER_THRESHOLD))
    except Exception as e:
        print("下單失敗: {}".format(e))

# =====================================================
# 平倉（正確使用 reduceOnly）
# =====================================================
def close_all():
    try:
        n=0
        positions=exchange.fetch_positions()
        for p in positions:
            c=float(p.get('contracts',0) or 0)
            if abs(c)>0:
                sym=p['symbol']
                side='sell' if p['side']=='long' else 'buy'
                try:
                    exchange.create_order(sym,'market',side,abs(c),params={
                        'reduceOnly':True,
                        'marginMode':'cross',
                    })
                    n+=1
                    print("平倉成功: {} {}口".format(sym,abs(c)))
                except Exception as pe:
                    print("平倉失敗 {}: {}".format(sym,pe))
        return n
    except Exception as e:
        print("平倉整體失敗: {}".format(e)); return 0

# =====================================================
# 主掃描執行緒
# =====================================================
def scan_thread():
    print("掃描執行緒啟動，等待10秒讓其他執行緒就緒...")
    update_state(scan_progress="掃描執行緒啟動中，10秒後開始第1輪...")
    time.sleep(10)
    _refresh_learn_summary()
    while True:
        try:
            # 每輪開始清空本輪下單記錄
            with _ORDERED_LOCK:
                _ORDERED_THIS_SCAN.clear()
            AUTO_ORDER_AUDIT.clear()

            sc = STATE.get("scan_count", 0)
            print("=== 開始第{}輪掃描 === 時間:{}".format(sc+1, tw_now_str()))
            update_state(
                scan_progress="第{}輪：抓取市場數據... {}".format(sc+1, tw_now_str()),
                last_update=tw_now_str()
            )
            try:
                tickers = exchange.fetch_tickers()
                print("fetch_tickers 成功，共 {} 個幣".format(len(tickers)))
            except Exception as ft_e:
                print("fetch_tickers 失敗，10秒後重試: {}".format(ft_e))
                time.sleep(10)
                continue

            ranked=sorted(tickers.items(),key=lambda x:x[1].get('quoteVolume',0),reverse=True)

            # 排除股票代幣（只保留加密貨幣）
            STOCK_TOKENS = {
                'AAPL','GOOGL','GOOG','AMZN','TSLA','MSFT','META','NVDA','NFLX',
                'BABA','BIDU','JD','PDD','NIO','XPEV','LI','SNAP','TWTR','UBER',
                'LYFT','ABNB','COIN','HOOD','AMC','GME','SPY','QQQ','DJI',
                'MSTR','PLTR','SQ','PYPL','SHOP','INTC','AMD','QCOM','AVGO',
            }
            def is_crypto(sym):
                base = sym.split('/')[0].split(':')[0]
                if base in STOCK_TOKENS:
                    return False
                # 排除含小數點或看起來像股票的（如 1000BONK 是幣）
                return True

            symbols=[s[0] for s in ranked
                     if s[0].endswith(':USDT') and is_crypto(s[0])][:70]
            print("本輪掃描 {} 個幣".format(len(symbols)))

            sigs=[]
            with LEARN_LOCK:
                sym_stats=LEARN_DB.get("symbol_stats",{})
            blocked_syms={s for s,v in sym_stats.items()
                          if v.get("count",0)>=7 and v.get("win",0)/v["count"]<0.4}

            for i,sym in enumerate(symbols):
                update_state(scan_progress="掃描 {}/70：{}".format(i+1,sym))
                try:
                    time.sleep(0.5)  # 幣與幣之間間隔0.5秒，避免rate limit
                    sc,desc,pr,sl,tp,ep,bd,atr,atr15,atr4h,sl_m,tp_m = analyze(sym)
                    allowed,sym_n,sym_wr=is_symbol_allowed(sym)
                    status="觀察中(勝率{}%)".format(sym_wr) if not allowed else ""
                    if abs(sc)>=8:
                        stable_score = smooth_signal_score(sym, sc)
                        SIGNAL_META_CACHE[sym] = {"atr": atr, "atr15": atr15, "atr4h": atr4h, "price": pr, "raw_score": sc, "stable_score": stable_score, "updated_at": tw_now_str()}
                        sigs.append({
                            "symbol":sym,"score":stable_score,"raw_score":sc,"desc":desc,"price":pr,
                            "stop_loss":sl,"take_profit":tp,"est_pnl":ep,
                            "direction":"做多 ▲" if stable_score>0 else "做空 ▼",
                            "breakdown": bd,
"atr": atr,
"atr15": atr15,
"atr4h": atr4h,
"sl_mult": sl_m,
"tp_mult": tp_m,
                            "allowed":allowed,"status":status,
                            "sym_trades":sym_n,"sym_wr":sym_wr,
                            "margin_pct": 0,
                            "entry_quality": bd.get("進場品質", 0),
                            "rr_ratio": bd.get("RR", 0),
                            "regime_bias": bd.get("RegimeBias", 0),
                            "setup_label": bd.get("Setup", ""),
                            "signal_grade": bd.get("等級", ""),
                            "direction_confidence": bd.get("方向信心", 0),
                            "score_jump": score_jump_alert(sym, sc, stable_score),
                        })
                except Exception as sym_e:
                    print("分析 {} 失敗跳過: {}".format(sym, sym_e))
                if i%5==0: gc.collect()

            for s in sigs:
                try:
                    ctx = infer_margin_context(s, same_side_count=0)
                    s['margin_pct'] = ctx.get('margin_pct', RISK_PCT)
                    s['margin_ctx'] = ctx
                except:
                    s['margin_pct'] = RISK_PCT

            # 分開排序：多頭取前6，空頭取前4，排行榜顯示10個
            long_sigs  = sorted([s for s in sigs if s['score']>0], key=lambda x:x['score'], reverse=True)[:6]
            short_sigs = sorted([s for s in sigs if s['score']<0], key=lambda x:x['score'])[:4]
            top10 = sorted(long_sigs + short_sigs, key=lambda x:abs(x['score']), reverse=True)[:10]
            top7  = top10  # 排行榜顯示10個
            print("步驟A: 排行榜排序完成，共{}個信號".format(len(top7)))
            with STATE_LOCK:
                STATE["top_signals"]=top10; STATE["scan_count"]+=1
                STATE["last_update"]=tw_now_str()
                STATE["scan_progress"]="第{}輪完成 | {} | 門檻:{}分".format(STATE["scan_count"],STATE["last_update"],ORDER_THRESHOLD)
                STATE["auto_order_audit"]=dict(AUTO_ORDER_AUDIT)
            print("步驟B: STATE更新完成")

            with STATE_LOCK:
                active_pos = list(STATE["active_positions"])
                pos_syms   = {p['symbol'] for p in active_pos}
                pos_cnt    = len(active_pos)
            print("步驟C: 持倉讀取完成，共{}個".format(pos_cnt))

            # ── 反向偵測：持倉方向與新訊號方向相反 → 只平倉，不開新倉 ──
            sig_map = {s['symbol']: s['score'] for s in top7}
            already_closing = set()  # 防止重複平倉同一個幣
            for pos in active_pos:
                sym_p = pos['symbol']
                if sym_p in already_closing:
                    continue
                pos_side = (pos.get('side') or '').lower()   # 'long' or 'short'
                new_score = sig_map.get(sym_p, None)
                if new_score is None:
                    continue  # 這輪沒掃到這個幣，跳過
                # 判斷方向衝突
                is_reverse = (pos_side == 'long'  and new_score < -ORDER_THRESHOLD) or                              (pos_side == 'short' and new_score >  ORDER_THRESHOLD)
                if is_reverse:
                    contracts = float(pos.get('contracts', 0) or 0)
                    if abs(contracts) > 0:
                        close_side = 'sell' if pos_side == 'long' else 'buy'
                        already_closing.add(sym_p)
                        def _do_reverse_close(s, c, cs, score, mprice):
                            try:
                                exchange.create_order(s,'market',cs,abs(c),params={'reduceOnly':True})
                                print("反向平倉成功: {} 新分數:{:.1f}".format(s, score))
                                with STATE_LOCK:
                                    STATE["trade_history"].insert(0,{
                                        "symbol":s,"side":"反向平倉","score":score,
                                        "price":mprice,"stop_loss":0,"take_profit":0,
                                        "est_pnl":0,"order_usdt":0,
                                        "time":tw_now_str(),
                                    })
                            except Exception as re:
                                print("反向平倉失敗 {}: {}".format(s, re))
                        threading.Thread(
                            target=_do_reverse_close,
                            args=(sym_p, contracts, close_side, new_score, pos.get('markPrice',0)),
                            daemon=True
                        ).start()

            # ── 正常開倉邏輯（下單間隔5秒，避免rate limit）──
            # 下單只取分數前 7，排行榜顯示 10 個但短線總倉上限仍為 7
            top7_for_order = sorted(top10, key=lambda x:abs(x['score']), reverse=True)[:MAX_OPEN_POSITIONS]
            if pos_cnt < MAX_OPEN_POSITIONS:
                order_delay = 0
                for best in top7_for_order:
                    # 大盤方向過濾
                    with MARKET_LOCK:
                        mkt_dir = MARKET_STATE.get("direction", "中性")
                        mkt_str = MARKET_STATE.get("strength", 0)

                    # 強度 >= 60% 才過濾方向，弱空/弱多不過濾
                    signal_side = 'long' if best['score'] > 0 else 'short'
                    mkt_ok = True
                    if mkt_str >= 0.6:  # 只有強方向才過濾
                        if mkt_dir in ("強多", "多") and signal_side == 'short':
                            mkt_ok = False  # 強多頭不做空
                        elif mkt_dir in ("強空", "空") and signal_side == 'long':
                            mkt_ok = False  # 強空頭不做多

                    # 大盤中性時門檻提高5分
                    eff_threshold = ORDER_THRESHOLD + (5 if mkt_dir == "中性" and mkt_str >= 0.5 else 0)

                    same_dir_cnt_now = get_direction_position_count(signal_side)
                    entry_quality = float(best.get('entry_quality', 0) or 0)
                    rr_ratio = float(best.get('rr_ratio', 0) or 0)
                    regime_bias = float(best.get('regime_bias', 0) or 0)
                    side_ok = (signal_side == 'long' and regime_bias >= 0) or (signal_side == 'short' and regime_bias <= 0)
                    ai_decision = ai_decide_trade(best, eff_threshold, mkt_ok, side_ok, same_dir_cnt_now, pos_syms, already_closing)
                    allow_now = bool(ai_decision.get('allow_now'))
                    reasons = build_auto_order_reason(best, ai_decision.get('effective_threshold', eff_threshold), mkt_ok, side_ok, same_dir_cnt_now, pos_syms, already_closing, ai_decision=ai_decision)
                    reasons = list(dict.fromkeys((ai_decision.get('reasons') or []) + (reasons or [])))
                    AUTO_ORDER_AUDIT[best['symbol']] = {
                        'will_order': bool(allow_now),
                        'reasons': reasons or ['符合條件，自動下單'],
                        'threshold': round(ai_decision.get('effective_threshold', eff_threshold), 2),
                        'entry_quality': round(entry_quality, 2),
                        'rr_ratio': round(rr_ratio, 2),
                        'mkt_dir': mkt_dir,
                        'same_dir_cnt': same_dir_cnt_now,
                        'checked_at': tw_now_str(),
                        'ai_enabled': True,
                        'ai_ready': bool((ai_decision.get('profile') or {}).get('ready')),
                        'ai_source': (ai_decision.get('profile') or {}).get('source', 'none'),
                        'ai_strategy': (ai_decision.get('profile') or {}).get('strategy', ''),
                        'ai_sample_count': int((ai_decision.get('profile') or {}).get('sample_count', 0) or 0),
                        'ai_win_rate': round(float((ai_decision.get('profile') or {}).get('win_rate', 0) or 0), 2),
                        'ai_avg_pnl': round(float((ai_decision.get('profile') or {}).get('avg_pnl', 0) or 0), 2),
                        'ai_note': (ai_decision.get('profile') or {}).get('note', ''),
                        'ai_phase': (ai_decision.get('profile') or {}).get('phase', ''),
                    }
                    best['auto_order'] = AUTO_ORDER_AUDIT[best['symbol']]
                    best['ai_decision'] = AUTO_ORDER_AUDIT[best['symbol']]

                    if allow_now:  # 動態門檻（含 AI 接管）
                        def _make_delayed(sig, delay):
                            def _run():
                                time.sleep(delay)
                                place_order(sig)
                            return _run
                        threading.Thread(
                            target=_make_delayed(best, order_delay),
                            daemon=True
                        ).start()
                        order_delay += 5  # 每筆單間隔5秒

            # 更新動態門檻
            update_dynamic_threshold()

            # 每10輪更新一次大盤分析（不等1小時）
            if STATE.get("scan_count", 0) % 10 == 1:
                try:
                    result = analyze_btc_market_trend()
                    if result:
                        with MARKET_LOCK:
                            MARKET_STATE.update(result)
                        update_state(market_info=dict(MARKET_STATE))
                        print("📊 大盤(定期更新): {} | {}".format(
                            result["pattern"], result["direction"]))
                except Exception as me:
                    print("大盤定期更新失敗: {}".format(me))

            # 更新風控摘要
            print("步驟D: 準備更新風控... 當前門檻:{}分".format(ORDER_THRESHOLD))
            update_state(risk_status=get_risk_status())
            print("第{}輪掃描完成，60秒後開始下一輪".format(STATE["scan_count"]))
            time.sleep(60)  # 輪間隔60秒
            print("步驟E: 60秒休息結束，開始下一輪")
        except Exception as e:
            import traceback
            print("掃描異常: {}".format(e))
            print(traceback.format_exc())
            time.sleep(10)

# =====================================================
# Flask 路由
# =====================================================
@app.route('/')
def index(): return render_template('index.html')


def _calc_max_drawdown(equity_curve):
    peak = equity_curve[0] if equity_curve else 0
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
    return round(max_dd * 100, 2)

def run_simple_backtest(symbol="BTC/USDT:USDT", timeframe="15m", limit=800, fee_rate=0.0006):
    """
    輕量回測：納入
    - 蓄勢結構
    - 假突破過濾
    - 反追價
    - 分批進場
    - 分批止盈
    讓回測更貼近實盤邏輯。
    """
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
        if len(df) < 250:
            return {"ok": False, "error": "K線不足，至少需要250根"}

        df['ema21'] = ta.ema(df['c'], length=21)
        df['ema55'] = ta.ema(df['c'], length=55)
        macd = ta.macd(df['c'], fast=12, slow=26, signal=9)
        df['macd'] = macd.iloc[:, 0]
        df['macds'] = macd.iloc[:, 1]
        df['rsi'] = ta.rsi(df['c'], length=14)
        adx_df = ta.adx(df['h'], df['l'], df['c'], length=14)
        df['adx'] = adx_df.iloc[:, 0]
        df['atr'] = ta.atr(df['h'], df['l'], df['c'], length=14)
        df = df.dropna().reset_index(drop=True)
        if len(df) < 120:
            return {"ok": False, "error": "有效指標資料不足"}

        equity = 10000.0
        equity_curve = [equity]
        trades = []
        position = None

        for i in range(60, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            price = float(row['c'])
            atr = max(float(row['atr']), price * 0.003)
            ema_up = row['ema21'] > row['ema55']
            ema_dn = row['ema21'] < row['ema55']
            macd_up = row['macd'] > row['macds'] and prev['macd'] <= prev['macds']
            macd_dn = row['macd'] < row['macds'] and prev['macd'] >= prev['macds']
            adx_ok = row['adx'] >= 18
            win = df.iloc[max(0, i-BREAKOUT_LOOKBACK):i]
            hh = float(win['h'].max()) if len(win) else price
            ll = float(win['l'].min()) if len(win) else price
            recent_lows = df['l'].iloc[max(0, i-6):i].tolist()
            recent_highs = df['h'].iloc[max(0, i-6):i].tolist()
            squeeze_long = ema_up and len(recent_lows) >= 4 and _linreg_slope(recent_lows) > 0 and (hh - price) / max(atr, 1e-9) <= 0.5
            squeeze_short = ema_dn and len(recent_highs) >= 4 and _linreg_slope(recent_highs) < 0 and (price - ll) / max(atr, 1e-9) <= 0.5
            ext = (price - float(row['ema21'])) / max(atr, 1e-9)
            anti_chase_long = ext <= ANTI_CHASE_ATR
            anti_chase_short = ext >= -ANTI_CHASE_ATR
            sub = df.iloc[max(0, i-BREAKOUT_LOOKBACK-1):i+1].copy()
            fake_s, _, fake_meta = analyze_fake_breakout(sub, 1 if ema_up else -1 if ema_dn else 0)
            fakeout_long = fake_meta.get('fakeout') and fake_meta.get('direction') == 'up'
            fakeout_short = fake_meta.get('fakeout') and fake_meta.get('direction') == 'down'

            if position is None:
                if ema_up and (macd_up or squeeze_long) and row['rsi'] < 68 and adx_ok and anti_chase_long and not fakeout_long:
                    entry = min(price, max(float(row['ema21']), hh - atr * PULLBACK_BUFFER_ATR)) if price >= hh * 0.999 else price
                    sl = entry - atr * 1.55
                    tp = entry + atr * (3.6 if squeeze_long else 3.0)
                    pseudo_score = 55 if squeeze_long and adx_ok else 52 if macd_up else 50
                    margin_pct = calc_dynamic_margin_pct(pseudo_score, atr / max(price,1e-9), True, squeeze_long, not anti_chase_long, 0)
                    scale_in = 0.4 if squeeze_long and pseudo_score >= 55 else 0.0
                    blended_entry = entry * (1 - scale_in) + max(float(row['ema21']), entry - atr * PULLBACK_BUFFER_ATR) * scale_in
                    risk_budget = equity * ATR_RISK_PCT
                    cap_qty = (equity * margin_pct) / max(blended_entry, 1e-9)
                    risk_qty = risk_budget / max(blended_entry - sl, price * 0.002)
                    qty = max(min(risk_qty, cap_qty), 0)
                    position = {"side": "long", "entry": blended_entry, "sl": sl, "tp": tp, "qty": qty, "bar": i, "margin_pct": margin_pct, "partial_done": 0}
                elif ema_dn and (macd_dn or squeeze_short) and row['rsi'] > 32 and adx_ok and anti_chase_short and not fakeout_short:
                    entry = max(price, min(float(row['ema21']), ll + atr * PULLBACK_BUFFER_ATR)) if price <= ll * 1.001 else price
                    sl = entry + atr * 1.55
                    tp = entry - atr * (3.6 if squeeze_short else 3.0)
                    pseudo_score = 55 if squeeze_short and adx_ok else 52 if macd_dn else 50
                    margin_pct = calc_dynamic_margin_pct(pseudo_score, atr / max(price,1e-9), True, squeeze_short, not anti_chase_short, 0)
                    scale_in = 0.4 if squeeze_short and pseudo_score >= 55 else 0.0
                    blended_entry = entry * (1 - scale_in) + min(float(row['ema21']), entry + atr * PULLBACK_BUFFER_ATR) * scale_in
                    risk_budget = equity * ATR_RISK_PCT
                    cap_qty = (equity * margin_pct) / max(blended_entry, 1e-9)
                    risk_qty = risk_budget / max(sl - blended_entry, price * 0.002)
                    qty = max(min(risk_qty, cap_qty), 0)
                    position = {"side": "short", "entry": blended_entry, "sl": sl, "tp": tp, "qty": qty, "bar": i, "margin_pct": margin_pct, "partial_done": 0}
                continue

            exit_reason = None
            exit_price = price
            bars_held = i - position['bar']
            pnl = None

            if position['side'] == 'long':
                profit_atr = (price - position['entry']) / max(atr, 1e-9)
                if profit_atr >= 1.2 and position['partial_done'] == 0:
                    realized_qty = position['qty'] * 0.25
                    net = (price - position['entry']) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "long", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP1', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.75
                    position['sl'] = max(position['sl'], position['entry'])
                    position['partial_done'] = 1
                elif profit_atr >= 2.4 and position['partial_done'] == 1:
                    realized_qty = position['qty'] * 0.35
                    net = (price - position['entry']) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "long", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP2', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.65
                    position['sl'] = max(position['sl'], position['entry'] + atr * 0.8)
                    position['partial_done'] = 2
                if row['l'] <= position['sl']:
                    exit_price = position['sl']; exit_reason = 'SL'
                elif row['h'] >= position['tp']:
                    exit_price = position['tp']; exit_reason = 'TP'
                elif bars_held >= TIME_STOP_BARS_15M and abs(price - position['entry']) / position['entry'] < 0.006:
                    exit_reason = 'TIME'
                elif ema_dn and macd_dn:
                    exit_reason = 'REVERSE'
                pnl = (exit_price - position['entry']) * position['qty'] if exit_reason else None
            else:
                profit_atr = (position['entry'] - price) / max(atr, 1e-9)
                if profit_atr >= 1.2 and position['partial_done'] == 0:
                    realized_qty = position['qty'] * 0.25
                    net = (position['entry'] - price) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "short", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP1', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.75
                    position['sl'] = min(position['sl'], position['entry'])
                    position['partial_done'] = 1
                elif profit_atr >= 2.4 and position['partial_done'] == 1:
                    realized_qty = position['qty'] * 0.35
                    net = (position['entry'] - price) * realized_qty - (position['entry'] + price) * realized_qty * fee_rate
                    equity += net
                    trades.append({"side": "short", "entry": round(position['entry'], 6), "exit": round(price, 6), "pnl": round(net, 4), "reason": 'TP2', "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                    position['qty'] *= 0.65
                    position['sl'] = min(position['sl'], position['entry'] - atr * 0.8)
                    position['partial_done'] = 2
                if row['h'] >= position['sl']:
                    exit_price = position['sl']; exit_reason = 'SL'
                elif row['l'] <= position['tp']:
                    exit_price = position['tp']; exit_reason = 'TP'
                elif bars_held >= TIME_STOP_BARS_15M and abs(price - position['entry']) / position['entry'] < 0.006:
                    exit_reason = 'TIME'
                elif ema_up and macd_up:
                    exit_reason = 'REVERSE'
                pnl = (position['entry'] - exit_price) * position['qty'] if exit_reason else None

            if exit_reason:
                fee = (position['entry'] + exit_price) * position['qty'] * fee_rate
                net = pnl - fee
                equity += net
                trades.append({"side": position['side'], "entry": round(position['entry'], 6), "exit": round(exit_price, 6), "pnl": round(net, 4), "reason": exit_reason, "bars": bars_held, "margin_pct": round(position.get('margin_pct', RISK_PCT) * 100, 2)})
                equity_curve.append(equity)
                position = None

        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = sum(1 for t in trades if t['pnl'] <= 0)
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else None
        win_rate = round((wins / len(trades) * 100), 2) if trades else 0.0
        avg_margin_pct = round((sum(t.get('margin_pct', RISK_PCT * 100) for t in trades) / len(trades)), 2) if trades else round(RISK_PCT * 100, 2)
        return {"ok": True, "symbol": symbol, "timeframe": timeframe, "trades": len(trades), "wins": wins, "losses": losses, "win_rate": win_rate, "profit_factor": profit_factor, "avg_margin_pct": avg_margin_pct, "margin_range_pct": [round(MIN_MARGIN_PCT * 100, 2), round(MAX_MARGIN_PCT * 100, 2)], "net_profit": round(equity - 10000.0, 2), "return_pct": round((equity / 10000.0 - 1) * 100, 2), "max_drawdown_pct": _calc_max_drawdown(equity_curve), "last_10_trades": trades[-10:]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.route('/api/backtest', methods=['GET'])
def api_backtest():
    symbol = _resolve_backtest_symbol(request.args.get('symbol', 'auto'))
    timeframe = request.args.get('timeframe', '15m')
    try:
        limit = int(request.args.get('limit', 800))
    except:
        limit = 800
    result = run_simple_backtest(symbol=symbol, timeframe=timeframe, limit=max(250, min(limit, 2000)))
    if isinstance(result, dict):
        result['selected_symbol'] = symbol
    return jsonify(result)

@app.route('/api/state')
def api_state():
    try:
        with STATE_LOCK:
            s = dict(STATE)

        # 清理 active_positions（移除不能 JSON 序列化的欄位）
        clean_pos = []
        for p in s.get('active_positions', []):
            try:
                clean_pos.append({
                    'symbol':        str(p.get('symbol','') or ''),
                    'side':          str(p.get('side','') or ''),
                    'contracts':     float(p.get('contracts',0) or 0),
                    'entryPrice':    float(p.get('entryPrice',0) or 0),
                    'markPrice':     float(p.get('markPrice',0) or 0),
                    'unrealizedPnl': float(p.get('unrealizedPnl',0) or 0),
                    'percentage':    float(p.get('percentage',0) or 0),
                    'leverage':      float(p.get('leverage',1) or 1),
                    'drawdown_pct':  _position_drawdown_pct(p),
                    'leveraged_pnl_pct': _position_leveraged_pnl_pct(p),
                })
            except:
                pass
        s['active_positions'] = clean_pos

        # 補上即時風控狀態
        s['risk_status'] = get_risk_status()

        # 補上大盤和長期倉位
        with MARKET_LOCK:
            s['market_info'] = dict(MARKET_STATE)
        with LT_LOCK:
            s['lt_info'] = dict(LT_STATE)
        with FVG_LOCK:
            s['fvg_orders'] = dict(FVG_ORDERS)

        # 即時組合 trailing_info（不等 trailing_stop_thread 更新）
        try:
            ui_trail = {}
            with TRAILING_LOCK:
                for sym, ts in TRAILING_STATE.items():
                    side_t = ts.get('side','')
                    trail  = ts.get('trail_pct', 0.05)
                    highest = ts.get('highest_price', 0)
                    lowest  = ts.get('lowest_price', float('inf'))
                    entry   = ts.get('entry_price', 0)
                    sl      = ts.get('initial_sl', 0)
                    if side_t in ('buy','long') and highest > 0:
                        trail_price = highest * (1 - trail)
                        stage = '保本' if abs(sl - entry) < entry * 0.001 else '鎖利'
                        ui_trail[sym] = {
                            'side': '做多',
                            'peak': round(highest, 6),
                            'trail_price': round(trail_price, 6),
                            'trail_pct': round(trail * 100, 1),
                            'initial_sl': round(sl, 6),
                            'stage': stage,
                        }
                    elif side_t in ('sell','short') and lowest != float('inf'):
                        trail_price = lowest * (1 + trail)
                        ui_trail[sym] = {
                            'side': '做空',
                            'peak': round(lowest, 6),
                            'trail_price': round(trail_price, 6),
                            'trail_pct': round(trail * 100, 1),
                            'initial_sl': round(sl, 6),
                        }
            s['trailing_info'] = ui_trail
        except:
            pass

        # 補上時段資訊（歐美盤橘色條）
        with SESSION_LOCK:
            s['session_info'] = {
                "phase":    SESSION_STATE.get("session_phase", "normal"),
                "note":     SESSION_STATE.get("session_note", ""),
                "eu_score": SESSION_STATE.get("eu_score", 0),
                "us_score": SESSION_STATE.get("us_score", 0),
                "eu_time":  SESSION_STATE.get("eu_score_time", ""),
                "us_time":  SESSION_STATE.get("us_score_time", ""),
            }

        # 補上動態門檻資訊
        with _DT_LOCK:
            s['threshold_info'] = {
                "current":      _DT.get("current", 38),
                "phase":        "嚴格" if _DT.get("current",38) >= 50 else "調降中" if _DT.get("current",38) < 38 else "預設",
                "full_rounds":  _DT.get("full_rounds", 0),
                "empty_rounds": _DT.get("empty_rounds", 0),
            }

        return jsonify(s)
    except Exception as e:
        print("api_state 錯誤: {}".format(e))
        return jsonify({"error": str(e), "scan_progress": "API錯誤: {}".format(str(e)[:50])})

@app.route('/api/learn_db')
def api_learn_db():
    with LEARN_LOCK: return jsonify(LEARN_DB)

@app.route('/api/close_all',methods=['POST'])
def api_close(): return jsonify({"status":"ok","closed":close_all()})

@app.route('/api/fvg_cancel', methods=['POST'])
def api_fvg_cancel():
    data   = request.get_json() or {}
    symbol = data.get('symbol','')
    if not symbol:
        return jsonify({"ok": False, "msg": "缺少 symbol"})
    with FVG_LOCK:
        if symbol not in FVG_ORDERS:
            return jsonify({"ok": False, "msg": "找不到 {} 的掛單".format(symbol)})
        order    = FVG_ORDERS.get(symbol, {})
        order_id = order.get("order_id","")
    if order_id:
        try:
            exchange.cancel_order(order_id, symbol)
            print("手動取消掛單: {} order_id={}".format(symbol, order_id))
        except Exception as e:
            print("取消失敗(可能已成交): {}".format(e))
    with FVG_LOCK:
        FVG_ORDERS.pop(symbol, None)
    update_state(fvg_orders=dict(FVG_ORDERS))
    return jsonify({"ok": True, "msg": "{} 掛單已取消".format(symbol)})

@app.route('/api/lt_open', methods=['POST'])
def api_lt_open():
    data      = request.get_json() or {}
    direction = data.get('direction', 'long')
    reason    = data.get('reason', '手動操作')
    ok = open_long_term_position(direction, reason)
    return jsonify({"ok": ok, "msg": "長期倉位已開啟" if ok else "開倉失敗"})

@app.route('/api/lt_close', methods=['POST'])
def api_lt_close():
    ok = close_long_term_position("手動平倉")
    return jsonify({"ok": ok, "msg": "長期倉位已平倉" if ok else "平倉失敗"})

@app.route('/api/lt_analyze', methods=['POST'])
def api_lt_analyze():
    result = analyze_btc_market_trend()
    if result:
        with MARKET_LOCK:
            MARKET_STATE.update(result)
        update_state(market_info=dict(MARKET_STATE))
        check_long_term_position()
        return jsonify({"ok": True, "result": result})
    return jsonify({"ok": False, "msg": "分析失敗"})

@app.route('/api/reset_cooldown',methods=['POST'])
def api_reset_cooldown():
    with RISK_LOCK:
        RISK_STATE["cooldown_until"]    = None
        RISK_STATE["consecutive_loss"]  = 0
        RISK_STATE["trading_halted"]    = False
        RISK_STATE["halt_reason"]       = ""
    update_state(risk_status=get_risk_status(), halt_reason="")
    print("冷靜期已手動解除")
    return jsonify({"status":"ok","msg":"冷靜期已解除，恢復交易"})

# =====================================================
# Gunicorn hook（單 worker）
# =====================================================
# =====================================================
# 執行緒守護：任何執行緒死掉自動重啟
# =====================================================
def watchdog(target_func, name):
    """包裹執行緒函數，死掉自動重啟（捕捉所有錯誤）"""
    while True:
        try:
            print("=== 執行緒啟動: {} ===".format(name))
            target_func()
            print("=== 執行緒正常結束（不應發生）: {} ===".format(name))
        except BaseException as e:
            import traceback
            print("=== 執行緒崩潰 {} : {} ===".format(name, e))
            print(traceback.format_exc())
        print("=== 執行緒5秒後重啟: {} ===".format(name))
        time.sleep(5)

def start_all_threads():
    # 啟動時恢復備份狀態
    load_full_state()
    load_risk_state()
    threads = [
        (news_thread,            "news"),
        (position_thread,        "position"),
        (scan_thread,            "scan"),
        (trailing_stop_thread,   "trailing"),
        (session_monitor_thread, "session"),
        (market_analysis_thread,  "market"),
        (fvg_order_monitor_thread,"fvg_monitor"),
    ]
    for fn, name in threads:
        t = threading.Thread(
            target=watchdog,
            args=(fn, name),
            daemon=True,
            name=name
        )
        t.start()
    print("=== 所有執行緒已啟動（含守護重啟機制）===")

def post_fork(server, worker):
    start_all_threads()
    print("=== [worker {}] 啟動完成 ===".format(worker.pid))



# =====================================================
# V6 強化版：方向先行 + 結構觸發 + 風報比過濾
# =====================================================
DIRECTION_STRONG_GATE = 3.2
DIRECTION_WEAK_GATE   = 2.0
NO_TRADE_CHOP_ADX     = 17
MAX_SIGNAL_AGE_BARS   = 3


def _clip(v, lo, hi):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return lo


def _ema_bias(close_s, fast=9, mid=21, slow=55):
    curr = float(close_s.iloc[-1])
    e1 = safe_last(ta.ema(close_s, length=fast), curr)
    e2 = safe_last(ta.ema(close_s, length=mid), curr)
    e3 = safe_last(ta.ema(close_s, length=slow), curr)
    slope = _linreg_slope(close_s.tail(8).tolist()) / max(curr, 1e-9) * 100
    if curr > e1 > e2 > e3 and slope > 0.03:
        return 1
    if curr < e1 < e2 < e3 and slope < -0.03:
        return -1
    return 0


def _detect_pullback_trigger(d15, side):
    c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
    curr = float(c.iloc[-1])
    atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
    ema9 = safe_last(ta.ema(c, length=9), curr)
    ema21 = safe_last(ta.ema(c, length=21), curr)
    ema55 = safe_last(ta.ema(c, length=55), curr)
    vol_now = float(v.tail(2).mean()) if len(v) >= 2 else float(v.iloc[-1])
    vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else vol_now
    low1 = float(l.iloc[-1]); high1 = float(h.iloc[-1]); close1 = float(c.iloc[-1]); open1 = float(o.iloc[-1])
    body = abs(close1-open1)
    candle_range = max(high1-low1, 1e-9)
    close_pos = (close1-low1)/candle_range
    bearish_close_pos = (high1-close1)/candle_range
    ext = abs(curr-ema21)/max(atr,1e-9)
    hh = float(h.tail(20).iloc[:-1].max()) if len(h) > 21 else float(h.max())
    ll = float(l.tail(20).iloc[:-1].min()) if len(l) > 21 else float(l.min())

    if side > 0:
        ok = curr > ema21 > ema55 and ema9 >= ema21 and ext <= 0.95 and close_pos > 0.58 and body > atr * 0.18
        if ok:
            sl = min(float(l.tail(4).min()), ema21 - atr * 0.55)
            entry = curr
            tp = entry + max(entry - sl, atr * 0.8) * 2.4
            quality = 7 + (1 if vol_now > vol_avg * 1.05 else 0) + (1 if curr >= hh * 0.995 else 0)
            return True, '趨勢回踩續攻', quality, entry, sl, tp
    else:
        ok = curr < ema21 < ema55 and ema9 <= ema21 and ext <= 0.95 and bearish_close_pos > 0.58 and body > atr * 0.18
        if ok:
            sl = max(float(h.tail(4).max()), ema21 + atr * 0.55)
            entry = curr
            tp = entry - max(sl - entry, atr * 0.8) * 2.4
            quality = 7 + (1 if vol_now > vol_avg * 1.05 else 0) + (1 if curr <= ll * 1.005 else 0)
            return True, '趨勢反彈續跌', quality, entry, sl, tp
    return False, '', 0, curr, curr, curr


def _detect_squeeze_break_trigger(d15, side):
    c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
    curr = float(c.iloc[-1])
    atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
    bb = ta.bbands(c, length=20, std=2.0)
    if bb is None or bb.empty:
        return False, '', 0, curr, curr, curr
    bb_up = safe_last(bb.iloc[:, 0], curr); bb_low = safe_last(bb.iloc[:, 2], curr)
    width = (bb_up - bb_low) / max(curr, 1e-9)
    width_med = float(((bb.iloc[:, 0] - bb.iloc[:, 2]) / c).tail(30).median()) if len(c) >= 30 else width
    vol_now = float(v.tail(2).mean()) if len(v) >= 2 else float(v.iloc[-1])
    vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else vol_now
    body = abs(float(c.iloc[-1]) - float(o.iloc[-1]))
    hh = float(h.tail(20).iloc[:-1].max()) if len(h) > 21 else float(h.max())
    ll = float(l.tail(20).iloc[:-1].min()) if len(l) > 21 else float(l.min())
    squeeze = width < width_med * 0.82

    if side > 0 and squeeze and curr >= hh * 0.999 and body > atr * 0.55 and vol_now > vol_avg * 1.18:
        sl = min(float(l.tail(3).min()), curr - atr * 1.1)
        tp = curr + max(curr - sl, atr * 0.9) * 2.9
        return True, '收斂突破啟動', 9, curr, sl, tp
    if side < 0 and squeeze and curr <= ll * 1.001 and body > atr * 0.55 and vol_now > vol_avg * 1.18:
        sl = max(float(h.tail(3).max()), curr + atr * 1.1)
        tp = curr - max(sl - curr, atr * 0.9) * 2.9
        return True, '收斂跌破啟動', 9, curr, sl, tp
    return False, '', 0, curr, curr, curr


def _detect_sweep_reclaim_trigger(d15, side):
    c = d15['c'].astype(float); o = d15['o'].astype(float); h = d15['h'].astype(float); l = d15['l'].astype(float); v = d15['v'].astype(float)
    curr = float(c.iloc[-1])
    atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
    ema9 = safe_last(ta.ema(c, length=9), curr)
    ema21 = safe_last(ta.ema(c, length=21), curr)
    vol_now = float(v.tail(2).mean()) if len(v) >= 2 else float(v.iloc[-1])
    vol_avg = float(v.tail(20).mean()) if len(v) >= 20 else vol_now
    prior_low = float(l.tail(24).iloc[:-1].min()) if len(l) > 25 else float(l.min())
    prior_high = float(h.tail(24).iloc[:-1].max()) if len(h) > 25 else float(h.max())
    candle_range = max(float(h.iloc[-1] - l.iloc[-1]), 1e-9)
    close_pos = (float(c.iloc[-1]) - float(l.iloc[-1])) / candle_range
    upper_close_pos = (float(h.iloc[-1]) - float(c.iloc[-1])) / candle_range

    if side > 0:
        swept = float(l.iloc[-1]) < prior_low * 0.999 and curr > prior_low and curr > ema9 and close_pos > 0.65
        if swept and vol_now > vol_avg * 0.95:
            sl = float(l.iloc[-1]) - atr * 0.2
            tp = curr + max(curr - sl, atr * 0.75) * 2.2
            q = 8 + (1 if curr > ema21 else 0)
            return True, '流動性掃低回收', q, curr, sl, tp
    else:
        swept = float(h.iloc[-1]) > prior_high * 1.001 and curr < prior_high and curr < ema9 and upper_close_pos > 0.65
        if swept and vol_now > vol_avg * 0.95:
            sl = float(h.iloc[-1]) + atr * 0.2
            tp = curr - max(sl - curr, atr * 0.75) * 2.2
            q = 8 + (1 if curr < ema21 else 0)
            return True, '流動性掃高回落', q, curr, sl, tp
    return False, '', 0, curr, curr, curr


def _best_setup_v6(d15, preferred_side):
    candidates = []
    for fn in (_detect_pullback_trigger, _detect_squeeze_break_trigger, _detect_sweep_reclaim_trigger):
        ok, label, quality, entry, sl, tp = fn(d15, preferred_side)
        if ok:
            candidates.append((quality, label, entry, sl, tp))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    quality, label, entry, sl, tp = candidates[0]
    return {'setup_label': label, 'setup_quality': quality, 'entry': entry, 'sl': sl, 'tp': tp}


def _direction_profile_v6(d15, d4h, d1d):
    c15 = d15['c'].astype(float); h15 = d15['h'].astype(float); l15 = d15['l'].astype(float)
    c4 = d4h['c'].astype(float); h4 = d4h['h'].astype(float); l4 = d4h['l'].astype(float)
    c1 = d1d['c'].astype(float)
    curr = float(c15.iloc[-1])
    adx15_df = ta.adx(h15, l15, c15, length=14)
    adx4_df = ta.adx(h4, l4, c4, length=14)
    adx15 = safe_last(adx15_df['ADX_14'], 20) if adx15_df is not None and 'ADX_14' in adx15_df else 20
    adx4 = safe_last(adx4_df['ADX_14'], 20) if adx4_df is not None and 'ADX_14' in adx4_df else 20
    bias15 = _ema_bias(c15)
    bias4 = _ema_bias(c4)
    bias1 = _ema_bias(c1, fast=9, mid=20, slow=50)
    consensus = bias15 * 1.1 + bias4 * 1.5 + bias1 * 1.7
    bb = ta.bbands(c15, length=20, std=2.0)
    bb_up = safe_last(bb.iloc[:, 0], curr) if bb is not None and not bb.empty else curr
    bb_low = safe_last(bb.iloc[:, 2], curr) if bb is not None and not bb.empty else curr
    width = (bb_up - bb_low) / max(curr, 1e-9)
    atr15 = max(safe_last(ta.atr(h15, l15, c15, length=14), curr * 0.004), curr * 0.003)
    range20 = float(h15.tail(20).max() - l15.tail(20).min()) if len(h15) >= 20 else float(h15.max() - l15.min())
    chop = adx15 < NO_TRADE_CHOP_ADX and width < 0.022 and range20 < atr15 * 7.5 and abs(consensus) < DIRECTION_WEAK_GATE
    if chop:
        return 0, 0.0, '震盪雜訊區', False, adx15, adx4
    side = 1 if consensus > 0 else -1 if consensus < 0 else 0
    strong = abs(consensus) >= DIRECTION_STRONG_GATE
    label = ('強多共振' if side > 0 else '強空共振') if strong else ('偏多結構' if side > 0 else '偏空結構') if side != 0 else '方向不足'
    return side, abs(consensus), label, strong, adx15, adx4


def _grade_signal_v6(direction_conf, setup_q, rr, anti_chase_penalty, htf_penalty):
    raw = direction_conf * 6.2 + setup_q * 4.6 + min(rr, 3.0) * 7.2 - anti_chase_penalty - htf_penalty
    if raw >= 84:
        return 'A+'
    if raw >= 74:
        return 'A'
    if raw >= 64:
        return 'B+'
    if raw >= 54:
        return 'B'
    if raw >= 46:
        return 'C'
    return 'D'


def analyze(symbol):
    is_major = symbol in MAJOR_COINS
    try:
        d15 = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=120), columns=['t','o','h','l','c','v'])
        time.sleep(0.18)
        d4h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '4h', limit=80), columns=['t','o','h','l','c','v'])
        time.sleep(0.18)
        d1d = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1d', limit=60), columns=['t','o','h','l','c','v'])
        time.sleep(0.08)
        if len(d15) < 80 or len(d4h) < 40 or len(d1d) < 40:
            return 0, '資料不足', 0, 0, 0, 0, {}, 0, 0, 0, 2.0, 3.0

        curr = float(d15['c'].iloc[-1])
        atr15 = max(safe_last(ta.atr(d15['h'], d15['l'], d15['c'], length=14), curr * 0.004), curr * 0.003)
        atr4h = max(safe_last(ta.atr(d4h['h'], d4h['l'], d4h['c'], length=14), curr * 0.008), curr * 0.006)
        atr = atr15
        breakdown = {}
        tags = []

        side, direction_conf, direction_label, direction_strong, adx15, adx4 = _direction_profile_v6(d15, d4h, d1d)
        breakdown['方向信心'] = round(direction_conf, 2)
        breakdown['ADX15'] = round(adx15, 1)
        breakdown['ADX4H'] = round(adx4, 1)
        tags.append(direction_label)

        if side == 0:
            return 0, '震盪過濾|方向不足', curr, 0, 0, 0, {'方向信心':0, 'Setup':'NoTrade', '等級':'D'}, atr, atr15, atr4h, 2.0, 3.0

        setup = _best_setup_v6(d15, side)
        if not setup:
            # 沒有明確觸發，維持觀察，不讓純方向分數硬進場
            base = 18 + direction_conf * 4
            capped = min(base, 34)
            return side * capped, '方向有但未到觸發位|等待回踩/突破確認', curr, 0, 0, 0, {
                '方向信心': round(direction_conf,2), 'Setup':'等待觸發', '進場品質':0, 'RR':0, '等級':'D'
            }, atr, atr15, atr4h, 2.0, 3.0

        setup_label = setup['setup_label']
        entry = float(setup['entry'])
        sl = float(setup['sl'])
        tp = float(setup['tp'])
        setup_q = float(setup['setup_quality'])
        tags.append(setup_label)

        ema21 = safe_last(ta.ema(d15['c'], length=21), curr)
        ext_atr = abs(curr - ema21) / max(atr15, 1e-9)
        anti_chase_penalty = 0
        if ext_atr > 1.35:
            anti_chase_penalty += 9
            tags.append('追價風險高')
        elif ext_atr > 1.05:
            anti_chase_penalty += 4
            tags.append('偏離均線')

        # 靠近4H反向極值時降權
        hh4 = float(d4h['h'].tail(30).max())
        ll4 = float(d4h['l'].tail(30).min())
        htf_penalty = 0
        if side > 0 and (hh4 - curr) / max(atr4h, 1e-9) < 0.55:
            htf_penalty += 5
            tags.append('接近4H壓力')
        if side < 0 and (curr - ll4) / max(atr4h, 1e-9) < 0.55:
            htf_penalty += 5
            tags.append('接近4H支撐')

        rr_ratio = abs(tp - entry) / max(abs(entry - sl), 1e-9)
        if rr_ratio < 1.55:
            htf_penalty += 8
            tags.append('風報比不足')
        elif rr_ratio >= 2.3:
            tags.append('風報比優秀')

        # 補上少量輔助因子，但不再讓它們主導方向
        rsi = safe_last(ta.rsi(d15['c'], length=14), 50)
        macd = ta.macd(d15['c'])
        hist = safe_last(macd['MACDh_12_26_9'], 0) if macd is not None and 'MACDh_12_26_9' in macd else 0
        helper = 0
        if side > 0:
            if 46 <= rsi <= 66:
                helper += 5; tags.append('RSI多頭甜蜜區')
            elif rsi > 74:
                helper -= 4; tags.append('RSI過熱')
            if hist > 0:
                helper += 4; tags.append('MACD順多')
        else:
            if 34 <= rsi <= 54:
                helper += 5; tags.append('RSI空頭甜蜜區')
            elif rsi < 26:
                helper -= 4; tags.append('RSI過冷')
            if hist < 0:
                helper += 4; tags.append('MACD順空')

        # 大盤同向加分，逆向扣分
        try:
            with MARKET_LOCK:
                mdir = MARKET_STATE.get('direction', '中性')
            if side > 0 and mdir in ('多', '強多'):
                helper += 4; tags.append('大盤順風')
            elif side < 0 and mdir in ('空', '強空'):
                helper += 4; tags.append('大盤順風')
            elif mdir != '中性':
                helper -= 3; tags.append('大盤逆風')
        except Exception:
            pass

        raw = direction_conf * 9.5 + setup_q * 5.8 + helper - anti_chase_penalty - htf_penalty
        if not direction_strong:
            raw *= 0.92
        score_abs = _clip(raw, 12, 92)
        if score_abs < 44:
            score_abs *= 0.78
        score = round(score_abs if side > 0 else -score_abs, 1)

        sl_mult = round(abs(entry - sl) / max(atr15, 1e-9), 2)
        tp_mult = round(abs(tp - entry) / max(atr15, 1e-9), 2)
        est_pnl = round(abs(tp - entry) / max(entry, 1e-9) * 100 * 20, 2)
        grade = _grade_signal_v6(direction_conf, setup_q, rr_ratio, anti_chase_penalty, htf_penalty)

        breakdown['進場品質'] = round(setup_q, 1)
        breakdown['RR'] = round(rr_ratio, 2)
        breakdown['Setup'] = setup_label
        breakdown['RegimeBias'] = side * round(direction_conf, 2)
        breakdown['追價風險'] = -anti_chase_penalty if side > 0 else anti_chase_penalty
        breakdown['高階位階壓力'] = -htf_penalty if side > 0 else htf_penalty
        breakdown['等級'] = grade
        breakdown['輔助因子'] = helper if side > 0 else -helper

        desc = '|'.join(tags[:8])
        return score, desc, round(entry, 6), round(sl, 6), round(tp, 6), est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult

    except Exception as e:
        import traceback
        print('analyze {} 失敗(v6): {}\n{}'.format(symbol, e, traceback.format_exc()[-400:]))
        return 0, '錯誤:{}'.format(str(e)[:40]), 0, 0, 0, 0, {}, 0, 0, 0, 2.0, 3.0


# =====================================================
# V7 AI 強化層：市場識別 / 自動回測 / 30筆自學習 / 記憶體維護
# 這層直接疊加在原本系統上，不拿掉既有功能。
# =====================================================
AI_DB_PATH = "/app/data/ai_learning_db.json"
AUTO_BACKTEST_STATE = {
    "running": False,
    "last_run": "--",
    "summary": "尚未啟動",
    "results": [],
    "target_count": 70,
    "scanned_markets": 0,
    "data_timeframes": ["5m", "15m", "1h", "4h", "1d"],
    "db_last_update": "--",
    "db_symbols": 0,
    "last_duration_sec": 0,
    "errors": [],
}
AI_PANEL = {
    "regime": "初始化中",
    "symbol_regimes": {},
    "best_strategies": [],
    "params": {
        "sl_mult": 2.0,
        "tp_mult": 3.5,
        "breakeven_atr": 0.9,
        "trail_trigger_atr": 1.4,
        "trail_pct": 0.035,
        "score_boost": {},
    },
    "memory": {
        "score_cache": 0,
        "signal_meta_cache": 0,
        "entry_locks": 0,
        "protection_state": 0,
        "fvg_orders": 0,
    },
    "last_learning": "--",
    "last_backtest": "--",
    "market_db_info": {
        "symbols": 0,
        "timeframes": ["5m", "15m", "1h", "4h", "1d"],
        "last_update": "--",
    },
}
AI_LOCK = threading.Lock()
AI_MARKET_TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
AI_MARKET_LIMIT = int(os.getenv('AI_MARKET_LIMIT', '70'))
AI_BACKTEST_LIMIT = int(os.getenv('AI_BACKTEST_KLINE_LIMIT', '320'))
AI_SNAPSHOT_LIMIT = int(os.getenv('AI_SNAPSHOT_KLINE_LIMIT', '240'))
AI_BACKTEST_SLEEP_SEC = int(os.getenv('AI_BACKTEST_SLEEP_SEC', '7200'))

def _default_ai_db():
    return {
        "regime_stats": {},
        "symbol_regime_stats": {},
        "indicator_weights": {},
        "combo_stats": {},
        "param_sets": {
            "trend":  {"sl_mult": 1.9, "tp_mult": 3.8, "breakeven_atr": 1.0, "trail_trigger_atr": 1.8, "trail_pct": 0.032},
            "range":  {"sl_mult": 1.5, "tp_mult": 2.2, "breakeven_atr": 0.7, "trail_trigger_atr": 1.1, "trail_pct": 0.022},
            "news":   {"sl_mult": 2.5, "tp_mult": 4.8, "breakeven_atr": 1.2, "trail_trigger_atr": 2.0, "trail_pct": 0.045},
            "neutral":{"sl_mult": 2.0, "tp_mult": 3.3, "breakeven_atr": 0.9, "trail_trigger_atr": 1.5, "trail_pct": 0.035},
        },
        "backtests": [],
        "strategy_scoreboard": [],
        "market_snapshots": {},
        "market_history_meta": {
            "symbols": 0,
            "timeframes": AI_MARKET_TIMEFRAMES,
            "last_update": "--",
        },
        "last_learning": "--",
        "version": 2,
    }

def load_ai_db():
    try:
        with open(AI_DB_PATH, 'r', encoding='utf-8') as f:
            db = json.load(f)
        base = _default_ai_db()
        for k, v in base.items():
            db.setdefault(k, v)
        return db
    except Exception:
        return _default_ai_db()

def save_ai_db(db):
    try:
        os.makedirs(os.path.dirname(AI_DB_PATH), exist_ok=True)
        with open(AI_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('AI DB 儲存失敗:', e)

AI_DB = load_ai_db()


def _is_crypto_usdt_swap_symbol(symbol):
    try:
        if not isinstance(symbol, str):
            return False
        if not symbol.endswith(':USDT'):
            return False
        base = symbol.split('/')[0].split(':')[0].upper()
        banned = {
            'AAPL','GOOGL','GOOG','AMZN','TSLA','MSFT','META','NVDA','NFLX','BABA','BIDU','JD','PDD','NIO','XPEV','LI',
            'SNAP','TWTR','UBER','LYFT','ABNB','COIN','HOOD','AMC','GME','SPY','QQQ','DJI','MSTR','PLTR','SQ','PYPL',
            'SHOP','INTC','AMD','QCOM','AVGO'
        }
        return base not in banned
    except Exception:
        return False


def fetch_top_volume_symbols(limit=70):
    try:
        tickers = exchange.fetch_tickers()
        ranked = sorted(
            [(sym, data) for sym, data in tickers.items() if _is_crypto_usdt_swap_symbol(sym)],
            key=lambda x: float((x[1] or {}).get('quoteVolume', 0) or 0),
            reverse=True,
        )
        symbols = [sym for sym, _ in ranked[:max(1, int(limit))]]
        return symbols, len(ranked)
    except Exception as e:
        print('抓前{}成交量市場失敗: {}'.format(limit, e))
        return [], 0


def _safe_fetch_ohlcv_df(symbol, timeframe, limit):
    try:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(rows, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        if df.empty:
            return None
        return df
    except Exception as e:
        print('抓K線失敗 {} {}: {}'.format(symbol, timeframe, e))
        return None


def _snapshot_from_df(df):
    if df is None or len(df) < 20:
        return None
    c = df['c'].astype(float)
    h = df['h'].astype(float)
    l = df['l'].astype(float)
    v = df['v'].astype(float)
    last = float(c.iloc[-1])
    atr = safe_last(ta.atr(h, l, c, length=14), 0)
    ema20 = safe_last(ta.ema(c, length=20), last)
    ema50 = safe_last(ta.ema(c, length=50), last)
    rsi = safe_last(ta.rsi(c, length=14), 50)
    adx_df = ta.adx(h, l, c, length=14)
    adx = safe_last(adx_df.iloc[:, 0], 0) if adx_df is not None and not adx_df.empty else 0
    vol_ratio = float(v.tail(5).mean()) / max(float(v.tail(30).mean()), 1e-9)
    ret = 0.0
    if len(c) >= 25:
        base = float(c.iloc[-25])
        ret = (last - base) / max(base, 1e-9) * 100
    return {
        'bars': int(len(df)),
        'last_close': round(last, 6),
        'atr': round(float(atr or 0), 6),
        'rsi': round(float(rsi or 0), 2),
        'adx': round(float(adx or 0), 2),
        'ema20': round(float(ema20 or 0), 6),
        'ema50': round(float(ema50 or 0), 6),
        'ret_24bars_pct': round(ret, 2),
        'vol_ratio': round(vol_ratio, 2),
    }


def _persist_market_snapshot(db, symbol, regime_info, timeframe_data):
    snap_root = db.setdefault('market_snapshots', {})
    snap_root[symbol] = {
        'symbol': symbol,
        'regime': dict(regime_info or {}),
        'timeframes': dict(timeframe_data or {}),
        'updated_at': tw_now_str('%Y-%m-%d %H:%M:%S'),
    }
    meta = db.setdefault('market_history_meta', {})
    meta['symbols'] = len(snap_root)
    meta['timeframes'] = AI_MARKET_TIMEFRAMES
    meta['last_update'] = tw_now_str('%Y-%m-%d %H:%M:%S')


def _refresh_ai_panel_market_meta():
    with AI_LOCK:
        meta = dict((AI_DB.get('market_history_meta', {}) or {}))
        AI_PANEL['market_db_info'] = {
            'symbols': int(meta.get('symbols', 0) or 0),
            'timeframes': list(meta.get('timeframes', AI_MARKET_TIMEFRAMES) or AI_MARKET_TIMEFRAMES),
            'last_update': meta.get('last_update', '--') or '--',
        }
        AUTO_BACKTEST_STATE['db_symbols'] = AI_PANEL['market_db_info']['symbols']
        AUTO_BACKTEST_STATE['db_last_update'] = AI_PANEL['market_db_info']['last_update']
        AUTO_BACKTEST_STATE['data_timeframes'] = AI_PANEL['market_db_info']['timeframes']


_refresh_ai_panel_market_meta()

def get_margin_learning_multiplier(symbol, score, breakdown):
    try:
        with LEARN_LOCK:
            ss = LEARN_DB.get('symbol_stats', {}).get(symbol, {})
        count = int(ss.get('count', 0) or 0)
        if count < 5:
            return 1.0
        wr = float(ss.get('win', 0)) / max(count, 1)
        mult = 1.0
        if wr >= 0.62:
            mult += 0.08
        elif wr < 0.4:
            mult -= 0.10
        if abs(float(score or 0)) >= 70:
            mult += 0.04
        if isinstance(breakdown, dict):
            rr = float(breakdown.get('RR', 0) or 0)
            if rr >= 2.0:
                mult += 0.04
        return round(clamp(mult, 0.82, 1.15), 4)
    except Exception:
        return 1.0

def classify_market_regime(df15, df1h=None):
    try:
        c = df15['c'].astype(float)
        h = df15['h'].astype(float)
        l = df15['l'].astype(float)
        v = df15['v'].astype(float)
        curr = float(c.iloc[-1])
        adx = safe_last(ta.adx(h, l, c, length=14).iloc[:, 0], 18)
        atr = max(safe_last(ta.atr(h, l, c, length=14), curr * 0.004), curr * 0.003)
        atr_ratio = atr / max(curr, 1e-9)
        bb = ta.bbands(c, length=20, std=2)
        bb_up = safe_last(bb.iloc[:, 0], curr) if bb is not None and not bb.empty else curr
        bb_low = safe_last(bb.iloc[:, 2], curr) if bb is not None and not bb.empty else curr
        bb_width = (bb_up - bb_low) / max(curr, 1e-9)
        ret3 = abs(float(c.iloc[-1] - c.iloc[-4]) / max(c.iloc[-4], 1e-9) * 100) if len(c) >= 4 else 0
        vol_now = float(v.tail(3).mean()) if len(v) >= 3 else float(v.iloc[-1])
        vol_base = float(v.tail(30).head(20).mean()) if len(v) >= 30 else max(vol_now, 1.0)
        vol_ratio = vol_now / max(vol_base, 1e-9)
        slope = _linreg_slope(c.tail(14).tolist()) / max(curr, 1e-9) * 100
        dir_hint = '中性'
        if slope > 0.06:
            dir_hint = '多'
        elif slope < -0.06:
            dir_hint = '空'
        if ret3 >= 2.2 and vol_ratio >= 1.8 and atr_ratio >= 0.01:
            regime = 'news'; confidence = 0.9; note = '短時間爆量急拉急殺'
        elif adx >= 23 and abs(slope) >= 0.08 and bb_width >= 0.018:
            regime = 'trend'; confidence = min(0.95, 0.55 + adx / 50); note = 'ADX與斜率同步，屬於趨勢盤'
        elif adx <= 18 and bb_width <= 0.02:
            regime = 'range'; confidence = 0.72; note = '低ADX低波動，偏區間盤'
        else:
            regime = 'neutral'; confidence = 0.55; note = '混合結構，走勢未完全定型'
        return {'regime': regime,'direction': dir_hint,'confidence': round(confidence, 3),'adx': round(adx, 2),'atr_ratio': round(atr_ratio, 5),'bb_width': round(bb_width, 5),'vol_ratio': round(vol_ratio, 2),'move_3bars_pct': round(ret3, 2),'note': note}
    except Exception as e:
        return {'regime': 'neutral', 'direction': '中性', 'confidence': 0.4, 'note': f'判定失敗:{e}'}

def get_regime_params(regime):
    with AI_LOCK:
        return dict(AI_DB.get('param_sets', {}).get(regime, AI_DB.get('param_sets', {}).get('neutral', {})))

_BASE_ANALYZE = analyze
_BASE_LEARN_FROM_CLOSED_TRADE = learn_from_closed_trade
_BASE_RUN_SIMPLE_BACKTEST = run_simple_backtest
_BASE_API_STATE = api_state

def _fetch_regime_for_symbol(symbol):
    try:
        d15 = pd.DataFrame(exchange.fetch_ohlcv(symbol, '15m', limit=160), columns=['t','o','h','l','c','v'])
        d1h = pd.DataFrame(exchange.fetch_ohlcv(symbol, '1h', limit=160), columns=['t','o','h','l','c','v'])
        return classify_market_regime(d15, d1h)
    except Exception as e:
        return {'regime': 'neutral', 'direction': '中性', 'confidence': 0.3, 'note': f'抓K失敗:{e}'}

def _safe_num(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _signal_quality_from_breakdown(breakdown, side):
    bd = dict(breakdown or {})
    quality = 0.0
    notes = []
    rr = _safe_num(bd.get('RR', 0))
    entry_gate = _safe_num(bd.get('EntryGate', bd.get('進場品質', 0)))
    regime_bias = _safe_num(bd.get('RegimeBias', bd.get('方向品質', 0)))
    if rr >= 2.0:
        quality += 2.2
        notes.append('RR佳')
    elif rr >= 1.5:
        quality += 1.2
    elif 0 < rr < 1.2:
        quality -= 2.5
        notes.append('RR弱')
    if entry_gate >= 4:
        quality += 2.0
        notes.append('進場佳')
    elif entry_gate <= 0:
        quality -= 2.2
        notes.append('進場弱')
    if regime_bias * side > 0:
        quality += min(abs(regime_bias) * 0.35, 2.0)
        notes.append('方向同向')
    elif regime_bias * side < 0:
        quality -= min(abs(regime_bias) * 0.45, 3.0)
        notes.append('方向逆風')
    if '高波動過熱' in bd:
        quality -= 1.6
        notes.append('波動過熱')
    if '4H趨勢不順' in bd:
        quality -= 2.2
        notes.append('逆4H')
    if '風報比不足' in bd:
        quality -= 2.0
    return round(quality, 2), notes


def _learning_edge(symbol, regime):
    edge = 0.0
    note = ''
    try:
        with LEARN_LOCK:
            ss = dict(LEARN_DB.get('symbol_stats', {}).get(symbol, {}) or {})
        n = int(ss.get('count', 0) or 0)
        if n >= 8:
            wr = float(ss.get('win', 0) or 0) / max(n, 1)
            if wr >= 0.60:
                edge += 2.0
                note = '該幣歷史較強'
            elif wr < 0.40:
                edge -= 2.5
                note = '該幣歷史偏弱'
        with AI_LOCK:
            sr = dict((AI_DB.get('symbol_regime_stats', {}) or {}).get(f'{symbol}|{regime}', {}) or {})
        rn = int(sr.get('count', 0) or 0)
        if rn >= 4:
            rwr = float(sr.get('win', 0) or 0) / max(rn, 1)
            ravg = float(sr.get('pnl_sum', 0) or 0) / max(rn, 1)
            edge += max(min((rwr - 0.5) * 10 + ravg * 0.2, 3.5), -3.5)
            if not note:
                note = '該幣在此市場型態有統計優勢' if rwr >= 0.55 else '該幣在此市場型態需保守'
    except Exception:
        pass
    return round(edge, 2), note


def _apply_regime_to_signal(symbol, score, desc, entry, sl, tp, est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult):
    regime_info = _fetch_regime_for_symbol(symbol)
    regime = regime_info.get('regime', 'neutral')
    params = get_regime_params(regime)
    breakdown = dict(breakdown or {})
    score = _safe_num(score)
    entry = _safe_num(entry)
    sl = _safe_num(sl)
    tp = _safe_num(tp)
    atr15 = max(_safe_num(atr15), 0.0)
    side = 1 if score >= 0 else -1
    rr = abs(tp - entry) / max(abs(entry - sl), 1e-9) if entry and sl and tp else _safe_num(breakdown.get('RR', 0))
    direction = regime_info.get('direction', '中性')
    conf = _safe_num(regime_info.get('confidence', 0.5))
    slope_dir = 1 if direction == '多' else -1 if direction == '空' else 0

    score_boost = 0.0
    extra = []

    # 1) 先看 base 分析品質
    quality_boost, quality_notes = _signal_quality_from_breakdown(breakdown, side)
    score_boost += quality_boost
    extra.extend(quality_notes)

    # 2) 市場型態智能加權
    if regime == 'trend':
        if slope_dir == side:
            score_boost += 3.0 + conf * 2.5
            extra.append('趨勢同向')
            if rr >= 1.6:
                score_boost += 1.6
                extra.append('趨勢盤RR佳')
        elif slope_dir != 0:
            score_boost -= 4.5 + conf * 2.0
            extra.append('逆趨勢')
        else:
            score_boost -= 0.8
    elif regime == 'range':
        if abs(score) >= 60 and rr >= 1.4:
            score_boost += 0.8 * side
            extra.append('區間強反轉')
        else:
            score_boost -= 2.6 * side
            extra.append('區間盤降追價')
        if abs(_safe_num(regime_info.get('bb_width', 0))) < 0.018:
            score_boost -= 0.8 * side
    elif regime == 'news':
        move = _safe_num(regime_info.get('move_3bars_pct', 0))
        volr = _safe_num(regime_info.get('vol_ratio', 1))
        if abs(score) >= 64 and rr >= 1.8:
            score_boost += (3.5 + min(move, 4) * 0.7) * side
            extra.append('消息盤強訊號')
        else:
            score_boost -= (4.0 + min(volr, 3.0) * 0.8) * side
            extra.append('消息盤保守')
    else:
        if rr >= 1.7:
            score_boost += 1.2 * side
            extra.append('中性盤留強勢')
        elif rr < 1.25:
            score_boost -= 1.4 * side
            extra.append('中性盤淘汰弱RR')

    # 3) 學習資料加權
    learn_boost, learn_note = _learning_edge(symbol, regime)
    if learn_boost:
        score_boost += learn_boost * side
        if learn_note:
            extra.append(learn_note)

    # 4) 依市場型態覆蓋風控參數
    new_sl_mult = float(params.get('sl_mult', sl_mult or 2.0))
    new_tp_mult = float(params.get('tp_mult', tp_mult or 3.5))
    if regime == 'news' and conf >= 0.8:
        new_sl_mult = max(new_sl_mult, 2.4)
        new_tp_mult = max(new_tp_mult, 4.2)
    elif regime == 'range':
        new_sl_mult = min(new_sl_mult, 1.7)
        new_tp_mult = min(new_tp_mult, 2.6)
    elif regime == 'trend' and slope_dir == side and rr >= 1.5:
        new_tp_mult = max(new_tp_mult, 3.8)

    if entry and atr15:
        if side > 0:
            sl = round(entry - atr15 * new_sl_mult, 6)
            tp = round(entry + atr15 * new_tp_mult, 6)
        else:
            sl = round(entry + atr15 * new_sl_mult, 6)
            tp = round(entry - atr15 * new_tp_mult, 6)
        sl_mult = round(new_sl_mult, 2)
        tp_mult = round(new_tp_mult, 2)
        rr = abs(tp - entry) / max(abs(entry - sl), 1e-9)
        est_pnl = round(abs(tp - entry) / max(entry, 1e-9) * 100 * 20, 2)

    final_score = round(max(min(score + score_boost, 100), -100), 1)
    breakdown['Regime'] = regime
    breakdown['RegimeConf'] = round(conf, 3)
    breakdown['RegimeDir'] = direction
    breakdown['RegimeScoreAdj'] = round(score_boost, 2)
    breakdown['SignalQuality'] = round(quality_boost, 2)
    breakdown['LearnEdge'] = round(learn_boost, 2)
    breakdown['RR'] = round(rr, 2)
    breakdown['AdaptiveSL'] = round(sl_mult, 2)
    breakdown['AdaptiveTP'] = round(tp_mult, 2)

    desc = (desc + '|' if desc else '') + '市場:{}({}/{:.0%})'.format(regime, direction, conf)
    if extra:
        desc += '|' + '|'.join(dict.fromkeys(extra))

    with AI_LOCK:
        AI_PANEL['symbol_regimes'][symbol] = dict(regime_info, score_adjust=round(score_boost, 2), quality=round(quality_boost, 2), learn_edge=round(learn_boost, 2))
        AI_PANEL['regime'] = regime
        AI_PANEL['params'].update({'sl_mult': sl_mult,'tp_mult': tp_mult,'breakeven_atr': float(params.get('breakeven_atr', 0.9)),'trail_trigger_atr': float(params.get('trail_trigger_atr', 1.4)),'trail_pct': float(params.get('trail_pct', 0.035))})
    return final_score, desc, entry, sl, tp, est_pnl, breakdown, atr, atr15, atr4h, sl_mult, tp_mult

def analyze(symbol):
    base = _BASE_ANALYZE(symbol)
    try:
        return _apply_regime_to_signal(symbol, *base)
    except Exception as e:
        print('AI regime overlay失敗 {}: {}'.format(symbol, e))
        return base

def _extract_strategy_key(trade):
    bd = trade.get('breakdown', {}) or {}
    regime = bd.get('Regime', 'neutral')
    setup = bd.get('Setup', 'unknown')
    symbol = trade.get('symbol', 'NA')
    return f'{regime}|{setup}|{symbol}'


def _enhanced_auto_learn():
    with LEARN_LOCK:
        closed = [t for t in LEARN_DB.get('trades', []) if _is_live_source(t.get('source')) and t.get('result') in ('win', 'loss')]
        total = len(closed)
    if total < 20:
        return
    db = AI_DB
    combo_stats = db.setdefault('combo_stats', {})
    regime_stats = db.setdefault('regime_stats', {})
    symbol_regime_stats = db.setdefault('symbol_regime_stats', {})
    blocked_strategy_keys = set(db.setdefault('blocked_strategy_keys', []))
    blocked_symbols = set(db.setdefault('blocked_symbols', []))

    combo_stats.clear(); regime_stats.clear(); symbol_regime_stats.clear()

    recent_closed = closed[-240:]
    for t in recent_closed:
        key = _extract_strategy_key(t)
        metric = float(_trade_learn_metric(t) or 0.0)
        rec = combo_stats.setdefault(key, {
            'count': 0, 'win': 0, 'loss': 0,
            'pnl_sum': 0.0, 'pnl_list': [],
            'gross_win': 0.0, 'gross_loss': 0.0,
        })
        rec['count'] += 1
        if t.get('result') == 'win':
            rec['win'] += 1
            rec['gross_win'] += max(metric, 0.0)
        else:
            rec['loss'] += 1
            rec['gross_loss'] += abs(min(metric, 0.0))
        rec['pnl_sum'] += metric
        rec['pnl_list'].append(metric)

        regime = (t.get('breakdown', {}) or {}).get('Regime', 'neutral')
        rs = regime_stats.setdefault(regime, {'count': 0, 'win': 0, 'pnl_sum': 0.0})
        rs['count'] += 1
        if t.get('result') == 'win':
            rs['win'] += 1
        rs['pnl_sum'] += metric

        sym = t.get('symbol', 'NA')
        sk = f'{sym}|{regime}'
        sr = symbol_regime_stats.setdefault(sk, {'count': 0, 'win': 0, 'pnl_sum': 0.0})
        sr['count'] += 1
        if t.get('result') == 'win':
            sr['win'] += 1
        sr['pnl_sum'] += metric

    board = []
    new_blocked_strategy_keys = set()
    new_blocked_symbols = set()
    for key, rec in combo_stats.items():
        count = int(rec.get('count', 0) or 0)
        if count < 5:
            continue
        wins = int(rec.get('win', 0) or 0)
        wr = wins / max(count, 1)
        avg = float(rec.get('pnl_sum', 0.0) or 0.0) / max(count, 1)
        pnl_list = list(rec.get('pnl_list', []) or [])
        eq = 100.0
        peak = 100.0
        max_dd = 0.0
        for p in pnl_list:
            step = max(0.01, 1.0 + (float(p) / 100.0))
            eq *= step
            peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak * 100.0)
        gross_win = float(rec.get('gross_win', 0.0) or 0.0)
        gross_loss = abs(float(rec.get('gross_loss', 0.0) or 0.0))
        pf = (gross_win / max(gross_loss, 1e-9)) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
        std = (sum((float(p) - avg) ** 2 for p in pnl_list) / max(len(pnl_list), 1)) ** 0.5
        conf = min(count / 50.0, 1.0) * max(0.0, 1.0 - min(std / 3.0, 1.0))

        if count >= 12 and (wr < 0.30 and avg < -0.20 and max_dd >= 25):
            new_blocked_strategy_keys.add(key)

        sym = str(key).split('|')[-1]
        if count >= 14 and wr < 0.28 and avg < -0.25:
            new_blocked_symbols.add(sym)

        score = (
            avg * 35.0 +
            ((wr * 100.0) - 50.0) * 0.6 +
            min(pf, 3.0) * 12.0 +
            min(count, 30) * 0.5 -
            max_dd * 0.35 +
            conf * 8.0
        )

        board.append({
            'strategy': key,
            'count': count,
            'win_rate': round(wr * 100, 1),
            'avg_pnl': round(avg, 4),
            'score': round(score, 3),
            'ev_per_trade': round(avg, 4),
            'profit_factor': round(pf, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'confidence': round(conf, 3),
        })

    board.sort(key=lambda x: (x['score'], x['count'], x['profit_factor'], x['win_rate']), reverse=True)
    db['strategy_scoreboard'] = board[:20]
    db['blocked_strategy_keys'] = sorted(new_blocked_strategy_keys)
    db['blocked_symbols'] = sorted(new_blocked_symbols)

    recent = recent_closed[-60:]
    if recent:
        avg = sum(_trade_learn_metric(t) for t in recent) / len(recent)
        wr = sum(1 for t in recent if t.get('result') == 'win') / len(recent)
        for regime, p in db.get('param_sets', {}).items():
            if wr >= 0.60 and avg > 0:
                p['tp_mult'] = round(min(float(p.get('tp_mult', 3.0)) * 1.02, 5.2), 2)
                p['trail_pct'] = round(max(float(p.get('trail_pct', 0.03)) * 0.99, 0.018), 4)
            elif wr < 0.45:
                p['sl_mult'] = round(min(float(p.get('sl_mult', 2.0)) * 1.02, 3.0), 2)
                p['tp_mult'] = round(max(float(p.get('tp_mult', 3.0)) * 0.98, 2.0), 2)

    db['last_learning'] = tw_now_str('%Y-%m-%d %H:%M:%S')
    save_ai_db(db)
    with AI_LOCK:
        AI_PANEL['best_strategies'] = db.get('strategy_scoreboard', [])[:8]
        AI_PANEL['last_learning'] = db['last_learning']
        AI_PANEL['params'].update(db.get('param_sets', {}).get('neutral', {}))


def learn_from_closed_trade(trade_id):
    _BASE_LEARN_FROM_CLOSED_TRADE(trade_id)
    try:
        _enhanced_auto_learn()
    except Exception as e:
        print('增強學習失敗:', e)

def run_simple_backtest(symbol='BTC/USDT:USDT', timeframe='15m', limit=800, fee_rate=0.0006):
    base = _BASE_RUN_SIMPLE_BACKTEST(symbol=symbol, timeframe=timeframe, limit=limit, fee_rate=fee_rate)
    if not base.get('ok'):
        return base
    regime = _fetch_regime_for_symbol(symbol)
    params = get_regime_params(regime.get('regime', 'neutral'))
    base['market_regime'] = regime
    base['ai_params'] = params
    base['ai_comment'] = f"{symbol} 當前屬於 {regime.get('regime')}，回測以 {regime.get('note')} 參考調參"
    return base

def run_multi_market_backtest(symbols=None):
    started_at = time.time()
    if symbols is None:
        symbols, eligible_count = fetch_top_volume_symbols(AI_MARKET_LIMIT)
    else:
        eligible_count = len(symbols)
    if not symbols:
        with AI_LOCK:
            AUTO_BACKTEST_STATE['running'] = False
            AUTO_BACKTEST_STATE['summary'] = '找不到可回測市場'
            AUTO_BACKTEST_STATE['scanned_markets'] = 0
            AUTO_BACKTEST_STATE['errors'] = ['無可用市場']
        update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE))
        return []

    results = []
    errors = []
    scoreboard = []
    scanned = 0
    for idx, sym in enumerate(symbols, start=1):
        try:
            timeframe_data = {}
            regime_seed_df = None
            for tf in AI_MARKET_TIMEFRAMES:
                df_tf = _safe_fetch_ohlcv_df(sym, tf, AI_SNAPSHOT_LIMIT)
                if df_tf is not None:
                    timeframe_data[tf] = _snapshot_from_df(df_tf)
                    if tf == '15m':
                        regime_seed_df = df_tf.rename(columns={'o': 'o', 'h': 'h', 'l': 'l', 'c': 'c', 'v': 'v'})
            if not timeframe_data:
                errors.append(f'{sym}: 無法抓取多週期K線')
                continue

            bt = run_simple_backtest(symbol=sym, timeframe='15m', limit=AI_BACKTEST_LIMIT)
            if not bt.get('ok'):
                errors.append(f"{sym}: {bt.get('error', '回測失敗')}")
                continue

            regime_info = bt.get('market_regime') if isinstance(bt.get('market_regime'), dict) else None
            if not regime_info and regime_seed_df is not None:
                try:
                    regime_info = classify_market_regime(regime_seed_df)
                except Exception:
                    regime_info = {'regime': 'neutral', 'direction': '中性', 'confidence': 0.4}
            regime_info = regime_info or {'regime': 'neutral', 'direction': '中性', 'confidence': 0.4}

            scanned += 1
            trades_n = int(bt.get('trades', bt.get('total_trades', 0)) or 0)
            return_pct = round(float(bt.get('return_pct', bt.get('net_profit_pct', 0)) or 0), 2)
            pf_val = round(float(bt.get('profit_factor', 0) or 0), 3) if bt.get('profit_factor') is not None else None
            dd_val = round(float(bt.get('max_drawdown_pct', 0) or 0), 2)
            ev_per_trade = round(return_pct / max(trades_n, 1), 4)
            strategy_mode = 'breakout' if regime_info.get('regime') in ('news', 'breakout') else 'range' if regime_info.get('regime') == 'range' else 'main'
            result_row = {
                'symbol': sym,
                'win_rate': round(float(bt.get('win_rate', 0) or 0), 2),
                'return_pct': return_pct,
                'profit_factor': pf_val,
                'max_drawdown_pct': dd_val,
                'ev_per_trade': ev_per_trade,
                'trades': trades_n,
                'market_regime': regime_info.get('regime', 'neutral'),
                'strategy_mode': strategy_mode,
                'regime_confidence': round(float(regime_info.get('confidence', 0) or 0), 3),
                'timeframes': list(timeframe_data.keys()),
                'updated_at': tw_now_str('%Y-%m-%d %H:%M:%S'),
            }
            results.append(result_row)
            sample_conf = min(max(trades_n / 30.0, 0), 1)
            ev_component = ev_per_trade * 180.0
            pf_component = ((pf_val or 1.0) - 1.0) * 18.0
            dd_penalty = dd_val * 1.6
            win_component = ((result_row['win_rate'] - 50.0) * sample_conf * 0.35)
            score_val = ev_component + pf_component + win_component - dd_penalty + min(trades_n, 30) * 0.45
            scoreboard.append({
                'strategy': '{}|{}'.format(sym, regime_info.get('regime', 'neutral')),
                'strategy_mode': strategy_mode,
                'count': trades_n,
                'win_rate': result_row['win_rate'],
                'avg_pnl': return_pct,
                'ev_per_trade': ev_per_trade,
                'profit_factor': pf_val,
                'max_drawdown_pct': dd_val,
                'score': round(score_val, 2),
                'confidence': round(sample_conf, 2),
                'timeframes': '/'.join(result_row['timeframes']),
                'updated_at': result_row['updated_at'],
            })
            with AI_LOCK:
                _persist_market_snapshot(AI_DB, sym, regime_info, timeframe_data)
                AUTO_BACKTEST_STATE['scanned_markets'] = scanned
                AUTO_BACKTEST_STATE['summary'] = 'AI回測進行中 {}/{}｜成功 {}｜失敗 {}'.format(idx, len(symbols), scanned, len(errors))
        except Exception as e:
            errors.append(f'{sym}: {str(e)[:90]}')
            print('multi backtest失敗 {}: {}'.format(sym, e))

    results.sort(key=lambda x: (x.get('ev_per_trade', 0), x.get('profit_factor') or 0, -(x.get('max_drawdown_pct', 0) or 0), x.get('trades', 0)), reverse=True)
    scoreboard.sort(key=lambda x: (x.get('score', 0), x.get('ev_per_trade', 0), x.get('count', 0)), reverse=True)
    with AI_LOCK:
        AUTO_BACKTEST_STATE['running'] = False
        AUTO_BACKTEST_STATE['last_run'] = tw_now_str('%Y-%m-%d %H:%M:%S')
        AUTO_BACKTEST_STATE['target_count'] = len(symbols)
        AUTO_BACKTEST_STATE['scanned_markets'] = scanned
        AUTO_BACKTEST_STATE['last_duration_sec'] = round(time.time() - started_at, 1)
        AUTO_BACKTEST_STATE['errors'] = errors[:12]
        AUTO_BACKTEST_STATE['summary'] = '完成前{}成交量市場回測｜成功 {}｜失敗 {}｜候選總數 {}'.format(len(symbols), scanned, len(errors), eligible_count)
        AUTO_BACKTEST_STATE['results'] = results[:12]
        AI_PANEL['last_backtest'] = AUTO_BACKTEST_STATE['last_run']
        AI_PANEL['best_strategies'] = scoreboard[:12]
        AI_DB['strategy_scoreboard'] = scoreboard[:20]
        AI_DB.setdefault('backtests', []).append({
            'time': AUTO_BACKTEST_STATE['last_run'],
            'target_count': len(symbols),
            'scanned_markets': scanned,
            'results': results[:20],
            'errors': errors[:20],
            'duration_sec': AUTO_BACKTEST_STATE['last_duration_sec'],
        })
        AI_DB['backtests'] = AI_DB['backtests'][-30:]
        meta = AI_DB.setdefault('market_history_meta', {})
        meta['symbols'] = len((AI_DB.get('market_snapshots', {}) or {}))
        meta['timeframes'] = AI_MARKET_TIMEFRAMES
        meta['last_update'] = AUTO_BACKTEST_STATE['last_run']
        AI_PANEL['market_db_info'] = {
            'symbols': meta['symbols'],
            'timeframes': meta['timeframes'],
            'last_update': meta['last_update'],
        }
        AUTO_BACKTEST_STATE['db_symbols'] = meta['symbols']
        AUTO_BACKTEST_STATE['db_last_update'] = meta['last_update']
        AUTO_BACKTEST_STATE['data_timeframes'] = meta['timeframes']
    save_ai_db(AI_DB)
    update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE))
    return results

def auto_backtest_thread():
    while True:
        try:
            with AI_LOCK:
                AUTO_BACKTEST_STATE['running'] = True
                AUTO_BACKTEST_STATE['summary'] = '自動回測中...'
            sync_ai_state_to_dashboard(force_regime=True)
            run_multi_market_backtest()
            sync_ai_state_to_dashboard(force_regime=True)
        except Exception as e:
            print('自動回測執行緒失敗:', e)
            with AI_LOCK:
                AUTO_BACKTEST_STATE['running'] = False
                AUTO_BACKTEST_STATE['summary'] = '回測失敗: {}'.format(str(e)[:80])
        time.sleep(AI_BACKTEST_SLEEP_SEC)

def memory_guard_thread():
    while True:
        try:
            for cache in [SIGNAL_META_CACHE, SCORE_CACHE, ENTRY_LOCKS, PROTECTION_STATE]:
                if len(cache) > 500:
                    for k in list(cache.keys())[:200]:
                        cache.pop(k, None)
            with AI_LOCK:
                AI_PANEL['memory'] = {'score_cache': len(SCORE_CACHE),'signal_meta_cache': len(SIGNAL_META_CACHE),'entry_locks': len(ENTRY_LOCKS),'protection_state': len(PROTECTION_STATE),'fvg_orders': len(FVG_ORDERS)}
            gc.collect()
            update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE))
        except Exception as e:
            print('記憶體守護失敗:', e)
        time.sleep(120)

def enhanced_position_thread():
    while True:
        try:
            with TRAILING_LOCK:
                for sym, ts in list(TRAILING_STATE.items()):
                    side = str(ts.get('side', '')).lower()
                    entry = float(ts.get('entry_price', 0) or 0)
                    atr = float(ts.get('atr', 0) or 0)
                    if entry <= 0 or atr <= 0: continue
                    ticker = exchange.fetch_ticker(sym)
                    mark = float(ticker.get('last', 0) or 0)
                    if mark <= 0: continue
                    params = get_regime_params((AI_PANEL.get('symbol_regimes', {}).get(sym) or {}).get('regime', 'neutral'))
                    breakeven_atr = float(params.get('breakeven_atr', 0.9))
                    trail_trigger_atr = float(params.get('trail_trigger_atr', 1.4))
                    trail_pct = float(params.get('trail_pct', ts.get('trail_pct', 0.035)))
                    if side in ('buy', 'long'):
                        profit_atr = (mark - entry) / max(atr, 1e-9)
                        if profit_atr >= breakeven_atr: ts['initial_sl'] = max(float(ts.get('initial_sl', 0) or 0), entry)
                        if profit_atr >= trail_trigger_atr: ts['trail_pct'] = min(float(ts.get('trail_pct', trail_pct) or trail_pct), trail_pct)
                    elif side in ('sell', 'short'):
                        profit_atr = (entry - mark) / max(atr, 1e-9)
                        if profit_atr >= breakeven_atr: ts['initial_sl'] = min(float(ts.get('initial_sl', entry * 9) or entry * 9), entry)
                        if profit_atr >= trail_trigger_atr: ts['trail_pct'] = min(float(ts.get('trail_pct', trail_pct) or trail_pct), trail_pct)
        except Exception as e:
            print('強化保本/動態止盈失敗:', e)
        time.sleep(8)



def extract_analysis_score(result):
    """相容 analyze() 不同回傳格式，穩定取出分數。"""
    try:
        if isinstance(result, (list, tuple)):
            if len(result) >= 1:
                return float(result[0] or 0)
        if isinstance(result, dict):
            for key in ('score', 'final_score', 'stable_score', 'raw_score'):
                if key in result:
                    return float(result.get(key) or 0)
        return 0.0
    except Exception:
        return 0.0


def sync_ai_state_to_dashboard(force_regime=False):
    """把 AI 面板/回測狀態強制同步進 STATE，避免前端全是 --。"""
    try:
        with AI_LOCK:
            ai_panel = dict(AI_PANEL)
            auto_bt = dict(AUTO_BACKTEST_STATE)
            params = dict((ai_panel.get('params') or {}))
            market_db = dict((ai_panel.get('market_db_info') or {}))
            if force_regime and (not ai_panel.get('regime') or ai_panel.get('regime') in ('初始化中', '--')):
                patt = ((STATE.get('market_info') or {}).get('pattern') or '').strip()
                if patt:
                    ai_panel['regime'] = patt
            ai_panel.setdefault('regime', 'neutral')
            ai_panel['params'] = {
                'sl_mult': params.get('sl_mult', 2.0),
                'tp_mult': params.get('tp_mult', 3.3),
                'breakeven_atr': params.get('breakeven_atr', 0.9),
                'trail_trigger_atr': params.get('trail_trigger_atr', 1.5),
                'trail_pct': params.get('trail_pct', 0.035),
            }
            ai_panel['market_db_info'] = {
                'symbols': market_db.get('symbols', auto_bt.get('db_symbols', 0)),
                'timeframes': market_db.get('timeframes', auto_bt.get('data_timeframes', ['5m','15m','1h','4h','1d'])),
                'last_update': market_db.get('last_update', auto_bt.get('db_last_update', '--')),
            }
            auto_bt.setdefault('target_count', AI_MARKET_LIMIT)
            auto_bt.setdefault('data_timeframes', ['5m','15m','1h','4h','1d'])
            update_state(ai_panel=ai_panel, auto_backtest=auto_bt)
    except Exception as e:
        print('同步 AI 狀態失敗:', e)


@app.route('/api/ai_status')
def api_ai_status():
    sync_ai_state_to_dashboard(force_regime=True)
    with AI_LOCK:
        return jsonify({
            'ok': True,
            'ai_panel': dict(AI_PANEL),
            'auto_backtest': dict(AUTO_BACKTEST_STATE),
        })


@app.route('/api/ai_db_stats')
def api_ai_db_stats():
    live_open = get_live_trades(closed_only=False)
    live_closed = get_live_trades(closed_only=True)
    symbols = sorted(list({str(t.get('symbol') or '') for t in live_closed if t.get('symbol')}))
    latest = None
    if live_closed:
        lt = live_closed[-1]
        latest = {
            'symbol': lt.get('symbol'),
            'result': lt.get('result'),
            'pnl_pct': lt.get('pnl_pct'),
            'exit_time': lt.get('exit_time') or lt.get('time'),
            'source': lt.get('source'),
        }
    closed_count = len(live_closed)
    phase = 'learning' if closed_count < 20 else 'semi' if closed_count < 50 else 'full'
    return jsonify({
        'ok': True,
        'data_scope': 'live_only',
        'open_live_count': len([t for t in live_open if t.get('result') == 'open']),
        'closed_live_count': closed_count,
        'last_learning': AI_PANEL.get('last_learning', '--'),
        'symbols': symbols,
        'latest': latest,
        'backtest_run_count': len((BACKTEST_DB.get('runs') or [])),
        'ai_phase': phase,
        'ai_ready': closed_count >= 50,
        'mode': phase,
    })

def api_state_enhanced():
    resp = _BASE_API_STATE()
    try:
        payload = resp.get_json() or {}
    except Exception:
        payload = {}
    try:
        sync_ai_state_to_dashboard(force_regime=True)
        with AI_LOCK:
            payload['ai_panel'] = dict(AI_PANEL)
            payload['auto_backtest'] = dict(AUTO_BACKTEST_STATE)
        if 'learn_summary' not in payload:
            payload['learn_summary'] = dict(STATE.get('learn_summary', {}))
        return jsonify(payload)
    except Exception as e:
        payload['api_state_fix_error'] = str(e)
        return jsonify(payload)


@app.route('/api/cancel_fvg_order', methods=['POST'])
def api_cancel_fvg_order_alias():
    return api_fvg_cancel()

@app.route('/api/force_backtest', methods=['POST'])
def api_force_backtest():
    try:
        results = run_multi_market_backtest()
        sync_ai_state_to_dashboard(force_regime=True)
        return jsonify({'ok': True, 'results': results, 'summary': AUTO_BACKTEST_STATE.get('summary', ''), 'auto_backtest': dict(AUTO_BACKTEST_STATE), 'ai_panel': dict(AI_PANEL)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/risk_override', methods=['POST'])
def api_risk_override():
    try:
        payload = request.get_json(silent=True) or {}
        action = str(payload.get('action') or 'release').lower()
        if action == 'release':
            return jsonify(_manual_release_risk_state())
        return jsonify({'ok': False, 'message': '未知 action'}), 400
    except Exception as e:
        return jsonify({'ok': False, 'message': str(e)}), 500

@app.route('/api/params', methods=['GET'])
def api_params():
    with AI_LOCK:
        return jsonify({'ok': True,'params': AI_PANEL.get('params', {}),'best_strategies': AI_PANEL.get('best_strategies', []),'auto_backtest': dict(AUTO_BACKTEST_STATE)})

def api_state():
    resp = _BASE_API_STATE()
    try:
        payload = resp.get_json()
        payload['ai_panel'] = dict(AI_PANEL)
        payload['auto_backtest'] = dict(AUTO_BACKTEST_STATE)
        return jsonify(payload)
    except Exception:
        return resp

def start_all_threads():
    load_full_state()
    load_risk_state()
    _refresh_ai_panel_market_meta()
    sync_ai_state_to_dashboard(force_regime=True)
    threads = [(news_thread,'news'),(position_thread,'position'),(enhanced_position_thread,'enhanced_position'),(scan_thread,'scan'),(trailing_stop_thread,'trailing'),(session_monitor_thread,'session'),(market_analysis_thread,'market'),(fvg_order_monitor_thread,'fvg_monitor'),(auto_backtest_thread,'auto_backtest'),(memory_guard_thread,'memory_guard')]
    for fn, name in threads:
        t = threading.Thread(target=watchdog, args=(fn, name), daemon=True, name=name)
        t.start()
    app.view_functions['api_state'] = api_state_enhanced
    update_state(ai_panel=dict(AI_PANEL), auto_backtest=dict(AUTO_BACKTEST_STATE))
    print('=== V11 AI / UI 修正版執行緒已啟動 ===')

if __name__=='__main__':
    app.view_functions['api_state'] = api_state_enhanced
    start_all_threads()
    app.run(host='0.0.0.0',port=int(os.environ.get("PORT",8080)),threaded=True)

# =========================

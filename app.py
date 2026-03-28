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
    'apiKey':   'bg_d94506261d1e5337d124ddade02149c6',
    'secret':   '3aa1590f429f87f2d25e6024938a10cb003cf78e3783007540abaea6f1fa6ed2',
    'password': 'Jeff5466',
    'enableRateLimit': True,
    'options': {'defaultType': 'swap', 'defaultMarginMode': 'cross'}
}
exchange = ccxt.bitget(bitget_config)
exchange.timeout = 10000   # 10秒 API 超時，絕不無限等待
exchange.enableRateLimit = True
PANIC_API_KEY   = "cde101d65a00a418e451ae0118ed0835275781e1"
OPENAI_API_KEY  = "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"
ANTHROPIC_KEY   = "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"
ORDER_THRESHOLD      = 50    # 預設38，同時也是最低
ORDER_THRESHOLD_DEFAULT = 50 # 預設值
ORDER_THRESHOLD_HIGH    = 55 # 滿倉5輪後升到50（維持）
ORDER_THRESHOLD_DROP    = 1  # 保留（不再使用時間，改用輪數）
ORDER_THRESHOLD_FLOOR   = 48 # 最低38（預設值，不再往下降）

# =====================================================
# 核心交易參數
# =====================================================
RISK_PCT        = 0.05      # 每單使用總資產 5%
LEARN_DB_PATH      = "/app/data/learn_db.json"
STATE_BACKUP_PATH  = "/app/data/state_backup.json"
RISK_STATE_PATH    = "/app/data/risk_state.json"

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
    "current":          48,   # 當前門檻（最低也是38）
    "last_order_time":  None, # 最近下單時間
    "full_rounds":      0,    # 連續滿倉輪數
    "empty_rounds":     0,    # 門檻50時連續空倉輪數
    "no_order_rounds":  0,    # 連續無下單輪數（整數，避免None+1錯誤）
}
_DT_LOCK = threading.Lock()

def update_dynamic_threshold(top_sigs=None):
    """
    動態門檻循環（每輪掃描結束呼叫）：
    預設40
    → 滿倉連續5輪 → 升到50
    → 50門檻下空倉連續5輪 → 回到40
    → 有空位且1小時無新單 → 每小時降5（最低35）
    → 下單後回到40
    """
    global ORDER_THRESHOLD
    with _DT_LOCK:
        dt  = _DT
        now = datetime.now()

        with STATE_LOCK:
            pos_count = len(STATE["active_positions"])
            sigs = top_sigs or STATE.get("top_signals", [])

        is_full   = pos_count >= 7
        has_space = pos_count < 7

        # ── 情況A：滿倉 → 計算連續滿倉輪數 ──
        if is_full:
            dt["full_rounds"]  += 1
            dt["empty_rounds"]  = 0
            dt["no_order_rounds"] = 0  # 滿倉重置無單計數

            # 滿倉5輪後升到50
            if dt["full_rounds"] >= 5 and dt["current"] < ORDER_THRESHOLD_HIGH:
                dt["current"] = ORDER_THRESHOLD_HIGH
                ORDER_THRESHOLD = ORDER_THRESHOLD_HIGH
                print("📈 門檻升至50（連續滿倉{}輪）".format(dt["full_rounds"]))
                update_state(threshold_info={
                    "current": ORDER_THRESHOLD_HIGH,
                    "phase": "嚴格（持續滿倉）",
                    "full_rounds": dt["full_rounds"]
                })
            return

        # ── 有空位 ──
        dt["full_rounds"] = 0  # 重置滿倉計數

        # ── 情況B：門檻50，空倉連續5輪 → 回到40 ──
        if dt["current"] >= ORDER_THRESHOLD_HIGH:
            dt["empty_rounds"] += 1
            if dt["empty_rounds"] >= 5:
                dt["current"]      = ORDER_THRESHOLD_DEFAULT
                ORDER_THRESHOLD    = ORDER_THRESHOLD_DEFAULT
                dt["empty_rounds"] = 0
                print("📉 門檻回到40（50門檻下空倉{}輪）".format(dt["empty_rounds"]))
                update_state(threshold_info={
                    "current": ORDER_THRESHOLD_DEFAULT,
                    "phase": "預設",
                    "empty_rounds": dt["empty_rounds"]
                })
            else:
                print("⏱ 門檻50空倉中 {}/5 輪".format(dt["empty_rounds"]))
            return
        else:
            dt["empty_rounds"] = 0

        # ── 情況C：門檻50時有空倉，每3輪降5，降到38為止 ──
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
            dt["no_order_rounds"] = 0  # 已在最低38，重置

def record_order_placed():
    """下單後呼叫，重置計時並回到預設40"""
    global ORDER_THRESHOLD
    with _DT_LOCK:
        _DT["last_order_time"] = datetime.now()
        _DT["no_order_rounds"] = 0  # 重置輪數計數
        _DT["empty_rounds"]    = 0
        # 下單後若門檻曾降低（<40），回到40
        if _DT["current"] < ORDER_THRESHOLD_DEFAULT:
            _DT["current"] = ORDER_THRESHOLD_DEFAULT
            ORDER_THRESHOLD = ORDER_THRESHOLD_DEFAULT
            print("↩️ 門檻回到40（有新下單）")
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
            key = "{}_obs".format(session)
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

            score_key = "{}_score".format(session)
            date_key  = "{}_score_date".format(session)
            time_key  = "{}_score_time".format(session)
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
                    print("🔔 開盤保護：平倉所有持倉")
                    close_all()
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
                    sc,_,_,_,_,_,_,_,_,_ = analyze(symbol)

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

        # 時段保護檢查
        status, note = get_session_status()
        if status in ("eu_closed", "us_closed"):
            return False, "開盤保護期暫停下單：{}".format(note)
        if status in ("eu_pause", "us_pause"):
            return False, "開盤前暫停下單：{}".format(note)

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
            "current_threshold": _DT.get("current", 40),
        }
    except Exception as e:
        return {"trading_ok": True, "halt_reason": "", "consecutive_loss": 0,
                "daily_loss_usdt": 0, "daily_loss_pct": 0, "max_daily_loss_pct": 15}

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
    "threshold_info":    {"current": 38, "phase": "預設"},  # 動態門檻資訊
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

def safe_last(series, default=0):
    try:
        v = series.iloc[-1]
        return float(v) if v == v else default
    except:
        return default

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

        # ===== 用 15m ATR 算 SL / TP =====
        if score > 0:
            sl = round(curr - atr * sl_mult, 6)
            tp = round(curr + atr * tp_mult, 6)
        else:
            sl = round(curr + atr * sl_mult, 6)
            tp = round(curr - atr * tp_mult, 6)

        ep = round((atr * tp_mult) / curr * 100 * 20, 2)
        score = min(max(round(score, 1), -100), 100)

        del d15,d4h,d1d; gc.collect()
        return score,"|".join(tags),curr,sl,tp,ep,breakdown,atr,atr15,atr4h,sl_mult,tp_mult

    except Exception as e:
        import traceback
        print("analyze {} 失敗: {} \n{}".format(symbol, e, traceback.format_exc()[-300:]))
        return 0,"錯誤:{}".format(str(e)[:40]),0,0,0,0,{},0,0,0,2.0,3.0

# =====================================================
# 新聞執行緒
# =====================================================
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
    """用 Claude AI 分析新聞對加密市場的影響，回傳評分和摘要"""
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

    # 先試 Anthropic API
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            timeout=15,
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        result = r.json()
        text = result['content'][0]['text'].strip()
        import json as _json
        data = _json.loads(text)
        return data.get('score', 0), data.get('sentiment', '中性'), data.get('summary', '')
    except Exception as e:
        print("Anthropic 分析失敗: {}".format(e))
    
    # 備用：OpenAI API
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer {}".format(OPENAI_API_KEY),
                "Content-Type": "application/json"
            },
            timeout=15,
            json={
                "model": "gpt-3.5-turbo",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        result = r.json()
        text = result['choices'][0]['message']['content'].strip()
        import json as _json
        data = _json.loads(text)
        return data.get('score', 0), data.get('sentiment', '中性'), data.get('summary', '')
    except Exception as e:
        print("OpenAI 分析失敗: {}".format(e))
    
    # 最後備用：關鍵字分析
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
    目標1（+1.5ATR）→ 平倉30%，止損移至保本
    目標2（+3ATR）  → 再平倉30%，止損移至+1ATR
    目標3（+5ATR）  → 剩餘跟著走，止損鎖最高點90%
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
            if profit_atr >= 1.5 and partial_done == 0:
                ts["partial_done"]  = 1
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), entry)
                ts["trail_pct"]     = 0.05
                print("🎯 目標1達成 {} +{:.1f}ATR → 平30%，止損移保本".format(sym, profit_atr))
                return True, "目標1平倉30% +{:.1f}ATR".format(profit_atr), 0.30

            # 目標2：+3ATR → 再平30%，止損移到+1ATR
            elif profit_atr >= 3.0 and partial_done == 1:
                ts["partial_done"]  = 2
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), entry + atr_val)
                ts["trail_pct"]     = 0.04
                print("🎯 目標2達成 {} +{:.1f}ATR → 再平30%，止損+1ATR".format(sym, profit_atr))
                return True, "目標2平倉30% +{:.1f}ATR".format(profit_atr), 0.30

            # 目標3：+5ATR → 剩餘跟緊，止損鎖90%
            elif profit_atr >= 5.0 and partial_done == 2:
                ts["partial_done"]  = 3
                ts["initial_sl"]    = max(ts.get("initial_sl", 0), current_price * 0.90)
                ts["trail_pct"]     = 0.03
                print("🎯 目標3達成 {} +{:.1f}ATR → 止損鎖90%高點".format(sym, profit_atr))

            # ── 止損移動（只升不降）──
            if profit_atr >= 5.0:
                new_sl = current_price * 0.90
                ts["trail_pct"] = 0.03
            elif profit_atr >= 3.0:
                new_sl = entry + atr_val * 2.0
                ts["trail_pct"] = max(ts.get("trail_pct", 0.05) * 0.85, 0.03)
            elif profit_atr >= 1.5:
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

            if profit_atr >= 1.5 and partial_done == 0:
                ts["partial_done"] = 1
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), entry)
                ts["trail_pct"]    = 0.05
                return True, "目標1平倉30% +{:.1f}ATR".format(profit_atr), 0.30

            elif profit_atr >= 3.0 and partial_done == 1:
                ts["partial_done"] = 2
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), entry - atr_val)
                ts["trail_pct"]    = 0.04
                return True, "目標2平倉30% +{:.1f}ATR".format(profit_atr), 0.30

            elif profit_atr >= 5.0 and partial_done == 2:
                ts["partial_done"] = 3
                ts["initial_sl"]   = min(ts.get("initial_sl", float('inf')), current_price * 1.10)
                ts["trail_pct"]    = 0.03

            if profit_atr >= 5.0:
                new_sl = current_price * 1.10
                ts["trail_pct"] = 0.03
            elif profit_atr >= 3.0:
                new_sl = entry - atr_val * 2.0
            elif profit_atr >= 1.5:
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
    """獨立執行緒，每15秒追蹤所有持倉"""
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
                        # 用進場價的1%估算ATR，不發API請求
                        atr = entry * 0.01
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
                print("偵測到平倉: {}，開始學習分析...".format(sym))
                with LEARN_LOCK:
                    db=LEARN_DB; trade_id=None
                    for t in db["trades"]:
                        if t["symbol"]==sym and t["result"]=="open":
                            try:
                                ticker=exchange.fetch_ticker(sym)
                                t["exit_price"]=ticker['last']
                                print("平倉價格: {} @{}".format(sym, ticker['last']))
                            except: pass
                            trade_id=t["id"]; break
                    if not trade_id:
                        print("警告: {} 無學習紀錄（可能是手動下單）".format(sym))
                if trade_id:
                    print("啟動學習執行緒: {}".format(trade_id))
                    threading.Thread(target=learn_from_closed_trade,args=(trade_id,),daemon=True).start()
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
        trade=next((t for t in LEARN_DB["trades"] if t["id"]==trade_id),None)
    if not trade or trade["result"]!="open":
        return
    time.sleep(5)
    try:
        sym=trade["symbol"]; side=trade["side"]
        exit_p=trade["exit_price"]; entry_p=trade["entry_price"]
        # 用實際損益計算（不假設槓桿）
        raw_pct = (exit_p-entry_p)/entry_p*100 if side=="buy" else (entry_p-exit_p)/entry_p*100
        pnl_pct = raw_pct  # 百分比變動（不含槓桿，學習用）
        result="win" if pnl_pct>0 else "loss"

        time.sleep(60)
        post_ohlcv=exchange.fetch_ohlcv(sym,'15m',limit=12)
        post_closes=[c[4] for c in post_ohlcv[-10:]]
        future_max=max(post_closes); future_min=min(post_closes)
        missed_pct=(future_max-exit_p)/exit_p*100*20 if side=="buy" else (exit_p-future_min)/exit_p*100*20

        bd=trade.get("breakdown",{})
        active_keys=[k for k,v in bd.items() if v!=0]
        pkey="|".join(sorted(active_keys))

        with LEARN_LOCK:
            db=LEARN_DB
            for t in db["trades"]:
                if t["id"]==trade_id:
                    t["result"]=result; t["pnl_pct"]=round(pnl_pct,2)
                    t["post_candles"]=post_closes; t["missed_move_pct"]=round(missed_pct,2)
                    t["exit_time"]=tw_now_str("%Y-%m-%d %H:%M:%S"); break

            # 更新指標組合統計
            if pkey not in db["pattern_stats"]:
                db["pattern_stats"][pkey]={"win":0,"loss":0,"sample_count":0,"total_pnl":0.0,
                    "avg_pnl":0.0,"best_sl":trade.get("atr_mult_sl",2.0),
                    "best_tp":trade.get("atr_mult_tp",3.0),"tp_candidates":[],"sl_candidates":[]}
            ps=db["pattern_stats"][pkey]
            ps["sample_count"]+=1; ps["total_pnl"]+=pnl_pct
            ps["avg_pnl"]=round(ps["total_pnl"]/ps["sample_count"],2)
            if result=="win": ps["win"]+=1; ps["tp_candidates"].append(trade.get("atr_mult_tp",3.0))
            else:             ps["loss"]+=1; ps["sl_candidates"].append(trade.get("atr_mult_sl",2.0))
            if ps["sample_count"]>=5:
                wr=ps["win"]/ps["sample_count"]
                if wr>=0.6 and ps["tp_candidates"]:
                    ps["best_tp"]=round(min(max(ps["tp_candidates"])*1.1,5.0),2)
                    ps["best_sl"]=round(max(ps.get("best_sl",2.0)*0.95,1.8),2)
                elif wr<0.4:
                    ps["best_sl"]=round(min(ps.get("best_sl",2.0)*0.85,1.8),2)
                    ps["best_tp"]=round(max(ps.get("best_tp",3.5)*0.9,2.8),2)

            # 更新幣種統計
            ss=db.setdefault("symbol_stats",{})
            if sym not in ss: ss[sym]={"win":0,"loss":0,"count":0,"total_pnl":0.0}
            ss[sym]["count"]+=1; ss[sym]["total_pnl"]+=pnl_pct
            if result=="win": ss[sym]["win"]+=1
            else:             ss[sym]["loss"]+=1

            # 全域統計
            all_closed=[t for t in db["trades"] if t["result"] in("win","loss")]
            if all_closed:
                db["total_trades"]=len(all_closed)
                wins=sum(1 for t in all_closed if t["result"]=="win")
                db["win_rate"]=round(wins/len(all_closed)*100,1)
                db["avg_pnl"]=round(sum(t.get("pnl_pct",0) for t in all_closed)/len(all_closed),2)
                recent=all_closed[-20:]
                if len(recent)>=10:
                    rwr=sum(1 for t in recent if t["result"]=="win")/len(recent)
                    if rwr>=0.65:
                        db["atr_params"]["default_tp"]=round(min(db["atr_params"]["default_tp"]*1.05,5.0),2)
                    elif rwr<0.35:
                        db["atr_params"]["default_sl"]=round(max(db["atr_params"]["default_sl"]*0.92,1.2),2)
                        db["atr_params"]["default_tp"]=round(max(db["atr_params"]["default_tp"]*0.95,1.5),2)
            # 累積50筆後自動調整指標權重
            all_closed_count = len([t for t in db["trades"] if t["result"] in ("win","loss")])
            if all_closed_count >= 50 and all_closed_count % 10 == 0:
                _auto_adjust_weights(db)

            save_learn_db(db)
        # 更新風控連損狀態
        pnl_usdt = pnl_pct / 100 * STATE.get("equity", 10)
        record_trade_result(pnl_usdt)
        # 更新風控摘要到 STATE
        update_state(risk_status=get_risk_status())
        _refresh_learn_summary()
        print("✅ 學習完成 {} | 損益:{:.1f}% | {} | 錯過:{:.1f}%".format(sym,pnl_pct,result,missed_pct))
    except Exception as e:
        print("學習失敗: {}".format(e))

def _auto_adjust_weights(db):
    """50筆後自動分析哪些指標最有預測力，調整權重"""
    try:
        trades = [t for t in db["trades"] if t["result"] in ("win","loss") and t.get("breakdown")]
        if len(trades) < 50:
            return

        # 統計每個指標在勝利/失敗時的平均分數
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

        # 計算每個指標的「勝率貢獻度」
        contrib = {}
        for key, st in indicator_stats.items():
            if st["win_n"] + st["loss_n"] < 10:
                continue
            win_avg = st["win_sum"] / max(st["win_n"], 1)
            loss_avg = st["loss_sum"] / max(st["loss_n"], 1)
            # 指標在勝利時分數高於失敗時 → 有預測力
            contrib[key] = win_avg - loss_avg

        if contrib:
            total = sum(max(v, 0.1) for v in contrib.values())
            # 儲存貢獻度供參考
            db["indicator_contrib"] = {k: round(v, 3) for k, v in contrib.items()}
            print("📊 指標貢獻度更新完成，共{}個指標分析".format(len(contrib)))
    except Exception as e:
        print("權重調整失敗: {}".format(e))

def _refresh_learn_summary():
    with LEARN_LOCK:
        db=LEARN_DB
        stats=db.get("pattern_stats",{})
        sym_stats=db.get("symbol_stats",{})
        ranked=sorted([(k,v) for k,v in stats.items() if v.get("sample_count",0)>=3],
                      key=lambda x:x[1].get("avg_pnl",0),reverse=True)
        top3=[{"pattern":k[:45],"avg_pnl":v["avg_pnl"],
               "win_rate":round(v["win"]/v["sample_count"]*100,0),"count":v["sample_count"]}
              for k,v in ranked[:3]] if ranked else []
        worst3=[{"pattern":k[:45],"avg_pnl":v["avg_pnl"],
                 "win_rate":round(v["win"]/v["sample_count"]*100,0),"count":v["sample_count"]}
                for k,v in ranked[-3:]] if len(ranked)>=3 else []
        # 勝率<40% 且樣本>=7 的封鎖幣種
        blocked=[{"symbol":s,"win_rate":round(v["win"]/v["count"]*100,1),"count":v["count"]}
                 for s,v in sym_stats.items() if v["count"]>=7 and v["win"]/v["count"]<0.4]
        summary={
            "total_trades":db.get("total_trades",0),
            "win_rate":db.get("win_rate",0.0),
            "avg_pnl":db.get("avg_pnl",0.0),
            "current_sl_mult":db["atr_params"]["default_sl"],
            "current_tp_mult":db["atr_params"]["default_tp"],
            "top_patterns":top3,"worst_patterns":worst3,
            "blocked_symbols":blocked,
        }
    update_state(learn_summary=summary)

# =====================================================
# 下單（使用總資產 5% + 最高槓桿）
# =====================================================
def get_fvg_entry_price(symbol, side, current_price, atr):
    """
    計算 FVG 最優進場價格
    - 找最近未填補的 FVG 缺口
    - 如果價格在缺口附近（2ATR以內）→ 掛限價等回到缺口
    - 如果太遠（>2ATR）→ 回傳 None，直接市價單
    """
    try:
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
        if current_pos_count >= 7:
            print("持倉已達7個上限，取消下單: {}".format(sig['symbol']))
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

        # 開倉金額 = 總資產 × 5%
        with STATE_LOCK: equity=STATE["equity"]
        if equity<=0: equity=10.0
        order_usdt=max(equity*RISK_PCT, 1.0)
        amt=exchange.amount_to_precision(sym, order_usdt*lev/sig['price'])
        sl_price=sig['stop_loss']; tp_price=sig['take_profit']
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
                # 重新計算止損止盈基於FVG價
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
                # 登記到 FVG 追蹤系統
                register_fvg_order(
                    sym, order_id, sig['side'], fvg_price,
                    sig['score'], sl_price, tp_price,
                    sig.get('support', 0), sig.get('resist', 0)
                )
                print("📌 FVG限價掛單: {} {} @{:.6f} | {}".format(sym, side, fvg_price, fvg_note))
                return  # 掛單完成，等成交後再處理止損止盈
            except Exception as fvg_err:
                print("FVG限價下單失敗，改用市價: {}".format(fvg_err))
                order = exchange.create_order(sym, 'market', side, amt, params=order_params)
        else:
            # 直接市價單前，先取消同幣的 FVG 掛單（避免重複）
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
            order = exchange.create_order(sym, 'market', side, amt, params=order_params)

        print("主單成功: {} {} {}口 | {}".format(sym, side, amt, fvg_note))

        # Step2+3: Bitget 止損止盈（移動止盈系統為主，這裡掛初始止損）
        sl_side = 'sell' if side == 'buy' else 'buy'

        # 止損單（三種格式依序嘗試）
        sl_ok = False
        # 格式1：Bitget TPSL
        try:
            exchange.create_order(sym, 'market', sl_side, amt, params={
                'reduceOnly':  True,
                'stopPrice':   str(sl_price),
                'orderType':   'stop',
                'posSide':     pos_side,
                'tdMode':      'cross',
            })
            print("止損單成功(格式1): {} @{}".format(sym, sl_price))
            sl_ok = True
        except Exception as e1:
            pass

        if not sl_ok:
            # 格式2：stopLossPrice
            try:
                exchange.create_order(sym, 'market', sl_side, amt, params={
                    'reduceOnly':    True,
                    'stopLossPrice': str(sl_price),
                    'posSide':       pos_side,
                    'tdMode':        'cross',
                })
                print("止損單成功(格式2): {} @{}".format(sym, sl_price))
                sl_ok = True
            except Exception as e2:
                pass

        if not sl_ok:
            # 格式3：triggerPrice（最後備用）
            try:
                exchange.create_order(sym, 'market', sl_side, amt, params={
                    'reduceOnly':    True,
                    'triggerPrice':  str(sl_price),
                    'triggerType':   'mark_price',
                    'posSide':       pos_side,
                })
                print("止損單成功(格式3): {} @{}".format(sym, sl_price))
                sl_ok = True
            except Exception as e3:
                print("止損三種格式都失敗，依賴移動止損系統: {}".format(e3))

        # 止盈單（三種格式依序嘗試）
        tp_ok = False
        try:
            exchange.create_order(sym, 'market', sl_side, amt, params={
                'reduceOnly':  True,
                'stopPrice':   str(tp_price),
                'orderType':   'takeProfit',
                'posSide':     pos_side,
                'tdMode':      'cross',
            })
            print("止盈單成功(格式1): {} @{}".format(sym, tp_price))
            tp_ok = True
        except:
            pass

        if not tp_ok:
            try:
                exchange.create_order(sym, 'market', sl_side, amt, params={
                    'reduceOnly':       True,
                    'takeProfitPrice':  str(tp_price),
                    'posSide':          pos_side,
                    'tdMode':           'cross',
                })
                print("止盈單成功(格式2): {} @{}".format(sym, tp_price))
                tp_ok = True
            except Exception as tp_err:
                print("止盈掛單失敗，依賴移動止盈系統: {}".format(tp_err))

        trade_id="{}_{}" .format(sym.replace('/','').replace(':',''),int(time.time()))
        rec={"symbol":sym,"side":"做多" if side=='buy' else "做空","score":sig['score'],
             "price":sig['price'],"stop_loss":sl_price,"take_profit":tp_price,
             "est_pnl":sig['est_pnl'],"order_usdt":round(order_usdt,2),"leverage":lev,
             "time":tw_now_str(),"learn_id":trade_id}
        with STATE_LOCK:
            STATE["trade_history"].insert(0,rec)
            if len(STATE["trade_history"])>30: STATE["trade_history"]=STATE["trade_history"][:30]

        learn_rec={"id":trade_id,"symbol":sym,"side":side,"entry_price":sig['price'],
                   "entry_score":sig['score'],"breakdown":sig.get('breakdown',{}),
                   "atr_mult_sl":sig.get('sl_mult',2.0),"atr_mult_tp":sig.get('tp_mult',3.0),
                   "entry_time":tw_now_str("%Y-%m-%d %H:%M:%S"),
                   "exit_price":None,"exit_time":None,"pnl_pct":None,
                   "result":"open","post_candles":[],"missed_move_pct":None}
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
            }
        record_order_placed()  # 通知門檻系統有新下單
        print("下單成功: {} {} @{} {}U x{}倍 SL:{} TP:{} 移動回撤:{:.1f}% 門檻:{}".format(
            sym,side,sig['price'],round(order_usdt,2),lev,sl_price,tp_price,trail_pct*100,ORDER_THRESHOLD))
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
                        sigs.append({
                            "symbol":sym,"score":sc,"desc":desc,"price":pr,
                            "stop_loss":sl,"take_profit":tp,"est_pnl":ep,
                            "direction":"做多 ▲" if sc>0 else "做空 ▼",
                            "breakdown": bd,
"atr": atr,
"atr15": atr15,
"atr4h": atr4h,
"sl_mult": sl_m,
"tp_mult": tp_m,
                            "allowed":allowed,"status":status,
                            "sym_trades":sym_n,"sym_wr":sym_wr,
                        })
                except Exception as sym_e:
                    print("分析 {} 失敗跳過: {}".format(sym, sym_e))
                if i%5==0: gc.collect()

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
            # 下單只取分數前7，排行榜顯示10個但下單上限7個
            top7_for_order = sorted(top10, key=lambda x:abs(x['score']), reverse=True)[:7]
            if pos_cnt < 7:
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

                    if (abs(best['score']) >= eff_threshold  # 動態門檻（含大盤調整）
                            and best['symbol'] not in pos_syms
                            and best['symbol'] not in already_closing
                            and best['symbol'] not in SHORT_TERM_EXCLUDED
                            and best['allowed']
                            and mkt_ok):  # 大盤方向過濾
                        # 用 lambda 的預設參數固定當前值，避免閉包問題
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

if __name__=='__main__':
    start_all_threads()
    app.run(host='0.0.0.0',port=int(os.environ.get("PORT",8080)),threaded=True)

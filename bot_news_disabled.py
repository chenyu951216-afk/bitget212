import time

_DISABLED_TITLE = '新聞系統已停用'
_DISABLED_SENTIMENT = '已停用'
_DISABLED_SUMMARY = '已依設定移除新聞來源與AI新聞分析，不影響下單主流程。'


def disabled_news_state():
    return {
        'score': 0,
        'sentiment': _DISABLED_SENTIMENT,
        'summary': _DISABLED_SUMMARY,
        'latest_title': _DISABLED_TITLE,
        'updated_at': time.time(),
    }


def fetch_crypto_news():
    return []


def analyze_news_with_ai(news_list):
    return 0, _DISABLED_SENTIMENT, _DISABLED_SUMMARY


def news_thread(update_state=None, set_cached_news=None, sleep_sec: int = 300):
    state = disabled_news_state()
    if set_cached_news:
        set_cached_news(state['score'], state['sentiment'], state['summary'], state['latest_title'])
    if update_state:
        update_state(
            news_score=state['score'],
            news_sentiment=state['sentiment'],
            latest_news_title=state['latest_title'],
        )
    while True:
        time.sleep(max(30, int(sleep_sec)))
        if update_state:
            update_state(
                news_score=0,
                news_sentiment=_DISABLED_SENTIMENT,
                latest_news_title=_DISABLED_TITLE,
            )

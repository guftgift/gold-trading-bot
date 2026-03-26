"""
Gold Trading Bot — MA Cross (20/100) + MA200 + News Sentiment
================================================================
สัญญาณซื้อขายทองคำโดยรวม:
  1. MA Cross (20/100) — trend signal
  2. MA200 Filter      — ซื้อเมื่ออยู่เหนือ MA200 เท่านั้น
  3. News Sentiment    — กรองด้วยข่าวล่าสุด

แจ้งเตือนผ่าน Telegram พร้อมสรุปข่าวสำคัญ
"""

import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
import asyncio
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── ⚙️  CONFIG — รับค่าจาก Environment Variables (GitHub Actions Secrets) ──
import os
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")
DRY_RUN            = os.environ.get("DRY_RUN", "false").lower() == "true"
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
#  1. PRICE & INDICATOR ENGINE
# ════════════════════════════════════════════════════════════════════════════

def _stooq_fetch(symbol: str, days: int = 400) -> pd.DataFrame:
    """ดึงข้อมูล OHLC จาก Stooq (ฟรี ไม่ต้อง API key, ทำงานได้บน PythonAnywhere)"""
    import io
    from datetime import date, timedelta
    end   = date.today()
    start = end - timedelta(days=days)
    url   = (f"https://stooq.com/q/d/l/?s={symbol}"
             f"&d1={start.strftime('%Y%m%d')}"
             f"&d2={end.strftime('%Y%m%d')}&i=d")
    r  = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), index_col="Date", parse_dates=True)
    df.columns = [c.capitalize() for c in df.columns]
    df = df.sort_index().dropna()
    if "Close" not in df.columns or len(df) < 10:
        raise ValueError(f"No usable data from Stooq for {symbol}")
    return df


def get_usd_thb():
    """ดึงอัตราแลกเปลี่ยน USD/THB ล่าสุด พร้อม fallback หลายชั้น"""
    # วิธีที่ 1: Stooq USDTHB (ทำงานได้บน PythonAnywhere free)
    try:
        df = _stooq_fetch("usdthb", days=10)
        return round(float(df["Close"].iloc[-1]), 4)
    except Exception:
        pass

    # วิธีที่ 2: yfinance THBUSD=X (กลับค่า)
    try:
        fx = yf.download("THBUSD=X", period="5d", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(fx.columns, pd.MultiIndex):
            fx.columns = fx.columns.get_level_values(0)
        val = fx["Close"].dropna()
        if len(val) > 0:
            return round(1.0 / float(val.iloc[-1]), 4)
    except Exception:
        pass

    # วิธีที่ 3: exchangerate-api.com (ฟรี ไม่ต้อง key)
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        return round(float(r.json()["rates"]["THB"]), 4)
    except Exception:
        pass

    return 33.5   # fallback สุดท้าย

def usd_to_thb_gold(price_usd, usd_thb):
    """
    แปลงราคา XAU/USD → ราคาทองไทย (บาท/บาทน้ำหนัก)
    1 troy oz = 31.1035 g | 1 บาทน้ำหนัก = 15.244 g
    """
    TROY_OZ_TO_GRAM = 31.1035
    THAI_BAHT_GOLD  = 15.244          # กรัม/บาทน้ำหนัก
    price_per_gram_thb  = (price_usd / TROY_OZ_TO_GRAM) * usd_thb
    price_bar           = round(price_per_gram_thb * THAI_BAHT_GOLD * 0.965, 0)   # ทองแท่ง 96.5%
    price_ornament_sell = round(price_per_gram_thb * THAI_BAHT_GOLD + 450, 0)      # ทองรูปพรรณ (ขายออก +ค่ากำเหน็จ ~450)
    price_ornament_buy  = round(price_per_gram_thb * THAI_BAHT_GOLD - 200, 0)      # ทองรูปพรรณ (รับซื้อ)
    return {
        'bar':            price_bar,
        'ornament_sell':  price_ornament_sell,
        'ornament_buy':   price_ornament_buy,
        'per_gram':       round(price_per_gram_thb, 2),
    }

def get_gold_data(period="1y", interval="1d"):
    """
    ดึงราคาทองคำ XAU/USD และคำนวณ Indicators
    ลำดับการลอง (เรียงตามความน่าเชื่อถือบน PythonAnywhere):
      1. Stooq XAUUSD  — ฟรี ไม่ต้อง API key ✅ ทำงานได้บน PythonAnywhere
      2. Stooq GC.F    — Gold Futures บน Stooq
      3. yfinance GC=F — ใช้ได้บนเครื่อง local
      4. yfinance GLD  — ETF ×10
      5. yfinance IAU  — ETF ×50
    """
    df = None

    # ─── วิธีที่ 1 & 2: Stooq (ทำงานได้บน PythonAnywhere free) ───
    for stooq_sym in ["xauusd", "gc.f"]:
        try:
            raw = _stooq_fetch(stooq_sym, days=400)
            # เพิ่มคอลัมน์ Volume ถ้าไม่มี
            if "Volume" not in raw.columns:
                raw["Volume"] = 0
            raw = raw[["Close", "High", "Low", "Open", "Volume"]].dropna()
            if len(raw) >= 200:
                df = raw
                print(f"   ✅ ใช้ Stooq: {stooq_sym} ({len(raw)} rows)")
                break
            else:
                print(f"   ⚠️ Stooq {stooq_sym}: ข้อมูลไม่พอ ({len(raw)} rows)")
        except Exception as e:
            print(f"   ❌ Stooq {stooq_sym}: {e}")

    # ─── วิธีที่ 3-5: yfinance fallback (สำหรับ local / server อื่น) ───
    if df is None:
        yf_sources = [
            ("GC=F", 1.0),
            ("GLD",  10.0),
            ("IAU",  50.0),
        ]
        for sym, mult in yf_sources:
            try:
                raw = yf.download(sym, period="2y", interval="1d",
                                  auto_adjust=True, progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw[["Close", "High", "Low", "Open", "Volume"]].copy().dropna()
                if len(raw) >= 200:
                    for col in ["Close", "High", "Low", "Open"]:
                        raw[col] = raw[col] * mult
                    df = raw
                    print(f"   ✅ ใช้ yfinance: {sym} ×{mult} ({len(raw)} rows)")
                    break
                else:
                    print(f"   ⚠️ yfinance {sym}: ข้อมูลไม่พอ ({len(raw)} rows)")
            except Exception as e:
                print(f"   ❌ yfinance {sym}: {e}")

    if df is None or len(df) == 0:
        raise RuntimeError(
            "❌ ดึงข้อมูลราคาทองไม่ได้จากทุกแหล่ง\n"
            "   กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต"
        )

    c = df['Close']
    df['MA20']  = c.rolling(20).mean()
    df['MA100'] = c.rolling(100).mean()
    df['MA200'] = c.rolling(200).mean()

    # RSI (14) — เสริมข้อมูล
    delta     = c.diff()
    gain      = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss      = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    return df.dropna()


def get_ma_signal(df):
    """
    MA Cross (20/100) + MA200 Filter
    BUY  : MA20 ข้ามขึ้น MA100  AND  ราคา > MA200
    SELL : MA20 ข้ามลง MA100
    HOLD : อื่นๆ
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    above_ma200 = latest['Close'] > latest['MA200']
    golden_cross = (prev['MA20'] <= prev['MA100']) and (latest['MA20'] > latest['MA100'])
    death_cross  = (prev['MA20'] >= prev['MA100']) and (latest['MA20'] < latest['MA100'])

    # ยืนยัน trend ปัจจุบัน
    ma_bullish = latest['MA20'] > latest['MA100']

    if golden_cross and above_ma200:
        return "BUY", "🔔 Golden Cross + เหนือ MA200"
    elif death_cross:
        return "SELL", "🔔 Death Cross"
    elif ma_bullish and above_ma200:
        return "HOLD_BULLISH", "📈 Uptrend (MA20 > MA100 > ราคาเหนือ MA200)"
    else:
        return "HOLD_BEARISH", "📉 Downtrend หรือ ราคาต่ำกว่า MA200"


# ════════════════════════════════════════════════════════════════════════════
#  2. NEWS FETCHER — RSS หลายแหล่ง
# ════════════════════════════════════════════════════════════════════════════

RSS_FEEDS_EN = {
    "Kitco Gold News":      "https://www.kitco.com/rss/kitco-news.xml",
    "Reuters Business":     "https://feeds.reuters.com/reuters/businessNews",
    "MarketWatch":          "https://feeds.marketwatch.com/marketwatch/topstories",
    "Investing.com Gold":   "https://www.investing.com/rss/news_301.rss",
    "Yahoo Finance Gold":   "https://finance.yahoo.com/rss/headline?s=GC%3DF",
}

RSS_FEEDS_TH = {
    "Gold Around TH":   "https://goldaround.com/feed/",
    "ประชาชาติ":         "https://www.prachachat.net/feed",
    "Sanook Money":     "https://money.sanook.com/feed/",
}

GOLD_KEYWORDS_EN = [
    "gold", "XAU", "precious metal", "bullion", "safe haven",
    "Fed", "inflation", "interest rate", "dollar", "USD", "DXY",
    "geopolitical", "war", "crisis", "recession", "treasury",
]

GOLD_KEYWORDS_TH = [
    "ทอง", "ทองคำ", "ราคาทอง", "XAU",
    "เฟด", "เงินเฟ้อ", "ดอกเบี้ย", "ดอลลาร์",
    "ทุนสำรอง", "ความไม่แน่นอน", "สงคราม",
]

GOLD_KEYWORDS = GOLD_KEYWORDS_EN + GOLD_KEYWORDS_TH

def fetch_news(max_per_source=5, hours_back=24):
    """ดึงข่าวทองคำจาก RSS feeds ทั้งไทยและต่างประเทศ"""
    all_news = []
    cutoff   = datetime.utcnow() - timedelta(hours=hours_back)

    all_feeds = {**RSS_FEEDS_EN, **RSS_FEEDS_TH}
    for source_name, url in all_feeds.items():
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_source:
                    break

                title   = getattr(entry, 'title', '')
                summary = getattr(entry, 'summary', '')
                link    = getattr(entry, 'link', '')

                # Filter: เฉพาะข่าวที่เกี่ยวกับทองคำ
                combined = (title + " " + summary).lower()
                if not any(k.lower() in combined for k in GOLD_KEYWORDS):
                    continue

                # Filter: ข่าวใหม่เท่านั้น
                pub_time = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    import time
                    pub_time = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if pub_time < cutoff:
                        continue

                all_news.append({
                    'source':   source_name,
                    'title':    title,
                    'summary':  summary[:300],
                    'link':     link,
                    'time':     pub_time,
                })
                count += 1

        except Exception as e:
            print(f"  ⚠️ Error fetching {source_name}: {e}")

    # เรียงตามเวลา (ใหม่สุดก่อน)
    all_news.sort(key=lambda x: x['time'] or datetime.min, reverse=True)
    return all_news


# ════════════════════════════════════════════════════════════════════════════
#  3. SENTIMENT ANALYZER
# ════════════════════════════════════════════════════════════════════════════

analyzer = SentimentIntensityAnalyzer()

# เพิ่ม Gold-specific lexicon
GOLD_LEXICON = {
    # Bullish keywords
    "surge": 3.0, "rally": 2.5, "soar": 3.0, "jump": 2.0,
    "record": 2.0, "high": 1.5, "gain": 1.5, "rise": 1.5,
    "bullish": 3.0, "safe haven": 2.5, "demand": 1.5,
    "buy": 1.5, "strong": 1.5, "breakout": 2.5,
    "rate cut": 2.0, "dovish": 2.0, "weaker dollar": 2.5,
    "geopolitical": 1.5, "uncertainty": 1.5, "crisis": 1.5,
    # Bearish keywords
    "drop": -2.0, "fall": -1.5, "decline": -2.0, "plunge": -3.0,
    "bearish": -3.0, "sell": -1.5, "weak": -1.5, "pressure": -1.5,
    "rate hike": -2.5, "hawkish": -2.5, "stronger dollar": -2.5,
    "recovery": -1.0,
}
analyzer.lexicon.update(GOLD_LEXICON)


def analyze_sentiment(news_list):
    """วิเคราะห์ sentiment ของข่าวทองคำ"""
    if not news_list:
        return {
            'score': 0.0, 'label': 'NEUTRAL',
            'bullish': 0, 'bearish': 0, 'neutral': 0,
            'top_news': []
        }

    scores  = []
    details = []

    for news in news_list:
        text  = news['title'] + ". " + news['summary']
        score = analyzer.polarity_scores(text)['compound']
        scores.append(score)

        label = "🟢 Bullish" if score > 0.05 else ("🔴 Bearish" if score < -0.05 else "⚪ Neutral")
        details.append({
            **news,
            'sentiment_score': round(score, 3),
            'sentiment_label': label,
        })

    avg_score = np.mean(scores)
    bullish   = sum(1 for s in scores if s > 0.05)
    bearish   = sum(1 for s in scores if s < -0.05)
    neutral   = len(scores) - bullish - bearish

    # Sentiment label รวม
    if avg_score > 0.1:
        overall = "BULLISH"
    elif avg_score < -0.1:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    # Top 5 ข่าวสำคัญ (เรียงตาม absolute score)
    top_news = sorted(details, key=lambda x: abs(x['sentiment_score']), reverse=True)[:5]

    return {
        'score':    round(avg_score, 3),
        'label':    overall,
        'bullish':  bullish,
        'bearish':  bearish,
        'neutral':  neutral,
        'total':    len(scores),
        'top_news': top_news,
    }


# ════════════════════════════════════════════════════════════════════════════
#  4. SIGNAL COMBINER — รวม MA + Sentiment
# ════════════════════════════════════════════════════════════════════════════

def combine_signals(ma_signal, sentiment):
    """
    รวมสัญญาณ MA + Sentiment เป็น Final Decision

    Matrix:
    MA Signal      | Sentiment  → Final
    ─────────────────────────────────────
    BUY            | BULLISH    → ✅ STRONG BUY
    BUY            | NEUTRAL    → ✅ BUY
    BUY            | BEARISH    → ⚠️  WEAK BUY (ระวัง)
    HOLD_BULLISH   | BULLISH    → 📈 HOLD / ACCUMULATE
    HOLD_BULLISH   | NEUTRAL    → 📈 HOLD
    HOLD_BULLISH   | BEARISH    → ⚠️  CAUTION
    SELL           | BEARISH    → 🔴 STRONG SELL
    SELL           | NEUTRAL    → 🔴 SELL
    SELL           | BULLISH    → ⚠️  WEAK SELL
    HOLD_BEARISH   | any        → 🚫 STAY OUT
    """
    ma   = ma_signal
    sent = sentiment['label']

    if ma == "BUY":
        if sent == "BULLISH":
            return "STRONG_BUY",   "✅ STRONG BUY — MA Cross + ข่าวบวก"
        elif sent == "NEUTRAL":
            return "BUY",          "✅ BUY — MA Cross (ข่าวเป็นกลาง)"
        else:
            return "WEAK_BUY",     "⚠️  WEAK BUY — MA Cross แต่ข่าวลบ ระวัง!"

    elif ma == "HOLD_BULLISH":
        if sent == "BULLISH":
            return "ACCUMULATE",   "📈 ACCUMULATE — Uptrend + ข่าวบวก"
        elif sent == "NEUTRAL":
            return "HOLD",         "📈 HOLD — Uptrend ปกติ"
        else:
            return "CAUTION",      "⚠️  CAUTION — Uptrend แต่ข่าวลบ รอดูก่อน"

    elif ma == "SELL":
        if sent == "BEARISH":
            return "STRONG_SELL",  "🔴 STRONG SELL — Death Cross + ข่าวลบ"
        elif sent == "NEUTRAL":
            return "SELL",         "🔴 SELL — Death Cross"
        else:
            return "WEAK_SELL",    "⚠️  WEAK SELL — Death Cross แต่ข่าวบวก"

    else:  # HOLD_BEARISH
        return "STAY_OUT",         "🚫 STAY OUT — ราคาต่ำกว่า MA200"


# ════════════════════════════════════════════════════════════════════════════
#  5. TELEGRAM SENDER
# ════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str):
    """ส่งข้อความไป Telegram"""
    if DRY_RUN:
        print("\n" + "─" * 60)
        print("📱 [DRY RUN] Telegram Message:")
        print("─" * 60)
        print(message)
        print("─" * 60)
        return True

    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       message,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"❌ Telegram Error: {e}")
        return False


def calc_trade_levels(price, signal, atr, ma200):
    """คำนวณจุดซื้อ / เป้าหมาย / Stop Loss อัตโนมัติจากราคาและ ATR"""
    buf = atr * 0.5   # buffer ครึ่ง ATR

    if signal in ("STRONG_BUY", "BUY", "WEAK_BUY", "ACCUMULATE"):
        entry      = round(price, 2)
        entry_low  = round(price - buf, 2)          # zone ต่ำสุดที่ยังซื้อได้
        target1    = round(price + atr * 2, 2)       # +2 ATR
        target2    = round(price + atr * 4, 2)       # +4 ATR
        stop_loss  = round(max(price - atr * 1.5, ma200 * 0.995), 2)
        rr         = round((target1 - entry) / max(entry - stop_loss, 0.01), 1)
        return {
            'action':     "ซื้อ / BUY",
            'entry':      f"${entry_low:,.2f} – ${entry:,.2f}",
            'target1':    f"${target1:,.2f}",
            'target2':    f"${target2:,.2f}",
            'stop_loss':  f"${stop_loss:,.2f}",
            'rr':         f"1 : {rr}",
            'gain1_pct':  round((target1 - price) / price * 100, 2),
            'gain2_pct':  round((target2 - price) / price * 100, 2),
            'risk_pct':   round((price - stop_loss) / price * 100, 2),
        }
    elif signal in ("STRONG_SELL", "SELL", "WEAK_SELL"):
        entry      = round(price, 2)
        entry_high = round(price + buf, 2)
        target1    = round(price - atr * 2, 2)
        target2    = round(price - atr * 4, 2)
        stop_loss  = round(price + atr * 1.5, 2)
        rr         = round((entry - target1) / max(stop_loss - entry, 0.01), 1)
        return {
            'action':     "ขาย / SELL",
            'entry':      f"${entry:,.2f} – ${entry_high:,.2f}",
            'target1':    f"${target1:,.2f}",
            'target2':    f"${target2:,.2f}",
            'stop_loss':  f"${stop_loss:,.2f}",
            'rr':         f"1 : {rr}",
            'gain1_pct':  round((price - target1) / price * 100, 2),
            'gain2_pct':  round((price - target2) / price * 100, 2),
            'risk_pct':   round((stop_loss - price) / price * 100, 2),
        }
    else:
        return None


def build_telegram_message(df, ma_signal, ma_reason,
                            sentiment, final_signal, final_reason):
    """สร้างข้อความ Telegram พร้อมราคาบาทและจุดซื้อขายที่ชัดเจน"""
    latest    = df.iloc[-1]
    now       = datetime.now().strftime("%d/%m/%Y %H:%M")
    price_usd = float(latest['Close'])
    rsi       = float(latest['RSI'])
    atr14     = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
    ma200     = float(latest['MA200'])

    # อัตราแลกเปลี่ยนและราคาบาท
    usd_thb   = get_usd_thb()
    thb_price = usd_to_thb_gold(price_usd, usd_thb)

    # Signal emoji
    emoji_map = {
        "STRONG_BUY":  "🚀", "BUY": "✅", "WEAK_BUY": "⚠️",
        "ACCUMULATE":  "📈", "HOLD": "🟡", "CAUTION": "⚠️",
        "STRONG_SELL": "🔴", "SELL": "🔴", "WEAK_SELL": "🟠",
        "STAY_OUT":    "🚫",
    }
    sig_emoji = emoji_map.get(final_signal, "❓")

    # Sentiment bar
    total    = max(sentiment['total'], 1)
    bull_pct = int(sentiment['bullish'] / total * 10)
    bear_pct = int(sentiment['bearish'] / total * 10)
    sent_bar = "🟢" * bull_pct + "🔴" * bear_pct + "⚪" * (10 - bull_pct - bear_pct)

    # Top news (3 ข่าว)
    news_text = ""
    for i, n in enumerate(sentiment['top_news'][:3], 1):
        t = n['time'].strftime("%H:%M") if n['time'] else "?"
        news_text += f"\n  {i}. {n['sentiment_label']} {n['title'][:65]}..."
        news_text += f"\n     📰 {n['source']} ({t})"

    # จุดซื้อขาย
    levels = calc_trade_levels(price_usd, final_signal, atr14, ma200)

    if levels:
        trade_block = f"""
━━━━━━━━━━━━━━━━━━━━━━
{sig_emoji} <b>ACTION: {levels['action']}</b>

📌 <b>จุดเข้า</b>   : <b>{levels['entry']}</b>
🎯 <b>เป้าหมาย 1</b>: <b>{levels['target1']}</b>  (+{levels['gain1_pct']}%)
🎯 <b>เป้าหมาย 2</b>: <b>{levels['target2']}</b>  (+{levels['gain2_pct']}%)
🛑 <b>Stop Loss</b> : <b>{levels['stop_loss']}</b>  (-{levels['risk_pct']}%)
⚖️ <b>Risk/Reward</b>: {levels['rr']}"""
    else:
        trade_block = f"""
━━━━━━━━━━━━━━━━━━━━━━
{sig_emoji} <b>{final_signal}</b>
   {final_reason}
   💤 <i>ไม่มีจุดเข้าที่แนะนำตอนนี้ — รอสัญญาณต่อไป</i>"""

    msg = f"""
🥇 <b>GOLD TRADING SIGNAL</b>
📅 {now}  |  USD/THB: {usd_thb:.2f}

━━━━━━━━━━━━━━━━━━━━━━
💰 <b>ราคาทองคำ (ฮั่วเซ่งเฮง โดยประมาณ)</b>
   🌍 XAU/USD  : <b>${price_usd:,.2f}</b>
   🇹🇭 ทองแท่ง   : <b>฿{thb_price['bar']:,.0f}</b> บาท/บาทน้ำหนัก
   🇹🇭 รูปพรรณขาย: <b>฿{thb_price['ornament_sell']:,.0f}</b> บาท/บาทน้ำหนัก
   🇹🇭 รูปพรรณรับซื้อ: <b>฿{thb_price['ornament_buy']:,.0f}</b> บาท/บาทน้ำหนัก

   MA20  : ${float(latest['MA20']):,.2f}
   MA100 : ${float(latest['MA100']):,.2f}
   MA200 : ${float(latest['MA200']):,.2f}
   RSI(14): {rsi:.1f}  |  ATR(14): ${atr14:,.1f}

━━━━━━━━━━━━━━━━━━━━━━
📊 <b>Technical</b>: {ma_reason}
📰 <b>Sentiment</b>: {sent_bar}
   Bullish {sentiment['bullish']} / Bearish {sentiment['bearish']} / Neutral {sentiment['neutral']}  →  <b>{sentiment['label']}</b>
{trade_block}

━━━━━━━━━━━━━━━━━━━━━━
🗞 <b>ข่าวสำคัญ ({sentiment['total']} ข่าว / 24h)</b>{news_text}

⚠️ <i>ราคาบาทเป็นการประมาณการจาก XAU/USD×USD/THB
ราคาจริงอาจต่างจากฮั่วเซ่งเฮง ±200-500 บาท
ไม่ใช่คำแนะนำลงทุน — ลงทุนมีความเสี่ยง</i>
""".strip()

    return msg


# ════════════════════════════════════════════════════════════════════════════
#  6. MAIN RUN FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_bot():
    print(f"\n{'═'*60}")
    print(f"  🤖 Gold Trading Bot — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"{'═'*60}")

    # Step 1: ราคาและ indicators
    print("📥 [1/4] ดึงข้อมูลราคาทอง...")
    df = get_gold_data()
    ma_signal, ma_reason = get_ma_signal(df)
    latest    = df.iloc[-1]
    price_usd = float(latest['Close'])
    usd_thb   = get_usd_thb()
    thb       = usd_to_thb_gold(price_usd, usd_thb)
    print(f"   XAU/USD       : ${price_usd:,.2f}")
    print(f"   USD/THB       : {usd_thb:.2f}")
    print(f"   ทองแท่ง (THB)  : ฿{thb['bar']:,.0f}")
    print(f"   รูปพรรณขาย    : ฿{thb['ornament_sell']:,.0f}")
    print(f"   MA Signal: {ma_signal} — {ma_reason}")

    # Step 2: ข่าว
    print("\n📰 [2/4] ดึงข่าวทองคำจาก RSS feeds...")
    news = fetch_news(max_per_source=10, hours_back=24)
    print(f"   พบข่าวที่เกี่ยวข้อง: {len(news)} ข่าว")

    # Step 3: Sentiment
    print("\n🧠 [3/4] วิเคราะห์ Sentiment...")
    sentiment = analyze_sentiment(news)
    print(f"   Overall: {sentiment['label']} (score: {sentiment['score']:+.3f})")
    print(f"   Bullish: {sentiment['bullish']} | Bearish: {sentiment['bearish']} | Neutral: {sentiment['neutral']}")

    # Step 4: รวมสัญญาณ
    print("\n🎯 [4/4] คำนวณ Final Signal...")
    final_signal, final_reason = combine_signals(ma_signal, sentiment)
    print(f"   ➜ {final_signal}: {final_reason}")

    # ส่ง Telegram เสมอ (ทุก run) หรือกรองเฉพาะ signal ที่น่าสนใจ
    NOTIFY_SIGNALS = {
        "STRONG_BUY", "BUY", "WEAK_BUY",
        "STRONG_SELL", "SELL",
        "ACCUMULATE", "CAUTION",
    }

    if final_signal in NOTIFY_SIGNALS:
        msg = build_telegram_message(
            df, ma_signal, ma_reason,
            sentiment, final_signal, final_reason
        )
        print("\n📱 ส่ง Telegram Alert...")
        ok = send_telegram(msg)
        print(f"   {'✅ ส่งสำเร็จ' if ok else '❌ ส่งไม่สำเร็จ'}")
    else:
        print(f"\n💤 Signal: {final_signal} — ไม่ส่ง alert (ตลาดปกติ)")

    # แสดงข่าวสำคัญ
    if sentiment['top_news']:
        print(f"\n🗞 Top News:")
        for n in sentiment['top_news'][:5]:
            print(f"   [{n['sentiment_label']}] {n['title'][:65]}...")

    return {
        'signal':    final_signal,
        'price':     float(latest['Close']),
        'sentiment': sentiment['score'],
        'news_count': len(news),
    }


# ════════════════════════════════════════════════════════════════════════════
#  7. SCHEDULER — รันอัตโนมัติ
# ════════════════════════════════════════════════════════════════════════════

def run_scheduler():
    """รัน bot ตามตาราง — ทุก 4 ชั่วโมงในวันทำการ"""
    import schedule, time

    print("⏰ Scheduler เริ่มทำงาน — รันทุก 4 ชั่วโมง")
    print("   กด Ctrl+C เพื่อหยุด\n")

    schedule.every(4).hours.do(run_bot)

    # รันครั้งแรกทันที
    run_bot()

    while True:
        schedule.run_pending()
        time.sleep(60)


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if "--scheduler" in sys.argv:
        run_scheduler()
    else:
        result = run_bot()
        print(f"\n✅ Done: {result}")

"""
Gold Trading Bot — 5-Factor Weight Scoring System
===================================================
ระบบสัญญาณซื้อขายทองคำแบบ Weight Scoring 5 ปัจจัยหลัก:
  1. Macro (DXY + Real Yield 10Y)    — ทิศทางเงิน/ดอกเบี้ย
  2. Support/Resistance (1D + H4)    — แนวรับ/ต้านสำคัญ
  3. RSI Oversold/Overbought (1D+H4) — จังหวะกลับตัว
  4. MA Cross + Trend (20/100/200)   — ทิศทางเทรนด์
  5. War/Inflation News              — ปัจจัยหนุนทองคำ

แจ้งเตือนผ่าน Telegram พร้อมสรุปข่าวและ Score Breakdown
"""

import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
import asyncio
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

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

    # pandas_ta indicators (ATR, MACD, Stochastic, Bollinger Bands)
    if HAS_PANDAS_TA:
        try:
            df['ATR14'] = df.ta.atr(length=14)
        except Exception:
            pass
        try:
            macd = df.ta.macd()
            df['MACD']        = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
        except Exception:
            pass
        try:
            stoch = df.ta.stoch()
            df['STOCH_K'] = stoch.iloc[:, 0]
            df['STOCH_D'] = stoch.iloc[:, 1]
        except Exception:
            pass
        try:
            bbands = df.ta.bbands(length=20)
            df['BB_UPPER'] = bbands.iloc[:, 2]
            df['BB_LOWER'] = bbands.iloc[:, 0]
        except Exception:
            pass

    return df.dropna(subset=['MA20', 'MA100', 'MA200', 'RSI'])


def get_gold_h4():
    """
    ดึงข้อมูล XAU/USD Timeframe H4
    ใช้สำหรับ RSI H4 และ Support/Resistance ระยะกลาง
    """
    try:
        raw = yf.download("GC=F", period="60d", interval="4h",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Close","High","Low","Open","Volume"]].dropna()
        if len(raw) < 20:
            return None
        c = raw['Close']
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
        raw['RSI_H4'] = 100 - (100 / (1 + gain / loss))
        raw['ATR_H4'] = (raw['High'] - raw['Low']).rolling(14).mean()
        return raw.dropna()
    except Exception as e:
        print(f"  ⚠️ H4 data: {e}")
        return None


def detect_support_level(df_1d, df_h4=None, atr_multiplier=0.5):
    """
    ตรวจสอบว่าราคาอยู่ใกล้แนวรับสำคัญหรือไม่
    วิธี: หา Swing Low ใน 20 แท่งล่าสุด แล้วเช็คว่าราคาปัจจุบัน
          อยู่ใน zone (swing_low ± ATR * 0.5)
    Return: True = อยู่ที่แนวรับ (BUY zone)
    """
    result = {'at_support': False, 'support_price': None, 'label': ''}

    try:
        # Swing Low 1D: ต่ำสุดใน window 5 แท่ง
        window = 5
        lows = df_1d['Low'].values
        closes = df_1d['Close'].values
        atr = float(df_1d['ATR14'].iloc[-1]) if 'ATR14' in df_1d.columns else \
              float((df_1d['High'] - df_1d['Low']).rolling(14).mean().iloc[-1])
        current = float(df_1d['Close'].iloc[-1])

        # หา swing lows ใน 30 แท่งล่าสุด
        swing_lows = []
        data = df_1d.tail(30)
        for i in range(window, len(data) - window):
            low_i = float(data['Low'].iloc[i])
            if all(low_i <= float(data['Low'].iloc[i-j]) for j in range(1, window+1)) and \
               all(low_i <= float(data['Low'].iloc[i+j]) for j in range(1, window+1)):
                swing_lows.append(low_i)

        # nearest swing low
        if swing_lows:
            nearest_support = min(swing_lows, key=lambda x: abs(x - current))
            zone_size = atr * atr_multiplier
            if abs(current - nearest_support) <= zone_size:
                result['at_support'] = True
                result['support_price'] = round(nearest_support, 1)
                result['label'] = f"✅ At Support ${nearest_support:,.0f} (±${zone_size:.0f})"
            else:
                result['label'] = f"Support: ${nearest_support:,.0f} | Current: ${current:,.0f}"
        else:
            result['label'] = "No swing low found"
    except Exception as e:
        result['label'] = f"Support N/A ({e})"

    # Check H4 support as well
    if df_h4 is not None and not result['at_support']:
        try:
            current = float(df_1d['Close'].iloc[-1])
            atr_h4 = float(df_h4['ATR_H4'].iloc[-1])
            h4_lows = df_h4['Low'].tail(60).values
            # Simple: check if price is near recent H4 low
            recent_low = float(min(df_h4['Low'].tail(20)))
            if abs(current - recent_low) <= atr_h4 * 0.5:
                result['at_support'] = True
                result['support_price'] = round(recent_low, 1)
                result['label'] = f"✅ At H4 Support ${recent_low:,.0f}"
        except Exception:
            pass

    return result


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


WAR_KEYWORDS = [
    "war", "attack", "missile", "military", "conflict", "strike",
    "invasion", "sanction", "geopolitical", "crisis", "tension",
    "สงคราม", "ความขัดแย้ง", "โจมตี", "วิกฤต",
]

INFLATION_KEYWORDS = [
    "inflation", "CPI", "price surge", "cost of living", "hyperinflation",
    "rate hike", "Fed", "interest rate", "monetary", "stagflation",
    "เงินเฟ้อ", "ดอกเบี้ย", "เฟด", "ราคาพุ่ง",
]

def detect_war_inflation(news_list):
    """
    ตรวจสอบข่าวสงคราม / เงินเฟ้อ → ปัจจัยหนุนทอง
    Return dict with war_score (+1) and inflation_score (+1)
    """
    war_count       = 0
    inflation_count = 0

    for news in news_list:
        text = (news['title'] + ' ' + news['summary']).lower()
        if any(k.lower() in text for k in WAR_KEYWORDS):
            war_count += 1
        if any(k.lower() in text for k in INFLATION_KEYWORDS):
            inflation_count += 1

    war_score       = 1 if war_count >= 2       else 0
    inflation_score = 1 if inflation_count >= 3 else 0

    return {
        'war_count':       war_count,
        'inflation_count': inflation_count,
        'war_score':       war_score,
        'inflation_score': inflation_score,
        'total':           war_score + inflation_score,
        'label':           (
            f"🔫 War: {war_count} ข่าว (+{war_score}pt) | "
            f"💸 Inflation: {inflation_count} ข่าว (+{inflation_score}pt)"
        ),
    }


# ════════════════════════════════════════════════════════════════════════════
#  4. MACRO FACTORS — Real Yield · DXY · Futures · Fibonacci · TD Sequential
# ════════════════════════════════════════════════════════════════════════════

def get_real_yield():
    """
    Real Yield 10Y จาก yfinance (^TNX = US 10Y Treasury Nominal Yield)
    Real Yield ≈ Nominal Yield (^TNX) − CPI เฉลี่ย (~2.5%)

    Real Yield ↑ → เงินไหลเข้าพันธบัตร → ทองลง  (BEARISH)
    Real Yield ↓ → เงินหนีออกพันธบัตร  → ทองขึ้น (BULLISH)
    """
    ASSUMED_CPI = 2.5   # ค่าเฉลี่ย CPI สหรัฐ (%)

    try:
        raw = yf.download("^TNX", period="1mo", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw['Close'].dropna()
        if len(closes) < 5:
            raise ValueError("ข้อมูลไม่พอ")

        nominal = float(closes.iloc[-1])
        current = round(nominal - ASSUMED_CPI, 2)   # Real Yield
        prev    = float(closes.iloc[max(-6, -len(closes))]) - ASSUMED_CPI
        trend   = current - prev

        signal    = 'BULLISH' if current < 0.5 else ('BEARISH' if current > 1.5 else 'NEUTRAL')
        direction = '↑' if trend > 0.05 else ('↓' if trend < -0.05 else '→')
        return {
            'nominal':   round(nominal, 2),
            'value':     current,
            'trend':     round(trend,   2),
            'direction': direction,
            'signal':    signal,
            'label':     f"{current:+.2f}% {direction}  (^TNX {nominal:.2f}%)",
        }
    except Exception as e:
        print(f"  ⚠️ Real Yield (yfinance ^TNX): {e}")
        return {'nominal': None, 'value': None, 'signal': 'NEUTRAL',
                'label': 'N/A', 'direction': '→', 'trend': 0}


def get_dxy_data():
    """
    Dollar Index (DXY) จาก yfinance / Alpha Vantage

    ลำดับ fallback:
      1. DX-Y.NYB  — ICE US Dollar Index Spot (Yahoo Finance) ✅
      2. UUP       — Invesco DB USD Bull ETF (proxy, ×3.6 ≈ DXY)
      3. Alpha Vantage FX_DAILY (ต้องตั้ง env ALPHA_VANTAGE_KEY)

    DXY ↑ / เหนือ MA20 → ทองอ่อน (BEARISH)
    DXY ↓ / ต่ำกว่า MA20 → ทองแข็ง (BULLISH)
    """
    def _calc(closes):
        now   = float(closes.iloc[-1])
        prev  = float(closes.iloc[max(-6, -len(closes))])
        ma20  = float(closes.rolling(20).mean().iloc[-1]) if len(closes) >= 20 else now
        w_chg = (now - prev) / prev * 100
        signal    = ('BEARISH' if now > ma20 and w_chg >  0.2 else
                     'BULLISH' if now < ma20 and w_chg < -0.2 else 'NEUTRAL')
        direction = '↑' if w_chg > 0.2 else ('↓' if w_chg < -0.2 else '→')
        return {
            'value':         round(now,   2),
            'weekly_change': round(w_chg, 2),
            'above_ma20':    now > ma20,
            'signal':        signal,
            'direction':     direction,
            'label':         f"{now:.2f} ({w_chg:+.2f}%w) {direction}",
        }

    # ── วิธีที่ 1: DX-Y.NYB (ICE Dollar Index Spot บน Yahoo Finance) ──
    try:
        raw = yf.download("DX-Y.NYB", period="2mo", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw['Close'].dropna()
        if len(closes) >= 5:
            print("   ✅ DXY จาก yfinance: DX-Y.NYB")
            return _calc(closes)
    except Exception as e:
        print(f"  ⚠️ DXY DX-Y.NYB: {e}")

    # ── วิธีที่ 2: UUP ETF (Invesco DB USD Bullish) × scale ≈ DXY ──
    try:
        raw = yf.download("UUP", period="2mo", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        closes = raw['Close'].dropna()
        if len(closes) >= 5:
            # scale UUP → DXY approximate (UUP ~27-28 = DXY ~100)
            scale  = 3.59
            closes = closes * scale
            print("   ✅ DXY จาก yfinance: UUP (proxy ×3.59)")
            return _calc(closes)
    except Exception as e:
        print(f"  ⚠️ DXY UUP: {e}")

    # ── วิธีที่ 3: Alpha Vantage (ต้องตั้ง env ALPHA_VANTAGE_KEY) ──
    av_key = os.environ.get("ALPHA_VANTAGE_KEY", "")
    if av_key:
        try:
            url = (f"https://www.alphavantage.co/query"
                   f"?function=FX_DAILY&from_symbol=USD&to_symbol=EUR"
                   f"&outputsize=compact&apikey={av_key}")
            r   = requests.get(url, timeout=10)
            data = r.json().get("Time Series FX (Daily)", {})
            if data:
                dates  = sorted(data.keys(), reverse=True)[:30]
                # USD/EUR inverse ≈ rough DXY proxy (EUR weight 57.6%)
                closes = pd.Series(
                    {d: 1 / float(data[d]['4. close']) * 57.6 for d in dates}
                ).sort_index()
                print("   ✅ DXY จาก Alpha Vantage (USD/EUR proxy)")
                return _calc(closes)
        except Exception as e:
            print(f"  ⚠️ DXY Alpha Vantage: {e}")

    print("  ❌ DXY: ดึงข้อมูลไม่ได้จากทุกแหล่ง")
    return {'value': None, 'signal': 'NEUTRAL', 'direction': '→',
            'weekly_change': 0, 'above_ma20': None, 'label': 'N/A'}


def get_futures_analysis(df_spot):
    """
    วิเคราะห์ตลาด Futures / COMEX
    1. Basis spread (Futures - Spot): Backwardation = Bullish (demand เร่งด่วน)
    2. Stop Hunt detection: ไส้เทียนยาว > 2× ตัวเทียน = Liquidity sweep
    3. GLD ETF flow proxy: institutional money direction
    """
    result = {
        'basis': None, 'basis_signal': 'NEUTRAL',
        'stop_hunt': False, 'gld_signal': 'NEUTRAL',
        'overall': 'NEUTRAL', 'label': '',
    }

    # 1. Basis (Futures − Spot)
    try:
        gc      = _stooq_fetch("gc.f", days=5)
        spot    = float(df_spot['Close'].iloc[-1])
        futures = float(gc['Close'].iloc[-1])
        basis   = futures - spot
        result['basis'] = round(basis, 1)
        if basis < -5:
            result['basis_signal'] = 'BULLISH'   # Backwardation
        elif basis > 20:
            result['basis_signal'] = 'BEARISH'   # Extreme contango
    except Exception:
        pass

    # 2. Stop Hunt — ไส้เทียน > 2× body ใน 5 แท่งล่าสุด
    try:
        for _, row in df_spot.tail(5).iterrows():
            body       = abs(float(row['Close']) - float(row['Open']))
            upper_wick = float(row['High']) - max(float(row['Close']), float(row['Open']))
            lower_wick = min(float(row['Close']), float(row['Open'])) - float(row['Low'])
            if upper_wick > 2 * max(body, 1) or lower_wick > 2 * max(body, 1):
                result['stop_hunt'] = True
                break
    except Exception:
        pass

    # 3. GLD ETF 5-day flow (institutional proxy)
    try:
        gld = yf.download("GLD", period="1mo", interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(gld.columns, pd.MultiIndex):
            gld.columns = gld.columns.get_level_values(0)
        c = gld['Close'].dropna()
        if len(c) >= 5:
            chg = (float(c.iloc[-1]) - float(c.iloc[-5])) / float(c.iloc[-5]) * 100
            result['gld_signal'] = ('BULLISH' if chg > 1.0 else
                                    'BEARISH' if chg < -1.0 else 'NEUTRAL')
    except Exception:
        pass

    sigs = [result['basis_signal'], result['gld_signal']]
    b, br = sigs.count('BULLISH'), sigs.count('BEARISH')
    result['overall'] = 'BULLISH' if b > br else ('BEARISH' if br > b else 'NEUTRAL')

    basis_str       = f"{result['basis']:+.0f}" if result['basis'] is not None else "N/A"
    stop_warn       = " ⚠️ Stop Hunt" if result['stop_hunt'] else ""
    result['label'] = f"Basis: {basis_str} | Flow: {result['gld_signal']}{stop_warn}"
    return result


def calc_fibonacci(df, lookback: int = 60):
    """
    Fibonacci Retracement จาก swing high/low ล่าสุด
    WHERE: บอกว่าราคาอยู่ใน zone ไหน — เป้าหมายราคา
    """
    recent  = df.tail(lookback)
    high    = float(recent['High'].max())
    low     = float(recent['Low'].min())
    current = float(df['Close'].iloc[-1])
    diff    = high - low if high != low else 1.0

    levels = {
        'high':  round(high,  1),
        'low':   round(low,   1),
        '23.6%': round(high - 0.236 * diff, 1),
        '38.2%': round(high - 0.382 * diff, 1),
        '50.0%': round(high - 0.500 * diff, 1),
        '61.8%': round(high - 0.618 * diff, 1),
        '78.6%': round(high - 0.786 * diff, 1),
    }

    if   current >= levels['23.6%']:
        zone, signal = "Zone บน (0–23.6%)",             'NEUTRAL'
    elif current >= levels['38.2%']:
        zone, signal = "Shallow Pullback (23.6–38.2%)", 'BULLISH'
    elif current >= levels['50.0%']:
        zone, signal = "Mid Retracement (38.2–50%)",    'BULLISH'
    elif current >= levels['61.8%']:
        zone, signal = "🌟 Golden Zone (50–61.8%)",     'STRONG_BULLISH'
    elif current >= levels['78.6%']:
        zone, signal = "Deep Pullback (61.8–78.6%)",    'BULLISH'
    else:
        zone, signal = "ต่ำกว่า 78.6% — ระวัง Trend เปลี่ยน", 'BEARISH'

    nearest = min(levels.items(), key=lambda x: abs(x[1] - current))
    return {
        'levels':  levels,
        'zone':    zone,
        'signal':  signal,
        'nearest': nearest,
        'label':   f"Fib {nearest[0]} ≈ ${nearest[1]:,.0f} | {zone}",
    }


def calc_td_sequential(df):
    """
    TD Sequential (Tom DeMark) — WHEN: บอกจังหวะหมดแรง
    นับแท่งที่ Close > / < Close[i-4] ต่อเนื่อง
    9 แท่ง = Setup สมบูรณ์ → สัญญาณกลับตัว
    """
    closes    = df['Close'].values
    td_count  = 0
    direction = None

    for i in range(4, len(closes)):
        if closes[i] > closes[i - 4]:
            if direction == 'UP':
                td_count += 1
            else:
                td_count, direction = 1, 'UP'
        elif closes[i] < closes[i - 4]:
            if direction == 'DOWN':
                td_count += 1
            else:
                td_count, direction = 1, 'DOWN'
        else:
            td_count, direction = 0, None

    signal = 'NEUTRAL'
    if td_count >= 9:
        signal = 'SELL_SETUP' if direction == 'UP' else 'BUY_SETUP'
        label  = (f"⚠️ TD Sell Setup {td_count} — หมดแรงขาขึ้น"
                  if direction == 'UP'
                  else f"⚠️ TD Buy Setup {td_count} — หมดแรงขาลง")
    elif td_count >= 6:
        label = f"TD {direction}: {td_count}/9 — {'ระวังยอด' if direction == 'UP' else 'ระวังก้น'}"
    else:
        label = f"TD {direction or '—'}: {td_count}"

    return {
        'count':      td_count,
        'direction':  direction,
        'signal':     signal,
        'exhaustion': td_count >= 9,
        'label':      label,
    }


# ════════════════════════════════════════════════════════════════════════════
#  5. SIGNAL COMBINER — Score-based รวม 5 ปัจจัยหลัก
# ════════════════════════════════════════════════════════════════════════════

def combine_signals(ma_signal, sentiment,
                    real_yield=None, dxy=None,
                    futures=None, fib=None, td_seq=None,
                    rsi_1d=50.0, rsi_h4=50.0,
                    support=None, war_inflation=None):
    """
    ════════════════════════════════════════════════════
    Weight Scoring System — ตัดสินใจเข้าซื้อสะสมทอง
    ════════════════════════════════════════════════════

    ┌─────────────────────────────────────────┬──────┐
    │ ปัจจัย                                  │ คะแนน│
    ├─────────────────────────────────────────┼──────┤
    │ DXY ↓ AND Yield 10Y ↓ (พร้อมกัน)       │  +2  │
    │ ราคาแตะแนวรับสำคัญ (1D หรือ H4)        │  +3  │
    │ RSI < 30 Oversold (1D)                  │  +3  │
    │ RSI < 35 Oversold (H4)                  │  +2  │
    │ MA Golden Cross (20/100)                │  +2  │
    │ Trend Uptrend (MA20 > MA100 > MA200)    │  +1  │
    │ TD Sequential Buy Setup ≥ 9             │  +1  │
    │ Fibonacci Golden Zone (50–61.8%)        │  +1  │
    │ COMEX Backwardation / GLD inflow        │  +1  │
    │ ข่าวสงคราม / geopolitical               │  +1  │
    │ ข่าวเงินเฟ้อพุ่ง                        │  +1  │
    │ Sentiment BULLISH รวม                   │  +1  │
    ├─────────────────────────────────────────┼──────┤
    │ DXY ↑ AND Yield ↑ (พร้อมกัน)           │  -2  │
    │ ราคาแตะแนวต้านสำคัญ / RSI > 70 (1D)   │  -3  │
    │ RSI > 65 (H4)                           │  -2  │
    │ MA Death Cross                          │  -2  │
    │ TD Sequential Sell Setup ≥ 9           │  -2  │
    │ Sentiment BEARISH รวม                   │  -1  │
    └─────────────────────────────────────────┴──────┘

    Action:
      score > 5  → BUY / STRONG BUY
      score 3–5  → WEAK BUY / ACCUMULATE
      score < -5 → SELL / STRONG SELL
      อื่นๆ     → HOLD / STAY OUT
    """
    score     = 0
    breakdown = []

    # ── 1. MACRO: DXY & Yield 10Y (negative correlation กับทอง) ──
    dxy_falling   = dxy and dxy['signal'] == 'BULLISH'       # DXY ↓ = bullish gold
    yield_falling = real_yield and real_yield['signal'] == 'BULLISH'  # Yield ↓ = bullish gold
    dxy_rising    = dxy and dxy['signal'] == 'BEARISH'
    yield_rising  = real_yield and real_yield['signal'] == 'BEARISH'

    if dxy_falling and yield_falling:
        score += 2
        breakdown.append("DXY↓+Yield↓ +2")
    elif dxy_falling or yield_falling:
        score += 1
        breakdown.append("DXY/Yield↓ +1")
    elif dxy_rising and yield_rising:
        score -= 2
        breakdown.append("DXY↑+Yield↑ -2")
    elif dxy_rising or yield_rising:
        score -= 1
        breakdown.append("DXY/Yield↑ -1")

    # ── 2. TECHNICAL: Support Level ──
    if support and support['at_support']:
        score += 3
        breakdown.append("At Support +3")
    elif support and support['support_price']:
        # ใกล้ support แต่ยังไม่ถึง
        pass

    # ── 3. TECHNICAL: RSI Oversold / Overbought ──
    if rsi_1d < 30:
        score += 3
        breakdown.append(f"RSI1D {rsi_1d:.0f}<30 +3")
    elif rsi_1d < 40:
        score += 1
        breakdown.append(f"RSI1D {rsi_1d:.0f}<40 +1")
    elif rsi_1d > 70:
        score -= 3
        breakdown.append(f"RSI1D {rsi_1d:.0f}>70 -3")
    elif rsi_1d > 60:
        score -= 1
        breakdown.append(f"RSI1D {rsi_1d:.0f}>60 -1")

    if rsi_h4 < 35:
        score += 2
        breakdown.append(f"RSI_H4 {rsi_h4:.0f}<35 +2")
    elif rsi_h4 > 65:
        score -= 2
        breakdown.append(f"RSI_H4 {rsi_h4:.0f}>65 -2")

    # ── 4. TECHNICAL: MA Cross / Trend ──
    if ma_signal == 'BUY':
        score += 2
        breakdown.append("Golden Cross +2")
    elif ma_signal == 'HOLD_BULLISH':
        score += 1
        breakdown.append("Uptrend +1")
    elif ma_signal == 'SELL':
        score -= 2
        breakdown.append("Death Cross -2")
    elif ma_signal == 'HOLD_BEARISH':
        score -= 1
        breakdown.append("Downtrend -1")

    # ── 5. TECHNICAL: TD Sequential ──
    if td_seq:
        if td_seq['signal'] == 'BUY_SETUP':
            score += 1
            breakdown.append("TD BuySetup +1")
        elif td_seq['signal'] == 'SELL_SETUP':
            score -= 2
            breakdown.append("TD SellSetup -2")

    # ── 6. TECHNICAL: Fibonacci ──
    if fib and fib['signal'] == 'STRONG_BULLISH':
        score += 1
        breakdown.append("Fib GoldenZone +1")

    # ── 7. FUTURES: COMEX / Institutional ──
    if futures and futures['overall'] == 'BULLISH':
        score += 1
        breakdown.append("Futures Bull +1")
    elif futures and futures['overall'] == 'BEARISH':
        score -= 1
        breakdown.append("Futures Bear -1")

    # ── 8. SENTIMENT: War / Inflation / Overall ──
    if war_inflation:
        if war_inflation['war_score'] > 0:
            score += 1
            breakdown.append(f"War news +1")
        if war_inflation['inflation_score'] > 0:
            score += 1
            breakdown.append(f"Inflation news +1")

    if sentiment['label'] == 'BULLISH':
        score += 1
        breakdown.append("Sent Bull +1")
    elif sentiment['label'] == 'BEARISH':
        score -= 1
        breakdown.append("Sent Bear -1")

    # ── Final Decision ──
    score_str = f"Score: {score:+d}  [{', '.join(breakdown)}]"

    if score >= 9:
        return "STRONG_BUY",  f"🚀 STRONG BUY  — {score_str}"
    elif score >= 6:
        return "BUY",          f"✅ BUY          — {score_str}"
    elif score >= 3:
        return "WEAK_BUY",     f"⚠️  WEAK BUY    — {score_str}"
    elif score <= -9:
        return "STRONG_SELL",  f"🔴 STRONG SELL  — {score_str}"
    elif score <= -6:
        return "SELL",          f"🔴 SELL          — {score_str}"
    elif score <= -3:
        return "WEAK_SELL",    f"🟠 WEAK SELL    — {score_str}"
    elif ma_signal in ('HOLD_BULLISH', 'BUY') and score >= 0:
        return "HOLD",          f"📈 HOLD          — {score_str}"
    elif score < 0:
        return "CAUTION",       f"⚠️  CAUTION      — {score_str}"
    else:
        return "STAY_OUT",      f"🚫 STAY OUT     — {score_str}"


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


def calc_confidence(ma_signal, sentiment, rsi,
                    real_yield=None, dxy=None,
                    futures=None, fib=None, td_seq=None,
                    rsi_h4=50.0, support=None, war_inflation=None):
    """
    Confidence Score 0-99 อิงจาก weight scoring
    สูงสุดที่เป็นไปได้ ~90 pts (หาร ~0.9 = 0-100%)
    """
    pts = 0

    # Macro (max 15)
    dxy_ok   = dxy and dxy['signal'] == 'BULLISH'
    yield_ok = real_yield and real_yield['signal'] == 'BULLISH'
    if dxy_ok and yield_ok: pts += 15
    elif dxy_ok or yield_ok: pts += 8

    # RSI oversold bonus (max 20)
    if rsi < 30:      pts += 20
    elif rsi < 40:    pts += 12
    elif rsi < 50:    pts += 6
    if rsi_h4 < 35:   pts += 10
    elif rsi_h4 < 45: pts += 5

    # Support zone (max 20)
    if support and support['at_support']: pts += 20

    # MA / Trend (max 15)
    ma_pts = {'BUY': 15, 'HOLD_BULLISH': 10, 'HOLD': 6,
              'WEAK_BUY': 8, 'CAUTION': 4, 'SELL': 0,
              'STRONG_SELL': 0, 'STAY_OUT': 0}.get(ma_signal, 6)
    pts += ma_pts

    # Sentiment + War/Inflation (max 15)
    pts += min(15, max(0, int((sentiment['score'] + 1) / 2 * 10)))
    if war_inflation:
        pts += war_inflation['total'] * 3

    # Futures + Fib (max 10)
    if futures and futures['overall'] == 'BULLISH': pts += 5
    if fib and fib['signal'] == 'STRONG_BULLISH':   pts += 5

    # TD penalty
    if td_seq and td_seq['exhaustion']: pts = max(0, pts - 10)

    return min(99, pts)


def calc_trade_levels_thb(price_usd, signal, atr_usd, ma200_usd, usd_thb):
    """คำนวณจุดซื้อ/ขาย/SL/TP เป็นเงินบาท"""
    def to_thb(usd_val):
        """แปลง XAU/USD price → ราคาทองไทย บาท/บาทน้ำหนัก"""
        TROY_OZ_TO_GRAM = 31.1035
        THAI_BAHT_GOLD  = 15.244
        return round((usd_val / TROY_OZ_TO_GRAM) * usd_thb * THAI_BAHT_GOLD * 0.965, -2)

    buf  = atr_usd * 0.5
    atr2 = atr_usd * 2
    atr4 = atr_usd * 4
    atr15 = atr_usd * 1.5

    if signal in ("STRONG_BUY", "BUY", "WEAK_BUY", "ACCUMULATE"):
        entry     = price_usd
        entry_low = price_usd - buf
        t1        = price_usd + atr2
        t2        = price_usd + atr4
        sl        = max(price_usd - atr15, ma200_usd * 0.995)
        rr        = round((t1 - entry) / max(entry - sl, 0.01), 1)
        return {
            'action':   "🟢 ซื้อ",
            'entry':    f"฿{to_thb(entry_low):,.0f} – ฿{to_thb(entry):,.0f}",
            'target1':  f"฿{to_thb(t1):,.0f}",
            'target2':  f"฿{to_thb(t2):,.0f}",
            'stop':     f"฿{to_thb(sl):,.0f}",
            'gain1':    round((t1 - entry) / entry * 100, 1),
            'gain2':    round((t2 - entry) / entry * 100, 1),
            'risk':     round((entry - sl)  / entry * 100, 1),
            'rr':       f"1:{rr}",
        }
    elif signal in ("STRONG_SELL", "SELL", "WEAK_SELL"):
        entry     = price_usd
        entry_hi  = price_usd + buf
        t1        = price_usd - atr2
        t2        = price_usd - atr4
        sl        = price_usd + atr15
        rr        = round((entry - t1) / max(sl - entry, 0.01), 1)
        return {
            'action':   "🔴 ขาย",
            'entry':    f"฿{to_thb(entry):,.0f} – ฿{to_thb(entry_hi):,.0f}",
            'target1':  f"฿{to_thb(t1):,.0f}",
            'target2':  f"฿{to_thb(t2):,.0f}",
            'stop':     f"฿{to_thb(sl):,.0f}",
            'gain1':    round((entry - t1) / entry * 100, 1),
            'gain2':    round((entry - t2) / entry * 100, 1),
            'risk':     round((sl - entry) / entry * 100, 1),
            'rr':       f"1:{rr}",
        }
    return None


def build_telegram_message(df, ma_signal, ma_reason, sentiment, final_signal, final_reason,
                            real_yield=None, dxy=None, futures=None, fib=None, td_seq=None,
                            rsi_h4=50.0, support=None, war_inflation=None, score_breakdown=""):
    """สร้างข้อความ Telegram พร้อม 5 ปัจจัยหลัก"""
    latest    = df.iloc[-1]
    now       = datetime.now().strftime("%d/%m/%Y %H:%M")
    price_usd = float(latest['Close'])
    rsi       = float(latest['RSI'])
    atr14     = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
    ma200     = float(latest['MA200'])

    usd_thb   = get_usd_thb()
    thb_price = usd_to_thb_gold(price_usd, usd_thb)

    confidence = calc_confidence(final_signal, sentiment, rsi,
                                 real_yield, dxy, futures, fib, td_seq,
                                 rsi_h4=rsi_h4, support=support,
                                 war_inflation=war_inflation)
    conf_bar   = "█" * (confidence // 10) + "░" * (10 - confidence // 10)
    conf_label = ("สูงมาก" if confidence >= 80 else
                  "ดี"     if confidence >= 60 else
                  "ปานกลาง" if confidence >= 40 else "ต่ำ")

    emoji_map = {
        "STRONG_BUY":  "🚀", "BUY": "✅", "WEAK_BUY": "⚠️",
        "ACCUMULATE":  "📈", "HOLD": "🟡", "CAUTION": "⚠️",
        "STRONG_SELL": "🔴", "SELL": "🔴", "WEAK_SELL": "🟠",
        "STAY_OUT":    "🚫",
    }
    sig_emoji = emoji_map.get(final_signal, "❓")

    total     = max(sentiment['total'], 1)
    bull_pct  = round(sentiment['bullish'] / total * 100)
    bear_pct  = round(sentiment['bearish'] / total * 100)
    sent_icon = ("🟢" if sentiment['label'] == "BULLISH" else
                 "🔴" if sentiment['label'] == "BEARISH" else "⚪")
    news_lines = ""
    for n in sentiment['top_news'][:3]:
        icon = "▲" if "Bullish" in n['sentiment_label'] else "▼"
        news_lines += f"\n  {icon} {n['title'][:55]}…"

    levels = calc_trade_levels_thb(price_usd, final_signal, atr14, ma200, usd_thb)
    if levels:
        trade_block = (
            f"\n<b>{levels['action']}</b>\n"
            f"📌 เข้า  : <b>{levels['entry']}</b>\n"
            f"🎯 TP1  : <b>{levels['target1']}</b>  (+{levels['gain1']}%)\n"
            f"🎯 TP2  : <b>{levels['target2']}</b>  (+{levels['gain2']}%)\n"
            f"🛑 SL   : <b>{levels['stop']}</b>  (-{levels['risk']}%)\n"
            f"⚖️ R/R  : {levels['rr']}"
        )
    else:
        trade_block = f"\n{sig_emoji} <b>{final_signal}</b> — รอสัญญาณต่อไป"

    # ── Macro block ──
    ry_icon  = ("🟢" if real_yield and real_yield['signal'] == 'BULLISH' else
                "🔴" if real_yield and real_yield['signal'] == 'BEARISH' else "⚪")
    dxy_icon = ("🟢" if dxy and dxy['signal'] == 'BULLISH' else
                "🔴" if dxy and dxy['signal'] == 'BEARISH' else "⚪")
    fut_icon = ("🟢" if futures and futures['overall'] == 'BULLISH' else
                "🔴" if futures and futures['overall'] == 'BEARISH' else "⚪")

    ry_lbl  = real_yield['label']  if real_yield  else 'N/A'
    dxy_lbl = dxy['label']         if dxy         else 'N/A'
    fut_lbl = futures['label']     if futures      else 'N/A'
    fib_lbl = fib['label']         if fib          else 'N/A'
    td_lbl  = td_seq['label']      if td_seq       else 'N/A'

    msg = (
        f"🥇 <b>GOLD SIGNAL</b>  {now}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🇹🇭 ทองแท่ง  : <b>฿{thb_price['bar']:,.0f}</b>\n"
        f"🇹🇭 รูปพรรณ  : <b>฿{thb_price['ornament_sell']:,.0f}</b> (ขาย)\n"
        f"🌍 XAU/USD : ${price_usd:,.0f}  |  ฿/{usd_thb:.1f}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 <b>5 ปัจจัยหลัก</b>\n"
        f"  {ry_icon} Real Yield : {ry_lbl}\n"
        f"  {dxy_icon} DXY        : {dxy_lbl}\n"
        f"  {fut_icon} Futures    : {fut_lbl}\n"
        f"  📐 Fibonacci : {fib_lbl}\n"
        f"  🕰 TD Seq    : {td_lbl}\n"
        f"  📊 RSI H4    : {rsi_h4:.0f} {'🟢 Oversold' if rsi_h4 < 35 else '🔴 Overbought' if rsi_h4 > 65 else '⚪ Normal'}\n"
        f"  🛡 Support   : {support['label'] if support else 'N/A'}\n"
        f"  🔫 War/Infl  : {war_inflation['label'] if war_inflation else 'N/A'}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Technical</b> : {ma_reason}\n"
        f"RSI : {rsi:.0f}  |  ATR : ฿{(atr14/31.1035*usd_thb*15.244*0.965):,.0f}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>Confidence</b> : {conf_bar} <b>{confidence}%</b> ({conf_label})\n"
        f"📝 Score: <code>{score_breakdown[:80]}</code>\n"
        f"{sig_emoji} <b>{final_signal}</b>  {final_reason}\n"
        f"{trade_block}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📰 <b>News</b> {sent_icon} <b>{sentiment['label']}</b>"
        f"  (🟢{bull_pct}% / 🔴{bear_pct}%  |  {sentiment['total']} ข่าว)"
        f"{news_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ ประมาณการเท่านั้น ±500฿  ลงทุนมีความเสี่ยง</i>"
    )
    return msg


# ════════════════════════════════════════════════════════════════════════════
#  6. MAIN RUN FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def run_bot():
    print(f"\n{'═'*60}")
    print(f"  🤖 Gold Trading Bot — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"{'═'*60}")

    # Step 1: ราคาและ indicators
    print("📥 [1/6] ดึงข้อมูลราคาทอง...")
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
    print(f"   MA Signal     : {ma_signal} — {ma_reason}")

    # H4 data
    df_h4 = get_gold_h4()
    rsi_h4 = float(df_h4['RSI_H4'].iloc[-1]) if df_h4 is not None else 50.0

    # Support detection
    support = detect_support_level(df, df_h4)

    # Step 2: Macro Factors
    print("\n🌐 [2/6] วิเคราะห์ Macro Factors...")
    real_yield = get_real_yield()
    dxy        = get_dxy_data()
    futures    = get_futures_analysis(df)
    fib        = calc_fibonacci(df)
    td_seq     = calc_td_sequential(df)
    print(f"   Real Yield : {real_yield['label']}  ({real_yield['signal']})")
    print(f"   DXY        : {dxy['label']}  ({dxy['signal']})")
    print(f"   Futures    : {futures['label']}  ({futures['overall']})")
    print(f"   Fibonacci  : {fib['zone']}")
    print(f"   TD Seq     : {td_seq['label']}")
    print(f"   RSI H4        : {rsi_h4:.1f}")
    print(f"   Support       : {support['label']}")

    # Step 3: ข่าว
    print("\n📰 [3/6] ดึงข่าวทองคำจาก RSS feeds...")
    news = fetch_news(max_per_source=10, hours_back=24)
    print(f"   พบข่าวที่เกี่ยวข้อง: {len(news)} ข่าว")

    # Step 4: Sentiment
    print("\n🧠 [4/6] วิเคราะห์ Sentiment...")
    sentiment = analyze_sentiment(news)
    print(f"   Overall: {sentiment['label']} (score: {sentiment['score']:+.3f})")
    print(f"   Bullish: {sentiment['bullish']} | Bearish: {sentiment['bearish']} | Neutral: {sentiment['neutral']}")
    war_inflation = detect_war_inflation(news)
    print(f"   {war_inflation['label']}")

    # Step 5: รวมสัญญาณ (5 ปัจจัย)
    print("\n🎯 [5/6] คำนวณ Final Signal (5 ปัจจัย)...")
    final_signal, final_reason = combine_signals(
        ma_signal, sentiment,
        real_yield=real_yield, dxy=dxy,
        futures=futures, fib=fib, td_seq=td_seq,
        rsi_1d=float(df.iloc[-1]['RSI']),
        rsi_h4=rsi_h4,
        support=support,
        war_inflation=war_inflation,
    )
    print(f"   ➜ {final_signal}: {final_reason}")

    NOTIFY_SIGNALS = {
        "STRONG_BUY", "BUY", "WEAK_BUY",
        "STRONG_SELL", "SELL",
        "ACCUMULATE", "CAUTION",
    }

    if final_signal in NOTIFY_SIGNALS:
        msg = build_telegram_message(
            df, ma_signal, ma_reason, sentiment, final_signal, final_reason,
            real_yield=real_yield, dxy=dxy, futures=futures, fib=fib, td_seq=td_seq,
            rsi_h4=rsi_h4, support=support, war_inflation=war_inflation,
            score_breakdown=final_reason,
        )
        print("\n📱 [6/6] ส่ง Telegram Alert...")
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
        'price':     float(df.iloc[-1]['Close']),
        'sentiment': sentiment['score'],
        'news_count': len(news),
        'score':     final_reason,
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

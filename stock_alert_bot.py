"""
Stock Alert Bot — Daily Sentiment + Technical + Cut Loss
=========================================================
ติดตาม 3 หุ้น/ETF พร้อมส่ง Telegram วันละ 1 ครั้ง

  INDY  $39.00  (iShares MSCI India ETF)
  UNH   $250.00 (UnitedHealth Group)
  XLF   $46.50  (Financial Select Sector SPDR)

สัญญาณ:
  🚨 CUT LOSS  — ราคาต่ำกว่าเส้น Cut Loss
  🔴 SELL      — สัญญาณขาย (Score ≤ -4)
  ⚠️  WEAK SELL — ระวัง (Score -2 ถึง -4)
  🟡 HOLD      — รอดู
  ⚠️  WEAK BUY  — สะสมแบบระมัดระวัง (Score 2-4)
  ✅ BUY       — สัญญาณซื้อ (Score ≥ 4)
  🚀 STRONG BUY — สัญญาณแข็งแกร่ง (Score ≥ 7)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import requests
import os
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── CONFIG ────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")
DRY_RUN            = os.environ.get("DRY_RUN", "false").lower() == "true"

# ─── หุ้นที่ติดตาม + ราคา Cut Loss ────────────────────────────────────────
WATCHLIST = [
    {"symbol": "INDY",  "name": "iShares MSCI India ETF",          "cut_loss": 39.00},
    {"symbol": "UNH",   "name": "UnitedHealth Group",               "cut_loss": 250.00},
    {"symbol": "XLF",   "name": "Financial Select Sector SPDR ETF", "cut_loss": 46.50},
]
# ────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
#  1. PRICE & TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════

def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """ดึงราคา + คำนวณ indicators สำหรับหุ้น"""
    raw = yf.download(symbol, period=period, interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Close", "High", "Low", "Open", "Volume"]].dropna()

    if len(df) < 50:
        raise ValueError(f"ข้อมูลไม่พอสำหรับ {symbol} ({len(df)} rows)")

    c = df["Close"]

    # Moving Averages
    df["MA20"]  = c.rolling(20).mean()
    df["MA50"]  = c.rolling(50).mean()
    df["MA200"] = c.rolling(200).mean() if len(df) >= 200 else c.rolling(len(df)).mean()

    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # ATR (14)
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    # Volume MA20 (for volume surge detection)
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"]   = df["MACD"] - df["MACD_SIGNAL"]

    # Bollinger Bands (20, 2σ)
    df["BB_MID"]   = c.rolling(20).mean()
    bb_std         = c.rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * bb_std
    df["BB_LOWER"] = df["BB_MID"] - 2 * bb_std

    return df.dropna(subset=["MA20", "RSI", "ATR"])


def get_technical_signal(df: pd.DataFrame, cut_loss: float) -> dict:
    """
    วิเคราะห์ Technical signal + ระยะห่างจาก Cut Loss

    Weight Scoring:
      RSI < 30 (Oversold 1D)       : +3
      ราคา > MA20 > MA50            : +2  (Uptrend)
      MACD Bullish Cross            : +2
      ราคาใกล้ BB Lower (Bounce)   : +1
      Volume surge (>1.5× MA)      : +1
      RSI > 70 (Overbought)         : -3
      ราคา < MA20 < MA50            : -2  (Downtrend)
      MACD Bearish Cross            : -2
      ราคา < Cut Loss               : -5  (🚨 Emergency)
    """
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price      = float(latest["Close"])
    rsi        = float(latest["RSI"])
    ma20       = float(latest["MA20"])
    ma50       = float(latest["MA50"])
    ma200      = float(latest["MA200"])
    atr        = float(latest["ATR"])
    macd       = float(latest["MACD"])
    macd_sig   = float(latest["MACD_SIGNAL"])
    macd_prev  = float(prev["MACD"])
    msig_prev  = float(prev["MACD_SIGNAL"])
    bb_lower   = float(latest["BB_LOWER"])
    bb_upper   = float(latest["BB_UPPER"])
    volume     = float(latest["Volume"])
    vol_ma     = float(latest["VOL_MA20"])

    score      = 0
    breakdown  = []

    # ── 🚨 Cut Loss Check ──
    cut_loss_hit      = price <= cut_loss
    cut_loss_pct      = round((price - cut_loss) / cut_loss * 100, 2)
    cut_loss_distance = round(price - cut_loss, 2)

    if cut_loss_hit:
        score -= 5
        breakdown.append(f"🚨 CUT LOSS HIT -5")

    # ── RSI ──
    if rsi < 30:
        score += 3
        breakdown.append(f"RSI {rsi:.0f} Oversold +3")
    elif rsi < 40:
        score += 1
        breakdown.append(f"RSI {rsi:.0f}<40 +1")
    elif rsi > 70:
        score -= 3
        breakdown.append(f"RSI {rsi:.0f} Overbought -3")
    elif rsi > 60:
        score -= 1
        breakdown.append(f"RSI {rsi:.0f}>60 -1")

    # ── MA Trend ──
    if price > ma20 > ma50:
        score += 2
        breakdown.append("Uptrend MA +2")
    elif price < ma20 < ma50:
        score -= 2
        breakdown.append("Downtrend MA -2")
    elif price > ma20:
        score += 1
        breakdown.append("Above MA20 +1")

    # ── MACD Cross ──
    macd_bull_cross = (macd_prev < msig_prev) and (macd > macd_sig)
    macd_bear_cross = (macd_prev > msig_prev) and (macd < macd_sig)
    if macd_bull_cross:
        score += 2
        breakdown.append("MACD BullCross +2")
    elif macd_bear_cross:
        score -= 2
        breakdown.append("MACD BearCross -2")
    elif macd > macd_sig:
        score += 1
        breakdown.append("MACD Bull +1")
    elif macd < macd_sig:
        score -= 1
        breakdown.append("MACD Bear -1")

    # ── Bollinger Band Bounce ──
    if price <= bb_lower * 1.005:
        score += 1
        breakdown.append("BB Lower Bounce +1")
    elif price >= bb_upper * 0.995:
        score -= 1
        breakdown.append("BB Upper Reject -1")

    # ── Volume Surge ──
    if volume > vol_ma * 1.5:
        score += 1
        breakdown.append(f"Vol Surge +1")

    # ── MA200 Filter ──
    above_ma200 = price > ma200

    return {
        "price":              round(price, 2),
        "rsi":                round(rsi, 1),
        "ma20":               round(ma20, 2),
        "ma50":               round(ma50, 2),
        "ma200":              round(ma200, 2),
        "atr":                round(atr, 2),
        "macd":               round(macd, 3),
        "macd_signal":        round(macd_sig, 3),
        "bb_lower":           round(bb_lower, 2),
        "bb_upper":           round(bb_upper, 2),
        "above_ma200":        above_ma200,
        "cut_loss":           cut_loss,
        "cut_loss_hit":       cut_loss_hit,
        "cut_loss_pct":       cut_loss_pct,
        "cut_loss_distance":  cut_loss_distance,
        "score":              score,
        "breakdown":          breakdown,
    }


# ════════════════════════════════════════════════════════════════════════════
#  2. NEWS SENTIMENT
# ════════════════════════════════════════════════════════════════════════════

analyzer = SentimentIntensityAnalyzer()

STOCK_LEXICON = {
    "surge": 3.0, "rally": 2.5, "soar": 3.0, "beat": 2.0, "record": 2.0,
    "bullish": 3.0, "upgrade": 2.0, "buy": 1.5, "outperform": 2.0,
    "drop": -2.0, "fall": -1.5, "decline": -2.0, "plunge": -3.0,
    "bearish": -3.0, "sell": -1.5, "downgrade": -2.5, "miss": -2.0,
    "lawsuit": -2.0, "fraud": -3.0, "investigation": -2.0, "loss": -1.5,
    "recession": -2.0, "layoff": -2.0, "tariff": -1.5,
}
analyzer.lexicon.update(STOCK_LEXICON)

RSS_FEEDS = {
    "Yahoo Finance":     "https://finance.yahoo.com/rss/topstories",
    "MarketWatch":       "https://feeds.marketwatch.com/marketwatch/topstories",
    "Reuters Business":  "https://feeds.reuters.com/reuters/businessNews",
    "Seeking Alpha":     "https://seekingalpha.com/feed.xml",
}

def fetch_stock_news(symbol: str, company_name: str = "", hours_back: int = 24) -> list:
    """ดึงข่าวที่เกี่ยวกับหุ้นตัวนั้นๆ"""
    keywords = [symbol.upper(), company_name.lower()]
    # เพิ่ม keywords เฉพาะของแต่ละหุ้น
    extra = {
        "INDY":  ["india", "emerging market", "nifty", "sensex", "rupee"],
        "UNH":   ["unitedhealth", "health insurance", "medicare", "medicaid",
                  "optum", "insurance"],
        "XLF":   ["financial", "bank", "finance", "JPMorgan", "Goldman",
                  "interest rate", "Fed", "fed rate"],
    }
    keywords += extra.get(symbol.upper(), [])

    news    = []
    cutoff  = datetime.utcnow() - timedelta(hours=hours_back)

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title   = getattr(entry, "title",   "")
                summary = getattr(entry, "summary", "")
                link    = getattr(entry, "link",    "")
                text    = (title + " " + summary).lower()

                if not any(k.lower() in text for k in keywords if k):
                    continue

                pub_time = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    import time
                    pub_time = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                    if pub_time < cutoff:
                        continue

                news.append({
                    "source":  source,
                    "title":   title,
                    "summary": summary[:200],
                    "link":    link,
                    "time":    pub_time,
                })
        except Exception as e:
            print(f"  ⚠️ RSS {source}: {e}")

    news.sort(key=lambda x: x["time"] or datetime.min, reverse=True)
    return news


def analyze_stock_sentiment(news_list: list) -> dict:
    """วิเคราะห์ Sentiment ข่าวหุ้น"""
    if not news_list:
        return {"score": 0.0, "label": "NEUTRAL", "score_pt": 0,
                "bullish": 0, "bearish": 0, "neutral": 0,
                "total": 0, "top_news": []}

    scores  = []
    details = []
    for n in news_list:
        text  = n["title"] + ". " + n["summary"]
        s     = analyzer.polarity_scores(text)["compound"]
        scores.append(s)
        label = "🟢 Bullish" if s > 0.05 else ("🔴 Bearish" if s < -0.05 else "⚪ Neutral")
        details.append({**n, "sentiment_score": round(s, 3), "sentiment_label": label})

    avg       = float(np.mean(scores))
    bullish   = sum(1 for s in scores if s > 0.05)
    bearish   = sum(1 for s in scores if s < -0.05)
    neutral   = len(scores) - bullish - bearish
    label     = "BULLISH" if avg > 0.1 else ("BEARISH" if avg < -0.1 else "NEUTRAL")
    score_pt  = 1 if label == "BULLISH" else (-1 if label == "BEARISH" else 0)
    top_news  = sorted(details, key=lambda x: abs(x["sentiment_score"]), reverse=True)[:3]

    return {
        "score":    round(avg, 3),
        "label":    label,
        "score_pt": score_pt,
        "bullish":  bullish,
        "bearish":  bearish,
        "neutral":  neutral,
        "total":    len(scores),
        "top_news": top_news,
    }


# ════════════════════════════════════════════════════════════════════════════
#  3. COMBINED SIGNAL
# ════════════════════════════════════════════════════════════════════════════

def combine_stock_signals(tech: dict, sentiment: dict) -> tuple[str, str]:
    """
    รวม Technical Score + Sentiment Score → Final Signal

    Total Score threshold:
      ≥ 7  → STRONG BUY
      ≥ 4  → BUY
      ≥ 2  → WEAK BUY
      ≤ -6 → STRONG SELL / CUT LOSS
      ≤ -4 → SELL
      ≤ -2 → WEAK SELL
      else → HOLD
    """
    score = tech["score"] + sentiment["score_pt"]

    if tech["cut_loss_hit"]:
        return "CUT_LOSS", f"🚨 CUT LOSS — ราคา ${tech['price']:.2f} ต่ำกว่า ${tech['cut_loss']:.2f}"

    if score >= 7:
        return "STRONG_BUY",  f"🚀 STRONG BUY  (Score {score:+d})"
    elif score >= 4:
        return "BUY",          f"✅ BUY          (Score {score:+d})"
    elif score >= 2:
        return "WEAK_BUY",     f"⚠️  WEAK BUY    (Score {score:+d})"
    elif score <= -6:
        return "STRONG_SELL",  f"🔴 STRONG SELL  (Score {score:+d})"
    elif score <= -4:
        return "SELL",          f"🔴 SELL          (Score {score:+d})"
    elif score <= -2:
        return "WEAK_SELL",    f"🟠 WEAK SELL    (Score {score:+d})"
    else:
        return "HOLD",          f"🟡 HOLD          (Score {score:+d})"


# ════════════════════════════════════════════════════════════════════════════
#  4. TELEGRAM SENDER
# ════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    if DRY_RUN:
        print("\n" + "─" * 60)
        print("📱 [DRY RUN] Telegram Message:")
        print("─" * 60)
        print(message)
        print("─" * 60)
        return True

    # ตรวจสอบ secrets ก่อนส่ง
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
        print("❌ TELEGRAM_BOT_TOKEN ไม่ได้ตั้งค่า (ตรวจสอบ GitHub Secrets)")
        return False
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID":
        print("❌ TELEGRAM_CHAT_ID ไม่ได้ตั้งค่า (ตรวจสอบ GitHub Secrets)")
        return False

    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            return True
        else:
            # แสดง error จาก Telegram API
            print(f"❌ Telegram API Error {r.status_code}: {r.text[:200]}")
            # ถ้า HTML parse error → ลองส่งแบบ plain text
            if r.status_code == 400 and "parse" in r.text.lower():
                print("   🔄 ลองส่งแบบ plain text...")
                data["parse_mode"] = ""
                r2 = requests.post(url, data=data, timeout=10)
                if r2.status_code == 200:
                    print("   ✅ ส่ง plain text สำเร็จ")
                    return True
                print(f"   ❌ plain text ก็ fail: {r2.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Telegram Error: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
#  5. MESSAGE BUILDER
# ════════════════════════════════════════════════════════════════════════════

def build_stock_block(stock: dict, tech: dict, sentiment: dict,
                       signal: str, reason: str) -> str:
    """สร้าง block ข้อความสำหรับหุ้น 1 ตัว"""

    SIGNAL_EMOJI = {
        "STRONG_BUY":  "🚀", "BUY": "✅", "WEAK_BUY": "⚠️",
        "HOLD":        "🟡",
        "WEAK_SELL":   "🟠", "SELL": "🔴", "STRONG_SELL": "🔴",
        "CUT_LOSS":    "🚨",
    }
    emoji = SIGNAL_EMOJI.get(signal, "❓")

    price      = tech["price"]
    cut_loss   = tech["cut_loss"]
    cut_pct    = tech["cut_loss_pct"]
    rsi        = tech["rsi"]
    ma20       = tech["ma20"]
    ma50       = tech["ma50"]
    atr        = tech["atr"]

    # Distance from cut loss
    if tech["cut_loss_hit"]:
        cl_str = f"🚨 ต่ำกว่า Cut Loss {abs(cut_pct):.1f}%!"
    elif cut_pct < 5:
        cl_str = f"⚠️  ใกล้ Cut Loss {cut_pct:.1f}% เหนือ"
    else:
        cl_str = f"✅ ห่าง Cut Loss +{cut_pct:.1f}%"

    # RSI label
    rsi_label = ("🔴 Overbought" if rsi > 70 else
                 "🟢 Oversold"   if rsi < 30 else
                 "⚪ Normal")

    # Sentiment
    sent_icon  = ("🟢" if sentiment["label"] == "BULLISH" else
                  "🔴" if sentiment["label"] == "BEARISH" else "⚪")
    total_news = max(sentiment["total"], 1)
    bull_pct   = round(sentiment["bullish"] / total_news * 100)
    bear_pct   = round(sentiment["bearish"] / total_news * 100)

    # Top news (max 2)
    news_lines = ""
    for n in sentiment["top_news"][:2]:
        icon = "▲" if "Bullish" in n["sentiment_label"] else "▼"
        news_lines += f"\n    {icon} {n['title'][:50]}…"

    # Score breakdown (short)
    bd_str = " | ".join(tech["breakdown"][:4])

    block = (
        f"{'━'*35}\n"
        f"<b>{stock['symbol']}</b>  <i>{stock['name']}</i>\n"
        f"💵 ราคา   : <b>${price:.2f}</b>  (Cut Loss: ${cut_loss:.2f})\n"
        f"🛡 Status  : {cl_str}\n"
        f"📊 RSI     : {rsi:.0f}  {rsi_label}\n"
        f"📈 MA      : MA20 ${ma20:.2f}  |  MA50 ${ma50:.2f}\n"
        f"📉 MACD    : {'🟢 Bull' if tech['macd'] > tech['macd_signal'] else '🔴 Bear'}"
        f"  |  ATR ${atr:.2f}\n"
        f"📝 Score   : {bd_str}\n"
        f"📰 News    : {sent_icon} {sentiment['label']}"
        f"  ({sentiment['bullish']}🟢 {sentiment['bearish']}🔴  |  {sentiment['total']} ข่าว)"
        f"{news_lines}\n"
        f"➤ <b>{reason}</b>\n"
    )
    return block


def build_full_message(results: list) -> str:
    """รวม block ทุกหุ้นเป็น message เดียว"""
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Header — นับ alerts
    cut_loss_count = sum(1 for r in results if r["signal"] == "CUT_LOSS")
    sell_count     = sum(1 for r in results if "SELL" in r["signal"])
    buy_count      = sum(1 for r in results if "BUY"  in r["signal"])

    if cut_loss_count > 0:
        header_emoji = "🚨"
        header_title = f"STOCK ALERT — CUT LOSS! ({cut_loss_count} ตัว)"
    elif sell_count > 0:
        header_emoji = "🔴"
        header_title = f"STOCK ALERT — SELL Signal ({sell_count} ตัว)"
    elif buy_count > 0:
        header_emoji = "✅"
        header_title = f"STOCK ALERT — BUY Signal ({buy_count} ตัว)"
    else:
        header_emoji = "📊"
        header_title = "STOCK DAILY UPDATE"

    msg = f"{header_emoji} <b>{header_title}</b>  {now}\n"

    for r in results:
        msg += build_stock_block(
            r["stock"], r["tech"], r["sentiment"],
            r["signal"], r["reason"]
        )

    msg += (
        f"{'━'*35}\n"
        f"<i>⚠️ ไม่ใช่คำแนะนำการลงทุน  ลงทุนมีความเสี่ยง</i>"
    )
    return msg


# ════════════════════════════════════════════════════════════════════════════
#  6. MAIN
# ════════════════════════════════════════════════════════════════════════════

def run_stock_bot():
    print(f"\n{'═'*60}")
    print(f"  📈 Stock Alert Bot — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"{'═'*60}")

    results = []

    for stock in WATCHLIST:
        sym  = stock["symbol"]
        print(f"\n── {sym} ──────────────────────────────────────")

        # 1. Price & Technical
        try:
            df   = get_stock_data(sym)
            tech = get_technical_signal(df, stock["cut_loss"])
            print(f"   💵 ราคา      : ${tech['price']:.2f}")
            print(f"   📊 RSI       : {tech['rsi']:.1f}")
            print(f"   🛡 Cut Loss  : ${tech['cut_loss']:.2f}  ({tech['cut_loss_pct']:+.1f}%)")
            print(f"   📝 Tech Scr  : {tech['score']:+d}  [{', '.join(tech['breakdown'][:3])}]")
        except Exception as e:
            print(f"   ❌ Technical error: {e}")
            continue

        # 2. News Sentiment
        news      = fetch_stock_news(sym, stock["name"])
        sentiment = analyze_stock_sentiment(news)
        print(f"   📰 Sentiment : {sentiment['label']} ({sentiment['total']} ข่าว)")

        # 3. Combined Signal
        signal, reason = combine_stock_signals(tech, sentiment)
        print(f"   🎯 Signal    : {signal} — {reason}")

        results.append({
            "stock":     stock,
            "tech":      tech,
            "sentiment": sentiment,
            "signal":    signal,
            "reason":    reason,
        })

    if not results:
        print("\n❌ ไม่มีข้อมูล")
        return

    # 4. Build & Send Message
    msg = build_full_message(results)
    print("\n📱 ส่ง Telegram Daily Update...")
    ok = send_telegram(msg)
    print(f"   {'✅ ส่งสำเร็จ' if ok else '❌ ส่งไม่สำเร็จ'}")

    return results


if __name__ == "__main__":
    run_stock_bot()

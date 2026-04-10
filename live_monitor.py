"""
Gold Live Monitor — Continuous Signal Watcher
==============================================
ตรวจสัญญาณทองคำแบบ real-time ทุก N นาที
เมื่อพบ BUY/SELL signal → ส่ง Telegram + บันทึก trades.json

วิธีรัน:
  python live_monitor.py             # รันต่อเนื่อง (loop ตลอด)
  python live_monitor.py --once      # รันครั้งเดียว (สำหรับ cron/GitHub Actions)
  python live_monitor.py --status    # แสดง position ปัจจุบันแล้วออก
  DRY_RUN=true python live_monitor.py  # ไม่ส่ง Telegram จริง
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from gold_simulation import (
    SimConfig,
    fetch_historical_data,
    compute_indicators,
    generate_signal,
)

# ─── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID      = os.environ.get("TELEGRAM_CHAT_ID",   "")
DRY_RUN               = os.environ.get("DRY_RUN", "false").lower() == "true"

CHECK_INTERVAL_MIN    = int(os.environ.get("CHECK_INTERVAL_MIN", "60"))   # ตรวจทุก 60 นาที
INITIAL_CAPITAL       = float(os.environ.get("INITIAL_CAPITAL",  "100"))  # ทุนเริ่มต้น
TRANSACTION_COST      = float(os.environ.get("TRANSACTION_COST", "0.001")) # 0.1%

TRADES_FILE           = Path(__file__).parent / "trades.json"

# Signal config (ต้องตรงกับ dashboard/simulation)
RSI_BUY_THRESHOLD     = float(os.environ.get("RSI_BUY",  "35"))
RSI_SELL_THRESHOLD    = float(os.environ.get("RSI_SELL", "65"))
MA_FAST               = int(os.environ.get("MA_FAST",    "20"))
MA_SLOW               = int(os.environ.get("MA_SLOW",    "100"))
MA_TREND              = int(os.environ.get("MA_TREND",   "200"))
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("monitor")


# ════════════════════════════════════════════════════════════════════════════
#  TRADE RECORD  (trades.json)
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_STATE = {
    "state":         "CASH",   # "CASH" | "GOLD"
    "cash":          INITIAL_CAPITAL,
    "oz_held":       0.0,
    "entry_price":   None,
    "entry_date":    None,
    "entry_cost":    0.0,
    "trades":        [],       # list of completed + open trades
    "last_checked":  None,
    "last_signal":   None,
}


def load_state() -> dict:
    """โหลด state จาก trades.json (หรือ default ถ้าไม่มีไฟล์)"""
    if TRADES_FILE.exists():
        try:
            with open(TRADES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Merge กับ default เพื่อ handle missing keys จาก version เก่า
            merged = {**DEFAULT_STATE, **data}
            merged["trades"] = data.get("trades", [])
            log.info(f"Loaded state: {merged['state']} | "
                     f"cash=${merged['cash']:.2f} | "
                     f"oz={merged['oz_held']:.5f} | "
                     f"{len(merged['trades'])} trade(s) recorded")
            return merged
        except Exception as e:
            log.warning(f"Cannot load trades.json: {e} — using default state")
    else:
        log.info(f"No trades.json found — starting fresh with ${INITIAL_CAPITAL:.2f}")

    state = DEFAULT_STATE.copy()
    state["cash"]   = INITIAL_CAPITAL
    state["trades"] = []
    return state


def save_state(state: dict) -> None:
    """บันทึก state ลง trades.json"""
    state["last_saved"] = datetime.now().isoformat(timespec="seconds")
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def portfolio_value(state: dict, current_price: float) -> float:
    """คำนวณ portfolio value ณ ราคาปัจจุบัน"""
    if state["state"] == "CASH":
        return state["cash"]
    return state["oz_held"] * current_price


def total_return_pct(state: dict, current_price: float) -> float:
    val = portfolio_value(state, current_price)
    return (val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION
# ════════════════════════════════════════════════════════════════════════════

def get_current_signal() -> tuple[str, float, float, pd.Series]:
    """
    ดึงข้อมูล XAU/USD ล่าสุด คำนวณ indicators แล้วคืน signal
    Returns: (signal, current_price, rsi, latest_row)
    """
    cfg = SimConfig(
        rsi_buy_threshold  = RSI_BUY_THRESHOLD,
        rsi_sell_threshold = RSI_SELL_THRESHOLD,
        ma_fast  = MA_FAST,
        ma_slow  = MA_SLOW,
        ma_trend = MA_TREND,
    )
    df_raw = fetch_historical_data(cfg)
    df     = compute_indicators(df_raw, cfg)
    latest = df.iloc[-1]
    signal = generate_signal(latest, cfg)
    price  = float(latest["Close"])
    rsi    = float(latest["RSI"]) if not pd.isna(latest.get("RSI", float("nan"))) else 0.0
    return signal, price, rsi, latest


def describe_signal(signal: str, price: float, rsi: float, row: pd.Series, cfg_ref: SimConfig) -> str:
    """สร้างคำอธิบาย signal แบบละเอียด"""
    parts = []
    if rsi < cfg_ref.rsi_buy_threshold:
        parts.append(f"RSI={rsi:.1f} (oversold <{cfg_ref.rsi_buy_threshold})")
    elif rsi > cfg_ref.rsi_sell_threshold:
        parts.append(f"RSI={rsi:.1f} (overbought >{cfg_ref.rsi_sell_threshold})")
    else:
        parts.append(f"RSI={rsi:.1f}")

    ma_fast = row.get(f"MA{cfg_ref.ma_fast}", float("nan"))
    ma_slow = row.get(f"MA{cfg_ref.ma_slow}", float("nan"))
    if not pd.isna(ma_fast) and not pd.isna(ma_slow):
        parts.append(f"MA{cfg_ref.ma_fast}={ma_fast:.0f} / MA{cfg_ref.ma_slow}={ma_slow:.0f}")

    return " | ".join(parts)


# ════════════════════════════════════════════════════════════════════════════
#  TELEGRAM
# ════════════════════════════════════════════════════════════════════════════

def send_telegram(message: str) -> bool:
    if DRY_RUN:
        log.info("[DRY RUN] Telegram message:\n" + message)
        return True
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured — skipping")
        return False
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=10)
        if r.status_code == 200:
            log.info("Telegram sent ✓")
            return True
        log.warning(f"Telegram error {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        log.warning(f"Telegram exception: {e}")
        return False


def build_buy_message(price: float, oz: float, cost: float,
                      state: dict, reason: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    pnl_trades = [t for t in state["trades"] if t.get("pnl_usd") is not None]
    wins  = sum(1 for t in pnl_trades if t["pnl_usd"] > 0)
    total = len(pnl_trades)
    return (
        f"🟢 <b>BUY SIGNAL — XAU/USD</b>\n"
        f"{'─'*30}\n"
        f"📅 {now}\n"
        f"💰 Price   : <b>${price:,.2f}</b>\n"
        f"⚖️  Oz held  : {oz:.5f} oz\n"
        f"💵 Capital  : ${cost:.2f} (all-in)\n"
        f"{'─'*30}\n"
        f"📊 Signal   : {reason}\n"
        f"{'─'*30}\n"
        f"📈 Track record: {wins}W/{total-wins}L ({total} closed)\n"
        f"🏦 Portfolio: ${portfolio_value(state, price):.2f} "
        f"({total_return_pct(state, price):+.2f}% total)"
    )


def build_sell_message(price: float, pnl_usd: float, pnl_pct: float,
                       duration_days: int, state: dict, reason: str) -> str:
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    icon  = "✅" if pnl_usd >= 0 else "🔴"
    sign  = "+" if pnl_usd >= 0 else ""
    pnl_trades = [t for t in state["trades"] if t.get("pnl_usd") is not None]
    wins  = sum(1 for t in pnl_trades if t["pnl_usd"] > 0)
    total = len(pnl_trades)
    return (
        f"{icon} <b>SELL SIGNAL — XAU/USD</b>\n"
        f"{'─'*30}\n"
        f"📅 {now}\n"
        f"💰 Price    : <b>${price:,.2f}</b>\n"
        f"📊 P&L      : <b>{sign}${pnl_usd:.2f} ({sign}{pnl_pct:.2f}%)</b>\n"
        f"⏱️  Duration  : {duration_days} days\n"
        f"{'─'*30}\n"
        f"📊 Signal   : {reason}\n"
        f"{'─'*30}\n"
        f"📈 Track record: {wins}W/{total-wins}L ({total} closed)\n"
        f"🏦 Cash     : ${state['cash']:.2f} "
        f"({total_return_pct(state, price):+.2f}% total)"
    )


def build_hold_message(price: float, rsi: float, state: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    pos = state["state"]
    if pos == "GOLD":
        pnl = (price - state["entry_price"]) / state["entry_price"] * 100
        pos_str = f"GOLD | Entry ${state['entry_price']:,.2f} | Unrealized {pnl:+.1f}%"
    else:
        pos_str = f"CASH | ${state['cash']:.2f}"
    return (
        f"⚪ <b>HOLD — XAU/USD</b>\n"
        f"📅 {now}\n"
        f"💰 Price : ${price:,.2f} | RSI: {rsi:.1f}\n"
        f"📌 Position: {pos_str}"
    )


# ════════════════════════════════════════════════════════════════════════════
#  TRADE EXECUTION (virtual)
# ════════════════════════════════════════════════════════════════════════════

def execute_buy(state: dict, price: float, reason: str, row: pd.Series) -> dict:
    """จำลองการ BUY all-in"""
    cost       = state["cash"] * (1 - TRANSACTION_COST)
    oz         = cost / price
    now_str    = datetime.now().isoformat(timespec="seconds")

    state["state"]       = "GOLD"
    state["oz_held"]     = oz
    state["entry_price"] = price
    state["entry_cost"]  = cost
    state["entry_date"]  = now_str
    state["cash"]        = 0.0

    trade = {
        "id":         len(state["trades"]) + 1,
        "action":     "BUY",
        "date":       now_str,
        "price":      price,
        "oz":         oz,
        "cost":       cost,
        "reason":     reason,
        "pnl_usd":    None,
        "pnl_pct":    None,
        "duration":   None,
        "balance":    cost,
    }
    state["trades"].append(trade)
    log.info(f"BUY  @ ${price:,.2f} | {oz:.5f} oz | cost=${cost:.2f} | {reason}")
    return state


def execute_sell(state: dict, price: float, reason: str) -> tuple[dict, float, float, int]:
    """จำลองการ SELL all-out"""
    gross     = state["oz_held"] * price
    net       = gross * (1 - TRANSACTION_COST)
    pnl_usd   = net - state["entry_cost"]
    pnl_pct   = pnl_usd / state["entry_cost"] * 100
    entry_dt  = datetime.fromisoformat(state["entry_date"]) if state["entry_date"] else datetime.now()
    duration  = (datetime.now() - entry_dt).days
    now_str   = datetime.now().isoformat(timespec="seconds")

    # อัปเดต BUY trade ที่ match กัน
    for t in reversed(state["trades"]):
        if t["action"] == "BUY" and t.get("pnl_usd") is None:
            t["pnl_usd"]  = pnl_usd
            t["pnl_pct"]  = pnl_pct
            t["duration"] = duration
            t["closed_date"] = now_str
            t["close_price"] = price
            break

    sell_trade = {
        "id":       len(state["trades"]) + 1,
        "action":   "SELL",
        "date":     now_str,
        "price":    price,
        "oz":       state["oz_held"],
        "pnl_usd":  pnl_usd,
        "pnl_pct":  pnl_pct,
        "duration": duration,
        "reason":   reason,
        "balance":  net,
    }
    state["trades"].append(sell_trade)

    state["state"]       = "CASH"
    state["cash"]        = net
    state["oz_held"]     = 0.0
    state["entry_price"] = None
    state["entry_date"]  = None
    state["entry_cost"]  = 0.0

    log.info(f"SELL @ ${price:,.2f} | P&L=${pnl_usd:+.2f} ({pnl_pct:+.2f}%) | "
             f"cash=${net:.2f} | {duration}d")
    return state, pnl_usd, pnl_pct, duration


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CHECK CYCLE
# ════════════════════════════════════════════════════════════════════════════

def run_check(state: dict, send_hold_alert: bool = False) -> dict:
    """
    ตรวจสัญญาณ 1 รอบ อัปเดต state และส่ง Telegram ถ้าจำเป็น
    Returns: updated state
    """
    log.info("─── Checking signal ───")

    cfg_ref = SimConfig(
        rsi_buy_threshold  = RSI_BUY_THRESHOLD,
        rsi_sell_threshold = RSI_SELL_THRESHOLD,
        ma_fast = MA_FAST, ma_slow = MA_SLOW, ma_trend = MA_TREND,
    )

    try:
        signal, price, rsi, row = get_current_signal()
    except Exception as e:
        log.error(f"Failed to get signal: {e}")
        state["last_checked"] = datetime.now().isoformat(timespec="seconds")
        state["last_signal"]  = "ERROR"
        save_state(state)
        return state

    reason = describe_signal(signal, price, rsi, row, cfg_ref)
    log.info(f"Signal={signal} | Price=${price:,.2f} | {reason}")

    state["last_checked"] = datetime.now().isoformat(timespec="seconds")
    state["last_signal"]  = signal
    state["last_price"]   = price
    state["last_rsi"]     = rsi

    # ── BUY ───────────────────────────────────────────────────────────────────
    if signal == "BUY" and state["state"] == "CASH":
        state = execute_buy(state, price, reason, row)
        oz    = state["oz_held"]
        cost  = state["entry_cost"]
        msg   = build_buy_message(price, oz, cost, state, reason)
        send_telegram(msg)

    # ── SELL ──────────────────────────────────────────────────────────────────
    elif signal == "SELL" and state["state"] == "GOLD":
        state, pnl_usd, pnl_pct, duration = execute_sell(state, price, reason)
        msg = build_sell_message(price, pnl_usd, pnl_pct, duration, state, reason)
        send_telegram(msg)

    # ── HOLD ──────────────────────────────────────────────────────────────────
    else:
        pv = portfolio_value(state, price)
        rt = total_return_pct(state, price)
        log.info(f"HOLD | portfolio=${pv:.2f} ({rt:+.2f}%) | position={state['state']}")
        if send_hold_alert:
            send_telegram(build_hold_message(price, rsi, state))

    save_state(state)
    return state


def print_status(state: dict) -> None:
    """แสดง position ปัจจุบันใน terminal"""
    W = 50
    print("=" * W)
    print("  GOLD MONITOR — CURRENT STATUS")
    print("=" * W)

    last_price = state.get("last_price") or 0.0
    pv  = portfolio_value(state, last_price)
    ret = total_return_pct(state, last_price)
    closed = [t for t in state["trades"] if t["action"] == "SELL"]
    wins   = sum(1 for t in closed if (t.get("pnl_usd") or 0) > 0)

    print(f"  Position   : {state['state']}")
    if state["state"] == "GOLD":
        print(f"  Entry      : ${state['entry_price']:,.2f} @ {state['entry_date'][:10]}")
        if last_price:
            unreal = (last_price - state["entry_price"]) / state["entry_price"] * 100
            print(f"  Unrealized : {unreal:+.2f}%")
    print(f"  Cash       : ${state['cash']:.2f}")
    print(f"  Portfolio  : ${pv:.2f}  ({ret:+.2f}%)")
    print(f"  Last check : {state.get('last_checked', '—')}")
    print(f"  Last signal: {state.get('last_signal', '—')}")
    print(f"  Last price : ${last_price:,.2f}")
    print("─" * W)
    print(f"  Closed trades : {len(closed)}")
    print(f"  Win rate      : {wins}/{len(closed)} ({wins/len(closed)*100:.0f}%)" if closed else "  Win rate      : —")
    print("=" * W)

    if state["trades"]:
        print(f"\n  {'#':<4} {'Date':<12} {'Act':<5} {'Price':>10} {'P&L ($)':>10} {'P&L (%)':>8}")
        print("  " + "─" * 52)
        for t in state["trades"][-10:]:   # แสดง 10 อันล่าสุด
            dt_s  = str(t.get("date", ""))[:10]
            act   = t["action"]
            price_s = f"${t['price']:,.2f}"
            pnl_u = f"{t['pnl_usd']:+.2f}" if t.get("pnl_usd") is not None else "—"
            pnl_p = f"{t['pnl_pct']:+.2f}%" if t.get("pnl_pct") is not None else "—"
            print(f"  {t.get('id',0):<4} {dt_s:<12} {act:<5} {price_s:>10} {pnl_u:>10} {pnl_p:>8}")
        print()


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINTS
# ════════════════════════════════════════════════════════════════════════════

def run_continuous() -> None:
    """รันต่อเนื่อง loop ไม่หยุด"""
    log.info(f"Starting continuous monitor — interval={CHECK_INTERVAL_MIN}m | "
             f"RSI {RSI_BUY_THRESHOLD}/{RSI_SELL_THRESHOLD} | "
             f"MA {MA_FAST}/{MA_SLOW}/{MA_TREND} | "
             f"capital=${INITIAL_CAPITAL}")
    state = load_state()

    while True:
        try:
            state = run_check(state)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)

        next_check = datetime.now() + timedelta(minutes=CHECK_INTERVAL_MIN)
        log.info(f"Next check at {next_check.strftime('%H:%M:%S')} "
                 f"(sleeping {CHECK_INTERVAL_MIN}m)")
        try:
            time.sleep(CHECK_INTERVAL_MIN * 60)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break


def run_once() -> None:
    """รันครั้งเดียว (สำหรับ GitHub Actions / cron)"""
    log.info("Running single check (--once mode)")
    state = load_state()
    state = run_check(state)
    print_status(state)


def show_status() -> None:
    """แสดง status แล้วออก"""
    state = load_state()
    # ดึงราคาล่าสุดสำหรับ unrealized P&L
    try:
        import yfinance as yf
        raw = yf.download("GC=F", period="2d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        state["last_price"] = float(raw["Close"].dropna().iloc[-1])
    except Exception:
        pass
    print_status(state)


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Live Signal Monitor")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--once",   action="store_true",
                       help="Run one check and exit (for cron/GitHub Actions)")
    group.add_argument("--status", action="store_true",
                       help="Show current position and exit")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.once:
        run_once()
    else:
        run_continuous()

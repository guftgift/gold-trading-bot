"""
Gold Signal Monitor — แจ้งสัญญาณ BUY / SELL / HOLD ทาง Telegram
=================================================================
ไม่มีการ simulate การลงทุน — แจ้งสัญญาณ + อธิบาย algorithm อย่างเดียว

วิธีรัน:
  python live_monitor.py             # รันต่อเนื่อง (loop)
  python live_monitor.py --once      # รันครั้งเดียว (GitHub Actions / cron)
  python live_monitor.py --status    # แสดงสัญญาณล่าสุดแล้วออก
  DRY_RUN=true python live_monitor.py --once
"""

import os, sys, json, time, logging, argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from gold_simulation import (
    SimConfig,
    fetch_historical_data,
    compute_indicators,
    generate_signal,
)

# ─── Config ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")
DRY_RUN            = os.environ.get("DRY_RUN", "false").lower() == "true"
CHECK_INTERVAL_MIN = int(os.environ.get("CHECK_INTERVAL_MIN", "60"))
SEND_HOLD          = os.environ.get("SEND_HOLD", "false").lower() == "true"

RSI_BUY  = float(os.environ.get("RSI_BUY",  "35"))
RSI_SELL = float(os.environ.get("RSI_SELL", "65"))
MA_FAST  = int(os.environ.get("MA_FAST",    "20"))
MA_SLOW  = int(os.environ.get("MA_SLOW",    "100"))
MA_TREND = int(os.environ.get("MA_TREND",   "200"))

SIGNALS_FILE = Path(__file__).parent / "signals.json"
MAX_HISTORY  = 500
RETRY_SEC    = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("monitor")

# ─── Default state ────────────────────────────────────────────────────────────
DEFAULT_STATE = {
    "last_signal":    None,
    "last_checked":   None,
    "last_price":     None,
    "last_rsi":       None,
    "last_ma_fast":   None,
    "last_ma_slow":   None,
    "last_ma_trend":  None,
    "signal_reason":  None,
    "algorithm_used": None,
    "history":        [],
}


# ════════════════════════════════════════════════════════════════════════════
#  STATE  (signals.json)
# ════════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            merged = {**DEFAULT_STATE, **data}
            merged["history"] = data.get("history", [])
            return merged
        except Exception as e:
            log.warning(f"Cannot load signals.json: {e} — using default")
    return DEFAULT_STATE.copy()


def save_state(state: dict) -> None:
    state["last_saved"] = datetime.now().isoformat(timespec="seconds")
    with open(SIGNALS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL DETECTION  (พร้อม algorithm breakdown)
# ════════════════════════════════════════════════════════════════════════════

def get_signal_data() -> dict:
    """
    ดึงข้อมูล + คำนวณ indicators แล้วคืน dict ที่อธิบาย signal ครบถ้วน
    รวมถึงว่า algorithm ไหน trigger
    """
    cfg = SimConfig(
        rsi_buy_threshold  = RSI_BUY,
        rsi_sell_threshold = RSI_SELL,
        ma_fast=MA_FAST, ma_slow=MA_SLOW, ma_trend=MA_TREND,
    )
    df_raw = fetch_historical_data(cfg)
    df     = compute_indicators(df_raw, cfg)

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price     = float(latest["Close"])
    rsi       = float(latest.get("RSI",           float("nan")))
    ma_f      = float(latest.get(f"MA{MA_FAST}",  float("nan")))
    ma_s      = float(latest.get(f"MA{MA_SLOW}",  float("nan")))
    ma_t      = float(latest.get(f"MA{MA_TREND}", float("nan")))
    prev_ma_f = float(prev.get(f"MA{MA_FAST}",    float("nan")))
    prev_ma_s = float(prev.get(f"MA{MA_SLOW}",    float("nan")))

    # ── RSI sub-signal ────────────────────────────────────────────────────────
    if not pd.isna(rsi):
        if rsi < RSI_BUY:
            rsi_signal = "BUY"
            rsi_detail = f"RSI={rsi:.1f} → Oversold (ต่ำกว่า {RSI_BUY})"
        elif rsi > RSI_SELL:
            rsi_signal = "SELL"
            rsi_detail = f"RSI={rsi:.1f} → Overbought (สูงกว่า {RSI_SELL})"
        else:
            rsi_signal = "HOLD"
            rsi_detail = f"RSI={rsi:.1f} → Neutral (อยู่ใน {RSI_BUY}–{RSI_SELL})"
    else:
        rsi_signal, rsi_detail = "HOLD", "RSI=N/A (warm-up)"

    # ── MA Crossover sub-signal ───────────────────────────────────────────────
    ma_signal = "HOLD"
    ma_cross  = "ไม่มี crossover"

    if not any(pd.isna(v) for v in [ma_f, ma_s, ma_t, prev_ma_f, prev_ma_s]):
        crossed_up   = prev_ma_f <= prev_ma_s and ma_f > ma_s
        crossed_down = prev_ma_f >= prev_ma_s and ma_f < ma_s
        above_trend  = price > ma_t

        if crossed_up and above_trend:
            ma_signal = "BUY"
            ma_cross  = f"MA{MA_FAST} ข้ามขึ้นเหนือ MA{MA_SLOW} ✅ | ราคาเหนือ MA{MA_TREND} ✅"
        elif crossed_up and not above_trend:
            ma_cross  = f"MA{MA_FAST} ข้ามขึ้น แต่ราคาต่ำกว่า MA{MA_TREND} ⚠️ (ไม่นับ BUY)"
        elif crossed_down:
            ma_signal = "SELL"
            ma_cross  = f"MA{MA_FAST} ข้ามลงใต้ MA{MA_SLOW} ❌"
        else:
            rel      = "เหนือ" if ma_f > ma_s else "ใต้"
            ma_cross = f"MA{MA_FAST} อยู่{rel} MA{MA_SLOW} — ยังไม่ข้าม"

    ma_detail = (
        f"MA{MA_FAST}={ma_f:.0f} / MA{MA_SLOW}={ma_s:.0f} / MA{MA_TREND}={ma_t:.0f}"
        f" | {ma_cross}"
    )

    # ── Final signal (RSI wins over MA on tie-break) ──────────────────────────
    final_signal = generate_signal(latest, cfg)

    if rsi_signal != "HOLD":
        algorithm = "RSI"
        reason    = rsi_detail
    elif ma_signal != "HOLD":
        algorithm = f"MA{MA_FAST}/{MA_SLOW} Crossover"
        reason    = ma_detail
    else:
        algorithm = "—"
        reason    = f"{rsi_detail} | {ma_cross}"

    return {
        "signal":      final_signal,
        "price":       price,
        "rsi":         rsi,
        "ma_fast":     ma_f,
        "ma_slow":     ma_s,
        "ma_trend":    ma_t,
        "rsi_signal":  rsi_signal,
        "rsi_detail":  rsi_detail,
        "ma_signal":   ma_signal,
        "ma_detail":   ma_detail,
        "ma_cross":    ma_cross,
        "algorithm":   algorithm,
        "reason":      reason,
        "df":          df,
    }


# ════════════════════════════════════════════════════════════════════════════
#  TELEGRAM MESSAGE
# ════════════════════════════════════════════════════════════════════════════

def build_message(data: dict, state: dict) -> str:
    sig = data["signal"]
    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig, "⬜")
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")
    sep  = "─" * 28

    rsi_used = data["algorithm"] == "RSI"
    ma_used  = "MA" in data["algorithm"]

    lines = [
        f"{icon} <b>{sig} — XAU/USD</b>",
        sep,
        f"📅 {now}",
        f"💰 Price : <b>${data['price']:,.2f}</b>",
        sep,
        "🧮 <b>Algorithm Breakdown</b>",
        "",
        f"  📊 RSI → {data['rsi_detail']}",
        f"         {'✅ ใช้สัญญาณนี้' if rsi_used else ('⬜ RSI neutral' if data['rsi_signal']=='HOLD' else '⬜ ไม่ได้ใช้ (RSI wins)')}",
        "",
        f"  📈 MA  → {data['ma_cross']}",
        f"         {'✅ ใช้สัญญาณนี้' if ma_used else ('⬜ ไม่มี crossover' if data['ma_signal']=='HOLD' else '⬜ ไม่ได้ใช้ (RSI wins)')}",
        "",
        sep,
        f"🏆 <b>Decision: {sig}</b>",
        f"   Triggered by: {data['algorithm'] or 'ไม่มีสัญญาณชัดเจน'}",
    ]

    # แสดงสัญญาณ BUY/SELL ล่าสุด
    non_hold = [h for h in state.get("history", []) if h.get("signal") in ("BUY", "SELL")]
    if non_hold:
        last = non_hold[-1]
        lines += [
            sep,
            f"📌 สัญญาณ BUY/SELL ล่าสุด: {last['signal']} @ ${last['price']:,.2f}"
            f"  ({str(last['date'])[:10]})",
        ]

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    if DRY_RUN:
        log.info(f"[DRY RUN] Telegram:\n{message}")
        return True
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured — skipping")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if r.status_code == 200:
            log.info("Telegram sent ✓")
            return True
        log.warning(f"Telegram {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.warning(f"Telegram error: {e}")
    return False


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CHECK CYCLE
# ════════════════════════════════════════════════════════════════════════════

def run_check(state: dict) -> dict:
    """ตรวจสัญญาณ 1 รอบ → อัปเดต state → ส่ง Telegram"""
    log.info("─── Checking signal ───")

    data    = get_signal_data()
    sig     = data["signal"]
    now_str = datetime.now().isoformat(timespec="seconds")

    log.info(f"Signal={sig} | Price=${data['price']:,.2f} | "
             f"RSI={data['rsi']:.1f} | Algorithm={data['algorithm']}")

    state.update({
        "last_signal":    sig,
        "last_checked":   now_str,
        "last_price":     data["price"],
        "last_rsi":       data["rsi"],
        "last_ma_fast":   data["ma_fast"],
        "last_ma_slow":   data["ma_slow"],
        "last_ma_trend":  data["ma_trend"],
        "signal_reason":  data["reason"],
        "algorithm_used": data["algorithm"],
    })

    entry = {
        "date":      now_str,
        "signal":    sig,
        "price":     data["price"],
        "rsi":       round(data["rsi"], 2),
        "ma_fast":   round(data["ma_fast"], 2),
        "ma_slow":   round(data["ma_slow"], 2),
        "ma_trend":  round(data["ma_trend"], 2),
        "reason":    data["reason"],
        "algorithm": data["algorithm"],
    }
    state["history"] = (state.get("history", []) + [entry])[-MAX_HISTORY:]

    # ส่ง Telegram: BUY/SELL เสมอ, HOLD เฉพาะถ้า SEND_HOLD=true
    if sig in ("BUY", "SELL") or (sig == "HOLD" and SEND_HOLD):
        send_telegram(build_message(data, state))

    save_state(state)
    return state


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINTS
# ════════════════════════════════════════════════════════════════════════════

def run_continuous() -> None:
    log.info(f"Starting monitor | interval={CHECK_INTERVAL_MIN}m | "
             f"RSI {RSI_BUY}/{RSI_SELL} | MA {MA_FAST}/{MA_SLOW}/{MA_TREND} | "
             f"SEND_HOLD={SEND_HOLD}")
    state = load_state()
    while True:
        try:
            state = run_check(state)
        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as e:
            log.error(f"Error: {e}", exc_info=True)
            log.info(f"Retry in {RETRY_SEC}s …")
            try:
                time.sleep(RETRY_SEC)
            except KeyboardInterrupt:
                break
            continue

        next_t = datetime.now() + timedelta(minutes=CHECK_INTERVAL_MIN)
        log.info(f"Next check: {next_t.strftime('%H:%M:%S')} (sleep {CHECK_INTERVAL_MIN}m)")
        try:
            time.sleep(CHECK_INTERVAL_MIN * 60)
        except KeyboardInterrupt:
            log.info("Stopped.")
            break


def run_once() -> None:
    log.info("Running single check (--once)")
    state = load_state()
    state = run_check(state)
    _print_status(state)


def _print_status(state: dict) -> None:
    print("=" * 50)
    print("  GOLD SIGNAL MONITOR — STATUS")
    print("=" * 50)
    print(f"  Signal    : {state.get('last_signal', '—')}")
    print(f"  Price     : ${state.get('last_price') or 0:,.2f}")
    print(f"  RSI       : {state.get('last_rsi') or '—'}")
    print(f"  Algorithm : {state.get('algorithm_used', '—')}")
    print(f"  Reason    : {state.get('signal_reason', '—')}")
    print(f"  Checked   : {state.get('last_checked', '—')}")
    print(f"  History   : {len(state.get('history', []))} entries")
    print("=" * 50)
    hist = state.get("history", [])
    if hist:
        print(f"\n  {'Date':<22} {'Signal':<6} {'Price':>10} {'RSI':>6}  Algorithm")
        print("  " + "─" * 58)
        for h in hist[-10:]:
            print(f"  {str(h['date'])[:19]:<22} {h['signal']:<6} "
                  f"${h['price']:>9,.2f} {h['rsi']:>6.1f}  {h['algorithm']}")


def show_status() -> None:
    state = load_state()
    _print_status(state)


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Signal Monitor")
    grp    = parser.add_mutually_exclusive_group()
    grp.add_argument("--once",   action="store_true", help="Run one check and exit")
    grp.add_argument("--status", action="store_true", help="Show last status and exit")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.once:
        run_once()
    else:
        run_continuous()

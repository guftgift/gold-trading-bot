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
from io import BytesIO

import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")   # non-interactive — ไม่ต้องมี display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

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
#  CHART GENERATOR
# ════════════════════════════════════════════════════════════════════════════

# Dark theme colors
_BG      = "#0d1117"
_PANEL   = "#161b22"
_GOLD    = "#f9e2af"
_BLUE    = "#89b4fa"
_ORANGE  = "#fab387"
_RED     = "#f38ba8"
_GREEN   = "#a6e3a1"
_MUTED   = "#6c7086"
_TEXT    = "#cdd6f4"


def generate_chart(df: pd.DataFrame, state: dict, chart_days: int = 90) -> BytesIO:
    """
    สร้างกราฟ technical analysis:
      Panel 1: ราคา XAU/USD + MA lines + BUY/SELL markers
      Panel 2: RSI + oversold/overbought zones
    คืน BytesIO ของ PNG image
    """
    # ── ตัดข้อมูลเฉพาะ N วันล่าสุด ──────────────────────────────────────────
    df_plot = df.tail(chart_days).copy()

    sig    = state.get("last_signal", "HOLD")
    price  = state.get("last_price") or 0
    rsi_v  = state.get("last_rsi") or 0
    now_s  = str(state.get("last_checked", ""))[:16].replace("T", " ")

    sig_icon  = {"BUY": "▲ BUY", "SELL": "▼ SELL", "HOLD": "● HOLD"}.get(sig, sig)
    sig_color = {"BUY": _GREEN,  "SELL": _RED,      "HOLD": _MUTED}.get(sig, _TEXT)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 6.5), facecolor=_BG)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1],
                            hspace=0.06, top=0.88, bottom=0.08,
                            left=0.06, right=0.97)
    ax1 = fig.add_subplot(gs[0])   # Price
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # RSI

    for ax in (ax1, ax2):
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.spines[:].set_color("#30363d")
        ax.grid(color="#21262d", linewidth=0.6, linestyle="--")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    # ── Panel 1: Price + MAs ──────────────────────────────────────────────────
    ax1.plot(df_plot.index, df_plot["Close"],
             color=_GOLD, linewidth=1.6, label="XAU/USD", zorder=3)

    ma_styles = {
        f"MA{MA_FAST}":  (_BLUE,   "--", 1.1, f"MA{MA_FAST}"),
        f"MA{MA_SLOW}":  (_ORANGE, "--", 1.1, f"MA{MA_SLOW}"),
        f"MA{MA_TREND}": (_RED,    ":",  1.2, f"MA{MA_TREND}"),
    }
    for col, (color, ls, lw, lbl) in ma_styles.items():
        if col in df_plot.columns:
            ax1.plot(df_plot.index, df_plot[col],
                     color=color, linestyle=ls, linewidth=lw,
                     label=lbl, zorder=2)

    # BUY/SELL markers จาก history
    hist = state.get("history", [])
    for entry in hist:
        try:
            dt  = pd.Timestamp(entry["date"])
            if dt < df_plot.index[0]:
                continue
            ep  = entry["price"]
            esig = entry["signal"]
            if esig == "BUY":
                ax1.scatter(dt, ep, marker="^", s=120, color=_GREEN,
                            zorder=5, edgecolors=_BG, linewidths=0.8)
            elif esig == "SELL":
                ax1.scatter(dt, ep, marker="v", s=120, color=_RED,
                            zorder=5, edgecolors=_BG, linewidths=0.8)
        except Exception:
            pass

    # Current signal marker (ใหญ่กว่า)
    last_idx = df_plot.index[-1]
    if sig == "BUY":
        ax1.scatter(last_idx, price, marker="^", s=280, color=_GREEN,
                    zorder=6, edgecolors="white", linewidths=1.2)
    elif sig == "SELL":
        ax1.scatter(last_idx, price, marker="v", s=280, color=_RED,
                    zorder=6, edgecolors="white", linewidths=1.2)

    ax1.set_ylabel("Price (USD)", color=_MUTED, fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=7.5, facecolor=_PANEL,
               edgecolor="#30363d", labelcolor=_TEXT, framealpha=0.9)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2: RSI ──────────────────────────────────────────────────────────
    if "RSI" in df_plot.columns:
        ax2.plot(df_plot.index, df_plot["RSI"],
                 color="#cba6f7", linewidth=1.4, label="RSI")
        ax2.axhline(RSI_BUY,  color=_GREEN, linewidth=0.8,
                    linestyle="--", alpha=0.8)
        ax2.axhline(RSI_SELL, color=_RED,   linewidth=0.8,
                    linestyle="--", alpha=0.8)
        ax2.axhline(50,       color=_MUTED, linewidth=0.6,
                    linestyle=":", alpha=0.5)
        ax2.fill_between(df_plot.index, 0,       RSI_BUY,
                         alpha=0.07, color=_GREEN)
        ax2.fill_between(df_plot.index, RSI_SELL, 100,
                         alpha=0.07, color=_RED)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI", color=_MUTED, fontsize=8)
        ax2.text(df_plot.index[-1], RSI_BUY - 5,
                 f" {RSI_BUY}", color=_GREEN, fontsize=7, va="top")
        ax2.text(df_plot.index[-1], RSI_SELL + 2,
                 f" {RSI_SELL}", color=_RED,   fontsize=7, va="bottom")

    # X-axis date format
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center",
             color=_MUTED, fontsize=7.5)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.06, 0.93,
             f"XAU/USD — Technical Analysis  ({chart_days}d)",
             color=_TEXT, fontsize=11, fontweight="bold")
    fig.text(0.06, 0.895,
             f"Signal: ", color=_MUTED, fontsize=10)
    fig.text(0.155, 0.895,
             f"{sig_icon}  ${price:,.2f}  |  RSI {rsi_v:.1f}",
             color=sig_color, fontsize=10, fontweight="bold")
    fig.text(0.97, 0.895,
             now_s, color=_MUTED, fontsize=8, ha="right")

    # ── Export ────────────────────────────────────────────────────────────────
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130,
                facecolor=_BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


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
    """ส่งข้อความ text ธรรมดา"""
    if DRY_RUN:
        log.info(f"[DRY RUN] Telegram text:\n{message}")
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
            log.info("Telegram text sent ✓")
            return True
        log.warning(f"Telegram {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.warning(f"Telegram error: {e}")
    return False


def send_telegram_photo(image_buf: BytesIO, caption: str) -> bool:
    """ส่งกราฟ PNG พร้อม caption (HTML)"""
    if DRY_RUN:
        log.info(f"[DRY RUN] Telegram photo + caption:\n{caption}")
        # บันทึก PNG ไว้ดูในโฟลเดอร์ (dry run เท่านั้น)
        try:
            out = Path(__file__).parent / "dry_run_chart.png"
            out.write_bytes(image_buf.read())
            log.info(f"[DRY RUN] Chart saved → {out}")
            image_buf.seek(0)
        except Exception:
            pass
        return True
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured — skipping")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        r = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID,
                  "caption": caption,
                  "parse_mode": "HTML"},
            files={"photo": ("chart.png", image_buf, "image/png")},
            timeout=30,
        )
        if r.status_code == 200:
            log.info("Telegram photo sent ✓")
            return True
        log.warning(f"Telegram photo {r.status_code}: {r.text[:300]}")
        # fallback: ส่งแค่ text ถ้ารูปพัง
        return send_telegram(caption)
    except Exception as e:
        log.warning(f"Telegram photo error: {e} — falling back to text")
        return send_telegram(caption)


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
        caption = build_message(data, state)
        try:
            chart_buf = generate_chart(data["df"], state)
            send_telegram_photo(chart_buf, caption)
        except Exception as e:
            log.warning(f"Chart generation failed: {e} — sending text only")
            send_telegram(caption)

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

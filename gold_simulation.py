"""
Gold Trading Simulation — Backtest Engine
==========================================
จำลองการเทรดทองคำแบบ All-in ด้วยทุนเริ่มต้น $100 USD
พร้อม Validation ครบ 4 วิธี:
  1. Benchmark Metrics (Sharpe, Win Rate, Alpha vs B&H)
  2. Parameter Sweep (Grid Search RSI + MA)
  3. Walk-forward Validation (Out-of-sample Test)
  4. Monte Carlo Simulation (Skill vs Luck)

Signal Logic (ใช้ข้อมูลย้อนหลัง ไม่มี lookahead):
  - RSI(14) < 35 → BUY  | RSI > 65 → SELL
  - MA20 cross above MA100 + Price > MA200 → BUY
  - MA20 cross below MA100 → SELL
  - Priority: RSI > MA cross (tie-break)
"""

import os
import io
import random
import dataclasses
from datetime import date, timedelta, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ─── Telegram Config (same env vars as gold_trading_bot.py) ──────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")
DRY_RUN            = os.environ.get("DRY_RUN", "false").lower() == "true"
# ─────────────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class SimConfig:
    initial_capital:    float = 100.0    # USD
    transaction_cost:   float = 0.001    # 0.1% per trade (each side)
    rsi_period:         int   = 14
    rsi_buy_threshold:  float = 35.0     # RSI < threshold → BUY
    rsi_sell_threshold: float = 65.0     # RSI > threshold → SELL
    ma_fast:            int   = 20
    ma_slow:            int   = 100
    ma_trend:           int   = 200
    data_days:          int   = 500      # fetch N days of OHLC
    lookback_required:  int   = 200      # warm-up rows before trading starts
    risk_free_rate:     float = 0.05     # annual, for Sharpe calculation
    send_telegram:      bool  = False
    show_ascii_chart:   bool  = True


# ════════════════════════════════════════════════════════════════════════════
#  1. DATA FETCHING  (same fallback chain as gold_trading_bot.py)
# ════════════════════════════════════════════════════════════════════════════

def _stooq_fetch(symbol: str, days: int = 500) -> pd.DataFrame:
    """
    ดึง OHLC จาก Stooq
    หมายเหตุ: Stooq เริ่มต้องใช้ API key สำหรับบาง symbol (เช่น xauusd)
    ถ้า response ไม่ใช่ CSV ที่มีคอลัมน์ Date → raise ValueError ทันที
    """
    end   = date.today()
    start = end - timedelta(days=days)
    url   = (f"https://stooq.com/q/d/l/?s={symbol}"
             f"&d1={start.strftime('%Y%m%d')}"
             f"&d2={end.strftime('%Y%m%d')}&i=d")
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = r.text.strip()
    # Guard: Stooq ส่ง API-key error หรือข้อมูลว่างกลับมา
    if not text or not text.lower().startswith("date"):
        raise ValueError(f"Stooq {symbol}: non-CSV response (API key required or empty)")
    df = pd.read_csv(io.StringIO(text), index_col="Date", parse_dates=True)
    df.columns = [c.capitalize() for c in df.columns]
    df = df.sort_index().dropna()
    if "Close" not in df.columns or len(df) < 10:
        raise ValueError(f"Stooq {symbol}: insufficient data ({len(df)} rows)")
    return df


def fetch_historical_data(config: SimConfig) -> pd.DataFrame:
    """
    ดึงข้อมูล OHLC ย้อนหลังของ XAU/USD
    Fallback order (Stooq เป็น optional เพราะต้องการ API key บางตัว):
      1. yfinance GC=F  — Gold Futures (primary)
      2. yfinance GLD   — Gold ETF ×10
      3. yfinance IAU   — Gold ETF ×50
      4. Stooq gc.f     — ถ้ายังใช้ได้
    คืน raw DataFrame (ยังไม่ dropna — indicators คำนวณ later)
    """
    import yfinance as yf

    df = None

    # ─── yfinance (primary) ───────────────────────────────────────────────────
    for sym, mult in [("GC=F", 1.0), ("GLD", 10.0), ("IAU", 50.0)]:
        try:
            raw = yf.download(sym, period="2y", interval="1d",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
            if len(raw) >= config.lookback_required:
                if mult != 1.0:
                    for col in ["Open", "High", "Low", "Close"]:
                        raw[col] = raw[col] * mult
                df = raw
                print(f"  [DATA] yfinance:{sym}"
                      + (f" ×{mult}" if mult != 1.0 else "")
                      + f" — {len(raw)} rows "
                      f"({raw.index[0].date()} → {raw.index[-1].date()})")
                break
        except Exception as e:
            print(f"  [WARN] yfinance {sym}: {e}")

    # ─── Stooq gc.f fallback (บางครั้งยังใช้ได้) ─────────────────────────────
    if df is None:
        try:
            raw = _stooq_fetch("gc.f", days=config.data_days)
            if "Volume" not in raw.columns:
                raw["Volume"] = 0
            raw = raw[["Open", "High", "Low", "Close", "Volume"]]
            if len(raw) >= config.lookback_required:
                df = raw
                print(f"  [DATA] Stooq:gc.f — {len(raw)} rows")
        except Exception as e:
            print(f"  [WARN] Stooq gc.f: {e}")

    if df is None or len(df) == 0:
        raise RuntimeError("ดึงข้อมูลราคาทองไม่ได้จากทุกแหล่ง — ตรวจสอบการเชื่อมต่อ")

    return df.dropna(subset=["Close"])


# ════════════════════════════════════════════════════════════════════════════
#  2. INDICATOR ENGINE  (causal — no lookahead)
# ════════════════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame, config: SimConfig) -> pd.DataFrame:
    """
    เพิ่ม columns: MA_fast, MA_slow, MA_trend, RSI, ATR
    และ lagged columns สำหรับ cross detection
    ทุก operation ใช้ rolling/ewm → causal, ไม่มี lookahead
    """
    df = df.copy()
    c = df["Close"]

    df[f"MA{config.ma_fast}"]  = c.rolling(config.ma_fast).mean()
    df[f"MA{config.ma_slow}"]  = c.rolling(config.ma_slow).mean()
    df[f"MA{config.ma_trend}"] = c.rolling(config.ma_trend).mean()

    # RSI — Wilder EWM (เหมือนใน gold_trading_bot.py)
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=config.rsi_period - 1, adjust=False).mean()
    loss  = (-delta).clip(lower=0).ewm(com=config.rsi_period - 1, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # ATR (simplified — High-Low range)
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    # Lagged MA columns สำหรับ cross detection (shift(1) = ข้อมูลวันก่อนหน้า)
    df[f"MA{config.ma_fast}_prev"]  = df[f"MA{config.ma_fast}"].shift(1)
    df[f"MA{config.ma_slow}_prev"]  = df[f"MA{config.ma_slow}"].shift(1)

    return df


# ════════════════════════════════════════════════════════════════════════════
#  3. SIGNAL GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate_signal(row: pd.Series, config: SimConfig) -> str:
    """
    คืน 'BUY', 'SELL', หรือ 'HOLD' สำหรับ 1 วัน
    NaN guard: ถ้า indicator ใด NaN → HOLD (warm-up period)
    Priority: RSI > MA cross
    """
    maf  = f"MA{config.ma_fast}"
    mas  = f"MA{config.ma_slow}"
    mat  = f"MA{config.ma_trend}"
    mafp = f"MA{config.ma_fast}_prev"
    masp = f"MA{config.ma_slow}_prev"

    # NaN guard
    for col in ["RSI", maf, mas, mat, mafp, masp]:
        if pd.isna(row.get(col, float("nan"))):
            return "HOLD"

    rsi        = float(row["RSI"])
    ma_fast    = float(row[maf])
    ma_slow    = float(row[mas])
    ma_trend   = float(row[mat])
    ma_fast_p  = float(row[mafp])
    ma_slow_p  = float(row[masp])
    close      = float(row["Close"])

    # Factor 1 — RSI
    if rsi < config.rsi_buy_threshold:
        rsi_signal = "BUY"
    elif rsi > config.rsi_sell_threshold:
        rsi_signal = "SELL"
    else:
        rsi_signal = "NEUTRAL"

    # Factor 2 — MA Cross (with MA_trend filter for BUY only)
    golden_cross = (ma_fast_p <= ma_slow_p) and (ma_fast > ma_slow)
    death_cross  = (ma_fast_p >= ma_slow_p) and (ma_fast < ma_slow)
    above_trend  = close > ma_trend

    if golden_cross and above_trend:
        ma_signal = "BUY"
    elif death_cross:
        ma_signal = "SELL"
    else:
        ma_signal = "NEUTRAL"

    # Combined — RSI wins on tie
    if rsi_signal == "BUY":
        return "BUY"
    if rsi_signal == "SELL":
        return "SELL"
    if ma_signal == "BUY":
        return "BUY"
    if ma_signal == "SELL":
        return "SELL"
    return "HOLD"


# ════════════════════════════════════════════════════════════════════════════
#  4. PORTFOLIO STATE MACHINE
# ════════════════════════════════════════════════════════════════════════════

def run_simulation(
    df: pd.DataFrame,
    config: SimConfig,
) -> tuple[list[dict], list[dict]]:
    """
    วน loop ทุกวัน → apply signal → track state (CASH / GOLD)
    คืน (trade_log, equity_curve)

    trade_log   : list[dict] — 1 record ต่อ 1 executed trade (BUY หรือ SELL)
    equity_curve: list[dict] — 1 record ต่อ 1 วัน (portfolio value mark-to-market)
    """
    cash        = float(config.initial_capital)
    oz_held     = 0.0
    entry_cost  = 0.0   # total cash spent on current GOLD position
    entry_price = 0.0
    entry_date  = None
    state       = "CASH"   # "CASH" | "GOLD"

    trade_log:    list[dict] = []
    equity_curve: list[dict] = []

    rows = list(df.iterrows())

    for i, (date_idx, row) in enumerate(rows):
        close = float(row["Close"])

        # Mark-to-market portfolio value
        if state == "CASH":
            mtm_value = cash
        else:
            mtm_value = oz_held * close

        # Signal (only after warm-up)
        if i < config.lookback_required:
            signal = "WARM-UP"
        else:
            signal = generate_signal(row, config)

        # Execute trades
        if state == "CASH" and signal == "BUY":
            cost       = cash * (1 - config.transaction_cost)
            oz_held    = cost / close
            entry_cost = cost
            entry_price = close
            entry_date  = date_idx
            state      = "GOLD"
            cash       = 0.0
            trade_log.append({
                "date":      date_idx,
                "action":    "BUY",
                "price":     close,
                "oz":        oz_held,
                "cost":      entry_cost,
                "pnl_usd":   None,
                "pnl_pct":   None,
                "balance":   mtm_value,
                "signal":    signal,
            })

        elif state == "GOLD" and signal == "SELL":
            gross     = oz_held * close
            net       = gross * (1 - config.transaction_cost)
            pnl_usd   = net - entry_cost
            pnl_pct   = pnl_usd / entry_cost * 100
            cash      = net
            duration  = (date_idx - entry_date).days if entry_date else 0
            state     = "CASH"
            trade_log.append({
                "date":      date_idx,
                "action":    "SELL",
                "price":     close,
                "oz":        oz_held,
                "cost":      entry_cost,
                "pnl_usd":   pnl_usd,
                "pnl_pct":   pnl_pct,
                "balance":   net,
                "signal":    signal,
                "duration":  duration,
            })
            oz_held    = 0.0
            entry_cost = 0.0

        # Update equity curve (after trade execution)
        if state == "CASH":
            mtm_value = cash
        else:
            mtm_value = oz_held * close

        equity_curve.append({
            "date":   date_idx,
            "value":  mtm_value,
            "state":  state,
            "signal": signal,
            "close":  close,
        })

    # Force-close open position at last price (mark-to-market, not counted as win/loss)
    if state == "GOLD":
        last_close = float(df["Close"].iloc[-1])
        mtm        = oz_held * last_close * (1 - config.transaction_cost)
        trade_log.append({
            "date":    df.index[-1],
            "action":  "OPEN",   # still open at sim end
            "price":   last_close,
            "oz":      oz_held,
            "cost":    entry_cost,
            "pnl_usd": mtm - entry_cost,
            "pnl_pct": (mtm - entry_cost) / entry_cost * 100,
            "balance": mtm,
            "signal":  "END",
        })
        # Update last equity curve entry
        if equity_curve:
            equity_curve[-1]["value"] = mtm

    return trade_log, equity_curve


# ════════════════════════════════════════════════════════════════════════════
#  5. METRICS CALCULATOR
# ════════════════════════════════════════════════════════════════════════════

def calculate_metrics(
    trade_log:    list[dict],
    equity_curve: list[dict],
    df:           pd.DataFrame,
    config:       SimConfig,
) -> dict:
    """คำนวณ performance metrics ทั้งหมด"""

    if not equity_curve:
        return {}

    values = [e["value"] for e in equity_curve]
    final_value   = values[-1]
    total_return  = (final_value - config.initial_capital) / config.initial_capital * 100

    # Win / Loss จาก SELL trades เท่านั้น (ไม่นับ OPEN)
    sell_trades = [t for t in trade_log if t["action"] == "SELL"]
    wins   = sum(1 for t in sell_trades if (t["pnl_usd"] or 0) > 0)
    losses = sum(1 for t in sell_trades if (t["pnl_usd"] or 0) <= 0)
    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0

    # Max Drawdown
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd   = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    # Sharpe Ratio (annualised, daily returns)
    s = pd.Series(values)
    daily_ret = s.pct_change().dropna()
    rf_daily  = (1 + config.risk_free_rate) ** (1 / 252) - 1
    excess    = daily_ret - rf_daily
    sharpe    = float(excess.mean() / excess.std() * (252 ** 0.5)) if excess.std() > 0 else 0.0

    # Buy-and-Hold baseline (sim period only, after warm-up)
    sim_df   = df.iloc[config.lookback_required:]
    bh_entry = float(sim_df["Close"].iloc[0]) if len(sim_df) > 0 else 1.0
    bh_exit  = float(sim_df["Close"].iloc[-1]) if len(sim_df) > 0 else 1.0
    bh_return = (bh_exit - bh_entry) / bh_entry * 100
    alpha     = total_return - bh_return

    # Average trade duration (closed trades only)
    durations = [t.get("duration", 0) for t in sell_trades if t.get("duration") is not None]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    # Dates
    sim_start = equity_curve[config.lookback_required]["date"] if len(equity_curve) > config.lookback_required else equity_curve[0]["date"]
    sim_end   = equity_curve[-1]["date"]

    # Total trades including open
    all_buys = [t for t in trade_log if t["action"] == "BUY"]

    return {
        "initial_capital":     config.initial_capital,
        "final_balance":       final_value,
        "total_return_pct":    total_return,
        "buy_hold_return_pct": bh_return,
        "alpha_pct":           alpha,
        "num_trades":          len(all_buys),
        "num_closed":          total_closed,
        "num_wins":            wins,
        "num_losses":          losses,
        "win_rate_pct":        win_rate,
        "max_drawdown_pct":    max_dd,
        "sharpe_ratio":        sharpe,
        "avg_duration_days":   avg_duration,
        "sim_start":           sim_start,
        "sim_end":             sim_end,
        "open_position":       any(t["action"] == "OPEN" for t in trade_log),
        "low_trade_warning":   total_closed < 5,
    }


# ════════════════════════════════════════════════════════════════════════════
#  6. OUTPUT FORMATTERS
# ════════════════════════════════════════════════════════════════════════════

def print_trade_log(trade_log: list[dict]) -> None:
    """แสดงตาราง trade history"""
    if not trade_log:
        print("  (No trades executed)")
        return

    print(f"\n{'─'*90}")
    print(f"  {'#':<4} {'Date':<12} {'Action':<6} {'Price ($)':>10} "
          f"{'Oz':>8} {'P&L ($)':>10} {'P&L (%)':>8} {'Balance':>10}")
    print(f"{'─'*90}")

    trade_num = 0
    running_balance = None
    for t in trade_log:
        action  = t["action"]
        if action == "BUY":
            trade_num += 1
        dt_str  = t["date"].strftime("%Y-%m-%d") if hasattr(t["date"], "strftime") else str(t["date"])[:10]
        price   = f"${t['price']:,.2f}"
        oz_str  = f"{t['oz']:.5f}"
        pnl_u   = f"+${t['pnl_usd']:,.2f}" if t['pnl_usd'] and t['pnl_usd'] > 0 else (f"-${abs(t['pnl_usd']):,.2f}" if t['pnl_usd'] else "—")
        pnl_p   = f"+{t['pnl_pct']:.2f}%" if t['pnl_pct'] and t['pnl_pct'] > 0 else (f"{t['pnl_pct']:.2f}%" if t['pnl_pct'] else "—")
        bal_str = f"${t['balance']:,.2f}"
        tag     = " [OPEN]" if action == "OPEN" else ""

        print(f"  {trade_num:<4} {dt_str:<12} {action+tag:<8} {price:>10} "
              f"{oz_str:>8} {pnl_u:>10} {pnl_p:>8} {bal_str:>10}")

    print(f"{'─'*90}\n")


def print_summary(metrics: dict) -> str:
    """แสดงกล่องสรุปผล คืน string สำหรับ Telegram"""
    if not metrics:
        return "No metrics available."

    ret_sign   = "+" if metrics["total_return_pct"] >= 0 else ""
    bh_sign    = "+" if metrics["buy_hold_return_pct"] >= 0 else ""
    alpha_sign = "+" if metrics["alpha_pct"] >= 0 else ""
    start_s = metrics["sim_start"].strftime("%Y-%m-%d") if hasattr(metrics["sim_start"], "strftime") else str(metrics["sim_start"])[:10]
    end_s   = metrics["sim_end"].strftime("%Y-%m-%d")   if hasattr(metrics["sim_end"], "strftime")   else str(metrics["sim_end"])[:10]

    w = 48
    lines = [
        "╔" + "═" * w + "╗",
        "║" + "  GOLD SIMULATION RESULTS".center(w) + "║",
        "║" + f"  {start_s}  →  {end_s}".center(w) + "║",
        "╠" + "═" * w + "╣",
        "║" + f"  Initial Capital  :  ${metrics['initial_capital']:.2f}".ljust(w) + "║",
        "║" + f"  Final Balance    :  ${metrics['final_balance']:.2f}".ljust(w) + "║",
        "║" + f"  Total Return     :  {ret_sign}{metrics['total_return_pct']:.2f}%".ljust(w) + "║",
        "║" + f"  Buy-and-Hold     :  {bh_sign}{metrics['buy_hold_return_pct']:.2f}%".ljust(w) + "║",
        "║" + f"  Alpha (vs B&H)   :  {alpha_sign}{metrics['alpha_pct']:.2f}%".ljust(w) + "║",
        "╠" + "═" * w + "╣",
        "║" + f"  Trades (Closed)  :  {metrics['num_closed']}".ljust(w) + "║",
        "║" + f"  Win Rate         :  {metrics['win_rate_pct']:.1f}%  ({metrics['num_wins']}W / {metrics['num_losses']}L)".ljust(w) + "║",
        "║" + f"  Max Drawdown     :  {metrics['max_drawdown_pct']:.1f}%".ljust(w) + "║",
        "║" + f"  Sharpe Ratio     :  {metrics['sharpe_ratio']:.2f}".ljust(w) + "║",
        "║" + f"  Avg Duration     :  {metrics['avg_duration_days']:.1f} days".ljust(w) + "║",
    ]
    if metrics.get("open_position"):
        lines.append("║" + "  * Position still OPEN at sim end".ljust(w) + "║")
    if metrics.get("low_trade_warning"):
        lines.append("║" + "  ! Warning: <5 trades — stats may be weak".ljust(w) + "║")
    lines.append("╚" + "═" * w + "╝")

    summary = "\n".join(lines)
    print(summary)
    return summary


def print_ascii_chart(
    equity_curve: list[dict],
    width:  int = 60,
    height: int = 12,
    title:  str = "Portfolio Value ($)",
) -> None:
    """ASCII sparkline ของ equity curve"""
    values = [e["value"] for e in equity_curve]
    dates  = [e["date"] for e in equity_curve]

    if len(values) < 2:
        return

    # Downsample to `width` points
    if len(values) > width:
        idx    = [int(i * (len(values) - 1) / (width - 1)) for i in range(width)]
        values = [values[i] for i in idx]
        dates  = [dates[i] for i in idx]

    v_min = min(values)
    v_max = max(values)
    v_range = v_max - v_min if v_max != v_min else 1.0

    # Build grid
    grid = [[" "] * len(values) for _ in range(height)]
    for col, val in enumerate(values):
        row = int((val - v_min) / v_range * (height - 1))
        row = height - 1 - row   # flip Y
        grid[row][col] = "█"

    print(f"\n  {title}")
    print(f"  ${v_max:,.2f} ┐")
    for r, row_chars in enumerate(grid):
        prefix = "  " + " " * 9 + "│"
        print(prefix + "".join(row_chars))
    print(f"  ${v_min:,.2f} └" + "─" * len(values))
    start_s = dates[0].strftime("%Y-%m-%d")  if hasattr(dates[0],  "strftime") else str(dates[0])[:10]
    end_s   = dates[-1].strftime("%Y-%m-%d") if hasattr(dates[-1], "strftime") else str(dates[-1])[:10]
    pad = len(values) - len(start_s) - len(end_s)
    print(f"           {start_s}" + " " * max(0, pad) + f"{end_s}")
    print()


# ════════════════════════════════════════════════════════════════════════════
#  7. PARAMETER SWEEP  (Grid Search)
# ════════════════════════════════════════════════════════════════════════════

def run_parameter_sweep(df: pd.DataFrame, config: SimConfig) -> pd.DataFrame:
    """
    Grid search ผ่าน RSI threshold + MA periods
    คืน DataFrame เรียงตาม Sharpe ratio (ดีสุดขึ้นก่อน)
    """
    rsi_buy_grid  = [25, 30, 35, 40]
    rsi_sell_grid = [60, 65, 70, 75]
    ma_fast_grid  = [10, 20, 50]
    ma_slow_grid  = [50, 100, 200]

    results = []
    total   = len(rsi_buy_grid) * len(rsi_sell_grid) * len(ma_fast_grid) * len(ma_slow_grid)
    done    = 0

    print(f"\n  Running {total} parameter combinations...")

    for rb in rsi_buy_grid:
        for rs in rsi_sell_grid:
            if rb >= rs:
                continue
            for mf in ma_fast_grid:
                for ms in ma_slow_grid:
                    if mf >= ms:
                        continue
                    done += 1
                    cfg = dataclasses.replace(
                        config,
                        rsi_buy_threshold  = rb,
                        rsi_sell_threshold = rs,
                        ma_fast            = mf,
                        ma_slow            = ms,
                    )
                    try:
                        df_i       = compute_indicators(df.copy(), cfg)
                        tlog, ecurve = run_simulation(df_i, cfg)
                        m          = calculate_metrics(tlog, ecurve, df, cfg)
                        is_default = (rb == config.rsi_buy_threshold and
                                      rs == config.rsi_sell_threshold and
                                      mf == config.ma_fast and
                                      ms == config.ma_slow)
                        results.append({
                            "RSI_buy":  rb, "RSI_sell": rs,
                            "MA_fast":  mf, "MA_slow":  ms,
                            "Return%":  round(m.get("total_return_pct", 0), 2),
                            "WinRate%": round(m.get("win_rate_pct", 0), 1),
                            "Sharpe":   round(m.get("sharpe_ratio", 0), 2),
                            "MaxDD%":   round(m.get("max_drawdown_pct", 0), 1),
                            "Trades":   m.get("num_closed", 0),
                            "Default":  "*" if is_default else "",
                        })
                    except Exception:
                        pass

    result_df = pd.DataFrame(results).sort_values("Sharpe", ascending=False)
    return result_df


def print_sweep_results(sweep_df: pd.DataFrame, top_n: int = 10) -> None:
    """แสดง top-N parameter combinations จาก sweep"""
    print(f"\n{'─'*72}")
    print(f"  PARAMETER SWEEP — Top {top_n} Combinations (sorted by Sharpe)")
    print(f"{'─'*72}")
    print(f"  {'RSI-B':>6} {'RSI-S':>6} {'MA-F':>5} {'MA-S':>5} "
          f"{'Return%':>8} {'WinR%':>6} {'Sharpe':>7} {'MaxDD%':>7} {'Trades':>7} {'':>2}")
    print(f"{'─'*72}")
    for _, row in sweep_df.head(top_n).iterrows():
        flag = row.get("Default", "")
        print(f"  {int(row['RSI_buy']):>6} {int(row['RSI_sell']):>6} "
              f"{int(row['MA_fast']):>5} {int(row['MA_slow']):>5} "
              f"{row['Return%']:>+8.2f} {row['WinRate%']:>6.1f} "
              f"{row['Sharpe']:>7.2f} {row['MaxDD%']:>7.1f} "
              f"{int(row['Trades']):>7} {flag:>2}")
    print(f"{'─'*72}")
    print("  (* = current default config)\n")


# ════════════════════════════════════════════════════════════════════════════
#  8. WALK-FORWARD VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    df: pd.DataFrame,
    config: SimConfig,
    n_splits: int = 3,
) -> list[dict]:
    """
    Expanding-window walk-forward validation
    สำหรับแต่ละ split: train ขยาย, test = ช่วงถัดไป
    คืน list[dict] ผลแต่ละ split
    """
    sim_df   = df.iloc[config.lookback_required:]   # เริ่มหลัง warm-up
    n        = len(sim_df)
    seg_size = n // (n_splits + 1)

    results = []
    for i in range(n_splits):
        test_start = (i + 1) * seg_size
        test_end   = min((i + 2) * seg_size, n)

        # test slice (ใช้ warmup rows ด้วยเพื่อ indicator ไม่ NaN)
        # เริ่มจาก lookback_required + test_start
        abs_start = config.lookback_required + test_start
        abs_end   = config.lookback_required + test_end
        test_slice = df.iloc[:abs_end]  # expanding window

        cfg_tmp = dataclasses.replace(config, lookback_required=abs_start)
        try:
            df_i         = compute_indicators(test_slice.copy(), cfg_tmp)
            tlog, ecurve = run_simulation(df_i, cfg_tmp)
            m            = calculate_metrics(tlog, ecurve, test_slice, cfg_tmp)
            results.append({
                "split":        i + 1,
                "period_start": sim_df.index[test_start].date() if test_start < len(sim_df) else None,
                "period_end":   sim_df.index[min(test_end, len(sim_df)-1) - 1].date(),
                "return_pct":   m.get("total_return_pct", 0),
                "win_rate":     m.get("win_rate_pct", 0),
                "num_trades":   m.get("num_closed", 0),
                "passed":       m.get("total_return_pct", 0) > 0,
            })
        except Exception as e:
            results.append({
                "split": i + 1, "period_start": None, "period_end": None,
                "return_pct": 0, "win_rate": 0, "num_trades": 0,
                "passed": False, "error": str(e),
            })

    return results


def print_walk_forward(wf_results: list[dict]) -> None:
    """แสดงผล walk-forward validation"""
    print(f"\n{'─'*60}")
    print("  WALK-FORWARD VALIDATION")
    print(f"{'─'*60}")
    passes = 0
    for r in wf_results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        if r["passed"]:
            passes += 1
        ret_sign = "+" if r["return_pct"] >= 0 else ""
        print(f"  Split {r['split']}: {r['period_start']} → {r['period_end']}  "
              f"Return={ret_sign}{r['return_pct']:.1f}%  "
              f"WinRate={r['win_rate']:.0f}%  "
              f"Trades={r['num_trades']}  [{status}]")
    total = len(wf_results)
    print(f"{'─'*60}")
    print(f"  Result: {passes}/{total} periods profitable\n")


# ════════════════════════════════════════════════════════════════════════════
#  9. MONTE CARLO SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(
    trade_log:      list[dict],
    config:         SimConfig,
    n_simulations:  int = 1000,
) -> dict:
    """
    Shuffle P&L sequence n ครั้ง → ดูว่าผลจริงดีกว่า random แค่ไหน
    p_value < 0.05 → อาจเกิดจาก skill ไม่ใช่ luck
    """
    sell_trades = [t for t in trade_log if t["action"] == "SELL"]
    if len(sell_trades) < 2:
        return {
            "p_value": 1.0, "actual_return_pct": 0.0,
            "mean_random_return": 0.0, "percentile": 0.0,
            "distribution": [], "passed": False, "n_trades": 0,
        }

    # P&L ของแต่ละ trade เป็น % ต่อ capital
    pnl_pcts = [t["pnl_pct"] for t in sell_trades if t["pnl_pct"] is not None]
    n_trades  = len(pnl_pcts)

    # Actual cumulative return
    actual_final = config.initial_capital
    for p in pnl_pcts:
        actual_final *= (1 + p / 100)
    actual_return = (actual_final - config.initial_capital) / config.initial_capital * 100

    # Monte Carlo runs
    random_returns = []
    for _ in range(n_simulations):
        shuffled = random.sample(pnl_pcts, len(pnl_pcts))
        bal = config.initial_capital
        for p in shuffled:
            bal *= (1 + p / 100)
        random_returns.append((bal - config.initial_capital) / config.initial_capital * 100)

    # p-value = สัดส่วน random sims ที่ดีกว่าหรือเท่ากับ actual
    p_value    = sum(1 for r in random_returns if r >= actual_return) / n_simulations
    percentile = sum(1 for r in random_returns if r < actual_return) / n_simulations * 100

    return {
        "p_value":             p_value,
        "actual_return_pct":   actual_return,
        "mean_random_return":  sum(random_returns) / len(random_returns),
        "percentile":          percentile,
        "distribution":        random_returns,
        "passed":              p_value < 0.10,
        "n_trades":            n_trades,
        "n_simulations":       n_simulations,
    }


def print_monte_carlo(mc: dict) -> None:
    """แสดงผล Monte Carlo + ASCII histogram"""
    print(f"\n{'─'*60}")
    print(f"  MONTE CARLO SIMULATION ({mc.get('n_simulations', 1000)} runs)")
    print(f"{'─'*60}")
    if mc["n_trades"] < 2:
        print("  Not enough trades for Monte Carlo analysis.\n")
        return

    sign = "+" if mc["actual_return_pct"] >= 0 else ""
    print(f"  Actual Return   : {sign}{mc['actual_return_pct']:.2f}%")
    print(f"  Mean (random)   : {mc['mean_random_return']:+.2f}%")
    print(f"  Percentile      : {mc['percentile']:.1f}th")
    print(f"  p-value         : {mc['p_value']:.3f}  "
          f"({'Skill likely ✓' if mc['passed'] else 'Could be luck ✗'})")

    # ASCII histogram (20 bins)
    dist = mc["distribution"]
    if dist:
        lo, hi = min(dist), max(dist)
        bins = 20
        step = (hi - lo) / bins if hi != lo else 1
        counts = [0] * bins
        for v in dist:
            b = min(int((v - lo) / step), bins - 1)
            counts[b] += 1
        max_count = max(counts)
        bar_height = 8
        print(f"\n  Distribution of random returns (n={len(dist)}):")
        for row in range(bar_height, 0, -1):
            line = "  │"
            for c in counts:
                h = int(c / max_count * bar_height)
                line += "█" if h >= row else " "
            print(line)
        print("  └" + "─" * bins)
        print(f"    {lo:+.1f}%" + " " * (bins - 8) + f"{hi:+.1f}%")
        actual_bin = min(int((mc["actual_return_pct"] - lo) / step), bins - 1)
        marker = "  " + " " * (actual_bin + 2) + "^actual"
        print(marker)
    print()


# ════════════════════════════════════════════════════════════════════════════
#  10. COMBINED VALIDATION REPORT
# ════════════════════════════════════════════════════════════════════════════

def print_validation_report(
    metrics:    dict,
    sweep_df:   pd.DataFrame,
    wf_results: list[dict],
    mc:         dict,
) -> None:
    """สรุปผลการ validation ทั้ง 4 วิธีในรูปแบบ PASS/FAIL"""

    # Benchmark criteria
    bench_pass = (metrics.get("sharpe_ratio", 0) >= 0.5 and
                  metrics.get("win_rate_pct", 0) >= 50 and
                  metrics.get("alpha_pct", -999) >= 0)

    # Best params vs default
    best_params = {}
    if not sweep_df.empty:
        best = sweep_df.iloc[0]
        best_params = {
            "RSI_buy": int(best["RSI_buy"]), "RSI_sell": int(best["RSI_sell"]),
            "MA_fast": int(best["MA_fast"]), "MA_slow":  int(best["MA_slow"]),
        }

    # Walk-forward
    wf_passes = sum(1 for r in wf_results if r["passed"])
    wf_total  = len(wf_results)
    wf_pass   = wf_passes >= max(1, wf_total * 2 // 3)

    # Monte Carlo
    mc_pass = mc.get("passed", False)

    def fmt(passed: bool) -> str:
        return "[PASS ✓]" if passed else "[FAIL ✗]"

    w = 52
    print(f"\n{'═'*w}")
    print(f"{'  STRATEGY VALIDATION REPORT':^{w}}")
    print(f"{'═'*w}")
    print(f"  {fmt(bench_pass)} Benchmark  "
          f"— Sharpe={metrics.get('sharpe_ratio',0):.2f}, "
          f"WinRate={metrics.get('win_rate_pct',0):.0f}%, "
          f"Alpha={metrics.get('alpha_pct',0):+.1f}%")
    if best_params:
        print(f"  {'[INFO] ':>9} Best Params "
              f"— RSI {best_params['RSI_buy']}/{best_params['RSI_sell']}, "
              f"MA {best_params['MA_fast']}/{best_params['MA_slow']}")
    print(f"  {fmt(wf_pass)} Walk-fwd  "
          f"— {wf_passes}/{wf_total} periods profitable")
    print(f"  {fmt(mc_pass)} Monte Carlo"
          f"— p-value={mc.get('p_value', 1.0):.3f}")

    all_pass = bench_pass and wf_pass and mc_pass
    print(f"{'═'*w}")
    if all_pass:
        print(f"  >>> STRATEGY CONFIDENCE: HIGH   ✓ ✓ ✓")
    elif sum([bench_pass, wf_pass, mc_pass]) >= 2:
        print(f"  >>> STRATEGY CONFIDENCE: MEDIUM ✓ ✓ ✗")
    else:
        print(f"  >>> STRATEGY CONFIDENCE: LOW    ✗ — use caution")
    print(f"{'═'*w}\n")


# ════════════════════════════════════════════════════════════════════════════
#  11. TELEGRAM SENDER
# ════════════════════════════════════════════════════════════════════════════

def send_telegram_summary(summary_str: str) -> bool:
    """ส่ง summary ไป Telegram (เหมือน pattern ใน gold_trading_bot.py)"""
    if not TELEGRAM_BOT_TOKEN:
        print("  [SKIP] Telegram not configured.")
        return False
    if DRY_RUN:
        print("[DRY RUN] Telegram message:\n" + summary_str)
        return True
    url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": f"<pre>{summary_str}</pre>",
            "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  [ERROR] Telegram: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
#  12. MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_simulation_main(config: Optional[SimConfig] = None) -> dict:
    """
    Full pipeline — คืน dict ของ metrics สำหรับใช้ใน dashboard
    """
    if config is None:
        config = SimConfig()

    W = 60
    print("=" * W)
    print("  GOLD TRADING SIMULATION — ALL-IN $100")
    print("=" * W)

    # ── Step 1: Data ──────────────────────────────────────────────────────
    print("\n[1/6] Fetching historical XAU/USD data...")
    df_raw = fetch_historical_data(config)

    # ── Step 2: Indicators ────────────────────────────────────────────────
    print("[2/6] Computing indicators (RSI, MA, ATR)...")
    df = compute_indicators(df_raw, config)

    # ── Step 3: Simulation ────────────────────────────────────────────────
    print("[3/6] Running ALL-IN simulation...")
    trade_log, equity_curve = run_simulation(df, config)
    closed = sum(1 for t in trade_log if t["action"] == "SELL")
    print(f"      → {closed} round-trips completed | "
          f"{'Position OPEN at end' if any(t['action']=='OPEN' for t in trade_log) else 'All closed'}")

    # ── Step 4: Metrics ───────────────────────────────────────────────────
    print("[4/6] Calculating performance metrics...")
    metrics = calculate_metrics(trade_log, equity_curve, df, config)

    # ── Step 5: Validation ────────────────────────────────────────────────
    print("[5/6] Running strategy validation...")
    print("  → Parameter sweep...")
    sweep_df = run_parameter_sweep(df_raw, config)
    print("  → Walk-forward validation...")
    wf_results = run_walk_forward(df_raw, config)
    print("  → Monte Carlo simulation...")
    mc = run_monte_carlo(trade_log, config)

    # ── Step 6: Output ────────────────────────────────────────────────────
    print("[6/6] Generating output...\n")
    print_trade_log(trade_log)

    if config.show_ascii_chart:
        print_ascii_chart(equity_curve)

    summary_str = print_summary(metrics)
    print_sweep_results(sweep_df)
    print_walk_forward(wf_results)
    print_monte_carlo(mc)
    print_validation_report(metrics, sweep_df, wf_results, mc)

    if config.send_telegram:
        print("Sending to Telegram...")
        send_telegram_summary(summary_str)

    return {
        "metrics":     metrics,
        "trade_log":   trade_log,
        "equity_curve": equity_curve,
        "sweep_df":    sweep_df,
        "wf_results":  wf_results,
        "mc":          mc,
        "df":          df,
    }


if __name__ == "__main__":
    run_simulation_main()

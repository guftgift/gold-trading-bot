"""
Gold Trading Dashboard — Streamlit
====================================
Monitor & visualize gold trading simulation results
Deploy ฟรีบน Streamlit Cloud (streamlit.io)

Run locally:
    streamlit run dashboard.py

Deploy:
    1. Push to GitHub
    2. Go to share.streamlit.io → New app → select this file
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests

from gold_simulation import (
    SimConfig,
    fetch_historical_data,
    compute_indicators,
    run_simulation,
    calculate_metrics,
    run_parameter_sweep,
    run_walk_forward,
    run_monte_carlo,
)

TRADES_FILE = Path(__file__).parent / "trades.json"

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Gold Trading Simulator",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #313244;
    }
    div[data-testid="stSidebar"] { background: #181825; }
    .live-price-box {
        background: linear-gradient(135deg, #1e1e2e 0%, #313244 100%);
        border: 1px solid #f9e2af44;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def get_live_price() -> dict:
    """
    ดึงราคา XAU/USD ล่าสุด — cache 60 วินาที
    Fallback: yfinance GC=F → yfinance GLD×10 → yfinance IAU×50
    (Stooq ถูกลบออกเพราะต้องการ API key แล้ว)
    """
    import yfinance as yf

    for sym, mult in [("GC=F", 1.0), ("GLD", 10.0), ("IAU", 50.0)]:
        try:
            raw = yf.download(sym, period="5d", interval="1d",
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            closes = raw["Close"].dropna() * mult
            if len(closes) < 1:
                continue
            price = float(closes.iloc[-1])
            prev  = float(closes.iloc[-2]) if len(closes) >= 2 else price
            chg   = price - prev
            chg_p = chg / prev * 100
            ts    = closes.index[-1].strftime("%Y-%m-%d")
            return {
                "price": price, "change": chg, "change_pct": chg_p,
                "date": ts, "source": sym, "ok": True,
            }
        except Exception:
            continue

    return {"price": 0, "change": 0, "change_pct": 0, "date": "—", "source": "—", "ok": False}


@st.cache_data(ttl=3600, show_spinner="Fetching XAU/USD history...")
def cached_fetch(data_days: int) -> pd.DataFrame:
    cfg = SimConfig(data_days=data_days)
    return fetch_historical_data(cfg)


@st.cache_data(ttl=300, show_spinner="Running simulation...")
def cached_run(initial_capital, transaction_cost, rsi_buy, rsi_sell,
               ma_fast, ma_slow, ma_trend):
    cfg = SimConfig(
        initial_capital    = initial_capital,
        transaction_cost   = transaction_cost,
        rsi_buy_threshold  = float(rsi_buy),
        rsi_sell_threshold = float(rsi_sell),
        ma_fast = ma_fast, ma_slow = ma_slow, ma_trend = ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    df     = compute_indicators(df_raw.copy(), cfg)
    tlog, ecurve = run_simulation(df, cfg)
    metrics      = calculate_metrics(tlog, ecurve, df, cfg)
    return cfg, df_raw, df, tlog, ecurve, metrics


@st.cache_data(ttl=300, show_spinner="Parameter sweep...")
def cached_sweep(initial_capital, transaction_cost, rsi_buy, rsi_sell,
                 ma_fast, ma_slow, ma_trend):
    cfg = SimConfig(
        initial_capital=initial_capital, transaction_cost=transaction_cost,
        rsi_buy_threshold=float(rsi_buy), rsi_sell_threshold=float(rsi_sell),
        ma_fast=ma_fast, ma_slow=ma_slow, ma_trend=ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    return run_parameter_sweep(df_raw, cfg)


@st.cache_data(ttl=300, show_spinner="Walk-forward test...")
def cached_wf(initial_capital, transaction_cost, rsi_buy, rsi_sell,
              ma_fast, ma_slow, ma_trend):
    cfg = SimConfig(
        initial_capital=initial_capital, transaction_cost=transaction_cost,
        rsi_buy_threshold=float(rsi_buy), rsi_sell_threshold=float(rsi_sell),
        ma_fast=ma_fast, ma_slow=ma_slow, ma_trend=ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    return run_walk_forward(df_raw, cfg)


@st.cache_data(ttl=300, show_spinner="Monte Carlo (1000 runs)...")
def cached_mc(initial_capital, transaction_cost, rsi_buy, rsi_sell,
              ma_fast, ma_slow, ma_trend):
    cfg = SimConfig(
        initial_capital=initial_capital, transaction_cost=transaction_cost,
        rsi_buy_threshold=float(rsi_buy), rsi_sell_threshold=float(rsi_sell),
        ma_fast=ma_fast, ma_slow=ma_slow, ma_trend=ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    df     = compute_indicators(df_raw.copy(), cfg)
    tlog, _ = run_simulation(df, cfg)
    return run_monte_carlo(tlog, cfg)


def _ecurve_value_at(ecurve: list[dict], target_date) -> float | None:
    """หาค่า equity curve ที่ตรงกับ target_date (หรือใกล้ที่สุด)"""
    target = pd.Timestamp(target_date)
    best_val, best_diff = None, float("inf")
    for e in ecurve:
        diff = abs((pd.Timestamp(e["date"]) - target).total_seconds())
        if diff < best_diff:
            best_diff, best_val = diff, e["value"]
    return best_val


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🥇 Gold Simulator")
    st.caption("All-in backtest • XAU/USD")
    st.divider()

    st.subheader("💰 Capital")
    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=10.0, max_value=1_000_000.0,
        value=100.0, step=10.0, format="%.2f",
    )
    transaction_cost = st.slider(
        "Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05
    ) / 100

    st.divider()
    st.subheader("📊 Signal Parameters")
    c1, c2 = st.columns(2)
    rsi_buy  = c1.number_input("RSI Buy <",  10, 49, 35)
    rsi_sell = c2.number_input("RSI Sell >", 51, 90, 65)
    c3, c4 = st.columns(2)
    ma_fast  = c3.selectbox("MA Fast",  [5, 10, 20, 50],       index=2)
    ma_slow  = c4.selectbox("MA Slow",  [50, 100, 150, 200],    index=1)
    ma_trend = st.selectbox("MA Trend", [100, 150, 200, 250],   index=2)

    st.divider()
    st.subheader("🔬 Validation")
    run_sweep = st.checkbox("Parameter Sweep",        value=True)
    run_wf    = st.checkbox("Walk-forward Test",       value=True)
    run_mc    = st.checkbox("Monte Carlo (1000 runs)", value=True)

    st.divider()
    run_btn      = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    refresh_live = st.button("🔄 Refresh Live Price", use_container_width=True)

    if refresh_live:
        st.cache_data.clear()
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER — LIVE PRICE
# ═════════════════════════════════════════════════════════════════════════════

st.title("🥇 Gold Trading Simulator")

# ── Main navigation tabs ──────────────────────────────────────────────────────
tab_live, tab_sim = st.tabs(["📡 Live Monitor", "🔬 Backtest Simulation"])

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE MONITOR
# ═════════════════════════════════════════════════════════════════════════════

with tab_live:
    # ── Auto-refresh countdown ────────────────────────────────────────────────
    lm_col1, lm_col2, lm_col3 = st.columns([3, 1, 1])
    with lm_col1:
        st.subheader("📡 Live Signal Monitor")
        st.caption("ตรวจสัญญาณ XAU/USD แบบ real-time | อ่านจาก trades.json ที่ live_monitor.py อัปเดต")
    with lm_col2:
        auto_refresh = st.toggle("Auto-refresh", value=False)
    with lm_col3:
        refresh_interval = st.selectbox("ทุก (วิ)", [30, 60, 120, 300], index=1,
                                        label_visibility="collapsed")

    if auto_refresh:
        st.info(f"🔄 Auto-refresh ทุก {refresh_interval} วินาที — หน้าจะรีโหลดอัตโนมัติ")
        time_ph = st.empty()
        import time as _time
        _time.sleep(refresh_interval)
        st.rerun()

    # ── Load trades.json ──────────────────────────────────────────────────────
    @st.cache_data(ttl=10)
    def load_trades_json() -> dict | None:
        if TRADES_FILE.exists():
            with open(TRADES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    if st.button("🔄 Refresh", key="refresh_live_monitor"):
        st.cache_data.clear()
        st.rerun()

    trade_state = load_trades_json()

    if trade_state is None:
        st.warning(
            "ยังไม่มีข้อมูล `trades.json` — รัน live_monitor.py ก่อน:\n\n"
            "```bash\npython live_monitor.py\n```"
        )
    else:
        # ── Current position card ──────────────────────────────────────────────
        lp = trade_state.get("last_price") or 0.0
        pos = trade_state.get("state", "CASH")
        cash = trade_state.get("cash", 0.0)
        oz   = trade_state.get("oz_held", 0.0)
        ep   = trade_state.get("entry_price") or 0.0
        pv   = cash if pos == "CASH" else oz * lp
        initial_cap = 100.0  # ใช้ค่า default (trades.json ไม่เก็บ initial_capital)
        ret  = (pv - initial_cap) / initial_cap * 100 if initial_cap else 0

        pos_color = "#a6e3a1" if pos == "GOLD" else "#89b4fa"
        pos_icon  = "🟡" if pos == "GOLD" else "💵"

        st.markdown(f"""
        <div style="background:#1e1e2e;border:1px solid {pos_color}44;
                    border-radius:12px;padding:20px 28px;margin-bottom:16px">
          <span style="font-size:1.5em;font-weight:700;color:{pos_color}">
            {pos_icon} Position: {pos}
          </span><br>
          <span style="color:#cdd6f4;font-size:0.95em">
            Portfolio: <b>${pv:.2f}</b> &nbsp;|&nbsp;
            Return: <b style="color:{'#a6e3a1' if ret>=0 else '#f38ba8'}">{ret:+.2f}%</b>
            &nbsp;|&nbsp; Last price: <b>${lp:,.2f}</b>
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Position detail ────────────────────────────────────────────────────
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("State",       pos)
        pc2.metric("Cash",        f"${cash:.2f}")
        pc3.metric("Portfolio",   f"${pv:.2f}", f"{ret:+.2f}%")
        if pos == "GOLD" and ep and lp:
            unreal = (lp - ep) / ep * 100
            pc4.metric("Unrealized P&L",
                       f"${(lp - ep)*oz:.2f}",
                       f"{unreal:+.2f}% vs entry ${ep:,.2f}",
                       delta_color="normal" if unreal >= 0 else "inverse")
        else:
            pc4.metric("Oz Held", f"{oz:.5f}")

        # ── Last signal status ─────────────────────────────────────────────────
        last_sig   = trade_state.get("last_signal", "—")
        last_check = trade_state.get("last_checked", "—")
        last_rsi   = trade_state.get("last_rsi")

        sig_colors = {"BUY": "#a6e3a1", "SELL": "#f38ba8",
                      "HOLD": "#cdd6f4", "ERROR": "#fab387", "WARM-UP": "#6c7086"}
        sig_col = sig_colors.get(last_sig, "#cdd6f4")

        st.markdown(f"""
        <div style="background:#181825;border-radius:8px;padding:12px 20px;
                    border-left:4px solid {sig_col};margin:8px 0">
          <b style="color:{sig_col};font-size:1.1em">{last_sig}</b>
          &nbsp;&nbsp;<span style="color:#6c7086;font-size:0.85em">
            Last checked: {last_check} &nbsp;|&nbsp;
            RSI: {f"{last_rsi:.1f}" if last_rsi is not None else "—"}
          </span>
        </div>
        """, unsafe_allow_html=True)

        # ── Trade history table ────────────────────────────────────────────────
        st.divider()
        trades = trade_state.get("trades", [])
        if trades:
            st.subheader(f"📋 Trade History ({len(trades)} entries)")

            rows = []
            for t in reversed(trades):   # ล่าสุดขึ้นก่อน
                dt_s  = str(t.get("date", ""))[:19].replace("T", " ")
                pnl_u = f"{t['pnl_usd']:+.2f}" if t.get("pnl_usd") is not None else "—"
                pnl_p = f"{t['pnl_pct']:+.2f}%" if t.get("pnl_pct") is not None else "—"
                dur   = f"{t['duration']}d" if t.get("duration") is not None else "—"
                rows.append({
                    "#":         t.get("id", ""),
                    "Date":      dt_s,
                    "Action":    t["action"],
                    "Price ($)": f"${t['price']:,.2f}",
                    "Oz":        f"{t.get('oz', 0):.5f}",
                    "P&L ($)":   pnl_u,
                    "P&L (%)":   pnl_p,
                    "Duration":  dur,
                    "Balance":   f"${t.get('balance', 0):,.2f}",
                    "Reason":    t.get("reason", ""),
                })

            hist_df = pd.DataFrame(rows)

            def _live_row_style(row):
                act = row["Action"]
                pnl = str(row.get("P&L ($)", ""))
                if act == "BUY":
                    return ["background-color:rgba(166,227,161,0.07)"] * len(row)
                elif act == "SELL":
                    c = "rgba(166,227,161,0.13)" if "+" in pnl else "rgba(243,139,168,0.13)"
                    return [f"background-color:{c}"] * len(row)
                return [""] * len(row)

            st.dataframe(
                hist_df.style.apply(_live_row_style, axis=1),
                use_container_width=True, hide_index=True,
            )

            # ── Stats summary ──────────────────────────────────────────────────
            sells   = [t for t in trades if t["action"] == "SELL"]
            wins    = sum(1 for t in sells if (t.get("pnl_usd") or 0) > 0)
            total_p = sum(t.get("pnl_usd") or 0 for t in sells)
            n_s     = len(sells)
            avg_dur = (sum(t.get("duration") or 0 for t in sells) / n_s) if n_s else 0

            if sells:
                st.divider()
                st.subheader("📊 Live Trading Stats")
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Closed Trades",  n_s)
                sc2.metric("Win Rate",       f"{wins/n_s*100:.1f}%",  f"{wins}W / {n_s-wins}L")
                sc3.metric("Total P&L",      f"${total_p:+.2f}")
                sc4.metric("Avg Duration",   f"{avg_dur:.1f} days")

                # P&L equity curve from live trades
                if n_s >= 2:
                    cum_bal = []
                    bal = 100.0
                    for t in [x for x in trades if x["action"] in ("SELL",)]:
                        bal += t.get("pnl_usd") or 0
                        cum_bal.append({
                            "date":  str(t["date"])[:10],
                            "value": bal,
                            "pnl":   t.get("pnl_usd", 0),
                        })
                    if cum_bal:
                        eq_df = pd.DataFrame(cum_bal)
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(
                            x=eq_df["date"], y=eq_df["value"],
                            mode="lines+markers", name="Balance",
                            line=dict(color="#f9e2af", width=2),
                            marker=dict(size=8,
                                        color=["#a6e3a1" if v >= 0 else "#f38ba8"
                                               for v in eq_df["pnl"]]),
                        ))
                        fig_eq.add_hline(y=100, line_dash="dash",
                                         line_color="rgba(255,255,255,0.2)")
                        fig_eq.update_layout(
                            title="Live Portfolio Balance ($)",
                            height=260, template="plotly_dark",
                            paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                            margin=dict(t=40, b=10),
                            yaxis=dict(gridcolor="#313244"),
                            xaxis=dict(gridcolor="#313244"),
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("ยังไม่มี trade ถูกบันทึก — รอ BUY signal แรก")

        # ── How to run ─────────────────────────────────────────────────────────
        with st.expander("🖥️ วิธีรัน live_monitor.py", expanded=False):
            st.markdown("""
**รันต่อเนื่อง** (loop ไม่หยุด ตรวจทุก 60 นาที):
```bash
python live_monitor.py
```

**รันครั้งเดียว** (สำหรับ cron / GitHub Actions):
```bash
python live_monitor.py --once
```

**ดู position ปัจจุบัน**:
```bash
python live_monitor.py --status
```

**ปรับ interval และ RSI ผ่าน env vars**:
```bash
CHECK_INTERVAL_MIN=30 RSI_BUY=30 RSI_SELL=70 python live_monitor.py
```

**Dry run** (ไม่ส่ง Telegram จริง):
```bash
DRY_RUN=true python live_monitor.py --once
```
            """)

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BACKTEST SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

with tab_sim:

    # ── Live price bar ────────────────────────────────────────────────────────
    live = get_live_price()
    lc1, lc2, lc3, lc4 = st.columns([2, 1, 1, 2])
    if live["ok"]:
        lc1.metric(
            "XAU/USD (latest close)",
            f"${live['price']:,.2f}",
            f"{live['change']:+.2f} ({live['change_pct']:+.2f}%)",
            delta_color="normal" if live["change"] >= 0 else "inverse",
        )
        lc2.metric("Date",   live["date"])
        lc3.metric("Source", live.get("source", "yfinance"))
    else:
        lc1.warning("Live price unavailable")
    lc4.caption(f"Cache 60s · {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    # ── Load / run simulation ─────────────────────────────────────────────────
    if "results_loaded" not in st.session_state:
        st.session_state["results_loaded"] = False

    if run_btn or not st.session_state["results_loaded"]:
        try:
            cfg, df_raw, df, tlog, ecurve, metrics = cached_run(
                initial_capital, transaction_cost,
                rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
            )
            st.session_state.update({
                "cfg": cfg, "df_raw": df_raw, "df": df,
                "tlog": tlog, "ecurve": ecurve, "metrics": metrics,
                "results_loaded": True,
                "sweep_df": None, "wf_results": None, "mc": None,
            })
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

    cfg     = st.session_state["cfg"]
    df_raw  = st.session_state["df_raw"]
    df      = st.session_state["df"]
    tlog    = st.session_state["tlog"]
    ecurve  = st.session_state["ecurve"]
    metrics = st.session_state["metrics"]

    # ── KEY METRICS ───────────────────────────────────────────────────────────
    st.subheader("📈 Performance Summary")
    ret_pct  = metrics.get("total_return_pct", 0)
    bh_pct   = metrics.get("buy_hold_return_pct", 0)
    alpha    = metrics.get("alpha_pct", 0)
    win_rate = metrics.get("win_rate_pct", 0)
    sharpe   = metrics.get("sharpe_ratio", 0)
    max_dd   = metrics.get("max_drawdown_pct", 0)
    final_b  = metrics.get("final_balance", initial_capital)
    n_wins   = metrics.get("num_wins", 0)
    n_loss   = metrics.get("num_losses", 0)
    n_closed = metrics.get("num_closed", 0)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Final Balance",  f"${final_b:.2f}",   f"{ret_pct:+.2f}%")
    m2.metric("Total Return",   f"{ret_pct:+.2f}%",  f"B&H: {bh_pct:+.1f}%")
    m3.metric("Alpha",          f"{alpha:+.2f}%",    "vs buy-and-hold")
    m4.metric("Win Rate",       f"{win_rate:.1f}%",  f"{n_wins}W / {n_loss}L")
    m5.metric("Sharpe Ratio",   f"{sharpe:.2f}",     "annualised")
    m6.metric("Max Drawdown",   f"-{max_dd:.1f}%",   f"{n_closed} closed trades")

    if metrics.get("low_trade_warning"):
        st.warning("⚠️ Less than 5 closed trades — lower RSI thresholds to generate more signals.")

    # ── EQUITY CURVE ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💹 Portfolio Equity Curve")

    if ecurve:
        ec_df = pd.DataFrame(ecurve)
        ec_df["date"] = pd.to_datetime(ec_df["date"])
        sim_df    = df.iloc[cfg.lookback_required:].copy()
        bh_prices = sim_df["Close"].values
        bh_norm   = bh_prices / bh_prices[0] * initial_capital if len(bh_prices) > 0 else []
        ec_lookup = {pd.Timestamp(e["date"]): e["value"] for e in ecurve}

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.04,
            subplot_titles=("Portfolio Value vs Buy-and-Hold ($)", "XAU/USD Price"),
        )
        fig.add_trace(go.Scatter(
            x=ec_df["date"], y=ec_df["value"], name="Strategy",
            line=dict(color="#f9e2af", width=2),
            fill="tozeroy", fillcolor="rgba(249,226,175,0.06)",
            hovertemplate="$%{y:.2f}<extra>Strategy</extra>",
        ), row=1, col=1)
        if len(bh_norm) > 0:
            fig.add_trace(go.Scatter(
                x=sim_df.index[:len(bh_norm)], y=bh_norm, name="Buy & Hold",
                line=dict(color="#6c7086", width=1.5, dash="dot"),
                hovertemplate="$%{y:.2f}<extra>Buy & Hold</extra>",
            ), row=1, col=1)
        fig.add_hline(y=initial_capital, line_dash="dash",
                      line_color="rgba(255,255,255,0.2)", row=1, col=1)

        buy_trades  = [t for t in tlog if t["action"] == "BUY"]
        sell_trades = [t for t in tlog if t["action"] in ("SELL", "OPEN")]
        if buy_trades:
            bx = [pd.Timestamp(t["date"]) for t in buy_trades]
            by = [ec_lookup.get(pd.Timestamp(t["date"]),
                                _ecurve_value_at(ecurve, t["date"])) for t in buy_trades]
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode="markers", name="BUY",
                marker=dict(symbol="triangle-up", size=13, color="#a6e3a1",
                            line=dict(width=1, color="#1e1e2e")),
                hovertemplate="BUY @ $%{y:.2f}<extra></extra>",
            ), row=1, col=1)
        if sell_trades:
            sx = [pd.Timestamp(t["date"]) for t in sell_trades]
            sy = [ec_lookup.get(pd.Timestamp(t["date"]),
                                _ecurve_value_at(ecurve, t["date"])) for t in sell_trades]
            sc = ["#a6e3a1" if (t.get("pnl_usd") or 0) >= 0 else "#f38ba8" for t in sell_trades]
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode="markers", name="SELL",
                marker=dict(symbol="triangle-down", size=13, color=sc,
                            line=dict(width=1, color="#1e1e2e")),
                hovertemplate="SELL @ $%{y:.2f}<extra></extra>",
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="XAU/USD",
            line=dict(color="#cdd6f4", width=1),
            hovertemplate="$%{y:,.2f}<extra>XAU/USD</extra>",
        ), row=2, col=1)

        rsi_clean = df[df["RSI"].notna()]
        for zone_df, zname, zcol in [
            (rsi_clean[rsi_clean["RSI"] < cfg.rsi_buy_threshold],
             f"RSI<{cfg.rsi_buy_threshold}", "#a6e3a1"),
            (rsi_clean[rsi_clean["RSI"] > cfg.rsi_sell_threshold],
             f"RSI>{cfg.rsi_sell_threshold}", "#f38ba8"),
        ]:
            if not zone_df.empty:
                fig.add_trace(go.Scatter(
                    x=zone_df.index, y=zone_df["Close"], mode="markers", name=zname,
                    marker=dict(size=5, color=zcol, opacity=0.6),
                ), row=2, col=1)

        fig.update_layout(
            height=580, template="plotly_dark",
            paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
            legend=dict(orientation="h", y=-0.08, x=0),
            margin=dict(t=40, b=10, l=60, r=20), hovermode="x unified",
        )
        fig.update_yaxes(title_text="USD ($)",  row=1, col=1, gridcolor="#313244")
        fig.update_yaxes(title_text="XAU/USD", row=2, col=1, gridcolor="#313244")
        fig.update_xaxes(gridcolor="#313244")
        st.plotly_chart(fig, use_container_width=True)

    # ── RSI + MA ──────────────────────────────────────────────────────────────
    with st.expander("📉 RSI & Moving Averages", expanded=False):
        tab_rsi, tab_ma = st.tabs(["RSI(14)", "Moving Averages"])
        with tab_rsi:
            rsi_data = df[df["RSI"].notna()]
            fig_rsi  = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data["RSI"],
                                         name="RSI(14)", line=dict(color="#cba6f7", width=1.5)))
            fig_rsi.add_hrect(y0=0, y1=cfg.rsi_buy_threshold,
                              fillcolor="rgba(166,227,161,0.08)", line_width=0)
            fig_rsi.add_hrect(y0=cfg.rsi_sell_threshold, y1=100,
                              fillcolor="rgba(243,139,168,0.08)", line_width=0)
            fig_rsi.add_hline(y=cfg.rsi_buy_threshold, line_dash="dash", line_color="#a6e3a1",
                              annotation_text=f"Buy < {cfg.rsi_buy_threshold}")
            fig_rsi.add_hline(y=cfg.rsi_sell_threshold, line_dash="dash", line_color="#f38ba8",
                              annotation_text=f"Sell > {cfg.rsi_sell_threshold}")
            fig_rsi.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)")
            fig_rsi.update_layout(height=260, template="plotly_dark",
                                  paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                                  margin=dict(t=10, b=10),
                                  yaxis=dict(range=[0, 100], gridcolor="#313244"),
                                  xaxis=dict(gridcolor="#313244"))
            st.plotly_chart(fig_rsi, use_container_width=True)
        with tab_ma:
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=df.index, y=df["Close"], name="XAU/USD",
                                        line=dict(color="#cdd6f4", width=1), opacity=0.7))
            for ma_col, ma_color in [(f"MA{cfg.ma_fast}", "#f9e2af"),
                                     (f"MA{cfg.ma_slow}", "#89b4fa"),
                                     (f"MA{cfg.ma_trend}", "#f38ba8")]:
                if ma_col in df.columns:
                    fig_ma.add_trace(go.Scatter(x=df.index, y=df[ma_col], name=ma_col,
                                                line=dict(color=ma_color, width=1.5)))
            fig_ma.update_layout(height=260, template="plotly_dark",
                                 paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                                 margin=dict(t=10, b=10),
                                 yaxis=dict(gridcolor="#313244"), xaxis=dict(gridcolor="#313244"))
            st.plotly_chart(fig_ma, use_container_width=True)

    # ── TRADE LOG ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Trade Log")
    if tlog:
        rows = []
        for t in tlog:
            dt_s  = pd.Timestamp(t["date"]).strftime("%Y-%m-%d")
            pnl_u = f"{t['pnl_usd']:+.2f}" if t["pnl_usd"] is not None else "—"
            pnl_p = f"{t['pnl_pct']:+.2f}%" if t["pnl_pct"] is not None else "—"
            dur   = f"{t.get('duration','—')}d" if t.get("duration") is not None else "—"
            rows.append({"Date": dt_s, "Action": t["action"],
                         "Price ($)": f"${t['price']:,.2f}", "Oz": f"{t['oz']:.5f}",
                         "P&L ($)": pnl_u, "P&L (%)": pnl_p, "Duration": dur,
                         "Balance ($)": f"${t['balance']:,.2f}"})
        tlog_df = pd.DataFrame(rows)

        def _row_style(row):
            act = row["Action"]; pnl = str(row.get("P&L ($)", ""))
            if act == "BUY":   return ["background-color:rgba(166,227,161,0.07)"] * len(row)
            if act == "SELL":
                c = "rgba(166,227,161,0.13)" if "+" in pnl else "rgba(243,139,168,0.13)"
                return [f"background-color:{c}"] * len(row)
            if act == "OPEN":  return ["background-color:rgba(249,226,175,0.10)"] * len(row)
            return [""] * len(row)

        st.dataframe(tlog_df.style.apply(_row_style, axis=1),
                     use_container_width=True, hide_index=True)

        sell_only = [t for t in tlog if t["action"] in ("SELL","OPEN") and t["pnl_usd"] is not None]
        if sell_only:
            pnl_vals   = [t["pnl_usd"] for t in sell_only]
            pnl_labels = [pd.Timestamp(t["date"]).strftime("%Y-%m-%d") for t in sell_only]
            fig_pnl = go.Figure(go.Bar(
                x=pnl_labels, y=pnl_vals,
                marker_color=["#a6e3a1" if v >= 0 else "#f38ba8" for v in pnl_vals],
                hovertemplate="$%{y:+.2f}<extra></extra>",
            ))
            fig_pnl.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
            fig_pnl.update_layout(title="P&L per Trade ($)", height=220,
                                  template="plotly_dark", paper_bgcolor="#1e1e2e",
                                  plot_bgcolor="#1e1e2e", margin=dict(t=40, b=20),
                                  showlegend=False,
                                  yaxis=dict(gridcolor="#313244"), xaxis=dict(gridcolor="#313244"))
            st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.info("No trades executed. Try adjusting RSI thresholds.")

    # ── PARAMETER SWEEP ───────────────────────────────────────────────────────
    if run_sweep:
        st.divider()
        st.subheader("🔍 Parameter Sweep")
        sweep_df = cached_sweep(initial_capital, transaction_cost,
                                rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend)
        st.session_state["sweep_df"] = sweep_df
        if not sweep_df.empty:
            sl, sr = st.columns([3, 2])
            with sl:
                st.markdown("**Top 15 combinations (sorted by Sharpe)**")
                disp = sweep_df.head(15).copy().reset_index(drop=True)
                for c in ["RSI_buy","RSI_sell","MA_fast","MA_slow","Trades"]:
                    disp[c] = disp[c].astype(int)
                def _highlight_sweep(row):
                    if row.name == 0: return ["background-color:rgba(249,226,175,0.18)"]*len(row)
                    if row.get("Default","")=="*": return ["background-color:rgba(137,180,250,0.12)"]*len(row)
                    return [""]*len(row)
                st.dataframe(disp.style.apply(_highlight_sweep, axis=1),
                             use_container_width=True, hide_index=True)
            with sr:
                best = sweep_df.iloc[0]
                st.markdown("**Best vs Current**")
                st.metric("Best Sharpe",   f"{float(best['Sharpe']):.2f}",
                          f"{float(best['Sharpe'])-metrics.get('sharpe_ratio',0):+.2f}")
                st.metric("Best Return",   f"{float(best['Return%']):+.2f}%")
                st.metric("Best Win Rate", f"{float(best['WinRate%']):.1f}%")
                if best.get("Default") != "*":
                    st.info(f"💡 Suggest: RSI {int(best['RSI_buy'])}/{int(best['RSI_sell'])}, "
                            f"MA {int(best['MA_fast'])}/{int(best['MA_slow'])}")
                else:
                    st.success("Current config is already optimal ✓")
            pivot = sweep_df.pivot_table(values="Sharpe", index="RSI_buy",
                                         columns="RSI_sell", aggfunc="max")
            if not pivot.empty:
                fig_h = px.imshow(pivot, text_auto=".2f",
                                  labels=dict(x="RSI Sell", y="RSI Buy", color="Sharpe"),
                                  color_continuous_scale="RdYlGn",
                                  title="Sharpe Heatmap (RSI Buy × RSI Sell)")
                fig_h.update_layout(height=320, template="plotly_dark",
                                    paper_bgcolor="#1e1e2e", margin=dict(t=40, b=10))
                st.plotly_chart(fig_h, use_container_width=True)

    # ── WALK-FORWARD ──────────────────────────────────────────────────────────
    wf_results = None
    if run_wf:
        st.divider()
        st.subheader("🔄 Walk-forward Validation")
        wf_results = cached_wf(initial_capital, transaction_cost,
                               rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend)
        st.session_state["wf_results"] = wf_results
        if wf_results:
            wf_passes = 0
            wf_cols = st.columns(len(wf_results))
            for col, r in zip(wf_cols, wf_results):
                if r["passed"]: wf_passes += 1
                col.metric(f"Period {r['split']}",
                           f"{r['return_pct']:+.1f}%", f"Win: {r['win_rate']:.0f}%",
                           delta_color="normal" if r["return_pct"] >= 0 else "inverse")
                col.caption(f"{r['period_start']} → {r['period_end']}")
                col.write("✅ PASS" if r["passed"] else "❌ FAIL")
            wf_ok = wf_passes >= max(1, len(wf_results) * 2 // 3)
            (st.success if wf_ok else st.warning)(
                f"Walk-forward: **{wf_passes}/{len(wf_results)}** periods profitable"
                + (" — robust ✓" if wf_ok else " — may overfit ✗"))

    # ── MONTE CARLO ───────────────────────────────────────────────────────────
    mc = None
    if run_mc:
        st.divider()
        st.subheader("🎲 Monte Carlo Simulation")
        mc = cached_mc(initial_capital, transaction_cost,
                       rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend)
        st.session_state["mc"] = mc
        if mc.get("n_trades", 0) >= 2:
            mc1, mc2 = st.columns([2, 1])
            with mc1:
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(x=mc["distribution"], nbinsx=40,
                                              marker_color="rgba(137,180,250,0.55)",
                                              histnorm="probability density", name="Random"))
                fig_mc.add_vline(x=mc["actual_return_pct"], line_dash="dash",
                                 line_color="#f9e2af", line_width=2,
                                 annotation_text=f"Actual: {mc['actual_return_pct']:+.1f}%",
                                 annotation_position="top right")
                fig_mc.update_layout(title=f"Return Distribution ({mc['n_simulations']} runs)",
                                     height=300, template="plotly_dark",
                                     paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                                     xaxis_title="Return (%)", yaxis_title="Density",
                                     margin=dict(t=40, b=10),
                                     yaxis=dict(gridcolor="#313244"), xaxis=dict(gridcolor="#313244"))
                st.plotly_chart(fig_mc, use_container_width=True)
            with mc2:
                pv = mc["p_value"]
                st.metric("p-value", f"{pv:.3f}",
                          "Skill likely ✓" if pv < 0.1 else "May be luck ✗",
                          delta_color="normal" if pv < 0.1 else "inverse")
                st.metric("Percentile", f"{mc['percentile']:.1f}th", "vs 1000 random runs")
                st.metric("Actual Return", f"{mc['actual_return_pct']:+.1f}%")
                st.metric("Avg Random",    f"{mc['mean_random_return']:+.1f}%")
                (st.success if mc["passed"] else st.warning)(
                    "Skill-driven (p < 0.10)" if mc["passed"] else "Cannot rule out luck")
        else:
            st.info("Not enough closed trades for Monte Carlo (need ≥ 2).")

    # ── VALIDATION REPORT ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("✅ Strategy Validation Report")
    bench_pass = (metrics.get("sharpe_ratio",0) >= 0.5 and
                  metrics.get("win_rate_pct",0) >= 50 and
                  metrics.get("alpha_pct",-999) >= 0)
    wf_r = wf_results or st.session_state.get("wf_results")
    if wf_r:
        wp = sum(1 for r in wf_r if r.get("passed")); wt = len(wf_r)
        wf_ok_r = wp >= max(1, wt*2//3); wf_det = f"{wp}/{wt} periods positive"
    else:
        wf_ok_r, wf_det = None, "Enable Walk-forward above"
    mc_r = mc or st.session_state.get("mc")
    if mc_r and mc_r.get("n_trades",0) >= 2:
        mc_ok_r = mc_r.get("passed",False); mc_det = f"p-value = {mc_r.get('p_value',1):.3f}"
    else:
        mc_ok_r, mc_det = None, "Enable Monte Carlo above"
    sw_r = st.session_state.get("sweep_df")
    if sw_r is not None and not sw_r.empty:
        br = sw_r.iloc[0]
        sw_ok_r = True
        sw_det = (f"Best: RSI {int(br['RSI_buy'])}/{int(br['RSI_sell'])}, "
                  f"MA {int(br['MA_fast'])}/{int(br['MA_slow'])} (Sharpe={float(br['Sharpe']):.2f})")
    else:
        sw_ok_r, sw_det = None, "Enable Parameter Sweep above"

    checks = [
        ("Benchmark\nMetrics", bench_pass,
         f"Sharpe {sharpe:.2f} · WinRate {win_rate:.0f}% · Alpha {alpha:+.1f}%"),
        ("Parameter\nSweep",   sw_ok_r, sw_det),
        ("Walk-forward\nTest", wf_ok_r, wf_det),
        ("Monte Carlo",        mc_ok_r, mc_det),
    ]
    vc = st.columns(4)
    for col, (label, passed, detail) in zip(vc, checks):
        with col:
            if passed is True:   st.success(f"✅ {label}")
            elif passed is False: st.error(f"❌ {label}")
            else:                 st.info(f"ℹ️ {label}")
            st.caption(detail)

    definite = [(p, l) for l, p, _ in checks if p is not None]
    n_p = sum(1 for p, _ in definite if p); n_t = len(definite)
    if n_t > 0:
        ratio = n_p / n_t
        if ratio == 1.0:   st.success("🏆 **STRATEGY CONFIDENCE: HIGH** — All checks passed")
        elif ratio >= 0.5: st.warning(f"⚠️ **STRATEGY CONFIDENCE: MEDIUM** — {n_p}/{n_t} passed")
        else:              st.error(f"🚨 **STRATEGY CONFIDENCE: LOW** — {n_p}/{n_t} passed")

    # ── FOOTER ────────────────────────────────────────────────────────────────
    st.divider()
    f1, f2, f3 = st.columns(3)
    sim_s = metrics.get("sim_start"); sim_e = metrics.get("sim_end")
    f1.caption(f"📅 {pd.Timestamp(sim_s).strftime('%Y-%m-%d') if sim_s else '—'} → "
               f"{pd.Timestamp(sim_e).strftime('%Y-%m-%d') if sim_e else '—'}")
    f2.caption(f"📊 {len(df_raw)} trading days · {cfg.lookback_required}-day warmup")
    f3.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

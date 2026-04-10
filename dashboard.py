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
from datetime import datetime
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
    """ดึงราคา XAU/USD ล่าสุด — cache 60 วินาที"""
    try:
        from gold_simulation import _stooq_fetch, SimConfig
        df = _stooq_fetch("xauusd", days=5)
        price = float(df["Close"].iloc[-1])
        prev  = float(df["Close"].iloc[-2]) if len(df) >= 2 else price
        chg   = price - prev
        chg_p = chg / prev * 100
        ts    = df.index[-1].strftime("%Y-%m-%d")
        return {"price": price, "change": chg, "change_pct": chg_p, "date": ts, "ok": True}
    except Exception:
        return {"price": 0, "change": 0, "change_pct": 0, "date": "—", "ok": False}


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

live = get_live_price()
lc1, lc2, lc3, lc4 = st.columns([2, 1, 1, 2])
if live["ok"]:
    arrow = "▲" if live["change"] >= 0 else "▼"
    color = "green" if live["change"] >= 0 else "red"
    lc1.metric(
        "XAU/USD (latest close)",
        f"${live['price']:,.2f}",
        f"{arrow} {live['change']:+.2f} ({live['change_pct']:+.2f}%)",
        delta_color="normal" if live["change"] >= 0 else "inverse",
    )
    lc2.metric("Date", live["date"])
else:
    lc1.warning("Live price unavailable")

lc4.caption(f"Cache refreshes every 60s · Last checked: {datetime.now().strftime('%H:%M:%S')}")

st.divider()


# ═════════════════════════════════════════════════════════════════════════════
#  LOAD / RUN SIMULATION
# ═════════════════════════════════════════════════════════════════════════════

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
            # clear cached validation results when params change
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


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — KEY METRICS
# ═════════════════════════════════════════════════════════════════════════════

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
    st.warning("⚠️ Less than 5 closed trades — statistical metrics may be unreliable. "
               "Try lowering RSI thresholds to generate more signals.")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EQUITY CURVE
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("💹 Portfolio Equity Curve")

if ecurve:
    ec_df = pd.DataFrame(ecurve)
    ec_df["date"] = pd.to_datetime(ec_df["date"])

    # Buy-and-hold normalized to initial_capital
    sim_df    = df.iloc[cfg.lookback_required:].copy()
    bh_prices = sim_df["Close"].values
    bh_norm   = bh_prices / bh_prices[0] * initial_capital if len(bh_prices) > 0 else []

    # Equity curve lookup dict for marker positions (correct Y values)
    ec_lookup = {pd.Timestamp(e["date"]): e["value"] for e in ecurve}

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.04,
        subplot_titles=("Portfolio Value vs Buy-and-Hold ($)", "XAU/USD Price"),
    )

    # ── Equity curve ──
    fig.add_trace(go.Scatter(
        x=ec_df["date"], y=ec_df["value"],
        name="Strategy", line=dict(color="#f9e2af", width=2),
        fill="tozeroy", fillcolor="rgba(249,226,175,0.06)",
        hovertemplate="$%{y:.2f}<extra>Strategy</extra>",
    ), row=1, col=1)

    # ── Buy-and-hold ──
    if len(bh_norm) > 0:
        fig.add_trace(go.Scatter(
            x=sim_df.index[:len(bh_norm)], y=bh_norm,
            name="Buy & Hold", line=dict(color="#6c7086", width=1.5, dash="dot"),
            hovertemplate="$%{y:.2f}<extra>Buy & Hold</extra>",
        ), row=1, col=1)

    # ── Capital baseline ──
    fig.add_hline(y=initial_capital, line_dash="dash",
                  line_color="rgba(255,255,255,0.2)", row=1, col=1)

    # ── BUY markers — lookup actual equity curve value at that date ──
    buy_trades  = [t for t in tlog if t["action"] == "BUY"]
    sell_trades = [t for t in tlog if t["action"] in ("SELL", "OPEN")]

    if buy_trades:
        bx = [pd.Timestamp(t["date"]) for t in buy_trades]
        by = [ec_lookup.get(pd.Timestamp(t["date"]), _ecurve_value_at(ecurve, t["date"]))
              for t in buy_trades]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=13, color="#a6e3a1",
                        line=dict(width=1, color="#1e1e2e")),
            hovertemplate="BUY @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    if sell_trades:
        sx = [pd.Timestamp(t["date"]) for t in sell_trades]
        sy = [ec_lookup.get(pd.Timestamp(t["date"]), _ecurve_value_at(ecurve, t["date"]))
              for t in sell_trades]
        colors = ["#a6e3a1" if (t.get("pnl_usd") or 0) >= 0 else "#f38ba8"
                  for t in sell_trades]
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=13, color=colors,
                        line=dict(width=1, color="#1e1e2e")),
            hovertemplate="SELL @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Gold price (row 2) ──
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="XAU/USD",
        line=dict(color="#cdd6f4", width=1),
        hovertemplate="$%{y:,.2f}<extra>XAU/USD</extra>",
    ), row=2, col=1)

    # RSI signal dots on price chart
    rsi_clean = df[df["RSI"].notna()]
    buy_zone  = rsi_clean[rsi_clean["RSI"] < cfg.rsi_buy_threshold]
    sell_zone = rsi_clean[rsi_clean["RSI"] > cfg.rsi_sell_threshold]

    if not buy_zone.empty:
        fig.add_trace(go.Scatter(
            x=buy_zone.index, y=buy_zone["Close"], mode="markers",
            name=f"RSI<{cfg.rsi_buy_threshold}",
            marker=dict(size=5, color="#a6e3a1", opacity=0.6),
            hovertemplate="RSI Oversold<extra></extra>",
        ), row=2, col=1)

    if not sell_zone.empty:
        fig.add_trace(go.Scatter(
            x=sell_zone.index, y=sell_zone["Close"], mode="markers",
            name=f"RSI>{cfg.rsi_sell_threshold}",
            marker=dict(size=5, color="#f38ba8", opacity=0.6),
            hovertemplate="RSI Overbought<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        height=580, template="plotly_dark",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        legend=dict(orientation="h", y=-0.08, x=0),
        margin=dict(t=40, b=10, l=60, r=20),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="USD ($)",  row=1, col=1, gridcolor="#313244")
    fig.update_yaxes(title_text="XAU/USD", row=2, col=1, gridcolor="#313244")
    fig.update_xaxes(gridcolor="#313244")

    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — RSI + MA CHART
# ═════════════════════════════════════════════════════════════════════════════

with st.expander("📉 RSI & Moving Averages", expanded=False):
    tab_rsi, tab_ma = st.tabs(["RSI(14)", "Moving Averages"])

    with tab_rsi:
        rsi_data = df[df["RSI"].notna()]
        fig_rsi  = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=rsi_data.index, y=rsi_data["RSI"],
            name="RSI(14)", line=dict(color="#cba6f7", width=1.5),
        ))
        # Oversold/Overbought shading
        fig_rsi.add_hrect(y0=0,  y1=cfg.rsi_buy_threshold,
                          fillcolor="rgba(166,227,161,0.08)", line_width=0)
        fig_rsi.add_hrect(y0=cfg.rsi_sell_threshold, y1=100,
                          fillcolor="rgba(243,139,168,0.08)", line_width=0)
        fig_rsi.add_hline(y=cfg.rsi_buy_threshold,  line_dash="dash",
                          line_color="#a6e3a1", annotation_text=f"Buy < {cfg.rsi_buy_threshold}")
        fig_rsi.add_hline(y=cfg.rsi_sell_threshold, line_dash="dash",
                          line_color="#f38ba8", annotation_text=f"Sell > {cfg.rsi_sell_threshold}")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.15)")
        fig_rsi.update_layout(
            height=260, template="plotly_dark", paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e", margin=dict(t=10, b=10),
            yaxis=dict(range=[0, 100], gridcolor="#313244"),
            xaxis=dict(gridcolor="#313244"),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    with tab_ma:
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name="XAU/USD",
            line=dict(color="#cdd6f4", width=1), opacity=0.7,
        ))
        ma_cols = {
            f"MA{cfg.ma_fast}":  "#f9e2af",
            f"MA{cfg.ma_slow}":  "#89b4fa",
            f"MA{cfg.ma_trend}": "#f38ba8",
        }
        for col, color in ma_cols.items():
            if col in df.columns:
                fig_ma.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=col,
                    line=dict(color=color, width=1.5),
                ))
        fig_ma.update_layout(
            height=260, template="plotly_dark", paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e", margin=dict(t=10, b=10),
            yaxis=dict(gridcolor="#313244"), xaxis=dict(gridcolor="#313244"),
        )
        st.plotly_chart(fig_ma, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — TRADE LOG
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("📋 Trade Log")

if tlog:
    # Build display rows
    rows = []
    for t in tlog:
        dt_s  = pd.Timestamp(t["date"]).strftime("%Y-%m-%d")
        pnl_u = f"{t['pnl_usd']:+.2f}" if t["pnl_usd"] is not None else "—"
        pnl_p = f"{t['pnl_pct']:+.2f}%" if t["pnl_pct"] is not None else "—"
        dur   = f"{t.get('duration', '—')} d" if t.get("duration") is not None else "—"
        rows.append({
            "Date":       dt_s,
            "Action":     t["action"],
            "Price ($)":  f"${t['price']:,.2f}",
            "Oz":         f"{t['oz']:.5f}",
            "P&L ($)":    pnl_u,
            "P&L (%)":    pnl_p,
            "Duration":   dur,
            "Balance ($)": f"${t['balance']:,.2f}",
        })

    tlog_df = pd.DataFrame(rows)

    def _row_style(row):
        act = row["Action"]
        pnl = str(row.get("P&L ($)", ""))
        if act == "BUY":
            bg = "background-color: rgba(166,227,161,0.07)"
        elif act == "SELL":
            bg = ("background-color: rgba(166,227,161,0.13)"
                  if "+" in pnl else "background-color: rgba(243,139,168,0.13)")
        elif act == "OPEN":
            bg = "background-color: rgba(249,226,175,0.10)"
        else:
            bg = ""
        return [bg] * len(row)

    st.dataframe(
        tlog_df.style.apply(_row_style, axis=1),
        use_container_width=True, hide_index=True,
    )

    # P&L bar chart
    sell_only = [t for t in tlog if t["action"] in ("SELL", "OPEN") and t["pnl_usd"] is not None]
    if sell_only:
        pnl_vals   = [t["pnl_usd"] for t in sell_only]
        pnl_labels = [pd.Timestamp(t["date"]).strftime("%Y-%m-%d") for t in sell_only]
        colors     = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in pnl_vals]
        fig_pnl    = go.Figure(go.Bar(
            x=pnl_labels, y=pnl_vals, marker_color=colors,
            hovertemplate="$%{y:+.2f}<extra></extra>",
        ))
        fig_pnl.add_hline(y=0, line_color="rgba(255,255,255,0.3)")
        fig_pnl.update_layout(
            title="P&L per Trade ($)", height=220,
            template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
            margin=dict(t=40, b=20), showlegend=False,
            yaxis=dict(gridcolor="#313244"), xaxis=dict(gridcolor="#313244"),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
else:
    st.info("No trades executed. Try adjusting RSI thresholds.")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PARAMETER SWEEP
# ═════════════════════════════════════════════════════════════════════════════

if run_sweep:
    st.divider()
    st.subheader("🔍 Parameter Sweep")

    sweep_df = cached_sweep(
        initial_capital, transaction_cost,
        rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
    )
    st.session_state["sweep_df"] = sweep_df

    if not sweep_df.empty:
        sl, sr = st.columns([3, 2])

        with sl:
            st.markdown("**Top 15 combinations (sorted by Sharpe)**")
            disp = sweep_df.head(15).copy().reset_index(drop=True)
            for c in ["RSI_buy", "RSI_sell", "MA_fast", "MA_slow", "Trades"]:
                disp[c] = disp[c].astype(int)

            def _highlight_sweep(row):
                # row.name is the positional index after reset_index
                if row.name == 0:   # best row
                    return ["background-color: rgba(249,226,175,0.18)"] * len(row)
                if row.get("Default", "") == "*":
                    return ["background-color: rgba(137,180,250,0.12)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                disp.style.apply(_highlight_sweep, axis=1),
                use_container_width=True, hide_index=True,
            )

        with sr:
            best          = sweep_df.iloc[0]
            cur_sharpe    = metrics.get("sharpe_ratio", 0)
            best_sharpe   = float(best["Sharpe"])
            improvement   = best_sharpe - cur_sharpe
            st.markdown("**Best vs Current**")
            st.metric("Best Sharpe",   f"{best_sharpe:.2f}", f"{improvement:+.2f}")
            st.metric("Best Return",   f"{float(best['Return%']):+.2f}%")
            st.metric("Best Win Rate", f"{float(best['WinRate%']):.1f}%")
            if best.get("Default") != "*":
                st.info(
                    f"💡 Suggest: RSI {int(best['RSI_buy'])}/{int(best['RSI_sell'])}, "
                    f"MA {int(best['MA_fast'])}/{int(best['MA_slow'])}"
                )
            else:
                st.success("Current config is already optimal ✓")

        # Heatmap
        pivot = sweep_df.pivot_table(
            values="Sharpe", index="RSI_buy", columns="RSI_sell", aggfunc="max",
        )
        if not pivot.empty:
            fig_h = px.imshow(
                pivot, text_auto=".2f",
                labels=dict(x="RSI Sell Threshold", y="RSI Buy Threshold", color="Sharpe"),
                color_continuous_scale="RdYlGn",
                title="Sharpe Ratio Heatmap (RSI Buy × RSI Sell)",
            )
            fig_h.update_layout(
                height=320, template="plotly_dark",
                paper_bgcolor="#1e1e2e", margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_h, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — WALK-FORWARD VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

wf_results = None
if run_wf:
    st.divider()
    st.subheader("🔄 Walk-forward Validation")

    wf_results = cached_wf(
        initial_capital, transaction_cost,
        rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
    )
    st.session_state["wf_results"] = wf_results

    if wf_results:
        wf_passes = 0
        wf_cols   = st.columns(len(wf_results))
        for col, r in zip(wf_cols, wf_results):
            if r["passed"]:
                wf_passes += 1
            sign = "+" if r["return_pct"] >= 0 else ""
            col.metric(
                f"Period {r['split']}",
                f"{sign}{r['return_pct']:.1f}%",
                f"Win: {r['win_rate']:.0f}%",
                delta_color="normal" if r["return_pct"] >= 0 else "inverse",
            )
            col.caption(f"{r['period_start']} → {r['period_end']}")
            col.write("✅ PASS" if r["passed"] else "❌ FAIL")

        wf_ok = wf_passes >= max(1, len(wf_results) * 2 // 3)
        total = len(wf_results)
        msg   = f"Walk-forward: **{wf_passes}/{total}** periods profitable"
        (st.success if wf_ok else st.warning)(
            msg + (" — strategy is robust ✓" if wf_ok else " — may overfit ✗")
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════

mc = None
if run_mc:
    st.divider()
    st.subheader("🎲 Monte Carlo Simulation")

    mc = cached_mc(
        initial_capital, transaction_cost,
        rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
    )
    st.session_state["mc"] = mc

    if mc.get("n_trades", 0) >= 2:
        mc1, mc2 = st.columns([2, 1])

        with mc1:
            dist       = mc["distribution"]
            actual_ret = mc["actual_return_pct"]
            fig_mc     = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=dist, nbinsx=40, name="Random",
                marker_color="rgba(137,180,250,0.55)",
                histnorm="probability density",
            ))
            fig_mc.add_vline(
                x=actual_ret, line_dash="dash",
                line_color="#f9e2af", line_width=2,
                annotation_text=f"Actual: {actual_ret:+.1f}%",
                annotation_position="top right",
            )
            fig_mc.update_layout(
                title=f"Return Distribution ({mc['n_simulations']} random sequences)",
                height=300, template="plotly_dark",
                paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                xaxis_title="Return (%)", yaxis_title="Density",
                margin=dict(t=40, b=10),
                yaxis=dict(gridcolor="#313244"), xaxis=dict(gridcolor="#313244"),
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with mc2:
            pv  = mc["p_value"]
            pct = mc["percentile"]
            st.metric("p-value",           f"{pv:.3f}",
                      "Skill likely ✓" if pv < 0.1 else "May be luck ✗",
                      delta_color="normal" if pv < 0.1 else "inverse")
            st.metric("Percentile",        f"{pct:.1f}th", "vs 1000 random runs")
            st.metric("Actual Return",     f"{actual_ret:+.1f}%")
            st.metric("Avg Random Return", f"{mc['mean_random_return']:+.1f}%")

            if mc["passed"]:
                st.success("Returns appear skill-driven (p < 0.10)")
            else:
                st.warning("Cannot rule out luck")
    else:
        st.info("Not enough closed trades for Monte Carlo (need ≥ 2).")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — VALIDATION REPORT
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("✅ Strategy Validation Report")

bench_pass = (
    metrics.get("sharpe_ratio", 0) >= 0.5 and
    metrics.get("win_rate_pct", 0) >= 50 and
    metrics.get("alpha_pct", -999) >= 0
)

# Walk-forward pass/fail
wf_results_for_report = wf_results or st.session_state.get("wf_results")
if wf_results_for_report:
    wf_p  = sum(1 for r in wf_results_for_report if r.get("passed"))
    wf_t  = len(wf_results_for_report)
    wf_ok = wf_p >= max(1, wf_t * 2 // 3)
    wf_detail = f"{wf_p}/{wf_t} periods positive"
else:
    wf_ok, wf_detail = None, "Enable Walk-forward above"

# Monte Carlo pass/fail
mc_for_report = mc or st.session_state.get("mc")
if mc_for_report and mc_for_report.get("n_trades", 0) >= 2:
    mc_ok     = mc_for_report.get("passed", False)
    mc_detail = f"p-value = {mc_for_report.get('p_value', 1):.3f}"
else:
    mc_ok, mc_detail = None, "Enable Monte Carlo above"

# Sweep info
sweep_for_report = st.session_state.get("sweep_df")
if sweep_for_report is not None and not sweep_for_report.empty:
    best_r    = sweep_for_report.iloc[0]
    sw_detail = (f"Best: RSI {int(best_r['RSI_buy'])}/{int(best_r['RSI_sell'])}, "
                 f"MA {int(best_r['MA_fast'])}/{int(best_r['MA_slow'])} "
                 f"(Sharpe={float(best_r['Sharpe']):.2f})")
    sw_ok     = True   # sweep itself is informational
else:
    sw_ok, sw_detail = None, "Enable Parameter Sweep above"

checks = [
    ("Benchmark\nMetrics",  bench_pass, f"Sharpe {sharpe:.2f} · WinRate {win_rate:.0f}% · Alpha {alpha:+.1f}%"),
    ("Parameter\nSweep",    sw_ok,      sw_detail),
    ("Walk-forward\nTest",  wf_ok,      wf_detail),
    ("Monte Carlo",         mc_ok,      mc_detail),
]

vc = st.columns(4)
for col, (label, passed, detail) in zip(vc, checks):
    with col:
        if passed is True:
            st.success(f"✅ {label}")
        elif passed is False:
            st.error(f"❌ {label}")
        else:
            st.info(f"ℹ️ {label}")
        st.caption(detail)

# Overall verdict
definite_checks = [(p, l) for l, p, _ in checks if p is not None]
n_pass  = sum(1 for p, _ in definite_checks if p)
n_total = len(definite_checks)
if n_total > 0:
    ratio = n_pass / n_total
    if ratio == 1.0:
        st.success("🏆 **STRATEGY CONFIDENCE: HIGH** — All checks passed")
    elif ratio >= 0.5:
        st.warning(f"⚠️ **STRATEGY CONFIDENCE: MEDIUM** — {n_pass}/{n_total} checks passed")
    else:
        st.error(f"🚨 **STRATEGY CONFIDENCE: LOW** — {n_pass}/{n_total} checks passed")


# ═════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═════════════════════════════════════════════════════════════════════════════

st.divider()
f1, f2, f3 = st.columns(3)
sim_s = metrics.get("sim_start")
sim_e = metrics.get("sim_end")
s_str = pd.Timestamp(sim_s).strftime("%Y-%m-%d") if sim_s else "—"
e_str = pd.Timestamp(sim_e).strftime("%Y-%m-%d") if sim_e else "—"

f1.caption(f"📅 Sim period: {s_str} → {e_str}")
f2.caption(f"📊 {len(df_raw)} trading days · {cfg.lookback_required}-day warmup")
f3.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

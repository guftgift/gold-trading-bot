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
import dataclasses

# ── Import simulation engine ──────────────────────────────────────────────────
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
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .positive { color: #50fa7b; }
    .negative { color: #ff5555; }
    .neutral  { color: #f1fa8c; }
    .stProgress > div > div { background-color: #ffb86c; }
    div[data-testid="metric-container"] {
        background: #282a36;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #44475a;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Gold_nugget_%28Australia%29_4_%28hg%29.jpg/320px-Gold_nugget_%28Australia%29_4_%28hg%29.jpg", use_container_width=True)
    st.title("⚙️ Simulation Config")

    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=10.0, max_value=100000.0,
        value=100.0, step=10.0, format="%.2f",
    )
    transaction_cost = st.slider(
        "Transaction Cost (%)", min_value=0.0, max_value=1.0,
        value=0.1, step=0.05, format="%.2f",
    ) / 100

    st.divider()
    st.subheader("📊 Signal Parameters")

    col1, col2 = st.columns(2)
    with col1:
        rsi_buy  = st.number_input("RSI Buy <", min_value=10, max_value=49, value=35)
    with col2:
        rsi_sell = st.number_input("RSI Sell >", min_value=51, max_value=90, value=65)

    col3, col4 = st.columns(2)
    with col3:
        ma_fast = st.selectbox("MA Fast", [5, 10, 20, 50], index=2)
    with col4:
        ma_slow = st.selectbox("MA Slow", [50, 100, 150, 200], index=1)

    ma_trend = st.selectbox("MA Trend Filter", [100, 150, 200, 250], index=2)

    st.divider()
    st.subheader("🔬 Validation")
    run_sweep = st.checkbox("Parameter Sweep", value=True)
    run_wf    = st.checkbox("Walk-forward Test", value=True)
    run_mc    = st.checkbox("Monte Carlo (1000 runs)", value=True)

    st.divider()
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CACHE — ข้อมูลและ indicators
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Fetching XAU/USD data...")
def cached_fetch(data_days: int):
    cfg = SimConfig(data_days=data_days)
    return fetch_historical_data(cfg)


@st.cache_data(ttl=300, show_spinner="Running simulation...")
def cached_run(
    initial_capital, transaction_cost, rsi_buy, rsi_sell,
    ma_fast, ma_slow, ma_trend,
):
    cfg = SimConfig(
        initial_capital    = initial_capital,
        transaction_cost   = transaction_cost,
        rsi_buy_threshold  = rsi_buy,
        rsi_sell_threshold = rsi_sell,
        ma_fast            = ma_fast,
        ma_slow            = ma_slow,
        ma_trend           = ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    df     = compute_indicators(df_raw.copy(), cfg)
    tlog, ecurve = run_simulation(df, cfg)
    metrics      = calculate_metrics(tlog, ecurve, df, cfg)
    return cfg, df_raw, df, tlog, ecurve, metrics


@st.cache_data(ttl=300, show_spinner="Running parameter sweep...")
def cached_sweep(initial_capital, transaction_cost, rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend):
    cfg = SimConfig(
        initial_capital=initial_capital, transaction_cost=transaction_cost,
        rsi_buy_threshold=rsi_buy, rsi_sell_threshold=rsi_sell,
        ma_fast=ma_fast, ma_slow=ma_slow, ma_trend=ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    return run_parameter_sweep(df_raw, cfg)


@st.cache_data(ttl=300, show_spinner="Running walk-forward validation...")
def cached_wf(initial_capital, transaction_cost, rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend):
    cfg = SimConfig(
        initial_capital=initial_capital, transaction_cost=transaction_cost,
        rsi_buy_threshold=rsi_buy, rsi_sell_threshold=rsi_sell,
        ma_fast=ma_fast, ma_slow=ma_slow, ma_trend=ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    return run_walk_forward(df_raw, cfg)


@st.cache_data(ttl=300, show_spinner="Running Monte Carlo...")
def cached_mc(initial_capital, transaction_cost, rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend):
    cfg = SimConfig(
        initial_capital=initial_capital, transaction_cost=transaction_cost,
        rsi_buy_threshold=rsi_buy, rsi_sell_threshold=rsi_sell,
        ma_fast=ma_fast, ma_slow=ma_slow, ma_trend=ma_trend,
    )
    df_raw = cached_fetch(cfg.data_days)
    df     = compute_indicators(df_raw.copy(), cfg)
    tlog, _ = run_simulation(df, cfg)
    return run_monte_carlo(tlog, cfg)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════

st.title("🥇 Gold Trading Simulator")
st.caption("All-in simulation • XAU/USD • Strategy validation dashboard")

# ── Auto-load on first visit ──────────────────────────────────────────────────
if "results_loaded" not in st.session_state:
    st.session_state["results_loaded"] = False

if run_btn or not st.session_state["results_loaded"]:
    with st.spinner("Loading..."):
        try:
            cfg, df_raw, df, tlog, ecurve, metrics = cached_run(
                initial_capital, transaction_cost,
                rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
            )
            st.session_state["cfg"]     = cfg
            st.session_state["df_raw"]  = df_raw
            st.session_state["df"]      = df
            st.session_state["tlog"]    = tlog
            st.session_state["ecurve"]  = ecurve
            st.session_state["metrics"] = metrics
            st.session_state["results_loaded"] = True
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

# ── Pull from session state ───────────────────────────────────────────────────
cfg     = st.session_state["cfg"]
df_raw  = st.session_state["df_raw"]
df      = st.session_state["df"]
tlog    = st.session_state["tlog"]
ecurve  = st.session_state["ecurve"]
metrics = st.session_state["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1: KEY METRICS
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📈 Performance Summary")

c1, c2, c3, c4, c5, c6 = st.columns(6)

ret_pct  = metrics.get("total_return_pct", 0)
bh_pct   = metrics.get("buy_hold_return_pct", 0)
alpha    = metrics.get("alpha_pct", 0)
win_rate = metrics.get("win_rate_pct", 0)
sharpe   = metrics.get("sharpe_ratio", 0)
max_dd   = metrics.get("max_drawdown_pct", 0)
final_b  = metrics.get("final_balance", initial_capital)

c1.metric("Final Balance",    f"${final_b:.2f}",     f"{ret_pct:+.2f}%")
c2.metric("Total Return",     f"{ret_pct:+.2f}%",    f"vs B&H {bh_pct:+.1f}%")
c3.metric("Alpha",            f"{alpha:+.2f}%",      "vs buy-and-hold")
c4.metric("Win Rate",         f"{win_rate:.1f}%",    f"{metrics.get('num_wins',0)}W / {metrics.get('num_losses',0)}L")
c5.metric("Sharpe Ratio",     f"{sharpe:.2f}",       "annualised")
c6.metric("Max Drawdown",     f"{max_dd:.1f}%",      f"{metrics.get('num_closed',0)} trades")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2: EQUITY CURVE
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("💹 Portfolio Equity Curve")

if ecurve:
    ec_df = pd.DataFrame(ecurve)
    ec_df["date"] = pd.to_datetime(ec_df["date"])

    # Gold price normalized to start at initial_capital for comparison
    sim_start_idx = cfg.lookback_required
    df_sim = df.iloc[sim_start_idx:].copy()
    bh_prices = df_sim["Close"].values
    if len(bh_prices) > 0:
        bh_norm = bh_prices / bh_prices[0] * initial_capital
    else:
        bh_norm = []

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=("Portfolio Value vs Buy-and-Hold", "XAU/USD Price + Signals"),
    )

    # ── Equity curve ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ec_df["date"], y=ec_df["value"],
        name="Strategy ($)",
        line=dict(color="#ffb86c", width=2),
        fill="tozeroy", fillcolor="rgba(255,184,108,0.08)",
    ), row=1, col=1)

    # Buy-and-hold line
    if len(bh_norm) > 0:
        fig.add_trace(go.Scatter(
            x=df_sim.index[:len(bh_norm)],
            y=bh_norm,
            name="Buy-and-Hold ($)",
            line=dict(color="#6272a4", width=1.5, dash="dot"),
        ), row=1, col=1)

    # Initial capital horizontal line
    fig.add_hline(y=initial_capital, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", row=1, col=1)

    # ── BUY / SELL markers on equity curve ───────────────────────────────────
    buys  = [t for t in tlog if t["action"] == "BUY"]
    sells = [t for t in tlog if t["action"] in ("SELL", "OPEN")]

    if buys:
        buy_dates = pd.to_datetime([t["date"] for t in buys])
        buy_vals  = [float(df.loc[df.index.asof(d), "Close"]) / float(df_sim["Close"].iloc[0]) * initial_capital
                     if len(df_sim) > 0 else initial_capital for d in buy_dates]
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_vals,
            mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=12, color="#50fa7b"),
        ), row=1, col=1)

    if sells:
        sell_dates = pd.to_datetime([t["date"] for t in sells])
        sell_vals  = [t["balance"] for t in sells]
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_vals,
            mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=12, color="#ff5555"),
        ), row=1, col=1)

    # ── XAU/USD price ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        name="XAU/USD",
        line=dict(color="#f1fa8c", width=1),
    ), row=2, col=1)

    # RSI oversold/overbought shading
    rsi_df = df[df["RSI"].notna()].copy()
    buy_zones  = rsi_df[rsi_df["RSI"] < cfg.rsi_buy_threshold]
    sell_zones = rsi_df[rsi_df["RSI"] > cfg.rsi_sell_threshold]

    if not buy_zones.empty:
        fig.add_trace(go.Scatter(
            x=buy_zones.index, y=buy_zones["Close"],
            mode="markers", name=f"RSI<{cfg.rsi_buy_threshold}",
            marker=dict(size=4, color="#50fa7b", opacity=0.5),
        ), row=2, col=1)

    if not sell_zones.empty:
        fig.add_trace(go.Scatter(
            x=sell_zones.index, y=sell_zones["Close"],
            mode="markers", name=f"RSI>{cfg.rsi_sell_threshold}",
            marker=dict(size=4, color="#ff5555", opacity=0.5),
        ), row=2, col=1)

    fig.update_layout(
        height=550, template="plotly_dark",
        legend=dict(orientation="h", y=-0.1),
        margin=dict(t=40, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="USD ($)", row=1, col=1)
    fig.update_yaxes(title_text="XAU/USD", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3: RSI CHART
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📉 RSI Indicator", expanded=False):
    rsi_data = df[df["RSI"].notna()].copy()
    if not rsi_data.empty:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=rsi_data.index, y=rsi_data["RSI"],
            name="RSI(14)", line=dict(color="#bd93f9", width=1.5),
        ))
        fig_rsi.add_hline(y=cfg.rsi_buy_threshold,  line_dash="dash", line_color="#50fa7b",
                          annotation_text=f"Buy < {cfg.rsi_buy_threshold}")
        fig_rsi.add_hline(y=cfg.rsi_sell_threshold, line_dash="dash", line_color="#ff5555",
                          annotation_text=f"Sell > {cfg.rsi_sell_threshold}")
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig_rsi.update_layout(
            height=250, template="plotly_dark",
            margin=dict(t=20, b=20), yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig_rsi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4: TRADE LOG TABLE
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("📋 Trade Log")

if tlog:
    rows = []
    for t in tlog:
        dt_str = t["date"].strftime("%Y-%m-%d") if hasattr(t["date"], "strftime") else str(t["date"])[:10]
        pnl_u  = f"{t['pnl_usd']:+.2f}" if t["pnl_usd"] is not None else "—"
        pnl_p  = f"{t['pnl_pct']:+.2f}%" if t["pnl_pct"] is not None else "—"
        rows.append({
            "Date":     dt_str,
            "Action":   t["action"],
            "Price":    f"${t['price']:,.2f}",
            "Oz":       f"{t['oz']:.5f}",
            "P&L ($)":  pnl_u,
            "P&L (%)":  pnl_p,
            "Balance":  f"${t['balance']:,.2f}",
        })
    tlog_df = pd.DataFrame(rows)

    def color_rows(row):
        if row["Action"] == "BUY":
            return ["background-color: rgba(80,250,123,0.08)"] * len(row)
        elif row["Action"] == "SELL":
            color = "rgba(80,250,123,0.12)" if "+" in str(row["P&L ($)"]) else "rgba(255,85,85,0.12)"
            return [f"background-color: {color}"] * len(row)
        return [""] * len(row)

    st.dataframe(
        tlog_df.style.apply(color_rows, axis=1),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No trades executed. Try adjusting RSI thresholds.")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5: PARAMETER SWEEP
# ─────────────────────────────────────────────────────────────────────────────

if run_sweep:
    st.divider()
    st.subheader("🔍 Parameter Sweep (Grid Search)")

    with st.spinner("Running parameter sweep..."):
        sweep_df = cached_sweep(
            initial_capital, transaction_cost,
            rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
        )

    if not sweep_df.empty:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("**Top 15 Parameter Combinations (sorted by Sharpe)**")
            display_sweep = sweep_df.head(15).copy()
            display_sweep["RSI_buy"]  = display_sweep["RSI_buy"].astype(int)
            display_sweep["RSI_sell"] = display_sweep["RSI_sell"].astype(int)
            display_sweep["MA_fast"]  = display_sweep["MA_fast"].astype(int)
            display_sweep["MA_slow"]  = display_sweep["MA_slow"].astype(int)

            def highlight_best(row):
                if row.name == 0:
                    return ["background-color: rgba(255,184,108,0.2)"] * len(row)
                if row.get("Default") == "*":
                    return ["background-color: rgba(98,114,164,0.2)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                display_sweep.style.apply(highlight_best, axis=1),
                use_container_width=True, hide_index=True,
            )

        with col_right:
            best = sweep_df.iloc[0]
            current_sharpe = metrics.get("sharpe_ratio", 0)
            best_sharpe    = float(best["Sharpe"])
            improvement    = best_sharpe - current_sharpe

            st.markdown("**Best vs Current Config**")
            st.metric("Best Sharpe",    f"{best_sharpe:.2f}",  f"{improvement:+.2f} vs current")
            st.metric("Best Return",    f"{float(best['Return%']):+.2f}%")
            st.metric("Best Win Rate",  f"{float(best['WinRate%']):.1f}%")
            st.info(
                f"Best params: RSI {int(best['RSI_buy'])}/{int(best['RSI_sell'])}, "
                f"MA {int(best['MA_fast'])}/{int(best['MA_slow'])}"
            )

        # Heatmap: RSI_buy × RSI_sell → Sharpe
        pivot = sweep_df.pivot_table(
            values="Sharpe", index="RSI_buy", columns="RSI_sell", aggfunc="max"
        )
        if not pivot.empty:
            fig_heat = px.imshow(
                pivot, text_auto=".2f",
                labels=dict(x="RSI Sell", y="RSI Buy", color="Sharpe"),
                color_continuous_scale="RdYlGn",
                title="Sharpe Ratio Heatmap (RSI Buy × RSI Sell)",
            )
            fig_heat.update_layout(height=300, template="plotly_dark", margin=dict(t=40, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6: WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

if run_wf:
    st.divider()
    st.subheader("🔄 Walk-forward Validation")

    with st.spinner("Running walk-forward..."):
        wf_results = cached_wf(
            initial_capital, transaction_cost,
            rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
        )

    if wf_results:
        wf_cols = st.columns(len(wf_results))
        passes  = 0
        for i, (col, r) in enumerate(zip(wf_cols, wf_results)):
            with col:
                status = "✅ PASS" if r["passed"] else "❌ FAIL"
                if r["passed"]:
                    passes += 1
                sign = "+" if r["return_pct"] >= 0 else ""
                st.metric(
                    f"Period {r['split']}",
                    f"{sign}{r['return_pct']:.1f}%",
                    f"{r['win_rate']:.0f}% win rate",
                )
                st.caption(f"{r['period_start']} → {r['period_end']}")
                st.write(status)

        wf_pass = passes >= max(1, len(wf_results) * 2 // 3)
        total   = len(wf_results)
        if wf_pass:
            st.success(f"Walk-forward: {passes}/{total} periods profitable — Strategy is robust ✓")
        else:
            st.warning(f"Walk-forward: {passes}/{total} periods profitable — Strategy may overfit ✗")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7: MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

if run_mc:
    st.divider()
    st.subheader("🎲 Monte Carlo Simulation")

    with st.spinner("Running 1000 Monte Carlo simulations..."):
        mc = cached_mc(
            initial_capital, transaction_cost,
            rsi_buy, rsi_sell, ma_fast, ma_slow, ma_trend,
        )

    if mc["n_trades"] >= 2:
        mc_col1, mc_col2 = st.columns([2, 1])

        with mc_col1:
            dist = mc["distribution"]
            actual_ret = mc["actual_return_pct"]
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=dist, nbinsx=40, name="Random simulations",
                marker_color="rgba(98,114,164,0.7)",
                histnorm="probability density",
            ))
            fig_mc.add_vline(
                x=actual_ret,
                line_dash="dash", line_color="#ffb86c", line_width=2,
                annotation_text=f"Actual: {actual_ret:+.1f}%",
                annotation_position="top right",
            )
            fig_mc.update_layout(
                title="Return Distribution (1000 Shuffled Trade Sequences)",
                height=320, template="plotly_dark",
                xaxis_title="Return (%)", yaxis_title="Density",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        with mc_col2:
            pv = mc["p_value"]
            pct = mc["percentile"]
            st.metric("p-value",    f"{pv:.3f}", f"{'Skill ✓' if pv < 0.1 else 'Luck? ✗'}")
            st.metric("Percentile", f"{pct:.1f}th", "vs random")
            st.metric("Actual Return",  f"{mc['actual_return_pct']:+.1f}%")
            st.metric("Avg Random Return", f"{mc['mean_random_return']:+.1f}%")

            if mc["passed"]:
                st.success("Strategy returns appear skill-driven (p < 0.10)")
            else:
                st.warning("Cannot rule out luck as the cause of returns")
    else:
        st.info("Not enough closed trades for Monte Carlo analysis.")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8: VALIDATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("✅ Strategy Validation Report")

# Collect validation results
bench_pass = (metrics.get("sharpe_ratio", 0) >= 0.5 and
              metrics.get("win_rate_pct", 0) >= 50 and
              metrics.get("alpha_pct", -999) >= 0)

wf_results_local = st.session_state.get("wf_results_cache") or []
mc_local         = st.session_state.get("mc_cache") or {}

# Use cached if available
if run_wf:
    wf_results_local = wf_results
if run_mc:
    mc_local = mc

wf_passes = sum(1 for r in wf_results_local if r.get("passed"))
wf_total  = max(len(wf_results_local), 1)
wf_pass   = wf_passes >= max(1, wf_total * 2 // 3) if wf_results_local else None
mc_pass   = mc_local.get("passed") if mc_local else None

vcols = st.columns(4)
checks = [
    ("Benchmark\nMetrics",    bench_pass,
     f"Sharpe {metrics.get('sharpe_ratio',0):.2f}, Win {metrics.get('win_rate_pct',0):.0f}%, Alpha {metrics.get('alpha_pct',0):+.1f}%"),
    ("Parameter\nSweep",      None,   "Run sweep to evaluate best params"),
    ("Walk-forward\nTest",    wf_pass, f"{wf_passes}/{wf_total} periods positive" if wf_results_local else "Run walk-forward"),
    ("Monte Carlo\np-value",  mc_pass, f"p={mc_local.get('p_value',1):.3f}" if mc_local else "Run Monte Carlo"),
]

for col, (label, passed, detail) in zip(vcols, checks):
    with col:
        if passed is True:
            st.success(f"✅ {label}")
        elif passed is False:
            st.error(f"❌ {label}")
        else:
            st.info(f"ℹ️ {label}")
        st.caption(detail)

# Overall confidence
passed_count = sum(1 for _, p, _ in checks if p is True)
total_checks = sum(1 for _, p, _ in checks if p is not None)

if total_checks > 0:
    if passed_count == total_checks:
        st.success("🏆 **STRATEGY CONFIDENCE: HIGH** — All validation checks passed")
    elif passed_count >= total_checks / 2:
        st.warning(f"⚠️ **STRATEGY CONFIDENCE: MEDIUM** — {passed_count}/{total_checks} checks passed")
    else:
        st.error(f"🚨 **STRATEGY CONFIDENCE: LOW** — Only {passed_count}/{total_checks} checks passed")


# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
sim_start = metrics.get("sim_start")
sim_end   = metrics.get("sim_end")
start_s   = sim_start.strftime("%Y-%m-%d") if hasattr(sim_start, "strftime") else str(sim_start)[:10]
end_s     = sim_end.strftime("%Y-%m-%d")   if hasattr(sim_end,   "strftime") else str(sim_end)[:10]

col_f1.caption(f"📅 Simulation period: {start_s} → {end_s}")
col_f2.caption(f"📊 Data: {len(df_raw)} trading days | Warmup: {cfg.lookback_required} rows")
col_f3.caption(f"🕐 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

"""
Gold Signal Dashboard — Streamlit
===================================
แสดงสัญญาณ BUY/SELL/HOLD + กราฟ indicators + อธิบาย algorithm
ไม่มีการ simulate การลงทุน

Deploy บน Streamlit Cloud:
  1. Push to GitHub
  2. share.streamlit.io → New app → dashboard.py
  3. ใส่ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ใน Secrets
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
import time as _time_mod

from gold_simulation import (
    SimConfig,
    fetch_historical_data,
    compute_indicators,
)
from live_monitor import (
    run_check,
    load_state,
    save_state,
    get_signal_data,
    RSI_BUY, RSI_SELL, MA_FAST, MA_SLOW, MA_TREND,
)

SIGNALS_FILE = Path(__file__).parent / "signals.json"

# ─── Background Monitor Thread ────────────────────────────────────────────────
_monitor_lock  = threading.Lock()
_monitor_thread: threading.Thread | None = None
_monitor_stop  = threading.Event()
_monitor_meta: dict = {
    "running":       False,
    "last_check":    None,
    "next_check":    None,
    "error":         None,
    "interval":      60,
    "should_run":    True,
    "restart_count": 0,
}
_RETRY_SEC = 60


def _monitor_loop(interval_min: int) -> None:
    _monitor_meta["running"] = True
    state = load_state()
    while not _monitor_stop.is_set():
        try:
            state = run_check(state)
            _monitor_meta["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _monitor_meta["error"]      = None
            next_dt = datetime.now() + timedelta(minutes=interval_min)
            _monitor_meta["next_check"] = next_dt.strftime("%Y-%m-%d %H:%M:%S")
            _monitor_stop.wait(timeout=interval_min * 60)
        except Exception as exc:
            _monitor_meta["error"] = f"{type(exc).__name__}: {exc}"
            next_dt = datetime.now() + timedelta(seconds=_RETRY_SEC)
            _monitor_meta["next_check"] = next_dt.strftime("%Y-%m-%d %H:%M:%S") + " (retry)"
            _monitor_stop.wait(timeout=_RETRY_SEC)
    _monitor_meta["running"] = False


def start_monitor(interval_min: int = 60) -> None:
    global _monitor_thread
    with _monitor_lock:
        if _monitor_thread and _monitor_thread.is_alive():
            return
        _monitor_stop.clear()
        _monitor_meta["interval"]      = interval_min
        _monitor_meta["should_run"]    = True
        _monitor_meta["restart_count"] += 1
        _monitor_thread = threading.Thread(
            target=_monitor_loop, args=(interval_min,),
            daemon=True, name="gold-monitor",
        )
        _monitor_thread.start()


def stop_monitor() -> None:
    _monitor_meta["should_run"] = False
    _monitor_stop.set()
    _monitor_meta["running"] = False


def is_monitor_running() -> bool:
    return _monitor_thread is not None and _monitor_thread.is_alive()


def _watchdog() -> bool:
    """เรียกทุก rerun — restart thread อัตโนมัติถ้าตายแต่ควรรัน"""
    if _monitor_meta["should_run"] and not is_monitor_running():
        start_monitor(interval_min=_monitor_meta["interval"])
        return True
    return False


def check_now_once() -> None:
    state = load_state()
    run_check(state)


# Auto-start เมื่อ app โหลด
if not is_monitor_running() and _monitor_meta["should_run"]:
    start_monitor(interval_min=_monitor_meta["interval"])


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Gold Signal Monitor",
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
    .signal-card {
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 18px;
        font-size: 1.05em;
    }
    .algo-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #313244;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🥇 Gold Signal Monitor")
    st.caption("XAU/USD — Signal only, no simulation")
    st.divider()

    st.subheader("📊 Signal Parameters")
    c1, c2 = st.columns(2)
    rsi_buy  = c1.number_input("RSI Buy <",  10, 49, int(RSI_BUY))
    rsi_sell = c2.number_input("RSI Sell >", 51, 90, int(RSI_SELL))
    c3, c4   = st.columns(2)
    ma_fast  = c3.selectbox("MA Fast",  [5, 10, 20, 50],      index=[5,10,20,50].index(MA_FAST)  if MA_FAST  in [5,10,20,50]  else 2)
    ma_slow  = c4.selectbox("MA Slow",  [50, 100, 150, 200],  index=[50,100,150,200].index(MA_SLOW) if MA_SLOW in [50,100,150,200] else 1)
    ma_trend = st.selectbox("MA Trend", [100, 150, 200, 250], index=[100,150,200,250].index(MA_TREND) if MA_TREND in [100,150,200,250] else 2)

    st.divider()
    show_thb = st.toggle("💱 แสดงราคาเป็น ฿ THB", value=True)

    st.divider()
    send_hold_ui = st.toggle("ส่ง HOLD ทาง Telegram ด้วย", value=False)

    st.divider()
    data_days = st.slider("ข้อมูลย้อนหลัง (วัน)", 100, 1000, 500, 50)

    st.divider()
    check_interval = st.selectbox("ตรวจทุก (นาที)", [15, 30, 60, 120, 240], index=2)


# ═════════════════════════════════════════════════════════════════════════════
#  EXCHANGE RATE  USD → THB
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def get_usdthb() -> float:
    """ดึงอัตราแลกเปลี่ยน USD/THB — cache 1 ชั่วโมง"""
    import yfinance as yf, requests as _req

    # 1) yfinance
    try:
        raw = yf.download("USDTHB=X", period="5d", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        rate = float(raw["Close"].dropna().iloc[-1])
        if 25 < rate < 60:
            return rate
    except Exception:
        pass

    # 2) frankfurter.app (ฟรี ไม่ต้อง API key)
    try:
        r = _req.get("https://api.frankfurter.app/latest?from=USD&to=THB", timeout=5)
        rate = float(r.json()["rates"]["THB"])
        if 25 < rate < 60:
            return rate
    except Exception:
        pass

    return 34.0   # fallback


# ═════════════════════════════════════════════════════════════════════════════
#  CACHED DATA
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Fetching XAU/USD data…")
def cached_fetch(days: int, rsi_b: float, rsi_s: float,
                 maf: int, mas: int, mat: int) -> pd.DataFrame:
    cfg = SimConfig(
        data_days=days,
        rsi_buy_threshold=rsi_b, rsi_sell_threshold=rsi_s,
        ma_fast=maf, ma_slow=mas, ma_trend=mat,
    )
    df_raw = fetch_historical_data(cfg)
    return compute_indicators(df_raw, cfg)


@st.cache_data(ttl=10, show_spinner=False)
def load_signals_json() -> dict | None:
    if SIGNALS_FILE.exists():
        try:
            with open(SIGNALS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════

st.title("🥇 Gold Signal Monitor")

tab_signal, tab_history = st.tabs(["📡 Signal & Chart", "📋 Signal History"])


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SIGNAL & CHART
# ═════════════════════════════════════════════════════════════════════════════

with tab_signal:

    # ── Watchdog ─────────────────────────────────────────────────────────────
    _just_restarted = _watchdog()
    if _just_restarted:
        st.toast("♻️ Monitor restarted อัตโนมัติ", icon="♻️")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    ar_col1, ar_col2, ar_col3 = st.columns([4, 1, 1])
    with ar_col2:
        auto_refresh = st.toggle("Auto-refresh", value=False)
    with ar_col3:
        ar_interval = st.selectbox("วิ", [30, 60, 120, 300], index=1,
                                   label_visibility="collapsed")
    if auto_refresh:
        _time_mod.sleep(ar_interval)
        st.rerun()

    # ── Monitor Control ───────────────────────────────────────────────────────
    running = is_monitor_running()
    mc1, mc2, mc3, mc4 = st.columns(4)

    if mc1.button("▶ Start Monitor", type="primary",
                  disabled=running, use_container_width=True):
        start_monitor(interval_min=check_interval)
        st.rerun()

    if mc2.button("⏹ Stop", disabled=not running, use_container_width=True):
        stop_monitor()
        _time_mod.sleep(0.3)
        st.rerun()

    if mc3.button("⚡ Check Now", use_container_width=True):
        with st.spinner("กำลังตรวจสัญญาณ…"):
            try:
                check_now_once()
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Error: {e}")
        st.rerun()

    if mc4.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Status bar
    rc = _monitor_meta.get("restart_count", 0)
    if running:
        bar = (
            f'<span style="color:#a6e3a1;font-weight:700">● RUNNING</span>'
            f' — ตรวจทุก {_monitor_meta["interval"]} นาที'
            f' | Last: <b>{_monitor_meta["last_check"] or "กำลังตรวจ…"}</b>'
            f' | Next: {_monitor_meta["next_check"] or "—"}'
            + (f' | Restarts: {rc}' if rc > 1 else '')
        )
    else:
        bar = '<span style="color:#6c7086;font-weight:700">○ STOPPED</span>'
    if _monitor_meta.get("error"):
        bar += f' | <span style="color:#f38ba8">⚠ {_monitor_meta["error"]} — retrying…</span>'

    st.markdown(
        f'<div style="background:#181825;border-radius:8px;padding:8px 16px;'
        f'margin:6px 0 20px 0;font-size:0.88em">{bar}</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Exchange rate ─────────────────────────────────────────────────────────
    fx = get_usdthb() if show_thb else 1.0
    sym     = "฿" if show_thb else "$"
    cur_lbl = "THB" if show_thb else "USD"

    def fmt_price(usd_val: float, decimals: int = 2) -> str:
        v = usd_val * fx
        return f"{sym}{v:,.{decimals}f}"

    if show_thb:
        st.caption(f"💱 อัตราแลกเปลี่ยน USD/THB = **{fx:.2f}** (อัปเดตทุก 1 ชม.)")

    # ── Load latest signal from signals.json ──────────────────────────────────
    sig_state = load_signals_json()

    # ── Current signal card ───────────────────────────────────────────────────
    if sig_state:
        sig    = sig_state.get("last_signal", "—")
        price  = sig_state.get("last_price") or 0
        rsi_v  = sig_state.get("last_rsi") or 0
        ma_fv  = sig_state.get("last_ma_fast") or 0
        ma_sv  = sig_state.get("last_ma_slow") or 0
        ma_tv  = sig_state.get("last_ma_trend") or 0
        chk    = sig_state.get("last_checked", "—")
        algo   = sig_state.get("algorithm_used", "—")
        reason = sig_state.get("signal_reason", "—")

        colors = {"BUY": "#a6e3a1", "SELL": "#f38ba8", "HOLD": "#89b4fa"}
        icons  = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
        c      = colors.get(sig, "#cdd6f4")
        ic     = icons.get(sig, "⬜")

        st.markdown(f"""
        <div class="signal-card" style="background:linear-gradient(135deg,#1e1e2e,#313244);
             border:2px solid {c}66">
          <span style="font-size:2em;font-weight:800;color:{c}">{ic} {sig}</span>
          &nbsp;&nbsp;
          <span style="color:#cdd6f4;font-size:1.1em">XAU/USD &nbsp;|&nbsp;
            <b>{fmt_price(price)}</b>
            {'&nbsp;<span style="color:#6c7086;font-size:0.75em">($'+f"{price:,.2f}"+')</span>' if show_thb else ''}
          </span><br>
          <span style="color:#6c7086;font-size:0.88em">
            ตรวจล่าสุด: {chk} &nbsp;|&nbsp; Algorithm: <b style="color:{c}">{algo}</b>
          </span>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics
        km1, km2, km3, km4, km5 = st.columns(5)
        km1.metric(f"XAU/USD ({cur_lbl})", fmt_price(price))
        km2.metric(f"RSI({rsi_buy}/<{rsi_sell})", f"{rsi_v:.1f}",
                   "Oversold" if rsi_v < rsi_buy else ("Overbought" if rsi_v > rsi_sell else "Neutral"))
        km3.metric(f"MA{ma_fast} ({cur_lbl})",  fmt_price(ma_fv, 0))
        km4.metric(f"MA{ma_slow} ({cur_lbl})",  fmt_price(ma_sv, 0))
        km5.metric(f"MA{ma_trend} ({cur_lbl})", fmt_price(ma_tv, 0))
    else:
        st.info("กด **⚡ Check Now** หรือ **▶ Start Monitor** เพื่อดูสัญญาณ")
        sig = None

    st.divider()

    # ── Chart ─────────────────────────────────────────────────────────────────
    chart_title = f"XAU/USD — ราคาทองคำ ({cur_lbl})"
    st.subheader(f"📈 {chart_title}")

    with st.spinner("Loading chart data…"):
        try:
            df = cached_fetch(data_days, float(rsi_buy), float(rsi_sell),
                              ma_fast, ma_slow, ma_trend)
        except Exception as e:
            st.error(f"ดึงข้อมูลไม่ได้: {e}")
            df = None

    if df is not None and len(df) > 0:
        # แปลงราคาใน DataFrame → THB ถ้าเปิด toggle
        df_plot = df.copy()
        price_cols = ["Close", f"MA{ma_fast}", f"MA{ma_slow}", f"MA{ma_trend}"]
        for col in price_cols:
            if col in df_plot.columns:
                df_plot[col] = df_plot[col] * fx

        # ── BUY/SELL markers จาก signal history ──────────────────────────────
        hist = (sig_state or {}).get("history", [])
        buy_dates   = [h["date"][:10] for h in hist if h["signal"] == "BUY"]
        sell_dates  = [h["date"][:10] for h in hist if h["signal"] == "SELL"]
        buy_prices  = [h["price"] * fx for h in hist if h["signal"] == "BUY"]
        sell_prices = [h["price"] * fx for h in hist if h["signal"] == "SELL"]

        y_label = f"ราคา ({sym})"

        # ── 2-panel chart ─────────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.04,
            subplot_titles=(
                f"ราคาทองคำ + Moving Averages ({cur_lbl})",
                f"RSI({rsi_buy}/{rsi_sell})",
            ),
        )

        # Panel 1: Price
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot["Close"],
            name=f"XAU/USD ({cur_lbl})",
            line=dict(color="#f9e2af", width=1.8),
            hovertemplate=f"{sym}%{{y:,.0f}}<extra></extra>",
        ), row=1, col=1)

        ma_colors = {
            f"MA{ma_fast}":  "#89b4fa",
            f"MA{ma_slow}":  "#fab387",
            f"MA{ma_trend}": "#f38ba8",
        }
        for col_name, color in ma_colors.items():
            if col_name in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot.index, y=df_plot[col_name],
                    name=col_name,
                    line=dict(color=color, width=1.2, dash="dot"),
                    hovertemplate=f"{sym}%{{y:,.0f}}<extra>{col_name}</extra>",
                ), row=1, col=1)

        # BUY markers
        if buy_dates:
            fig.add_trace(go.Scatter(
                x=buy_dates, y=buy_prices,
                name="BUY Signal",
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color="#a6e3a1",
                            line=dict(color="#1e1e2e", width=1)),
                hovertemplate=f"BUY<br>{sym}%{{y:,.0f}}<extra></extra>",
            ), row=1, col=1)

        # SELL markers
        if sell_dates:
            fig.add_trace(go.Scatter(
                x=sell_dates, y=sell_prices,
                name="SELL Signal",
                mode="markers",
                marker=dict(symbol="triangle-down", size=14, color="#f38ba8",
                            line=dict(color="#1e1e2e", width=1)),
                hovertemplate=f"SELL<br>{sym}%{{y:,.0f}}<extra></extra>",
            ), row=1, col=1)

        # Panel 2: RSI (ไม่แปลง fx — RSI ไม่มีหน่วยเงิน)
        if "RSI" in df_plot.columns:
            fig.add_trace(go.Scatter(
                x=df_plot.index, y=df_plot["RSI"],
                name="RSI", line=dict(color="#cba6f7", width=1.5),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ), row=2, col=1)

            # Oversold / Overbought zones
            fig.add_hrect(y0=0,        y1=rsi_buy,  row=2, col=1,
                          fillcolor="rgba(166,227,161,0.08)", line_width=0)
            fig.add_hrect(y0=rsi_sell, y1=100,      row=2, col=1,
                          fillcolor="rgba(243,139,168,0.08)", line_width=0)
            fig.add_hline(y=rsi_buy,  row=2, col=1,
                          line=dict(color="#a6e3a1", width=1, dash="dash"),
                          annotation_text=f"Buy {rsi_buy}", annotation_position="left")
            fig.add_hline(y=rsi_sell, row=2, col=1,
                          line=dict(color="#f38ba8", width=1, dash="dash"),
                          annotation_text=f"Sell {rsi_sell}", annotation_position="left")
            fig.add_hline(y=50, row=2, col=1,
                          line=dict(color="#6c7086", width=0.7, dash="dot"))

        fig.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=60, b=20, l=0, r=0),
            hovermode="x unified",
            xaxis=dict(rangeslider=dict(visible=False), gridcolor="#313244"),
            yaxis=dict(title=y_label, gridcolor="#313244",
                       tickprefix=sym, tickformat=",.0f"),
            xaxis2=dict(gridcolor="#313244"),
            yaxis2=dict(gridcolor="#313244", range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Algorithm Explanation ─────────────────────────────────────────────────
    st.subheader("🧮 Algorithm Breakdown")

    if sig_state:
        reason = sig_state.get("signal_reason", "—")
        algo   = sig_state.get("algorithm_used", "—")
        rsi_v  = sig_state.get("last_rsi") or 0
        ma_fv  = sig_state.get("last_ma_fast") or 0
        ma_sv  = sig_state.get("last_ma_slow") or 0
        ma_tv  = sig_state.get("last_ma_trend") or 0

        # RSI condition
        rsi_fired = algo == "RSI"
        if rsi_v < rsi_buy:
            rsi_status = f"🟢 BUY — Oversold ({rsi_v:.1f} < {rsi_buy})"
            rsi_col    = "#a6e3a1"
        elif rsi_v > rsi_sell:
            rsi_status = f"🔴 SELL — Overbought ({rsi_v:.1f} > {rsi_sell})"
            rsi_col    = "#f38ba8"
        else:
            rsi_status = f"⚪ HOLD — Neutral ({rsi_v:.1f} อยู่ใน {rsi_buy}–{rsi_sell})"
            rsi_col    = "#6c7086"

        # MA condition
        ma_fired = "MA" in algo
        if "ข้ามขึ้น" in reason and "MA" in algo:
            ma_status = f"🟢 BUY — MA{ma_fast} ข้ามขึ้นเหนือ MA{ma_slow} + ราคาเหนือ MA{ma_trend}"
            ma_col    = "#a6e3a1"
        elif "ข้ามลง" in reason and "MA" in algo:
            ma_status = f"🔴 SELL — MA{ma_fast} ข้ามลงใต้ MA{ma_slow}"
            ma_col    = "#f38ba8"
        else:
            ma_status = f"⚪ HOLD — ไม่มี MA Crossover"
            ma_col    = "#6c7086"

        col_a, col_b = st.columns(2)

        with col_a:
            border_a = "#f9e2af44" if rsi_fired else "#31324466"
            st.markdown(f"""
            <div class="algo-card" style="border-color:{border_a}">
              <div style="font-size:1.1em;font-weight:700;margin-bottom:10px">
                📊 Algorithm 1: RSI
                {'&nbsp;<span style="background:#f9e2af22;color:#f9e2af;'
                 'font-size:0.7em;padding:2px 8px;border-radius:99px">✅ WINNER</span>'
                 if rsi_fired else ''}
              </div>
              <div style="color:{rsi_col};font-size:1em;font-weight:600">{rsi_status}</div>
              <hr style="border-color:#31324466;margin:10px 0">
              <div style="color:#6c7086;font-size:0.85em">
                RSI วัด momentum ของราคา<br>
                &lt; {rsi_buy} = Oversold → สัญญาณ <b>BUY</b><br>
                &gt; {rsi_sell} = Overbought → สัญญาณ <b>SELL</b><br>
                ค่าปัจจุบัน: <b style="color:#cdd6f4">{rsi_v:.1f}</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            border_b = "#f9e2af44" if ma_fired else "#31324466"
            st.markdown(f"""
            <div class="algo-card" style="border-color:{border_b}">
              <div style="font-size:1.1em;font-weight:700;margin-bottom:10px">
                📈 Algorithm 2: MA Crossover
                {'&nbsp;<span style="background:#f9e2af22;color:#f9e2af;'
                 'font-size:0.7em;padding:2px 8px;border-radius:99px">✅ WINNER</span>'
                 if ma_fired else ''}
              </div>
              <div style="color:{ma_col};font-size:1em;font-weight:600">{ma_status}</div>
              <hr style="border-color:#31324466;margin:10px 0">
              <div style="color:#6c7086;font-size:0.85em">
                MA{ma_fast}={fmt_price(ma_fv,0)} &nbsp;/&nbsp;
                MA{ma_slow}={fmt_price(ma_sv,0)} &nbsp;/&nbsp;
                MA{ma_trend}={fmt_price(ma_tv,0)}<br>
                MA{ma_fast} ข้ามขึ้นเหนือ MA{ma_slow} + ราคา &gt; MA{ma_trend} → <b>BUY</b><br>
                MA{ma_fast} ข้ามลงใต้ MA{ma_slow} → <b>SELL</b><br>
                (RSI wins ถ้า RSI มีสัญญาณด้วย)
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Final decision arrow
        sig_now = sig_state.get("last_signal", "HOLD")
        c_now   = colors.get(sig_now, "#cdd6f4")
        st.markdown(f"""
        <div style="background:#181825;border-radius:10px;padding:14px 20px;
                    margin-top:12px;border-left:4px solid {c_now}">
          <b>🏆 Final Decision: <span style="color:{c_now}">{sig_now}</span></b>
          &nbsp;&nbsp;|&nbsp;&nbsp;
          Triggered by: <b>{algo or "ไม่มีสัญญาณชัดเจน"}</b>
          <br><span style="color:#6c7086;font-size:0.88em">{reason}</span>
        </div>
        """, unsafe_allow_html=True)

        # Tie-break rule
        with st.expander("ℹ️ กฎ Tie-break: เมื่อทั้ง 2 algorithm ให้สัญญาณ"):
            st.markdown(f"""
| สถานการณ์ | ผลลัพธ์ |
|-----------|---------|
| RSI = BUY, MA = HOLD | **BUY** (RSI wins) |
| RSI = HOLD, MA = BUY | **BUY** (MA wins) |
| RSI = BUY, MA = BUY  | **BUY** (ตรงกัน) |
| RSI = SELL, MA = BUY | **SELL** (RSI wins) |
| RSI = BUY, MA = SELL | **BUY** (RSI wins) |
| RSI = HOLD, MA = HOLD | **HOLD** |

**RSI มีความสำคัญสูงกว่า MA** เพราะวัด momentum ได้เร็วกว่า
            """)
    else:
        st.info("ยังไม่มีข้อมูล — กด **⚡ Check Now** ก่อน")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SIGNAL HISTORY
# ═════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.subheader("📋 Signal History")

    sig_state2 = load_signals_json()

    if sig_state2 is None or not sig_state2.get("history"):
        st.info("ยังไม่มี history — รอ Monitor ทำงานสักครู่")
    else:
        hist = sig_state2["history"]

        # ── Filter ────────────────────────────────────────────────────────────
        fc1, fc2 = st.columns([2, 4])
        filter_sig = fc1.multiselect(
            "กรอง Signal", ["BUY", "SELL", "HOLD"],
            default=["BUY", "SELL", "HOLD"],
        )
        filtered = [h for h in hist if h.get("signal") in filter_sig]

        st.caption(f"แสดง {len(filtered)} รายการ จาก {len(hist)} ทั้งหมด")

        if filtered:
            price_col_label = f"Price ({cur_lbl})"
            rows = []
            for h in reversed(filtered):
                sig_h  = h.get("signal", "—")
                icon_h = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig_h, "")
                p      = h.get("price", 0) * fx
                mf     = h.get("ma_fast",  0) * fx
                ms     = h.get("ma_slow",  0) * fx
                mt     = h.get("ma_trend", 0) * fx
                rows.append({
                    "Date":             str(h.get("date", ""))[:19].replace("T", " "),
                    "Signal":           f"{icon_h} {sig_h}",
                    price_col_label:    f"{sym}{p:,.0f}",
                    "RSI":              f"{h.get('rsi', 0):.1f}",
                    f"MA{ma_fast}":     f"{sym}{mf:,.0f}",
                    f"MA{ma_slow}":     f"{sym}{ms:,.0f}",
                    f"MA{ma_trend}":    f"{sym}{mt:,.0f}",
                    "Algorithm":        h.get("algorithm", "—"),
                    "Reason":           h.get("reason", ""),
                })

            hist_df = pd.DataFrame(rows)

            def _row_style(row):
                s = str(row.get("Signal", ""))
                if "BUY"  in s: return ["background:rgba(166,227,161,0.07)"] * len(row)
                if "SELL" in s: return ["background:rgba(243,139,168,0.07)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                hist_df.style.apply(_row_style, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # ── Mini signal timeline chart ─────────────────────────────────────
            st.divider()
            st.markdown("**Signal Timeline**")

            non_hold_h = [h for h in filtered if h.get("signal") in ("BUY", "SELL")]
            if len(non_hold_h) >= 1:
                tl_fig = go.Figure()
                for sig_type, color, mkr_sym in [
                    ("BUY",  "#a6e3a1", "triangle-up"),
                    ("SELL", "#f38ba8", "triangle-down"),
                ]:
                    pts = [h for h in non_hold_h if h.get("signal") == sig_type]
                    if pts:
                        tl_fig.add_trace(go.Scatter(
                            x=[p["date"][:10]      for p in pts],
                            y=[p["price"] * fx     for p in pts],
                            name=sig_type,
                            mode="markers+text",
                            text=[sig_type] * len(pts),
                            textposition="top center",
                            marker=dict(symbol=mkr_sym, size=14, color=color),
                            hovertemplate=f"{sig_type}<br>{sym}%{{y:,.0f}}<extra></extra>",
                        ))

                tl_fig.update_layout(
                    height=220, template="plotly_dark",
                    paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                    margin=dict(t=10, b=20, l=0, r=0),
                    showlegend=True,
                    yaxis=dict(title=f"Price ({cur_lbl})", gridcolor="#313244",
                               tickprefix=sym, tickformat=",.0f"),
                    xaxis=dict(gridcolor="#313244"),
                )
                st.plotly_chart(tl_fig, use_container_width=True)

            # ── Stats ──────────────────────────────────────────────────────────
            st.divider()
            st.subheader("📊 Statistics")
            total   = len(hist)
            n_buy   = sum(1 for h in hist if h.get("signal") == "BUY")
            n_sell  = sum(1 for h in hist if h.get("signal") == "SELL")
            n_hold  = sum(1 for h in hist if h.get("signal") == "HOLD")

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Checks", total)
            sc2.metric("BUY signals",  n_buy)
            sc3.metric("SELL signals", n_sell)
            sc4.metric("HOLD signals", n_hold)

            algo_counts = {}
            for h in hist:
                a = h.get("algorithm") or "—"
                algo_counts[a] = algo_counts.get(a, 0) + 1
            if algo_counts:
                st.markdown("**Algorithm Trigger Count:**")
                for a, cnt in sorted(algo_counts.items(), key=lambda x: -x[1]):
                    st.markdown(f"- **{a}**: {cnt} ครั้ง")

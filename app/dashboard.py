"""
app/dashboard.py
================
Quant Predator v2 — Live Trading Dashboard.

Run with:
    streamlit run app/dashboard.py

WHAT MAKES IT FEEL ALIVE:
    - Client-side JS countdown to the next H1 bar close (ticks every second,
      no server round-trip, never flickers).
    - Pulsing LIVE dot that turns RED if the engine goes silent (> 10s stale).
    - Live evaluation feed: one row per H1 bar, written by live_trader.py.
      Shows every hourly check — MTF, RSI, FVG distance, signal, reason.
      Updates every 30s. Never quiet even on 38-hour gaps between trades.
    - Signal flash: brief glow animation when the signal changes state.
    - Subsystem health strip: MT5 ● NEWS ● OLLAMA ● coloured by real status.

FRAGMENT REFRESH SCHEDULE:
    render_hud()        every  2s   balance, equity, position, heartbeat, health
    render_signal()     every  2s   AI signal card with flash animation
    render_eval_feed()  every 30s   hourly bar evaluation log
    render_analytics()  every 30s   win-rate + capital flow donuts
    render_chart()      every 30s   H1 candlestick + SL/TP lines
    render_journal()    every 60s   trade ledger table
    Chat panel: event-driven only (no polling)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Paths
from ledger.state_manager import StateManager

# ==============================================================================
# PAGE CONFIG & GLOBAL CSS
# ==============================================================================

st.set_page_config(
    page_title     = "Quant Predator v2",
    page_icon      = "🤖",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

_CSS = """
<style>
/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body, [data-testid="stApp"] {
    background-color: #0d1117 !important;
    color: #e6edf3;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 600;
    color: #e6edf3;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: #7d8590 !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* ── Cards ─────────────────────────────────────────────────────────────────── */
.qp-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 18px;
}
.qp-card-title {
    font-size: 0.68rem;
    color: #7d8590;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 6px;
}

/* ── Signal card ────────────────────────────────────────────────────────────── */
.sig-card {
    border-radius: 8px;
    padding: 16px 22px;
    text-align: center;
    border: 1px solid;
    transition: border-color 0.4s ease;
}
.sig-text {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin: 0;
}
.sig-sub {
    font-size: 0.78rem;
    color: #7d8590;
    margin: 4px 0 0;
}

/* ── Signal flash animation ─────────────────────────────────────────────────── */
@keyframes sigFlash {
    0%   { box-shadow: 0 0 0px 0px currentColor; opacity: 0.6; }
    40%  { box-shadow: 0 0 18px 4px currentColor; opacity: 1.0; }
    100% { box-shadow: 0 0 0px 0px currentColor; opacity: 1.0; }
}
.sig-flash { animation: sigFlash 0.7s ease-out forwards; }

/* ── Heartbeat dot ──────────────────────────────────────────────────────────── */
@keyframes dotPulse {
    0%,100% { opacity: 1.0; }
    50%      { opacity: 0.25; }
}
.hb-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    vertical-align: middle;
    margin-right: 5px;
    animation: dotPulse 2s ease-in-out infinite;
}

/* ── Eval feed ──────────────────────────────────────────────────────────────── */
.eval-feed {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 14px;
    height: 220px;
    overflow-y: auto;
    font-size: 0.78rem;
    line-height: 1.9;
}
.eval-row { display: flex; gap: 10px; align-items: baseline; }
.eval-time { color: #7d8590; min-width: 40px; }
.eval-session { color: #7d8590; min-width: 54px; font-size: 0.70rem; }
.eval-tag {
    font-size: 0.68rem;
    border-radius: 3px;
    padding: 1px 6px;
    font-weight: 600;
}
.eval-meta { color: #7d8590; font-size: 0.72rem; }
.eval-hold .eval-tag  { background:#21262d; color:#7d8590; }
.eval-buy  .eval-tag  { background:#0d2f23; color:#00c48c; }
.eval-sell .eval-tag  { background:#2f0d0d; color:#ff4c4c; }
.eval-buy  .eval-time,
.eval-buy  .eval-session { color: #00c48c; }
.eval-sell .eval-time,
.eval-sell .eval-session { color: #ff4c4c; }

/* ── Health dots ────────────────────────────────────────────────────────────── */
.health-row { display: flex; gap: 12px; align-items: center; padding-top: 2px; }
.health-item { display: flex; align-items: center; gap: 4px; font-size: 0.72rem; color: #7d8590; }
.h-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.h-ok  { background: #00c48c; }
.h-err { background: #ff4c4c; }

/* ── Streamlit cleanup ──────────────────────────────────────────────────────── */
.block-container { padding-top: 1.2rem !important; }
hr { border-color: #21262d !important; }
.stDataFrame { font-size: 0.80rem; }
[data-testid="stDataFrameResizable"] th {
    background: #161b22 !important;
    color: #7d8590 !important;
    font-size: 0.70rem;
    text-transform: uppercase;
}
</style>
"""

# Colour palette (accessed by name in Python too)
_C = {
    "buy":      "#00c48c",
    "sell":     "#ff4c4c",
    "hold":     "#7d8590",
    "warning":  "#f0a500",
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "border":   "#21262d",
    "text":     "#e6edf3",
    "muted":    "#7d8590",
}

_STATE = StateManager()


# ==============================================================================
# HELPERS
# ==============================================================================

def _sig_colour(signal: str) -> str:
    s = signal.upper()
    if s == "BUY":        return _C["buy"]
    if s == "SELL":       return _C["sell"]
    if s in ("NEWS HALT","HALTED","ERROR"): return _C["warning"]
    return _C["hold"]


def _load_eval_log() -> List[Dict]:
    return _STATE.read_eval_log()

def _seconds_since(ts_str: str) -> float:
    """Returns seconds elapsed since a '%Y-%m-%d %H:%M:%S' timestamp string."""
    try:
        then = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - then).total_seconds()
    except Exception:
        return 9999.0

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Converts a 6-digit hex color to an rgba() string Plotly accepts
    for fillcolor — Plotly does not support 8-digit hex-with-alpha."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ==============================================================================
# CLIENT-SIDE COUNTDOWN COMPONENT
# ==============================================================================

def _countdown_html(height: int = 28) -> None:
    """
    Injects a pure client-side JS countdown to next H1 bar close.
    Ticks every second. Zero server round-trips — never flickers on refresh.
    Renders as a small inline snippet; invisible iframe borders suppressed.
    """
    html = """
    <div style="font-family:'JetBrains Mono',monospace;
                font-size:0.80rem;color:#7d8590;
                display:flex;align-items:center;gap:6px;">
        <span style="color:#21262d;">▐</span>
        <span>Next H1 close in</span>
        <span id="cd" style="color:#e6edf3;font-weight:600;
                              letter-spacing:0.05em;">--:--</span>
    </div>
    <script>
    (function(){
        function tick(){
            var now  = new Date();
            var next = new Date(now);
            next.setHours(now.getHours()+1, 0, 0, 0);
            var diff = Math.max(0, next - now);
            var m = Math.floor(diff/60000);
            var s = Math.floor((diff%60000)/1000);
            var el = document.getElementById('cd');
            if(el) el.textContent =
                String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
        }
        tick();
        setInterval(tick, 1000);
    })();
    </script>
    """
    components.html(html, height=height, scrolling=False)


# ==============================================================================
# SIDEBAR
# ==============================================================================

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"<h2 style='margin:0 0 4px;color:{_C['text']};font-size:1.1rem;'>"
            "🤖 Quant Predator v2</h2>"
            f"<p style='font-size:0.72rem;color:{_C['muted']};margin:0 0 16px;'>"
            "EURUSD H1 · PPO-LSTM · Run 4</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        config   = _STATE.read_config()
        new_auto = st.toggle("⚡ Auto-Trading", value=config.get("auto_trade", True))
        new_sym  = st.selectbox(
            "💱 Symbol",
            ["EURUSD","XAUUSD","GBPUSD","USDJPY"],
            index=["EURUSD","XAUUSD","GBPUSD","USDJPY"].index(
                config.get("symbol","EURUSD"))
        )

        if new_auto != config.get("auto_trade") or new_sym != config.get("symbol"):
            config.update({"auto_trade": new_auto, "symbol": new_sym})
            _STATE.write_config(config)
            st.rerun()

        if not new_auto:
            st.divider()
            st.markdown(
                f"<p style='font-size:0.72rem;color:{_C['muted']};'>"
                "MANUAL OVERRIDE</p>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            if c1.button("🟢 BUY",  width='stretch'):
                config["manual_command"] = "BUY"; _STATE.write_config(config)
                st.toast("BUY dispatched", icon="📤")
            if c2.button("🔴 SELL", width='stretch'):
                config["manual_command"] = "SELL"; _STATE.write_config(config)
                st.toast("SELL dispatched", icon="📤")
            if st.button("⬛ CLOSE ALL", width='stretch'):
                config["manual_command"] = "CLOSE"; _STATE.write_config(config)
                st.toast("CLOSE ALL dispatched", icon="📤")

        st.divider()
        st.markdown(
            f"<p style='font-size:0.68rem;color:{_C['muted']};'>RUN 4 BENCHMARK</p>"
            f"<p style='font-size:0.80rem;color:{_C['text']};margin:2px 0;'>"
            "Sharpe 1.090 · WR 29.1%</p>"
            f"<p style='font-size:0.80rem;color:{_C['text']};margin:2px 0;'>"
            "R:R 3.19:1 · Exp $36.34</p>",
            unsafe_allow_html=True,
        )


# ==============================================================================
# FRAGMENT: HUD — balance, equity, position, heartbeat, health (2s)
# ==============================================================================

@st.fragment(run_every="2s")
def render_hud() -> None:
    state   = _STATE.read_live_state()
    balance = state.get("balance",         0.0)
    equity  = state.get("equity",          0.0)
    fp      = state.get("floating_profit", 0.0)
    pos     = state.get("open_position",  "FLAT")
    updated = state.get("last_update",    "")
    mt5_ok  = state.get("mt5_ok",         False)
    oll_ok  = state.get("ollama_ok",       False)

    # Heartbeat: stale if last_update is > 10s ago
    stale   = _seconds_since(updated) > 10 if updated else True
    dot_col = "#ff4c4c" if stale else "#00c48c"
    dot_lbl = "STALE" if stale else "LIVE"

    today_pnl = fp   # Simplification: use floating as today's P&L proxy

    col_dot, col_b, col_e, col_p, col_td, col_h = st.columns([1.4,2,2,1.8,2,1.8])

    with col_dot:
        st.markdown(
            f"<div style='padding-top:22px;'>"
            f"<span class='hb-dot' style='background:{dot_col};'></span>"
            f"<span style='font-size:0.68rem;color:{dot_col};'>{dot_lbl}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.metric("Balance", f"${balance:,.2f}")
    with col_e:
        st.metric("Equity",  f"${equity:,.2f}",
                  delta=f"{fp:+.2f}", delta_color="normal")
    with col_p:
        icon = "🟢" if pos=="LONG" else "🔴" if pos=="SHORT" else "⚪"
        st.metric("Position", f"{icon} {pos}")
    with col_td:
        st.metric("Floating P&L", f"${today_pnl:+.2f}")

    # Subsystem health dots
    with col_h:
        news_ok = state.get("news_status","") != "ERROR"
        st.markdown(
            f"<div style='padding-top:14px;'>"
            f"<p style='font-size:0.67rem;color:{_C['muted']};margin:0 0 5px;'>SYSTEMS</p>"
            f"<div class='health-row'>"
            f"  <div class='health-item'>"
            f"    <span class='h-dot {'h-ok' if mt5_ok else 'h-err'}'></span>MT5</div>"
            f"  <div class='health-item'>"
            f"    <span class='h-dot {'h-ok' if news_ok else 'h-err'}'></span>NEWS</div>"
            f"  <div class='health-item'>"
            f"    <span class='h-dot {'h-ok' if oll_ok else 'h-err'}'></span>LLM</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()


# ==============================================================================
# FRAGMENT: SIGNAL CARD + RADAR + COUNTDOWN (2s)
# ==============================================================================

@st.fragment(run_every="2s")
def render_signal() -> None:
    state      = _STATE.read_live_state()
    signal     = state.get("ai_signal",   "WAITING")
    news       = state.get("news_status", "Checking...")
    prev_signal= st.session_state.get("_prev_signal", signal)

    changed  = signal != prev_signal
    st.session_state["_prev_signal"] = signal

    sig_col  = _sig_colour(signal)
    flash_cls= "sig-flash" if changed else ""

    # Determine sub-label
    eval_log = _load_eval_log()
    reason   = eval_log[0].get("reason","") if eval_log else ""
    sub_text = f"— {reason}" if reason else ""

    news_col = _C["warning"] if "HIGH IMPACT" in news else _C["muted"]

    col_sig, col_radar = st.columns([3, 3])

    with col_sig:
        st.markdown(
            f"<div class='sig-card {flash_cls}' "
            f"style='border-color:{sig_col}44;background:{sig_col}0d;'>"
            f"<p class='sig-text' style='color:{sig_col};'>🧠 {signal}</p>"
            f"<p class='sig-sub'>{sub_text}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_radar:
        st.markdown(
            f"<div class='sig-card' "
            f"style='border-color:{news_col}44;background:{news_col}0d;'>"
            f"<p class='sig-text' style='color:{news_col};font-size:1.15rem;'>"
            f"📡 {news}</p>"
            f"<p class='sig-sub'>ForexFactory radar</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()


# ==============================================================================
# FRAGMENT: LIVE EVALUATION FEED (30s)
# ==============================================================================

@st.fragment(run_every="30s")
def render_eval_feed() -> None:
    entries = _load_eval_log()

    col_feed, col_stats = st.columns([3, 1])

    with col_feed:
        st.markdown(
            f"<p class='qp-card-title'>⚡ Live Evaluation Feed</p>",
            unsafe_allow_html=True,
        )

        if not entries:
            st.markdown(
                "<div class='eval-feed'>"
                "<span style='color:#7d8590;'>Waiting for first bar evaluation…"
                "<br>Engine writes one entry per H1 bar close.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            rows_html = ""
            for e in entries[:30]:
                sig      = e.get("signal","HOLD")
                cls      = f"eval-{sig.lower()}" if sig in ("BUY","SELL") else "eval-hold"
                mtf_val  = e.get("mtf", 0)
                mtf_str  = f"{mtf_val:+d}" if isinstance(mtf_val, int) else "—"
                rsi_str  = f"{e.get('rsi',0):.0f}"
                bull     = e.get("fvg_bull", 0.0)
                bear     = e.get("fvg_bear", 0.0)
                fvg_str  = (f"↑{bull:.1f}" if bull > 0 else
                            f"↓{bear:.1f}" if bear > 0 else "—")
                reason   = e.get("reason","")
                reason_s = f"({reason})" if reason else ""
                date_s   = e.get("date","")

                rows_html += (
                    f"<div class='eval-row {cls}'>"
                    f"  <span class='eval-time'>{e.get('t','')}</span>"
                    f"  <span class='eval-session'>[{e.get('session','')}]</span>"
                    f"  <span class='eval-meta'>MTF:{mtf_str} RSI:{rsi_str}"
                    f" FVG:{fvg_str}</span>"
                    f"  <span class='eval-tag'>{sig}</span>"
                    f"  <span class='eval-meta' style='color:#484f58;'>{reason_s}</span>"
                    f"</div>"
                )

            st.markdown(
                f"<div class='eval-feed'>{rows_html}</div>",
                unsafe_allow_html=True,
            )

    with col_stats:
        if entries:
            n_today = len([e for e in entries if e.get("date") == entries[0].get("date")])
            n_buys  = len([e for e in entries if e.get("signal") == "BUY"])
            n_sells = len([e for e in entries if e.get("signal") == "SELL"])
            last_trade = next(
                (e for e in entries if e.get("signal") in ("BUY","SELL")), None
            )
            st.markdown(
                f"<div class='qp-card'>"
                f"<p class='qp-card-title'>Today</p>"
                f"<p style='font-size:1.4rem;font-weight:600;margin:0 0 10px;'>"
                f"{n_today} bars</p>"
                f"<p style='font-size:0.78rem;color:{_C['buy']};margin:2px 0;'>"
                f"▲ {n_buys} long signal{'s' if n_buys!=1 else ''}</p>"
                f"<p style='font-size:0.78rem;color:{_C['sell']};margin:2px 0;'>"
                f"▼ {n_sells} short signal{'s' if n_sells!=1 else ''}</p>"
                + (
                    f"<p style='font-size:0.72rem;color:{_C['muted']};margin:8px 0 0;'>"
                    f"Last trade:<br>{last_trade.get('date','')} "
                    f"{last_trade.get('t','')} "
                    f"({last_trade.get('signal','')})</p>"
                    if last_trade else
                    f"<p style='font-size:0.72rem;color:{_C['muted']};margin:8px 0 0;'>"
                    f"No trades yet</p>"
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    st.divider()


# ==============================================================================
# FRAGMENT: CHART (30s)
# ==============================================================================

@st.fragment(run_every="30s")
def render_chart() -> None:
    config = _STATE.read_config()
    state  = _STATE.read_live_state()
    pos    = state.get("open_position","FLAT")

    sl_price = tp_price = 0.0
    if pos != "FLAT":
        jdf = _STATE.read_journal()
        if not jdf.empty:
            entries = jdf[jdf["Action"].isin(["BUY","SELL"])]
            if not entries.empty:
                last = entries.iloc[-1]
                sl_price = float(last.get("SL",0.0))
                tp_price = float(last.get("TP",0.0))

    if not os.path.exists(Paths.CHART_FILE):
        st.info("Chart data loading — waiting for engine to write first bar.")
        return

    try:
        df = pd.read_csv(Paths.CHART_FILE)
        if df.empty:
            return

        fig = go.Figure(data=[go.Candlestick(
            x                    = df.get("time", df.index),
            open                 = df["open"],
            high                 = df["high"],
            low                  = df["low"],
            close                = df["close"],
            increasing_line_color= _C["buy"],
            decreasing_line_color= _C["sell"],
            increasing_fillcolor=_hex_to_rgba(_C["buy"], 0.33),
            decreasing_fillcolor=_hex_to_rgba(_C["sell"], 0.33),
            name                 = "EURUSD",
            line                 = dict(width=1),
        )])

        if pos != "FLAT" and sl_price > 0:
            fig.add_hline(
                y=sl_price, line_dash="dot",
                line_color=_C["sell"], line_width=1.5,
                annotation_text=f"SL {sl_price:.5f}",
                annotation_position="bottom right",
                annotation_font_color=_C["sell"],
                annotation_font_size=11,
            )
        if pos != "FLAT" and tp_price > 0:
            fig.add_hline(
                y=tp_price, line_dash="dot",
                line_color=_C["buy"], line_width=1.5,
                annotation_text=f"TP {tp_price:.5f}",
                annotation_position="top right",
                annotation_font_color=_C["buy"],
                annotation_font_size=11,
            )

        fig.update_layout(
            title      = dict(
                text=f"{config.get('symbol','EURUSD')} H1  ·  {len(df)} bars",
                font=dict(color=_C["muted"], size=12),
            ),
            template   = "plotly_dark",
            height     = 440,
            paper_bgcolor = _C["bg"],
            plot_bgcolor  = _C["bg"],
            xaxis_rangeslider_visible = False,
            uirevision = "chart",
            dragmode   = "pan",
            margin     = dict(l=0, r=0, t=36, b=0),
            xaxis = dict(
                showgrid=True, gridcolor="#21262d", gridwidth=0.5,
                zeroline=False, color=_C["muted"],
            ),
            yaxis = dict(
                showgrid=True, gridcolor="#21262d", gridwidth=0.5,
                zeroline=False, color=_C["muted"], side="right",
            ),
            legend = dict(font=dict(color=_C["muted"])),
        )

        st.plotly_chart(fig, width='stretch',
            config={"scrollZoom":True,"displaylogo":False,
                    "modeBarButtonsToRemove":["lasso2d","select2d"]})
    except Exception as exc:
        st.warning(f"Chart loading… ({exc})")

    st.divider()


# ==============================================================================
# FRAGMENT: ANALYTICS DONUTS (30s)
# ==============================================================================

@st.fragment(run_every="30s")
def render_analytics() -> None:
    df = _STATE.read_journal()

    if df.empty or "PnL" not in df.columns:
        closed = pd.DataFrame(columns=["PnL"])
    else:
        closed = df[df["PnL"] != 0.0].copy()

    wins   = len(closed[closed["PnL"] > 0]) if not closed.empty else 0
    losses = len(closed[closed["PnL"] < 0]) if not closed.empty else 0
    total  = wins + losses
    wr     = wins / total * 100 if total > 0 else 0
    gp     = closed[closed["PnL"] > 0]["PnL"].sum() if not closed.empty else 0.0
    gl     = abs(closed[closed["PnL"] < 0]["PnL"].sum()) if not closed.empty else 0.0
    net    = gp - gl

    col_wr, col_pnl = st.columns(2)

    def _donut(labels, values, colours, annotation, title, no_data=False):
        if no_data:
            # Single flat ring in muted grey, rather than a misleading
            # 50/50 split when there's genuinely nothing to show yet.
            labels, values, colours = ["No data"], [1], [_C["muted"] + "33"]
        fig = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.70,
            marker_colors=colours, textinfo="none",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(color=_C["muted"], size=12)),
            template="plotly_dark",
            paper_bgcolor=_C["bg"], plot_bgcolor=_C["bg"],
            height=220, margin=dict(t=36,b=8,l=8,r=8),
            showlegend=not no_data,
            legend=dict(font=dict(color=_C["muted"], size=11)),
            annotations=[dict(
                text=annotation, x=0.5, y=0.5, showarrow=False,
                font=dict(size=18, color=_C["text"] if not no_data else _C["muted"]),
            )],
        )
        return fig

    with col_wr:
        st.plotly_chart(
            _donut(
                ["Wins","Losses"],
                [max(wins,0.01), max(losses,0.01)],
                [_C["buy"], _C["sell"]],
                f"{wr:.1f}%" if total > 0 else "—",
                f"Win Rate  ({total} trades)",
                no_data=(total == 0),
            ),
            width='stretch',
        )

    with col_pnl:
        sign = "+" if net >= 0 else ""
        st.plotly_chart(
            _donut(
                ["Gross Profit","Gross Loss"],
                [max(gp, 0.01), max(gl, 0.01)],
                [_C["buy"], _C["sell"]],
                f"{sign}${net:.0f}" if total > 0 else "—",
                "Capital Flow",
                no_data=(total == 0),
            ),
            width='stretch',
        )

    st.divider()
# ==============================================================================
# FRAGMENT: TRADE JOURNAL (60s)
# ==============================================================================

@st.fragment(run_every="60s")
def render_journal() -> None:
    st.markdown(
        f"<p class='qp-card-title'>📋 Trade Ledger</p>",
        unsafe_allow_html=True,
    )
    try:
        df = pd.read_csv(Paths.JOURNAL_FILE)
    except Exception:
        st.info("No trades recorded yet.")
        return

    if df.empty:
        st.info("No trades recorded yet.")
        return

    df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

    def _col_action(v):
        if v == "BUY":       return f"color:{_C['buy']};font-weight:700"
        if v == "SELL":      return f"color:{_C['sell']};font-weight:700"
        if v == "BUY_CLOSE": return f"color:{_C['buy']}99"
        if v == "SELL_CLOSE":return f"color:{_C['sell']}99"
        return f"color:{_C['muted']}"

    def _col_pnl(v):
        try:
            val = float(v)
            if val > 0: return f"color:{_C['buy']};font-weight:600"
            if val < 0: return f"color:{_C['sell']};font-weight:600"
        except Exception:
            pass
        return f"color:{_C['muted']}"

    styled = (
        df.style
        .map(_col_action, subset=["Action"])
        .map(_col_pnl,    subset=["PnL"])
    )
    st.dataframe(styled, width='stretch', height=260)
    st.divider()


# ==============================================================================
# CHAT PANEL (event-driven)
# ==============================================================================

def render_chat() -> None:
    st.markdown(
        f"<p class='qp-card-title'>💬 Ask the AI</p>",
        unsafe_allow_html=True,
    )
    if "llm" not in st.session_state:
        try:
            from news.llm_analyst import LLMAnalyst
            st.session_state.llm = LLMAnalyst()
        except Exception:
            st.warning("LLM module unavailable.")
            return

    llm = st.session_state.llm
    if not llm.is_available():
        st.markdown(
            f"<p style='font-size:0.80rem;color:{_C['muted']};'>"
            "Ollama offline. Run <code>ollama serve</code> then "
            "<code>ollama pull llama3</code>.</p>",
            unsafe_allow_html=True,
        )
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask about signals, risk, strategy…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.write(prompt)
        context = _STATE.read_live_state()
        with st.chat_message("assistant"):
            response = st.write_stream(llm.stream_chat(prompt, context))
        st.session_state.chat_history.append({"role":"assistant","content":response})
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    # Inject global CSS once per page load
    st.markdown(_CSS, unsafe_allow_html=True)

    # Title row
    col_title, col_cd = st.columns([5, 1.4])
    with col_title:
        st.markdown(
            f"<h1 style='font-size:1.25rem;font-weight:700;color:{_C['text']};"
            f"margin:0 0 4px;letter-spacing:0.04em;'>"
            "🤖 QUANT PREDATOR v2 — LIVE TERMINAL</h1>"
            f"<p style='font-size:0.72rem;color:{_C['muted']};margin:0 0 16px;'>"
            "EURUSD H1 · PPO-LSTM · Run 4 · best_model.zip</p>",
            unsafe_allow_html=True,
        )
    with col_cd:
        # Rendered ONCE outside any fragment — the injected JS timer runs
        # forever in the browser, completely untouched by the 2s/30s/60s
        # fragment reruns happening elsewhere on the page. This is what
        # makes it tick smoothly instead of restarting every few seconds.
        st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
        _countdown_html(height=32)

    _render_sidebar()

    render_hud()
    render_signal()
    render_eval_feed()
    render_chart()
    render_analytics()
    render_journal()
    render_chat()


main()
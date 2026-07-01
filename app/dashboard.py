"""
app/dashboard.py
================
Quant Predator v2 — Live Trading Dashboard.

Run with:
    streamlit run app/dashboard.py

FRAGMENT REFRESH SCHEDULE:
    render_hud()        every  2s  — balance, equity, position, heartbeat
    render_signal()     every  2s  — AI signal banner + news radar
    render_analytics()  every 10s  — win rate and capital flow donuts
    render_chart()      every 30s  — H1 candlestick with SL/TP lines
    render_journal()    every 30s  — trade ledger table

Chat panel is event-driven (on user input only, no auto-refresh).
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Paths
from ledger.state_manager import StateManager

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title     = "Quant Predator v2",
    page_icon      = "🤖",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# Dark theme CSS overrides — matches the dark terminal aesthetic
st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .stMetricValue { font-size: 1.6rem !important; font-weight: 700; }
  .stMetricLabel { color: #8b949e !important; font-size: 0.75rem; }
  div[data-testid="stMetricDelta"] { font-size: 0.9rem; }
  .signal-card {
    padding: 14px 20px;
    border-radius: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid;
    text-align: center;
  }
  .stDataFrame { font-size: 0.82rem; }
  .chat-container { border-top: 1px solid #30363d; padding-top: 16px; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# Shared StateManager instance
_state_mgr = StateManager()


# ==============================================================================
# SIDEBAR
# ==============================================================================

def _render_sidebar() -> None:
    """Static sidebar — reloads only on st.rerun() or full page refresh."""
    with st.sidebar:
        st.markdown("## 🤖 Quant Predator v2")
        st.markdown("---")

        config = _state_mgr.read_config()

        # ── Trading controls ────────────────────────────────────────────────
        new_auto = st.toggle("⚡ Auto-Trading", value=config.get("auto_trade", True))
        new_sym  = st.selectbox(
            "💱 Pair",
            ["EURUSD", "XAUUSD", "GBPUSD", "USDJPY"],
            index=["EURUSD","XAUUSD","GBPUSD","USDJPY"].index(config.get("symbol","EURUSD")),
        )
        new_tf   = st.selectbox(
            "🕐 Timeframe",
            ["H1", "H4", "M15", "M30"],
            index=["H1","H4","M15","M30"].index(config.get("timeframe","H1")),
        )

        # Persist changes immediately
        if (
            new_auto != config.get("auto_trade")
            or new_sym  != config.get("symbol")
            or new_tf   != config.get("timeframe")
        ):
            config.update({"auto_trade": new_auto, "symbol": new_sym, "timeframe": new_tf})
            _state_mgr.write_config(config)
            st.rerun()

        # ── Manual override (only when auto-trade is OFF) ───────────────────
        if not new_auto:
            st.markdown("---")
            st.subheader("🕹️ Manual Override")
            st.caption("AI generates signals. You pull the trigger.")
            col_a, col_b = st.columns(2)
            if col_a.button("🟢 BUY", width='stretch'):
                config["manual_command"] = "BUY"
                _state_mgr.write_config(config)
                st.toast("BUY command dispatched to engine.", icon="📤")
            if col_b.button("🔴 SELL", width='stretch'):
                config["manual_command"] = "SELL"
                _state_mgr.write_config(config)
                st.toast("SELL command dispatched to engine.", icon="📤")
            if st.button("⬛ CLOSE ALL", width='stretch'):
                config["manual_command"] = "CLOSE"
                _state_mgr.write_config(config)
                st.toast("CLOSE ALL command dispatched.", icon="📤")

        # ── Reconcile button ────────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔄 Reconcile Journal", width='stretch'):
            st.toast("Journal reconciliation triggered. Check terminal output.", icon="🔄")

        # ── Circuit breaker status ──────────────────────────────────────────
        live = _state_mgr.read_live_state()
        signal = live.get("ai_signal", "N/A")
        if signal == "HALTED":
            st.error("⛔ CIRCUIT BREAKER ACTIVE — Daily loss limit reached.")
        elif signal == "NEWS HALT":
            st.warning("📰 NEWS HALT — High-impact event incoming.")


# ==============================================================================
# FRAGMENT: LIVE HUD (2s refresh)
# ==============================================================================

@st.fragment(run_every="2s")
def render_hud() -> None:
    """Top metrics row: balance, equity, position, heartbeat."""
    state = _state_mgr.read_live_state()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Account Balance", f"${state.get('balance', 0):,.2f}")
    with col2:
        fp = float(state.get("floating_profit", 0))
        st.metric(
            "📊 Live Equity",
            f"${state.get('equity', 0):,.2f}",
            delta=f"{fp:+.2f}",
            delta_color="normal",
        )
    with col3:
        pos  = str(state.get("open_position", "FLAT"))
        icon = "🟢" if pos == "LONG" else "🔴" if pos == "SHORT" else "⚪"
        st.metric("📌 Position", f"{icon} {pos}")
    with col4:
        hb = str(state.get("last_update", "No data"))[:19]
        st.metric("🫀 Heartbeat", hb)

    st.markdown("---")


# ==============================================================================
# FRAGMENT: SIGNAL BANNER (2s refresh)
# ==============================================================================

@st.fragment(run_every="2s")
def render_signal() -> None:
    """AI signal and news radar banners."""
    state  = _state_mgr.read_live_state()
    signal = state.get("ai_signal", "WAITING")
    news   = state.get("news_status", "Checking...")

    # Signal colour
    sig_colour = (
        "#00ffcc" if signal == "BUY"
        else "#ff4444" if signal == "SELL"
        else "#ffaa00" if signal in ("NEWS HALT", "HALTED", "ERROR")
        else "#8b949e"
    )
    # News colour
    news_colour = "#ffaa00" if "HIGH IMPACT" in news else "#8b949e"

    col_sig, col_news = st.columns(2)
    with col_sig:
        st.markdown(
            f"""<div class="signal-card" style="border-color:{sig_colour}33;">
            <h3 style="color:{sig_colour};margin:0;">🧠 AI SIGNAL: {signal}</h3>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_news:
        st.markdown(
            f"""<div class="signal-card" style="border-color:{news_colour}33;">
            <h3 style="color:{news_colour};margin:0;">📡 RADAR: {news}</h3>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")


# ==============================================================================
# FRAGMENT: ANALYTICS DONUTS (10s refresh)
# ==============================================================================

@st.fragment(run_every="10s")
def render_analytics() -> None:
    """Win-rate and capital flow donut charts."""
    df = _state_mgr.read_journal()
    if df.empty:
        return

    closed = df[df["PnL"] != 0.0].copy()
    if len(closed) == 0:
        st.info("No closed trades yet. Analytics will appear after the first completed trade.")
        return

    wins        = len(closed[closed["PnL"] > 0])
    losses      = len(closed[closed["PnL"] < 0])
    total       = wins + losses
    win_rate    = wins / total * 100 if total > 0 else 0.0
    gross_profit= closed[closed["PnL"] > 0]["PnL"].sum()
    gross_loss  = abs(closed[closed["PnL"] < 0]["PnL"].sum())
    net_pnl     = gross_profit - gross_loss

    col1, col2 = st.columns(2)

    with col1:
        fig_wr = go.Figure(data=[
            go.Pie(
                labels       = ["Wins", "Losses"],
                values       = [wins, losses],
                hole         = 0.72,
                marker_colors= ["#00ffcc", "#ff4444"],
                textinfo     = "none",
            )
        ])
        fig_wr.update_layout(
            title      = "AI Win Rate",
            template   = "plotly_dark",
            height     = 240,
            margin     = dict(t=40, b=10, l=10, r=10),
            showlegend = True,
            legend     = dict(font=dict(color="#8b949e")),
            annotations= [dict(
                text      = f"{win_rate:.1f}%",
                x=0.5, y=0.5, font_size=22, showarrow=False,
                font_color= "#00ffcc" if win_rate >= 50 else "#ff4444",
            )],
        )
        st.plotly_chart(fig_wr, width='stretch')

    with col2:
        pnl_colour = "#00ffcc" if net_pnl >= 0 else "#ff4444"
        pnl_sign   = "+" if net_pnl >= 0 else ""
        fig_pnl = go.Figure(data=[
            go.Pie(
                labels       = ["Gross Profit", "Gross Loss"],
                values       = [max(gross_profit, 0.01), max(gross_loss, 0.01)],
                hole         = 0.72,
                marker_colors= ["#00ffcc", "#ff4444"],
                textinfo     = "none",
            )
        ])
        fig_pnl.update_layout(
            title      = "Capital Flow ($)",
            template   = "plotly_dark",
            height     = 240,
            margin     = dict(t=40, b=10, l=10, r=10),
            showlegend = True,
            legend     = dict(font=dict(color="#8b949e")),
            annotations= [dict(
                text       = f"{pnl_sign}${net_pnl:.2f}",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font_color = pnl_colour,
            )],
        )
        st.plotly_chart(fig_pnl, width='stretch')


# ==============================================================================
# FRAGMENT: CANDLESTICK CHART (30s refresh)
# ==============================================================================

@st.fragment(run_every="30s")
def render_chart() -> None:
    """H1 candlestick chart with SL/TP lines when in a trade."""
    config = _state_mgr.read_config()
    state  = _state_mgr.read_live_state()

    # Fetch SL/TP from most recent entry record
    sl_price = tp_price = 0.0
    pos = state.get("open_position", "FLAT")
    if pos != "FLAT":
        journal = _state_mgr.read_journal()
        if not journal.empty:
            entries = journal[journal["Action"].isin(["BUY", "SELL"])]
            if not entries.empty:
                last_entry = entries.iloc[-1]
                sl_price = float(last_entry.get("SL", 0.0))
                tp_price = float(last_entry.get("TP", 0.0))

    if not os.path.exists(Paths.CHART_FILE):
        st.info("Waiting for chart data from the trading engine...")
        return

    try:
        chart_df = pd.read_csv(Paths.CHART_FILE)
        if chart_df.empty:
            return

        fig = go.Figure(data=[
            go.Candlestick(
                x                    = chart_df.get("time", chart_df.index),
                open                 = chart_df["open"],
                high                 = chart_df["high"],
                low                  = chart_df["low"],
                close                = chart_df["close"],
                increasing_line_color= "#00ffcc",
                decreasing_line_color= "#ff4444",
                name                 = "Price",
            )
        ])

        # SL / TP lines (only when in a trade)
        if pos != "FLAT" and sl_price > 0:
            fig.add_hline(
                y                  = sl_price,
                line_dash          = "dot",
                line_color         = "#ff4444",
                line_width         = 2,
                annotation_text    = f"SL {sl_price:.5f}",
                annotation_position= "bottom right",
                annotation_font_color="#ff4444",
            )
        if pos != "FLAT" and tp_price > 0:
            fig.add_hline(
                y                  = tp_price,
                line_dash          = "dot",
                line_color         = "#00ffcc",
                line_width         = 2,
                annotation_text    = f"TP {tp_price:.5f}",
                annotation_position= "top right",
                annotation_font_color="#00ffcc",
            )

        fig.update_layout(
            title                  = f"Live {config.get('symbol','EURUSD')} {config.get('timeframe','H1')} — Market Structure",
            template               = "plotly_dark",
            height                 = 520,
            xaxis_rangeslider_visible = False,
            uirevision             = "constant",   # Preserves zoom/pan on refresh
            dragmode               = "pan",
            margin                 = dict(l=0, r=0, t=40, b=0),
            xaxis                  = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis                  = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", side="right"),
            paper_bgcolor          = "rgba(0,0,0,0)",
            plot_bgcolor           = "rgba(0,0,0,0)",
        )

        st.plotly_chart(
            fig,
            use_container_width = True,
            config = {
                "scrollZoom":            True,
                "displayModeBar":        True,
                "modeBarButtonsToRemove":["lasso2d", "select2d"],
                "displaylogo":           False,
            },
        )
    except Exception as exc:
        st.warning(f"Chart data loading... ({exc})")


# ==============================================================================
# FRAGMENT: TRADE JOURNAL (30s refresh)
# ==============================================================================

@st.fragment(run_every="30s")
def render_journal() -> None:
    """Trade ledger table with colour-coded actions and P&L."""
    st.subheader("📋 Trade Ledger & AI Reasoning")

    df = _state_mgr.read_journal()
    if df.empty:
        st.info("No trades recorded yet.")
        return

    # Sort descending (most recent first), reset index for clean display
    df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

    def _colour_action(val: str) -> str:
        if val in ("BUY",):          return "color:#00ffcc;font-weight:700"
        if val in ("SELL",):         return "color:#ff4444;font-weight:700"
        if val in ("BUY_CLOSE",):    return "color:#7ee8a2;font-weight:600"
        if val in ("SELL_CLOSE",):   return "color:#ff8a80;font-weight:600"
        return "color:#8b949e"

    def _colour_pnl(val) -> str:
        try:
            v = float(val)
            if v > 0:  return "color:#00ffcc;font-weight:600"
            if v < 0:  return "color:#ff4444;font-weight:600"
        except Exception:
            pass
        return "color:#8b949e"

    styled = (
        df.style
        .map(_colour_action, subset=["Action"])
        .map(_colour_pnl,    subset=["PnL"])
    )
    st.dataframe(styled, width='stretch', height=280)


# ==============================================================================
# CHAT PANEL (event-driven, no auto-refresh)
# ==============================================================================

def render_chat() -> None:
    """
    Interactive chat panel powered by local Ollama Llama3.

    Uses Streamlit's native chat_message / chat_input components.
    Conversation history is stored in st.session_state for the duration
    of the browser session (not persisted to disk).
    """
    st.markdown("---")
    st.subheader("💬 Ask the AI")

    # Lazy-load LLMAnalyst to avoid importing at module level if Ollama is absent
    if "llm" not in st.session_state:
        from news.llm_analyst import LLMAnalyst
        st.session_state.llm = LLMAnalyst()

    llm = st.session_state.llm

    if not llm.is_available():
        st.warning(
            "Ollama is not running. "
            "Start it with `ollama serve`, then reload this page."
        )
        return

    # Initialise chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render message history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input box
    if prompt := st.chat_input("Ask about market conditions, signals, or strategy..."):
        # Show user message immediately
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Stream LLM response
        context = _state_mgr.read_live_state()
        with st.chat_message("assistant"):
            response = st.write_stream(llm.stream_chat(prompt, context))

        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Keep last 20 messages to avoid context bloat
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]


# ==============================================================================
# MAIN RENDER
# ==============================================================================

def main() -> None:
    """Assembles the full dashboard layout."""
    st.title("🤖 Quant Predator — Live Terminal")

    # Static sidebar (settings, manual controls)
    _render_sidebar()

    # Dynamic fragments (each refreshes on its own schedule)
    render_hud()
    render_signal()
    render_analytics()
    render_chart()
    render_journal()

    # Event-driven chat (no fragment wrapper — only updates on user input)
    render_chat()


main()

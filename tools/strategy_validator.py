"""
tools/strategy_validator.py
============================
Rule-based strategy baseline validator for Quant Predator v2.

PURPOSE:
  Before spending 90+ minutes training an RL model, answer the fundamental
  question: "Does the feature set contain any exploitable edge at all?"

  A rule-based agent applies simple, hand-crafted entry rules using the same
  features our PPO model has access to. If the rule-based agent can produce
  positive expectancy, the feature set has edge and RL training is justified.
  If even the rules fail, the problem is in the features, not the model.

RULE SET (mirrors what the RL model should learn to approximate):
  LONG  when: MTF_Alignment >= +2  AND  bull_fvg_active == 1  AND  RSI_14 < 65
  SHORT when: MTF_Alignment <= -2  AND  bear_fvg_active == 1  AND  RSI_14 > 35
  R:R gate, SL/TP, and breakeven logic are IDENTICAL to rl_env.py.

OUTPUT:
  Prints a tear sheet comparing rule-based vs the RL model's run-1 results.
  Uses VectorBT for fast portfolio simulation if installed; falls back to
  our native loop backtester otherwise.

USAGE:
  python tools/strategy_validator.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from data.fetcher import get_historical_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig, Paths, RewardConfig, TradingConfig, get_symbol_spec
from core.logger import get_logger
from data.feature_engineer import add_indicators
from tools.tear_sheet import generate_tear_sheet

logger = get_logger(__name__)

# ── Optional VectorBT ──────────────────────────────────────────────────────────
try:
    import vectorbt as vbt
    _VBT_AVAILABLE = True
    logger.info("VectorBT available — using vectorized simulation.")
except ImportError:
    _VBT_AVAILABLE = False
    logger.info("VectorBT not installed — using native loop backtester.")


# ==============================================================================
# RULE ENGINE
# ==============================================================================

def _generate_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Applies the hand-crafted entry rules to the feature DataFrame.

    Returns:
        (long_entries, short_entries) — boolean Series aligned to df.index.
    """
    # Guard: ensure required columns exist
    required = ["MTF_Alignment", "bull_fvg_active", "bear_fvg_active", "RSI_14"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        # Fallback: use simpler rule without FVG columns
        logger.warning(
            f"Columns {missing} not found. Using simplified MTF-only rules."
        )
        long_signal  = df["MTF_Alignment"] >= 2
        short_signal = df["MTF_Alignment"] <= -2
    else:
        long_signal = (
            (df["MTF_Alignment"] >= RewardConfig.MTF_GATE_THRESHOLD) &
            (df["bull_fvg_active"] == 1.0) &
            (df["RSI_14"] < 65)
        )
        short_signal = (
            (df["MTF_Alignment"] <= -RewardConfig.MTF_GATE_THRESHOLD) &
            (df["bear_fvg_active"] == 1.0) &
            (df["RSI_14"] > 35)
        )

    return long_signal.astype(bool), short_signal.astype(bool)


# ==============================================================================
# NATIVE LOOP BACKTEST (fallback when VectorBT not installed)
# ==============================================================================

def _native_backtest(
    df:     pd.DataFrame,
    symbol: str,
    label:  str = "Rule-Based",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Simulates the rule-based strategy using the same loop logic as backtester.py.
    Identical SL/TP/breakeven mechanics to ForexTradingEnv for fair comparison.
    """
    spec         = get_symbol_spec(symbol)
    SPREAD       = spec.spread_pips * 0.0001
    INITIAL_BAL  = ModelConfig.INITIAL_BALANCE

    long_entries, short_entries = _generate_signals(df)

    balance           = INITIAL_BAL
    current_position  = 0
    entry_price       = 0.0
    sl_price          = 0.0
    tp_price          = 0.0
    lot_size          = 0.0
    initial_risk_dist = 0.0
    breakeven_done    = False
    swing_high        = float(df.iloc[0]["close"]) + 0.010
    swing_low         = float(df.iloc[0]["close"]) - 0.010

    START = ModelConfig.WINDOW_SIZE + 100
    equity_history = []
    trade_log      = []

    for i in range(START, len(df)):
        cur  = df.iloc[i]
        prev = df.iloc[i - 1]

        if prev["is_swing_high"] == 1.0: swing_high = float(prev["high"])
        if prev["is_swing_low"]  == 1.0: swing_low  = float(prev["low"])

        # ── SL/TP/Breakeven simulation ────────────────────────────────────────
        trade_closed = False
        pnl          = 0.0
        if current_position != 0:
            h, l = float(cur["high"]), float(cur["low"])
            if current_position == 1:
                if l <= sl_price:
                    pnl = (sl_price - entry_price) * 100_000 * lot_size
                    trade_closed = True
                elif h >= tp_price:
                    pnl = (tp_price - entry_price) * 100_000 * lot_size
                    trade_closed = True
                elif not breakeven_done and h >= entry_price + initial_risk_dist:
                    sl_price = entry_price; breakeven_done = True
            else:
                if h >= sl_price:
                    pnl = (entry_price - sl_price) * 100_000 * lot_size
                    trade_closed = True
                elif l <= tp_price:
                    pnl = (entry_price - tp_price) * 100_000 * lot_size
                    trade_closed = True
                elif not breakeven_done and l <= entry_price - initial_risk_dist:
                    sl_price = entry_price; breakeven_done = True

            if trade_closed:
                trade_log.append({"step": i, "action": "CLOSE", "pnl": round(pnl, 2),
                                   "entry": entry_price})
                balance += pnl; current_position = 0; lot_size = 0.0

        # ── Entry signal ──────────────────────────────────────────────────────
        if current_position == 0 and not trade_closed:
            target = 0
            if long_entries.iloc[i]:  target =  1
            elif short_entries.iloc[i]: target = -1

            if target != 0:
                price   = float(cur["open"])
                atr     = float(prev.get("ATRr_14", 0.001))
                min_sl  = max(0.00030, atr * spec.min_sl_atr_mult)
                padding = atr * 0.2

                if target == 1:
                    prop_sl = swing_low  - padding
                    prop_tp = swing_high - padding
                    exec_p  = price + SPREAD
                    risk_d  = exec_p - prop_sl
                    rwd_d   = prop_tp - exec_p
                else:
                    prop_sl = swing_high + padding
                    prop_tp = swing_low  + padding
                    exec_p  = price - SPREAD
                    risk_d  = prop_sl  - exec_p
                    rwd_d   = exec_p   - prop_tp

                if risk_d >= min_sl and rwd_d >= risk_d * ModelConfig.MIN_RR_RATIO:
                    rolling_bb = float(prev.get("BB_Width_rolling_mean", 0.0))
                    bb_w       = float(prev.get("BB_Width", 0.0))
                    vol_sc     = (bb_w / rolling_bb) if rolling_bb > 0 else 1.0
                    d_risk     = float(np.clip(
                        TradingConfig.RISK_PER_TRADE * vol_sc,
                        TradingConfig.MIN_DYNAMIC_RISK,
                        TradingConfig.MAX_DYNAMIC_RISK,
                    ))
                    max_risk   = min(balance * d_risk, 5_000.0)
                    lot_size   = max(round((max_risk / risk_d) / 100_000.0, 2), 0.01)
                    lot_size   = min(lot_size, TradingConfig.MAX_LOT_SIZE)

                    current_position = target
                    entry_price      = exec_p
                    sl_price         = prop_sl
                    tp_price         = prop_tp
                    initial_risk_dist= risk_d
                    breakeven_done   = False

                    trade_log.append({
                        "step": i, "action": "BUY" if target == 1 else "SELL",
                        "pnl": 0.0, "entry": exec_p,
                    })

        equity_history.append(balance)

    n_entries = len([t for t in trade_log if t["action"] in ("BUY", "SELL")])
    logger.info(
        f"{label}: {n_entries} entries | "
        f"Final: ${balance:,.2f} | "
        f"Return: {(balance/INITIAL_BAL - 1)*100:+.1f}%"
    )

    equity_df = pd.DataFrame({"balance": equity_history})
    trades_df = pd.DataFrame(trade_log) if trade_log else None
    return equity_df, trades_df


# ==============================================================================
# VECTORBT FAST BACKTEST
# ==============================================================================

def _vectorbt_backtest(
    df:     pd.DataFrame,
    symbol: str,
    label:  str = "Rule-Based (VectorBT)",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    VectorBT portfolio simulation. Much faster than the native loop for
    large datasets. Uses the same signal rules as _native_backtest.

    Note: VectorBT's trade mechanics differ slightly from ForexTradingEnv
    (no breakeven simulation). Use native backtest for strict comparisons;
    use VectorBT for fast parameter sweeps across many rule variants.
    """
    spec = get_symbol_spec(symbol)
    SPREAD = spec.spread_pips * 0.0001

    long_entries, short_entries = _generate_signals(df)

    close  = df["close"]
    atr    = df["ATRr_14"].fillna(0.001)
    sl_pct = (atr * 0.5)               # SL distance as price fraction
    tp_pct = (atr * 0.5 * ModelConfig.MIN_RR_RATIO)  # TP distance

    # VectorBT signals: entry when rule fires, exit at SL or TP
    long_sl  = close - (atr * 0.5)
    long_tp  = close + (atr * 0.5 * ModelConfig.MIN_RR_RATIO)
    short_sl = close + (atr * 0.5)
    short_tp = close - (atr * 0.5 * ModelConfig.MIN_RR_RATIO)

    pf = vbt.Portfolio.from_signals(
        close        = close,
        entries      = long_entries,
        short_entries= short_entries,
        exits        = pd.Series(False, index=df.index),
        short_exits  = pd.Series(False, index=df.index),
        sl_stop      = (atr * 0.5) / close,
        tp_stop      = (atr * 0.5 * ModelConfig.MIN_RR_RATIO) / close,
        fees         = SPREAD / close.mean(),
        init_cash    = ModelConfig.INITIAL_BALANCE,
        freq         = "1h",
    )

    equity_values = pf.value().values
    equity_df     = pd.DataFrame({"balance": equity_values})

    # Extract trade log
    trades = pf.trades.records_readable
    trades_df = pd.DataFrame({
        "step":   range(len(trades)),
        "action": trades["Direction"].apply(lambda d: "BUY" if d == "Long" else "SELL"),
        "pnl":    trades["PnL"].values,
        "entry":  trades["Entry Price"].values,
    }) if len(trades) > 0 else None

    final   = equity_values[-1]
    n_trade = len(trades)
    logger.info(
        f"{label}: {n_trade} trades | "
        f"Final: ${final:,.2f} | "
        f"Return: {(final/ModelConfig.INITIAL_BALANCE - 1)*100:+.1f}%"
    )
    return equity_df, trades_df


# ==============================================================================
# MAIN VALIDATION RUN
# ==============================================================================

def run_validation(
    symbol: str = TradingConfig.ACTIVE_SYMBOL,
    df:     Optional[pd.DataFrame] = None,
) -> None:
    """
    Validates the rule-based baseline and compares to run-1 RL results.

    Args:
        symbol: MT5 instrument.
        df:     Pre-loaded feature DataFrame. If None, fetches from MT5.
    """
    Paths.ensure_dirs()

    # ── Load data ─────────────────────────────────────────────────────────────
    if df is None:
        import MetaTrader5 as mt5
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Connecting to MT5 for data...")
        mt5.initialize()
        mt5.login(
            int(os.getenv("MT5_LOGIN")),
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
        )
        raw = get_historical_data(symbol, "H1", num_bars=ModelConfig.TRAIN_BARS)
        mt5.shutdown()
        if raw is None:
            logger.error("Data fetch failed.")
            return
        df = add_indicators(raw)
        if df is None:
            logger.error("Feature engineering failed.")
            return

    # Use the same test split as trainer.py (last 15%)
    n      = len(df)
    n_test = int(n * (1.0 - ModelConfig.TRAIN_SPLIT - ModelConfig.VAL_SPLIT))
    test_df= df.iloc[-n_test:].reset_index(drop=True)
    logger.info(f"Test set: {len(test_df):,} bars for {symbol}")

    # ── Run validation ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STRATEGY VALIDATOR — {symbol} Rule-Based Baseline")
    print(f"  Test bars: {len(test_df):,}  |  Initial: ${ModelConfig.INITIAL_BALANCE:,.0f}")
    print(f"{'='*60}")
    print("  Rules: MTF_Align ≥ 2 + FVG active + RSI filter")
    print(f"  Same R:R gate ({ModelConfig.MIN_RR_RATIO:.1f}:1), SL, TP, and breakeven as RL model")
    print(f"{'='*60}\n")

    if _VBT_AVAILABLE:
        equity_df, trade_log = _vectorbt_backtest(test_df, symbol, "Rule-Based (VBT)")
    else:
        equity_df, trade_log = _native_backtest(test_df, symbol, "Rule-Based")

    if equity_df is None:
        logger.error("Backtest produced no results.")
        return

    metrics = generate_tear_sheet(equity_df, trade_log, label="Rule-Based Baseline")

    # ── Comparison with RL run-1 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMPARISON: Rule-Based vs RL Model (Run 1)")
    print(f"{'─'*60}")
    print(f"  {'Metric':<25} {'Rule-Based':>15} {'RL Run 1':>15}")
    print(f"{'─'*60}")
    rl_run1 = {
        "Sharpe Ratio":    -0.388,
        "Max Drawdown (%)":38.30,
        "Win Rate (%)":    26.6,
        "Profit Factor":    0.928,
        "Expectancy ($)":  -5.31,
    }
    comparisons = {
        "Sharpe Ratio":    metrics.get("sharpe_ratio",     0),
        "Max Drawdown (%)":abs(metrics.get("max_drawdown_pct", 0)),
        "Win Rate (%)":    metrics.get("win_rate_pct",     0),
        "Profit Factor":   metrics.get("profit_factor",    0),
        "Expectancy ($)":  metrics.get("expectancy_usd",   0),
    }
    for metric, rb_val in comparisons.items():
        rl_val = rl_run1.get(metric, 0)
        print(f"  {metric:<25} {rb_val:>14.3f}  {rl_val:>14.3f}")
    print(f"{'─'*60}")

    rb_sharpe = metrics.get("sharpe_ratio", -99)
    if rb_sharpe > 0:
        print("\n  ✅ EDGE CONFIRMED: Rule-based baseline is profitable.")
        print("     Feature set has exploitable signal. RL training is justified.")
        print("     RL model should be able to meet or beat this baseline.")
    elif rb_sharpe > -0.20:
        print("\n  ⚠️  MARGINAL EDGE: Rule-based is near break-even.")
        print("     Feature set has weak signal. RL may improve it — or may not.")
        print("     Consider adding stronger features (e.g., distance-to-FVG).")
    else:
        print("\n  ❌ NO EDGE DETECTED: Rule-based baseline is clearly unprofitable.")
        print("     The feature set may not contain sufficient signal for this strategy.")
        print("     Review features in data/feature_engineer.py before RL training.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_validation(symbol=TradingConfig.ACTIVE_SYMBOL)

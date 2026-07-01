"""
engine/backtester.py
====================
Out-of-sample simulation engine for Quant Predator v2.

CRITICAL FIXES vs v1 (backtest_ppo.py):
    1. floating_profit fed correctly into observations (was always 0.0).
    2. O(1) swing tracking via incremental update (was O(n²) per step).
    3. Operates on a separate test DataFrame — never the training data.
    4. Returns structured equity curve + trade log for tear_sheet analysis.

RUN STANDALONE:
    python engine/backtester.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import MetaTrader5 as mt5 # type: ignore
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig, Paths, TradingConfig, get_symbol_spec
from core.logger import get_logger
from data.feature_engineer import add_indicators, FEATURE_COLUMNS
from data.fetcher import get_historical_data
from engine.policy import LSTMFeatureExtractor   # Must be imported before PPO.load
from engine.rl_env import ForexTradingEnv

logger = get_logger(__name__)


def run_backtest(
    df:         pd.DataFrame,
    model_path: str,
    stats_path: str,
    symbol:     str  = TradingConfig.ACTIVE_SYMBOL,
    label:      str  = "Backtest",
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Runs a deterministic simulation of the trained model on a given DataFrame.

    The simulation replicates live_trader.py's execution logic exactly:
    same SL/TP rules, same breakeven trigger, same position sizing.

    Args:
        df:         Feature-engineered DataFrame (from add_indicators).
                    Should be the out-of-sample test set only.
        model_path: Path to the saved PPO .zip file.
        stats_path: Path to the VecNormalize .pkl file.
        symbol:     Symbol string for logging context.
        label:      Label prefix for log messages.

    Returns:
        (equity_curve_df, trade_log_df) — both None on failure.
        equity_curve_df columns: ['balance']
        trade_log_df columns:    ['step', 'action', 'pnl', 'entry', 'exit']
    """
    spec = get_symbol_spec(symbol)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return None, None
    if not os.path.exists(stats_path):
        logger.error(f"VecNormalize stats not found: {stats_path}")
        return None, None

    dummy_env = DummyVecEnv([lambda: ForexTradingEnv(
        df, commission=ModelConfig.COMMISSION,
        spread_pips=spec.spread_pips, window_size=ModelConfig.WINDOW_SIZE
    )])
    vec_env = VecNormalize.load(stats_path, dummy_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(model_path)
    feature_columns = dummy_env.envs[0].feature_columns

    logger.info(f"{label}: Simulating {len(df):,} bars on {symbol}...")

    # ── Simulation state ──────────────────────────────────────────────────────
    balance          = ModelConfig.INITIAL_BALANCE
    current_position = 0
    entry_price      = 0.0
    sl_price         = 0.0
    tp_price         = 0.0
    lot_size         = 0.0
    initial_risk_dist= 0.0
    breakeven_done   = False
    swing_high       = float(df.iloc[0]["close"]) + 0.010
    swing_low        = float(df.iloc[0]["close"]) - 0.010

    # Pre-scan for initial swing levels
    _init = df.iloc[:ModelConfig.WINDOW_SIZE + 100]
    _h = _init[_init["is_swing_high"] == 1.0]["high"]
    _l = _init[_init["is_swing_low"]  == 1.0]["low"]
    if not _h.empty: swing_high = float(_h.iloc[-1])
    if not _l.empty: swing_low  = float(_l.iloc[-1])

    SPREAD = spec.spread_pips * 0.0001
    equity_history = []
    trade_log      = []

    start = ModelConfig.WINDOW_SIZE + 100

    for i in range(start, len(df)):
        cur  = df.iloc[i]
        prev = df.iloc[i - 1]

        # O(1) swing update — checks only the previous closed bar
        if prev["is_swing_high"] == 1.0: swing_high = float(prev["high"])
        if prev["is_swing_low"]  == 1.0: swing_low  = float(prev["low"])

        # ── SL / TP / Breakeven simulation ────────────────────────────────────
        trade_closed = False
        pnl = 0.0

        if current_position != 0:
            high = float(cur["high"])
            low  = float(cur["low"])

            if current_position == 1:
                if low <= sl_price:
                    pnl = (sl_price - entry_price) * 100_000 * lot_size
                    trade_closed = True
                elif high >= tp_price:
                    pnl = (tp_price - entry_price) * 100_000 * lot_size
                    trade_closed = True
                elif not breakeven_done and high >= entry_price + initial_risk_dist:
                    sl_price     = entry_price
                    breakeven_done = True
            else:  # short
                if high >= sl_price:
                    pnl = (entry_price - sl_price) * 100_000 * lot_size
                    trade_closed = True
                elif low <= tp_price:
                    pnl = (entry_price - tp_price) * 100_000 * lot_size
                    trade_closed = True
                elif not breakeven_done and low <= entry_price - initial_risk_dist:
                    sl_price     = entry_price
                    breakeven_done = True

            if trade_closed:
                trade_log.append({
                    "step":   i,
                    "action": "CLOSE",
                    "pnl":    round(pnl, 2),
                    "entry":  entry_price,
                    "exit":   sl_price if pnl < 0 else tp_price,
                })
                balance          += pnl
                current_position  = 0
                lot_size          = 0.0

        # ── Model inference ───────────────────────────────────────────────────
        market_window = df.iloc[i - ModelConfig.WINDOW_SIZE:i][feature_columns].values
        market_flat   = market_window.flatten()

        # FIXED: compute actual floating profit (was 0.0 in v1)
        floating_pnl = 0.0
        if current_position == 1:
            floating_pnl = (float(cur["close"]) - entry_price) * 100_000 * lot_size
        elif current_position == -1:
            floating_pnl = (entry_price - float(cur["close"])) * 100_000 * lot_size

        raw_obs  = np.append(
            market_flat,
            [float(current_position), floating_pnl / 1000.0],
        ).astype(np.float32)
        norm_obs = vec_env.normalize_obs(raw_obs.reshape(1, -1))
        action, _= model.predict(norm_obs, deterministic=True)
        action   = int(action[0])

        # ── Entry logic ───────────────────────────────────────────────────────
        if current_position == 0 and action != 0 and not trade_closed:
            target  = 1 if action == 1 else -1
            price   = float(cur["open"])
            atr     = float(prev.get("ATRr_14", 0.001))
            min_sl  = max(0.00030, atr * 0.5)
            padding = atr * 0.2

            if target == 1:
                prop_sl   = swing_low  - padding
                prop_tp   = swing_high - padding
                exec_p    = price + SPREAD
                risk_dist = exec_p - prop_sl
                rwd_dist  = prop_tp - exec_p
            else:
                prop_sl   = swing_high + padding
                prop_tp   = swing_low  + padding
                exec_p    = price - SPREAD
                risk_dist = prop_sl  - exec_p
                rwd_dist  = exec_p   - prop_tp

            if risk_dist >= min_sl and rwd_dist >= risk_dist:
                rolling_bb = float(prev.get("BB_Width_rolling_mean", 0.0))
                bb_w       = float(prev.get("BB_Width", 0.0))
                vol_scalar = (bb_w / rolling_bb) if rolling_bb > 0 else 1.0
                d_risk     = float(np.clip(
                    TradingConfig.RISK_PER_TRADE * vol_scalar,
                    TradingConfig.MIN_DYNAMIC_RISK,
                    TradingConfig.MAX_DYNAMIC_RISK,
                ))
                max_risk   = min(balance * d_risk, 5_000.0)
                lot_size   = max(round((max_risk / risk_dist) / 100_000.0, 2), 0.01)
                lot_size   = min(lot_size, TradingConfig.MAX_LOT_SIZE)

                current_position = target
                entry_price      = exec_p
                sl_price         = prop_sl
                tp_price         = prop_tp
                initial_risk_dist= risk_dist
                breakeven_done   = False

                trade_log.append({
                    "step":   i,
                    "action": "BUY" if target == 1 else "SELL",
                    "pnl":    0.0,
                    "entry":  exec_p,
                    "exit":   0.0,
                })

        equity_history.append(balance)

    logger.info(
        f"{label} complete. "
        f"Final balance: ${balance:,.2f} | "
        f"Trades executed: {len([t for t in trade_log if t['action'] in ('BUY','SELL')])}"
    )

    equity_df = pd.DataFrame({"balance": equity_history})
    trades_df = pd.DataFrame(trade_log) if trade_log else None

    # Persist equity curve for tear_sheet standalone use
    equity_path = os.path.join(Paths.ROOT, "tools", f"equity_{symbol.lower()}.csv")
    equity_df.to_csv(equity_path, index=False)

    return equity_df, trades_df


if __name__ == "__main__":
    load_dotenv()
    mt5.initialize()
    mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
    )
    sym    = TradingConfig.ACTIVE_SYMBOL
    raw    = get_historical_data(sym, "H1", num_bars=15_000)
    mt5.shutdown()
    df     = add_indicators(raw)
    eq, tl = run_backtest(
        df         = df,
        #model_path = Paths.model_zip(sym),
        model_path  = Paths.best_model_dir(sym),
        stats_path  = Paths.vec_normalize_pkl(sym),
        symbol      = sym,
    )
    if eq is not None:
        from tools.tear_sheet import generate_tear_sheet
        generate_tear_sheet(eq, tl, label="Standalone Backtest")

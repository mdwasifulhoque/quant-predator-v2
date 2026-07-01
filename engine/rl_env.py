"""
engine/rl_env.py
================
Gymnasium-compatible Reinforcement Learning Trading Environment v2.1.

CHANGES FROM v2.0 (post training-run-1 diagnostics):
    1. REWARD OVERHAUL — Pure P&L reward caused 9.2× overtrading.
       New reward components (all configurable in RewardConfig):
         - Entry cost:       -0.30 per new trade opened
         - Patience bonus:   +0.002 per bar held flat
         - MTF quality gate: -0.20 penalty for |MTF_Alignment| < 2
         - Asian session:     reward × 0.5 multiplier during low-liquidity hours

    2. SESSION FILTER — Asian session entries now incur a reward penalty.
       The model learns that London/NY/Overlap setups are more reliable.
       (Hard block removed — penalty is softer and allows the model to learn.)

    3. MIN R:R RAISED — Gate increased from 1.0:1 to ModelConfig.MIN_RR_RATIO (2.0:1).
       This directly filters the low-quality entries driving 521 trades per test.

    4. MINIMUM HOLD PERIOD — Model cannot exit a position within 3 bars of entry.
       Prevents the 1-bar scalp behaviour observed in training run 1.

    5. LINEAR LR SCHEDULE — Learning rate now decays from 1e-4 to 1e-5 over
       training. Implemented in trainer.py via lr_schedule parameter.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from core.config import ModelConfig, RewardConfig, SessionConfig, TradingConfig
from core.logger import get_logger
from data.feature_engineer import FEATURE_COLUMNS

logger = get_logger(__name__)

_NON_FEATURE_COLS = {
    "open", "high", "low", "close",
    "tick_volume", "volume", "spread", "real_volume", "time",
    "BB_Width_rolling_mean",
}

# Minimum bars a position must be held before the model can exit it.
# Prevents 1-bar scalp behaviour (observed: many 1-2 bar trades in run 1).
_MIN_HOLD_BARS = 3


class ForexTradingEnv(gym.Env):
    """
    H1 Forex trading environment v2.1.

    Key architecture changes vs v2.0:
      - Reward function now penalises overtrading and rewards patience.
      - MTF alignment gate embedded in reward (soft constraint).
      - Session-aware reward multiplier for Asian vs London/NY hours.
      - Minimum holding period prevents degenerate 1-bar scalping.
      - Min R:R gate raised to 2.0:1.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df:              pd.DataFrame,
        initial_balance: float = ModelConfig.INITIAL_BALANCE,
        risk_per_trade:  float = TradingConfig.RISK_PER_TRADE,
        commission:      float = ModelConfig.COMMISSION,
        spread_pips:     float = 1.8,
        window_size:     int   = ModelConfig.WINDOW_SIZE,
    ) -> None:
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.risk_per_trade  = risk_per_trade
        self.commission      = commission
        self.window_size     = window_size
        self.spread_cost_raw = spread_pips * 0.0001

        self.feature_columns = [c for c in FEATURE_COLUMNS if c in self.df.columns]
        n_missing = len(FEATURE_COLUMNS) - len(self.feature_columns)
        if n_missing > 0:
            missing = [c for c in FEATURE_COLUMNS if c not in self.df.columns]
            raise ValueError(
                f"ForexTradingEnv: missing {n_missing} feature columns: {missing}. "
                f"Ensure add_indicators() was called on the input DataFrame."
            )

        self.action_space      = spaces.Discrete(3)
        obs_dim                = (len(self.feature_columns) * self.window_size) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State variables
        self.current_step:        int   = 0
        self.balance:             float = initial_balance
        self.net_worth:           float = initial_balance
        self.current_position:    int   = 0
        self.entry_price:         float = 0.0
        self.active_lot_size:     float = 0.0
        self.sl_price:            float = 0.0
        self.tp_price:            float = 0.0
        self.initial_risk_dist:   float = 0.0
        self.breakeven_triggered: bool  = False
        self.current_swing_high:  float = 0.0
        self.current_swing_low:   float = 0.0
        self.bars_in_trade:       int   = 0   # NEW: tracks minimum hold period

        logger.info(
            f"ForexTradingEnv v2.1 initialised: "
            f"{len(self.df):,} bars | {len(self.feature_columns)} features | "
            f"obs_dim={obs_dim} | min_rr={ModelConfig.MIN_RR_RATIO:.1f}:1 | "
            f"min_hold={_MIN_HOLD_BARS} bars"
        )

    # ==========================================================================
    # GYMNASIUM API
    # ==========================================================================

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Random episode start — see v2.0 docstring for rationale."""
        super().reset(seed=seed)

        min_start = self.window_size + 100
        max_start = max(min_start + 1, len(self.df) - 500)
        self.current_step = int(self.np_random.integers(min_start, max_start))

        self.balance             = self.initial_balance
        self.net_worth           = self.initial_balance
        self.current_position    = 0
        self.entry_price         = 0.0
        self.active_lot_size     = 0.0
        self.sl_price            = 0.0
        self.tp_price            = 0.0
        self.initial_risk_dist   = 0.0
        self.breakeven_triggered = False
        self.bars_in_trade       = 0

        _history = self.df.iloc[:self.current_step]
        _highs   = _history[_history["is_swing_high"] == 1.0]["high"]
        _lows    = _history[_history["is_swing_low"]  == 1.0]["low"]
        ref      = float(self.df.iloc[self.current_step]["close"])
        self.current_swing_high = float(_highs.iloc[-1]) if not _highs.empty else ref + 0.010
        self.current_swing_low  = float(_lows.iloc[-1])  if not _lows.empty  else ref - 0.010

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        One H1 bar step. Execution order:
            1. Update O(1) swing trackers.
            2. Simulate SL/TP/Breakeven on current candle.
            3. Apply reward shaping (entry cost, patience, MTF, session).
            4. Process entry action if flat and not in minimum hold period.
            5. Advance step and check termination.
        """
        current_candle = self.df.iloc[self.current_step]
        prev_candle    = self.df.iloc[self.current_step - 1]
        step_reward    = 0.0
        trade_closed_this_step = False

        # ── 1. O(1) Swing update ──────────────────────────────────────────────
        if prev_candle["is_swing_high"] == 1.0:
            self.current_swing_high = float(prev_candle["high"])
        if prev_candle["is_swing_low"] == 1.0:
            self.current_swing_low  = float(prev_candle["low"])

        # ── 2. SL / TP / Breakeven simulation ────────────────────────────────
        if self.current_position != 0:
            self.bars_in_trade += 1
            high = float(current_candle["high"])
            low  = float(current_candle["low"])
            pnl  = 0.0

            if self.current_position == 1:
                if low <= self.sl_price:
                    pnl = (self.sl_price - self.entry_price) * 100_000 * self.active_lot_size
                    trade_closed_this_step = True
                elif high >= self.tp_price:
                    pnl = (self.tp_price - self.entry_price) * 100_000 * self.active_lot_size
                    trade_closed_this_step = True
                elif not self.breakeven_triggered and high >= self.entry_price + self.initial_risk_dist:
                    self.sl_price            = self.entry_price
                    self.breakeven_triggered = True
            elif self.current_position == -1:
                if high >= self.sl_price:
                    pnl = (self.entry_price - self.sl_price) * 100_000 * self.active_lot_size
                    trade_closed_this_step = True
                elif low <= self.tp_price:
                    pnl = (self.entry_price - self.tp_price) * 100_000 * self.active_lot_size
                    trade_closed_this_step = True
                elif not self.breakeven_triggered and low <= self.entry_price - self.initial_risk_dist:
                    self.sl_price            = self.entry_price
                    self.breakeven_triggered = True

            # Manual exit (action=0) — only allowed after minimum hold period
            if (action == 0
                    and not trade_closed_this_step
                    and self.bars_in_trade >= _MIN_HOLD_BARS):
                price = float(current_candle["open"])
                pnl = (
                    (price - self.entry_price) * 100_000 * self.active_lot_size
                    if self.current_position == 1
                    else (self.entry_price - price) * 100_000 * self.active_lot_size
                )
                trade_closed_this_step = True

            if trade_closed_this_step:
                self.balance          += pnl
                self.current_position  = 0
                self.active_lot_size   = 0.0
                self.bars_in_trade     = 0

        # ── 3. PATIENCE BONUS — reward being flat (idle is not losing) ────────
        elif self.current_position == 0:
            step_reward += RewardConfig.PATIENCE_BONUS_FLAT

        # ── 4. ENTRY LOGIC GATE ───────────────────────────────────────────────
        if self.current_position == 0 and action != 0 and not trade_closed_this_step:
            target = 1 if action == 1 else -1
            price  = float(current_candle["open"])
            atr    = float(prev_candle.get("ATRr_14", 0.001))

            # ATR-based SL floor
            min_sl_dist = max(0.00030, atr * 0.5)
            padding     = atr * 0.2

            if target == 1:
                prop_sl    = self.current_swing_low  - padding
                prop_tp    = self.current_swing_high - padding
                exec_price = price + self.spread_cost_raw
                risk_dist  = exec_price - prop_sl
                reward_dist= prop_tp    - exec_price
            else:
                prop_sl    = self.current_swing_high + padding
                prop_tp    = self.current_swing_low  + padding
                exec_price = price - self.spread_cost_raw
                risk_dist  = prop_sl    - exec_price
                reward_dist= exec_price - prop_tp

            # ── GATE 1: Structural R:R (raised from 1.0 to MIN_RR_RATIO=2.0) ─
            rr_ok = (risk_dist >= min_sl_dist
                     and reward_dist >= risk_dist * ModelConfig.MIN_RR_RATIO)

            if rr_ok:
                # ── Volatility-adjusted position sizing ───────────────────────
                rolling_bb_mean  = float(prev_candle.get("BB_Width_rolling_mean", 0.0))
                current_bb_width = float(prev_candle.get("BB_Width", 0.0))
                vol_scalar       = (current_bb_width / rolling_bb_mean) if rolling_bb_mean > 0 else 1.0
                dynamic_risk     = float(np.clip(
                    self.risk_per_trade * vol_scalar,
                    TradingConfig.MIN_DYNAMIC_RISK,
                    TradingConfig.MAX_DYNAMIC_RISK,
                ))
                max_risk_dollars = min(self.balance * dynamic_risk, 5_000.0)
                lot_size = max(
                    round((max_risk_dollars / risk_dist) / 100_000.0, 2),
                    0.01,
                )
                lot_size = min(lot_size, TradingConfig.MAX_LOT_SIZE)

                self.current_position    = target
                self.entry_price         = exec_price
                self.sl_price            = prop_sl
                self.tp_price            = prop_tp
                self.initial_risk_dist   = risk_dist
                self.active_lot_size     = lot_size
                self.breakeven_triggered = False
                self.bars_in_trade       = 0

                # ── ENTRY COST — discourages overtrading ──────────────────────
                step_reward += RewardConfig.ENTRY_COST

                # ── GATE 2: MTF alignment quality check ───────────────────────
                # MTF_Alignment is NOT in FEATURE_COLUMNS (it's a linear combo of
                # H1/H4/D1 Trend which ARE individual features). Compute inline
                # from the three trend columns to avoid a silent KeyError.
                h1t = float(prev_candle.get("H1_Trend", 0.0))
                h4t = float(prev_candle.get("H4_Trend", 0.0))
                d1t = float(prev_candle.get("D1_Trend", 0.0))
                mtf_align = h1t + h4t + d1t   # range: -3.0 to +3.0
                if abs(mtf_align) < RewardConfig.MTF_GATE_THRESHOLD:
                    step_reward += RewardConfig.WEAK_MTF_PENALTY

                # ── SESSION MULTIPLIER — discourage Asian entries ──────────────
                # Cannot use self.df.index[i].hour — index is RangeIndex after
                # reset_index(drop=True) in __init__. Use the pre-computed
                # session_asian column from feature_engineer.py instead.
                is_asian = float(prev_candle.get("session_asian", 0.0)) > 0.5
                if is_asian:
                    step_reward += RewardConfig.ENTRY_COST * (1.0 - RewardConfig.ASIAN_SESSION_MULT)

            else:
                # Penalise structurally invalid entry attempt
                step_reward -= 0.1

        # ── 5. P&L REWARD ─────────────────────────────────────────────────────
        floating_pnl = 0.0
        if self.current_position != 0:
            open_price = float(current_candle["open"])
            floating_pnl = (
                (open_price - self.entry_price) * 100_000 * self.active_lot_size
                if self.current_position == 1
                else (self.entry_price - open_price) * 100_000 * self.active_lot_size
            )

        new_net_worth  = self.balance + floating_pnl
        step_reward   += (new_net_worth - self.net_worth) / 100.0
        self.net_worth = new_net_worth

        # ── 6. Advance & termination ──────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1

        if self.net_worth <= self.initial_balance * ModelConfig.BANKRUPTCY_FRACTION:
            terminated  = True
            step_reward = RewardConfig.BANKRUPTCY_PENALTY

        return self._get_observation(), float(step_reward), terminated, False, self._get_info()

    def render(self, mode: str = "human") -> None:
        pos = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(self.current_position, "?")
        print(
            f"  Step {self.current_step:>6} | "
            f"Balance: ${self.balance:>10,.2f} | "
            f"NetWorth: ${self.net_worth:>10,.2f} | "
            f"Pos: {pos} | Bars held: {self.bars_in_trade}"
        )

    # ==========================================================================
    # PRIVATE HELPERS
    # ==========================================================================

    def _get_observation(self) -> np.ndarray:
        start       = self.current_step - self.window_size
        market_win  = self.df.iloc[start:self.current_step][self.feature_columns].values
        market_flat = market_win.flatten()

        floating_pnl = 0.0
        if self.current_position != 0:
            cur = float(self.df.iloc[self.current_step]["open"])
            floating_pnl = (
                (cur - self.entry_price) * 100_000 * self.active_lot_size
                if self.current_position == 1
                else (self.entry_price - cur) * 100_000 * self.active_lot_size
            )
        obs = np.append(
            market_flat,
            [float(self.current_position), floating_pnl / 1000.0],
        )
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step":        self.current_step,
            "net_worth":   self.net_worth,
            "balance":     self.balance,
            "position":    self.current_position,
            "bars_held":   self.bars_in_trade,
            "swing_high":  self.current_swing_high,
            "swing_low":   self.current_swing_low,
        }

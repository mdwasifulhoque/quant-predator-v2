"""
live_trader.py
==============
Quant Predator v2 — Live Trading Execution Engine.

This is the main process. Run it once; it loops indefinitely until stopped.

EXECUTION FLOW (every H1 bar):
    1. Bar-close detection (gatekeeper): fetch 500-bar UI data every 2s.
       When the latest candle timestamp changes → new H1 bar opened.
    2. Deep MTF fetch: 6,025 bars (enough for D1 EMA200 warmup + 24-bar window).
    3. Feature engineering: full add_indicators() pipeline.
    4. Live observation validation: all 35 features present, no NaN.
    5. VecNormalize + PPO inference → action (0=Hold, 1=Buy, 2=Sell).
    6. News halt check: suspend entries if high-impact event within 30 min.
    7. Circuit breaker check: pause if daily loss > 3%.
    8. Trade execution via MT5Gateway.
    9. State persistence via StateManager (atomic JSON writes).

STARTUP:
    python live_trader.py

STOPPING:
    Ctrl+C — the KeyboardInterrupt handler closes open positions cleanly
    if CLOSE_ON_EXIT = True (configurable below).
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from broker.mt5_gateway import MT5Gateway
from core.config import (
    BrokerConfig, ModelConfig, NewsConfig, Paths,
    TradingConfig, get_symbol_spec,
)
from core.logger import get_logger
from data.feature_engineer import FEATURE_COLUMNS, add_indicators, validate_live_observation
from data.fetcher import get_historical_data
from engine.policy import LSTMFeatureExtractor   # MUST import before PPO.load
from engine.rl_env import ForexTradingEnv
from ledger.reconciler import rebuild_journal
from ledger.state_manager import StateManager
from news.forex_radar import ForexNewsRadar
from news.llm_analyst import LLMAnalyst

logger = get_logger(__name__)

# Set True to close any open positions when the script exits (Ctrl+C)
CLOSE_ON_EXIT = False

# Deep data window: enough bars for D1 EMA200 warmup (4800) + inference window (24) + buffer
DEEP_BARS = 6_025


class LiveTrader:
    """
    Full lifecycle manager for a single-symbol live trading session.

    One instance of this class is created in main() and runs until
    the process is terminated.
    """

    def __init__(self, symbol: str = TradingConfig.ACTIVE_SYMBOL) -> None:
        self.symbol   = symbol
        self.spec     = get_symbol_spec(symbol)
        self.gateway  = MT5Gateway()
        self.state    = StateManager()
        self.radar    = ForexNewsRadar(currencies=NewsConfig.EURUSD_CURRENCIES)
        self.llm      = LLMAnalyst()

        # PPO inference objects (loaded in _load_model)
        self.model:   Optional[PPO] = None
        self.vec_env: Optional[VecNormalize] = None

        # Live position state — restored from MT5 on startup
        self.current_position:    int   = 0      # -1 Short | 0 Flat | 1 Long
        self.entry_price:         float = 0.0
        self.active_lot_size:     float = 0.0
        self.last_processed_time: Optional[datetime] = None

        # Signal / status strings for the dashboard
        self.ai_signal    = "STARTING"
        self.news_status  = "Checking..."

        # Indicator snapshot at entry time — logged at close for full audit trail
        self._entry_indicators: Dict = {}

        # Circuit breaker
        self._daily_open_balance: float = 0.0
        self._halted:             bool  = False

    # ==========================================================================
    # STARTUP
    # ==========================================================================

    def start(self) -> None:
        """
        Connects to MT5, loads the trained model, recovers position state,
        then enters the main loop. Blocks until KeyboardInterrupt or fatal error.
        """
        Paths.ensure_dirs()

        # ── Connect to MT5 via gateway ─────────────────────────────────────────
        logger.info(f"Starting Quant Predator v2 — {self.symbol} H1")
        if not self.gateway.start():
            logger.error("MT5 connection failed. Check terminal and .env credentials.")
            return

        acc = self.gateway.account_info()
        self._daily_open_balance = acc.balance
        logger.info(f"Session opening balance: ${acc.balance:,.2f}")

        # ── Load PPO model ─────────────────────────────────────────────────────
        if not self._load_model():
            logger.error("Model load failed. Run engine/trainer.py first.")
            self.gateway.stop()
            return

        # ── Recover position state from MT5 ────────────────────────────────────
        self._recover_position_state()

        # ── Run reconciler to sync any offline close events ────────────────────
        logger.info("Reconciling trade journal with broker history...")
        rebuild_journal(symbol=self.symbol, lookback_days=90)

        logger.info("V2 EXECUTION ENGINE ONLINE — entering main loop.")
        self.ai_signal = "WAITING"

        # ── Main loop ──────────────────────────────────────────────────────────
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down cleanly...")
            if CLOSE_ON_EXIT and self.current_position != 0:
                logger.info("CLOSE_ON_EXIT = True — flattening open position.")
                self._close_all_positions(reason="Script exit")
        finally:
            self.gateway.stop()

    # ==========================================================================
    # MAIN LOOP
    # ==========================================================================

    def _main_loop(self) -> None:
        """Runs forever — one iteration every MAIN_LOOP_SLEEP_SECS seconds."""
        while True:
            try:
                self._loop_once()
            except Exception as exc:
                logger.error(f"Unhandled loop error: {exc}", exc_info=True)
                self.ai_signal = "ERROR"
            finally:
                time.sleep(TradingConfig.MAIN_LOOP_SLEEP_SECS)

    def _loop_once(self) -> None:
        """Single iteration of the main loop."""
        config     = self.state.read_config()
        symbol     = config.get("symbol", self.symbol)
        auto_trade = config.get("auto_trade", True)
        manual_cmd = config.get("manual_command", "NONE")

        # ── 1. Position status check (SL/TP hit detection) ────────────────────
        positions = self.gateway.positions_get(symbol)
        if (positions is None or len(positions) == 0) and self.current_position != 0:
            # MT5 has no open positions but we think we're in one.
            # SL or TP was hit — log the true broker close.
            logger.info("Position closed by broker (SL/TP). Reconciling journal...")
            self._log_broker_close(symbol)
            self.current_position = 0
            self.entry_price      = 0.0
            self.active_lot_size  = 0.0
            self._entry_indicators = {}
            rebuild_journal(symbol=symbol, lookback_days=2)

        elif positions and len(positions) > 0:
            # ── 2. Breakeven management on open position ───────────────────────
            self._manage_breakeven(symbol, positions)

        # ── 3. Circuit breaker check ───────────────────────────────────────────
        acc = self.gateway.account_info()
        if acc and self._daily_open_balance > 0:
            daily_dd = (acc.balance - self._daily_open_balance) / self._daily_open_balance
            if daily_dd <= -TradingConfig.DAILY_LOSS_LIMIT and not self._halted:
                logger.warning(
                    f"CIRCUIT BREAKER: Daily loss {daily_dd*100:.1f}% exceeds "
                    f"{TradingConfig.DAILY_LOSS_LIMIT*100:.0f}% limit. "
                    f"Auto-trading suspended."
                )
                self._halted   = True
                self.ai_signal = "HALTED"

        # ── 4. Manual override ─────────────────────────────────────────────────
        if manual_cmd != "NONE":
            logger.info(f"Manual command received: {manual_cmd}")
            self._handle_manual_command(manual_cmd, symbol, acc)
            self.state.clear_manual_command()

        # ── 5. Light UI data update (chart) ───────────────────────────────────
        # Fetch 500 bars every loop for the chart — lightweight call.
        ui_df = self._fetch_ui_data(symbol, config.get("timeframe", "H1"))
        if ui_df is not None:
            self.state.update_chart_data(ui_df, TradingConfig.CHART_BARS)
            latest_time = ui_df.index[-1]

            # ── 6. Bar-close gatekeeper ────────────────────────────────────────
            if latest_time != self.last_processed_time:
                logger.info(
                    f"New H1 bar detected at {latest_time}. "
                    f"Running MTF analysis..."
                )
                self._on_new_bar(symbol, config, auto_trade, acc)
                self.last_processed_time = latest_time

        # ── 7. Live state update for dashboard ────────────────────────────────
        if acc:
            tick         = self.gateway.symbol_info_tick(symbol)
            floating_pnl = self._compute_floating_pnl(tick)
            self.state.update_live_state(
                balance         = acc.balance,
                floating_profit = floating_pnl,
                open_position   = self.current_position,
                ai_signal       = self.ai_signal,
                news_status     = self.news_status,
                mt5_ok          = (acc is not None),
                ollama_ok       = self.llm.is_available(),
            )

    # ==========================================================================
    # NEW-BAR HANDLER
    # ==========================================================================

    def _on_new_bar(
        self,
        symbol:     str,
        config:     Dict,
        auto_trade: bool,
        acc,
    ) -> None:
        """
        Called once per new H1 bar. Runs the full MTF analysis pipeline
        and executes a trade if signals align and auto_trade is enabled.
        """
        # ── Fetch deep data ────────────────────────────────────────────────────
        raw_df = self._fetch_data_with_gateway(symbol, DEEP_BARS)
        if raw_df is None:
            logger.warning("Deep data fetch failed. Skipping this bar.")
            return

        # ── News halt check ────────────────────────────────────────────────────
        news_imminent, news_msg = self.radar.fetch_imminent_news(
            hours_ahead=NewsConfig.HALT_HOURS_AHEAD
        )
        self.news_status = news_msg

        # ── Feature engineering ────────────────────────────────────────────────
        df = add_indicators(raw_df)
        if df is None:
            logger.warning("Feature engineering failed. Skipping this bar.")
            return

        # ── Extract current-bar indicators (CLOSED bar = iloc[-2]) ─────────────
        # iloc[-1] is the newly opened bar. iloc[-2] is the last confirmed close.
        closed_bar = df.iloc[-2]
        current_rsi       = float(closed_bar.get("RSI_14",         0.0))
        current_atr       = float(closed_bar.get("ATRr_14",        0.0010))
        current_ema_dist  = float(closed_bar.get("ema_200_dist",   0.0))
        current_mtf       = float(closed_bar.get("MTF_Alignment",  0.0)) if "MTF_Alignment" in df.columns else 0.0
        current_bb_width  = float(closed_bar.get("BB_Width",       0.0))
        current_price     = float(closed_bar.get("close",          0.0))

        # Swing levels for SL/TP placement (from last confirmed swing points)
        recent_highs = df[df["is_swing_high"] == 1.0]["high"]
        recent_lows  = df[df["is_swing_low"]  == 1.0]["low"]
        swing_high   = float(recent_highs.iloc[-1]) if not recent_highs.empty else current_price + (current_atr * 3)
        swing_low    = float(recent_lows.iloc[-1])  if not recent_lows.empty  else current_price - (current_atr * 3)

        # Volatility-adjusted dynamic risk
        rolling_bb_mean = float(df["BB_Width_rolling_mean"].iloc[-2]) if "BB_Width_rolling_mean" in df.columns else current_bb_width
        vol_scalar      = (current_bb_width / rolling_bb_mean) if rolling_bb_mean > 0 else 1.0
        dynamic_risk    = float(np.clip(
            TradingConfig.RISK_PER_TRADE * vol_scalar,
            TradingConfig.MIN_DYNAMIC_RISK,
            TradingConfig.MAX_DYNAMIC_RISK,
        ))

        # ── Build and validate live observation ────────────────────────────────
        if not validate_live_observation(df):
            logger.warning("Observation validation failed. Skipping inference.")
            return

        # Use last window_size CLOSED bars (exclude the open bar at [-1])
        feature_window = df.iloc[-ModelConfig.WINDOW_SIZE - 1: -1][FEATURE_COLUMNS].values
        market_flat    = feature_window.flatten()

        floating_pnl   = self._compute_floating_pnl_from_price(current_price)
        raw_obs        = np.append(
            market_flat,
            [float(self.current_position), floating_pnl / 1000.0],
        ).astype(np.float32)
        norm_obs = self.vec_env.normalize_obs(raw_obs.reshape(1, -1))

        # ── PPO inference ──────────────────────────────────────────────────────
        action, _ = self.model.predict(norm_obs, deterministic=True)
        action    = int(action[0])

        # News halt overrides model signal
        if news_imminent:
            self.ai_signal = "NEWS HALT"
            if self.current_position != 0:
                logger.warning("News imminent and position open — emergency flatten.")
                self._close_all_positions(reason="Pre-news flatten")
            return

        # Circuit breaker override
        if self._halted:
            self.ai_signal = "HALTED"
            return

        signal_map     = {1: "BUY", 2: "SELL", 0: "HOLD"}
        self.ai_signal = signal_map.get(action, "HOLD")
        logger.info(f"Signal: {self.ai_signal} | MTF: {current_mtf:+.0f} | RSI: {current_rsi:.1f}")

        # ── Write hourly eval log entry for dashboard feed ─────────────────────
        fvg_bull_dist = float(closed_bar.get("bull_fvg_dist_atr", 0.0))
        fvg_bear_dist = float(closed_bar.get("bear_fvg_dist_atr", 0.0))
        self.state.log_eval(
            mtf=int(current_mtf), rsi=current_rsi,
            fvg_bull=fvg_bull_dist, fvg_bear=fvg_bear_dist,
            signal=self.ai_signal,
        )

        # Cache indicator snapshot for attaching to the close journal record
        self._entry_indicators = {
            "rsi": current_rsi, "ema_dist": current_ema_dist,
            "atr": current_atr, "mtf_align": int(current_mtf),
            "bb_width": current_bb_width,
            "bull_fvg_dist_atr": fvg_bull_dist, "bear_fvg_dist_atr": fvg_bear_dist,
        }

        # ── Execute ────────────────────────────────────────────────────────────
        if auto_trade and acc:
            self._execute_action(
                action       = action,
                symbol       = symbol,
                account      = acc,
                current_price= current_price,
                swing_high   = swing_high,
                swing_low    = swing_low,
                atr          = current_atr,
                dynamic_risk = dynamic_risk,
            )

    # ==========================================================================
    # TRADE EXECUTION
    # ==========================================================================

    def _execute_action(
        self,
        action:        int,
        symbol:        str,
        account,
        current_price: float,
        swing_high:    float,
        swing_low:     float,
        atr:           float,
        dynamic_risk:  float,
    ) -> None:
        """Processes a model action: Close/Hold (0), Buy (1), or Sell (2)."""
        target = 1 if action == 1 else (-1 if action == 2 else 0)

        # Already in the target direction — no action needed
        if target == self.current_position:
            return

        # Close existing position before opening a new one (no simultaneous trades)
        if self.current_position != 0:
            self._close_all_positions(reason="Direction change")

        if target == 0:
            return   # Pure close — nothing more to do

        # ── Calculate SL / TP ─────────────────────────────────────────────────
        tick        = self.gateway.symbol_info_tick(symbol)
        padding     = atr * 0.2
        min_sl_dist = max(0.00030, atr * self.spec.min_sl_atr_mult)

        if target == 1:   # Long
            exec_price  = tick.ask
            sl_price    = swing_low  - padding
            tp_price    = swing_high - padding
            risk_dist   = exec_price - sl_price
            reward_dist = tp_price   - exec_price
            order_type  = mt5.ORDER_TYPE_BUY
        else:              # Short
            exec_price  = tick.bid
            sl_price    = swing_high + padding
            tp_price    = swing_low  + padding
            risk_dist   = sl_price   - exec_price
            reward_dist = exec_price - tp_price
            order_type  = mt5.ORDER_TYPE_SELL

        # R:R gate — minimum 1:1 enforced here; model learns to seek 2.5:1+ via reward
        if risk_dist < min_sl_dist or reward_dist < risk_dist:
            logger.info(
                f"Trade REJECTED — bad structural R:R "
                f"(risk={risk_dist:.5f}, reward={reward_dist:.5f}, min={min_sl_dist:.5f})"
            )
            return

        # ── Position sizing ────────────────────────────────────────────────────
        max_risk_dollars = min(account.balance * dynamic_risk, 5_000.0)
        lot_size = max(
            round((max_risk_dollars / risk_dist) / self.spec.lot_unit, 2),
            0.01,
        )
        lot_size = min(lot_size, TradingConfig.MAX_LOT_SIZE)

        # ── Send order ─────────────────────────────────────────────────────────
        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      symbol,
            "volume":      lot_size,
            "type":        order_type,
            "price":       exec_price,
            "sl":          round(sl_price, 5),
            "tp":          round(tp_price, 5),
            "deviation":   BrokerConfig.DEVIATION,
            "magic":       BrokerConfig.MAGIC_NUMBER,
            "comment":     BrokerConfig.ORDER_COMMENT,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling":mt5.ORDER_FILLING_IOC,
        }
        result = self.gateway.order_send(request)

        if result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED):
            self.current_position = target
            self.entry_price      = result.price
            self.active_lot_size  = lot_size

            ind = self._entry_indicators
            self.state.log_entry(
                action    = "BUY" if target == 1 else "SELL",
                price     = result.price,
                lot_size  = lot_size,
                sl        = sl_price,
                tp        = tp_price,
                rsi       = ind.get("rsi", 0.0),
                ema_dist  = ind.get("ema_dist", 0.0),
                atr       = ind.get("atr", 0.0),
                mtf_align = ind.get("mtf_align", 0),
                bb_width  = ind.get("bb_width", 0.0),
                bull_fvg_dist_atr=ind.get("bull_fvg_dist_atr", 0.0),
                bear_fvg_dist_atr=ind.get("bear_fvg_dist_atr", 0.0),
            )
            logger.trade(
                "BUY" if target == 1 else "SELL",
                symbol  = symbol,
                price   = result.price,
                lot     = lot_size,
                sl      = sl_price,
                tp      = tp_price,
            )
        else:
            logger.error(
                f"Order failed — retcode: {result.retcode} | "
                f"comment: {result.comment}"
            )

    def _close_all_positions(self, reason: str = "") -> None:
        """Closes all open positions for the current symbol."""
        positions = self.gateway.positions_get(self.symbol)
        if not positions:
            return

        for pos in positions:
            if pos.magic != BrokerConfig.MAGIC_NUMBER:
                continue

            tick       = self.gateway.symbol_info_tick(self.symbol)
            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price= tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       self.symbol,
                "volume":       pos.volume,
                "type":         close_type,
                "position":     pos.ticket,
                "price":        close_price,
                "deviation":    BrokerConfig.DEVIATION,
                "magic":        BrokerConfig.MAGIC_NUMBER,
                "comment":      f"Close: {reason}",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = self.gateway.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {pos.ticket} closed ({reason}). Price: {result.price}")

    # ==========================================================================
    # BREAKEVEN MANAGEMENT
    # ==========================================================================

    def _manage_breakeven(self, symbol: str, positions) -> None:
        """
        Moves SL to entry price when a position has moved 1R in our favour.
        Mirrors the breakeven logic in ForexTradingEnv.step() exactly.
        """
        for pos in positions:
            if pos.magic != BrokerConfig.MAGIC_NUMBER:
                continue

            entry  = pos.price_open
            cur    = pos.price_current
            sl     = pos.sl
            tp     = pos.tp

            moved_to_breakeven = False

            if pos.type == mt5.ORDER_TYPE_BUY:
                if sl > 0 and sl < entry:
                    initial_risk = entry - sl
                    if cur >= entry + initial_risk:
                        moved_to_breakeven = True
                        new_sl = entry
            else:
                if sl > 0 and sl > entry:
                    initial_risk = sl - entry
                    if cur <= entry - initial_risk:
                        moved_to_breakeven = True
                        new_sl = entry

            if moved_to_breakeven:
                req = {
                    "action":   mt5.TRADE_ACTION_SLTP,
                    "symbol":   symbol,
                    "position": pos.ticket,
                    "sl":       new_sl,
                    "tp":       tp,
                }
                result = self.gateway.order_send(req)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"Breakeven secured for ticket {pos.ticket} "
                        f"at {new_sl:.5f}. Trade is now risk-free."
                    )

    # ==========================================================================
    # MODEL LOADING
    # ==========================================================================

    def _load_model(self) -> bool:
        """Loads the PPO model and VecNormalize stats. Returns True on success."""
        #model_path = Paths.model_zip(self.symbol)
        model_path = Paths.best_model_dir(self.symbol)
        stats_path = Paths.vec_normalize_pkl(self.symbol)

        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False
        if not os.path.exists(stats_path):
            logger.error(f"VecNormalize stats not found: {stats_path}")
            return False

        logger.info("Loading PPO model and VecNormalize stats...")

        # Dummy env needed to reconstruct VecNormalize observation spaces
        raw_df   = self._fetch_data_with_gateway(self.symbol, DEEP_BARS)
        df       = add_indicators(raw_df) if raw_df is not None else None
        if df is None:
            logger.error("Cannot load model: data fetch for dummy env failed.")
            return False

        dummy_env       = DummyVecEnv([lambda: ForexTradingEnv(df)])
        self.vec_env    = VecNormalize.load(stats_path, dummy_env)
        self.vec_env.training    = False
        self.vec_env.norm_reward = False
        self.model = PPO.load(model_path, device="cpu")

        logger.info("Model loaded successfully.")
        return True

    # ==========================================================================
    # STATE RECOVERY & HELPERS
    # ==========================================================================

    def _recover_position_state(self) -> None:
        """
        Restores in-memory position state from MT5 on startup.

        This fixes the P0 bug where a crash/restart caused the bot to lose
        track of any open position (breakeven, floating P&L all wrong).
        """
        positions = self.gateway.positions_get(self.symbol)
        if not positions:
            logger.info("Startup: no open positions found. Starting flat.")
            return

        for pos in positions:
            if pos.magic == BrokerConfig.MAGIC_NUMBER:
                self.current_position = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                self.entry_price      = pos.price_open
                self.active_lot_size  = pos.volume
                logger.info(
                    f"Startup: recovered open position — "
                    f"{'LONG' if self.current_position == 1 else 'SHORT'} "
                    f"{pos.volume} lots @ {pos.price_open:.5f}"
                )
                return

    def _log_broker_close(self, symbol: str) -> None:
        """
        Queries MT5 for the true close record (filled price, commissions, swaps)
        and writes it to the trade journal via StateManager.
        """
        from datetime import timedelta
        from_date = datetime.now() - timedelta(days=2)
        to_date   = datetime.now() + timedelta(seconds=10)
        deals     = self.gateway.history_deals_get(from_date, to_date)
        if not deals:
            return

        bot_deals = sorted(
            [d for d in deals
             if d.magic == BrokerConfig.MAGIC_NUMBER and d.symbol == symbol],
            key=lambda x: x.time,
        )
        if not bot_deals:
            return

        exit_deal    = bot_deals[-1]
        pos_deals    = [d for d in bot_deals if d.position_id == exit_deal.position_id]

        if len(pos_deals) >= 2:
            true_comm  = sum(d.commission for d in pos_deals)
            true_swap  = sum(d.swap       for d in pos_deals)
            raw_profit = sum(d.profit     for d in pos_deals)
            net_pnl    = raw_profit + true_comm + true_swap

            from core.config import JournalConfig
            close_action = (
                JournalConfig.BUY_CLOSE
                if exit_deal.type == mt5.DEAL_TYPE_SELL
                else JournalConfig.SELL_CLOSE
            )
            ind = self._entry_indicators
            self.state.log_close(
                action    = close_action,
                price     = exit_deal.price,
                lot_size  = exit_deal.volume,
                pnl       = round(net_pnl, 2),
                rsi       = ind.get("rsi", 0.0),
                ema_dist  = ind.get("ema_dist", 0.0),
                atr       = ind.get("atr", 0.0),
                mtf_align = ind.get("mtf_align", 0),
                bb_width  = ind.get("bb_width", 0.0),
                bull_fvg  = ind.get("bull_fvg", 0),
                bear_fvg  = ind.get("bear_fvg", 0),
            )
            outcome = "TP HIT" if raw_profit > 0 else "SL HIT" if raw_profit < 0 else "BREAKEVEN"
            logger.trade(
                close_action,
                symbol = symbol,
                price  = exit_deal.price,
                lot    = exit_deal.volume,
                sl     = 0.0, tp = 0.0,
                pnl    = net_pnl,
            )
            logger.info(f"Outcome: {outcome} | Net P&L (with fees): ${net_pnl:+.2f}")

    def _handle_manual_command(self, cmd: str, symbol: str, acc) -> None:
        """Executes a manual BUY/SELL/CLOSE command from the dashboard."""
        if cmd == "CLOSE":
            self._close_all_positions(reason="Manual close")
            self.current_position = 0
            self.entry_price      = 0.0
            self.active_lot_size  = 0.0
            return

        action = 1 if cmd == "BUY" else 2 if cmd == "SELL" else 0
        if action == 0:
            return

        tick    = self.gateway.symbol_info_tick(symbol)
        ref_atr = 0.0010   # Safe fallback for manual trades (model ATR not available)

        self._execute_action(
            action        = action,
            symbol        = symbol,
            account       = acc,
            current_price = tick.ask,
            swing_high    = tick.ask + (ref_atr * 3),
            swing_low     = tick.ask - (ref_atr * 3),
            atr           = ref_atr,
            dynamic_risk  = TradingConfig.RISK_PER_TRADE,
        )

    def _fetch_data_with_gateway(self, symbol: str, n_bars: int):
        """Fetches OHLCV data using the gateway's serialised MT5 call."""
        rates = self.gateway.copy_rates_from_pos(
            symbol, mt5.TIMEFRAME_H1, 0, n_bars
        )
        if rates is None or len(rates) == 0:
            return None
        import pandas as pd
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    def _fetch_ui_data(self, symbol: str, timeframe: str):
        """Light fetch for UI chart update. Uses get_historical_data pattern."""
        rates = self.gateway.copy_rates_from_pos(
            symbol, mt5.TIMEFRAME_H1, 0, TradingConfig.CHART_BARS
        )
        if rates is None or len(rates) == 0:
            return None
        import pandas as pd
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        return df

    def _compute_floating_pnl(self, tick) -> float:
        """Computes current floating P&L from live tick prices."""
        if self.current_position == 0 or tick is None:
            return 0.0
        if self.current_position == 1:
            return (tick.bid - self.entry_price) * self.spec.lot_unit * self.active_lot_size
        return (self.entry_price - tick.ask) * self.spec.lot_unit * self.active_lot_size

    def _compute_floating_pnl_from_price(self, price: float) -> float:
        """Computes floating P&L from a mid price (for observation construction)."""
        if self.current_position == 0:
            return 0.0
        if self.current_position == 1:
            return (price - self.entry_price) * self.spec.lot_unit * self.active_lot_size
        return (self.entry_price - price) * self.spec.lot_unit * self.active_lot_size


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main() -> None:
    load_dotenv()
    trader = LiveTrader(symbol=TradingConfig.ACTIVE_SYMBOL)
    trader.start()


if __name__ == "__main__":
    main()
"""
engine/trainer.py
=================
Walk-forward PPO training pipeline — Quant Predator v2.1.

Changes from v2.0:
  - Linear learning-rate schedule (1e-4 → 1e-5).
  - Halved n_steps (4096 → 2048), reduced batch_size (512 → 256).
  - Tighter clip_range (0.20 → 0.15) + explicit max_grad_norm=0.5.
  - EvalCallback saves best checkpoint on validation set.
  - Full Weights & Biases integration (graceful no-op if not installed).
  - Convergence targets printed at startup — watch these in the W&B dashboard.


RUN:
  python engine/trainer.py
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback, ProgressBarCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig, Paths, RewardConfig, TradingConfig, WandbConfig, get_symbol_spec
from core.logger import get_logger
from data.feature_engineer import add_indicators
from data.fetcher import get_historical_data
from engine.policy import LSTMFeatureExtractor   # must import before PPO.load
from engine.rl_env import ForexTradingEnv

logger = get_logger(__name__)

# ── W&B optional import ────────────────────────────────────────────────────────
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.warning(
        "wandb not installed — experiment tracking disabled. "
        "Install with: pip install wandb"
    )


# ==============================================================================
# CALLBACKS
# ==============================================================================

class ConvergenceMonitorCallback(BaseCallback):
    """
    Prints per-rollout convergence warnings to the console and log file.

    W&B metric logging is handled EXCLUSIVELY by WandbCallback
    (from wandb.integration.sb3) to avoid duplicate metric entries.
    This callback is purely for human-readable console alerts.

    Fires after every PPO update (every n_steps × n_envs steps).
    Note: SB3 logs training metrics during train() which runs AFTER
    rollout collection, so metrics reflect the previous iteration's
    update — a one-rollout lag that is normal and acceptable.
    """

    _WARN_CLIP_ABOVE = ModelConfig.TARGET_CLIP_FRACTION * 1.5   # warn if >0.225
    _WARN_KL_ABOVE   = ModelConfig.TARGET_KL_DIVERGENCE * 2.0   # warn if >0.040

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.iteration  = 0
        self._last_clip = 0.0
        self._last_kl   = 0.0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Prints convergence metrics every 10 rollouts; warns on instability."""
        self.iteration += 1
        metrics = dict(self.model.logger.name_to_value)
        if not metrics:
            return

        clip = metrics.get("train/clip_fraction",     0.0)
        kl   = metrics.get("train/approx_kl",         0.0)
        ev   = metrics.get("train/explained_variance", 0.0)
        ent  = metrics.get("train/entropy_loss",       0.0)
        lr   = metrics.get("train/learning_rate",      0.0)

        self._last_clip = clip
        self._last_kl   = kl

        if self.iteration % 10 == 0:
            clip_tag = "✅" if clip <= self._WARN_CLIP_ABOVE else "⚠️  HIGH"
            kl_tag   = "✅" if kl   <= self._WARN_KL_ABOVE   else "⚠️  HIGH"
            logger.info(
                f"[iter {self.iteration:>4}] "
                f"clip={clip:.3f} {clip_tag}  "
                f"kl={kl:.4f} {kl_tag}  "
                f"ev={ev:.3f}  ent={ent:.3f}  lr={lr:.2e}"
            )
            if clip > self._WARN_CLIP_ABOVE:
                logger.warning(
                    f"clip_fraction {clip:.3f} still above target "
                    f"{ModelConfig.TARGET_CLIP_FRACTION:.2f}. "
                    f"Linear LR schedule will reduce this — watch next 10 iters."
                )


# ==============================================================================
# LEARNING RATE SCHEDULE
# ==============================================================================

def _linear_lr_schedule(initial_lr: float, end_lr: float):
    """
    Linear decay from initial_lr (start) to end_lr (end of training).

    SB3 passes remaining_progress: 1.0 at start, 0.0 at end.
    WHY: Constant 3e-4 caused clip_fraction=0.374 in run 1.
    Decaying to 1e-5 allows fine-tuning once policy stabilises.
    """
    def schedule(remaining: float) -> float:
        done = 1.0 - remaining
        return initial_lr + (end_lr - initial_lr) * done
    return schedule


# ==============================================================================
# ENV FACTORY  (module-level for Windows multiprocessing pickling)
# ==============================================================================

def _make_env(df, spread_pips: float):
    def _init() -> ForexTradingEnv:
        return ForexTradingEnv(
            df,
            commission  = ModelConfig.COMMISSION,
            spread_pips = spread_pips,
            window_size = ModelConfig.WINDOW_SIZE,
        )
    return _init


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_agent(symbol: str = TradingConfig.ACTIVE_SYMBOL) -> None:
    """
    Full v2.1 training pipeline with W&B tracking.

    Args:
        symbol: MT5 instrument to train on.
    """
    Paths.ensure_dirs()
    spec      = get_symbol_spec(symbol)
    run_name  = f"{WandbConfig.RUN_NAME_PREFIX}_{datetime.now():%Y%m%d_%H%M%S}"
    use_wandb = WandbConfig.ENABLED and _WANDB_AVAILABLE

    # ── W&B run init ──────────────────────────────────────────────────────────
    if use_wandb:
        wandb.init(
            project  = WandbConfig.PROJECT,
            entity   = WandbConfig.ENTITY,
            name     = run_name,
            config   = {
                # Model
                "symbol":           symbol,
                "window_size":      ModelConfig.WINDOW_SIZE,
                "n_features":       ModelConfig.N_FEATURES,
                "lstm_hidden":      ModelConfig.LSTM_HIDDEN,
                "lstm_layers":      ModelConfig.LSTM_LAYERS,
                "policy_net":       ModelConfig.POLICY_NET,
                # PPO v2.1
                "total_timesteps":  ModelConfig.TOTAL_TIMESTEPS,
                "learning_rate":    ModelConfig.LEARNING_RATE,
                "lr_end":           ModelConfig.LR_END,
                "ent_coef":         ModelConfig.ENT_COEF,
                "batch_size":       ModelConfig.BATCH_SIZE,
                "n_steps":          ModelConfig.N_STEPS,
                "clip_range":       ModelConfig.CLIP_RANGE,
                "max_grad_norm":    ModelConfig.MAX_GRAD_NORM,
                "min_rr_ratio":     ModelConfig.MIN_RR_RATIO,
                "n_envs":           ModelConfig.N_ENVS,
                # Reward v2.1
                "entry_cost":       RewardConfig.ENTRY_COST,
                "patience_bonus":   RewardConfig.PATIENCE_BONUS_FLAT,
                "mtf_penalty":      RewardConfig.WEAK_MTF_PENALTY,
                "mtf_threshold":    RewardConfig.MTF_GATE_THRESHOLD,
                "asian_session_mult": RewardConfig.ASIAN_SESSION_MULT,
                # Broker
                "spread_pips":      spec.spread_pips,
                "commission":       ModelConfig.COMMISSION,
            },
            tags     = [symbol, "v2.1", "ppo", "lstm"],
            notes    = (
                "v2.1: LR 1e-4 linear→1e-5, n_steps 2048, "
                "entry_cost -0.05, min_rr 2.0, clip 0.15"
            ),
        )
        logger.info(f"W&B run '{run_name}' initialised — {wandb.run.url}")

    # ── Connect to MT5 ────────────────────────────────────────────────────────
    load_dotenv()
    logger.info("Connecting to MT5...")
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return
    if not mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
    ):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    # ── Fetch and engineer features ───────────────────────────────────────────
    logger.info(f"Fetching {ModelConfig.TRAIN_BARS:,} {symbol} H1 bars...")
    raw_df = get_historical_data(symbol, "H1", num_bars=ModelConfig.TRAIN_BARS)
    mt5.shutdown()

    if raw_df is None:
        logger.error("Data fetch failed.")
        if use_wandb: wandb.finish(exit_code=1)
        return

    logger.info("Building Multi-Timeframe Feature Matrix...")
    df = add_indicators(raw_df)
    if df is None:
        logger.error("Feature engineering failed.")
        if use_wandb: wandb.finish(exit_code=1)
        return

    logger.info(f"Effective dataset after warmup: {len(df):,} bars")

    # ── Temporal split: 70 / 15 / 15 ─────────────────────────────────────────
    n         = len(df)
    n_train   = int(n * ModelConfig.TRAIN_SPLIT)
    n_val     = int(n * ModelConfig.VAL_SPLIT)
    n_test    = n - n_train - n_val

    train_df  = df.iloc[:n_train].reset_index(drop=True)
    val_df    = df.iloc[n_train: n_train + n_val].reset_index(drop=True)
    test_df   = df.iloc[n_train + n_val:].reset_index(drop=True)

    logger.info(
        f"Split — train: {n_train:,} | val: {n_val:,} | OOS test: {n_test:,} bars"
    )
    if use_wandb:
        wandb.config.update({
            "n_train_bars": n_train,
            "n_val_bars":   n_val,
            "n_test_bars":  n_test,
        })

    # ── Build training environment ────────────────────────────────────────────
    logger.info(f"Creating {ModelConfig.N_ENVS} parallel training environments...")
    env_fns = [_make_env(train_df, spec.spread_pips) for _ in range(ModelConfig.N_ENVS)]

    try:
        env = SubprocVecEnv(env_fns, start_method="spawn")
        logger.info("SubprocVecEnv active (parallel CPU workers).")
    except Exception as exc:
        logger.warning(f"SubprocVecEnv failed ({exc}). Falling back to DummyVecEnv.")
        env = DummyVecEnv([env_fns[0]])

    env = VecNormalize(
        env, norm_obs=True, norm_reward=True, clip_obs=ModelConfig.CLIP_OBS,
    )

    # Validation env for EvalCallback (separate from test set)
    val_env = VecNormalize(
        DummyVecEnv([_make_env(val_df, spec.spread_pips)]),
        norm_obs=True, norm_reward=False,
        clip_obs=ModelConfig.CLIP_OBS, training=False,
    )

    # ── PPO model with LSTM policy ────────────────────────────────────────────
    import torch
    policy_kwargs = dict(
        features_extractor_class  = LSTMFeatureExtractor,
        features_extractor_kwargs = dict(
            lstm_hidden = ModelConfig.LSTM_HIDDEN,
            n_layers    = ModelConfig.LSTM_LAYERS,
            dropout     = ModelConfig.LSTM_DROPOUT,
            n_features  = ModelConfig.N_FEATURES,
            window_size = ModelConfig.WINDOW_SIZE,
        ),
        net_arch      = dict(pi=ModelConfig.POLICY_NET, vf=ModelConfig.POLICY_NET),
        activation_fn = torch.nn.Tanh,
    )

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        verbose       = 1,
        learning_rate = _linear_lr_schedule(ModelConfig.LEARNING_RATE, ModelConfig.LR_END),
        ent_coef      = ModelConfig.ENT_COEF,
        batch_size    = ModelConfig.BATCH_SIZE,
        n_steps       = ModelConfig.N_STEPS,
        n_epochs      = ModelConfig.N_EPOCHS,
        gamma         = ModelConfig.GAMMA,
        gae_lambda    = ModelConfig.GAE_LAMBDA,
        clip_range    = ModelConfig.CLIP_RANGE,
        max_grad_norm = ModelConfig.MAX_GRAD_NORM,
        policy_kwargs = policy_kwargs,
        device        = "auto",
    )

    device_str = str(model.device)
    if "cuda" in device_str:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({device_str})")
    else:
        logger.warning(
            f"Training on CPU ({device_str}). "
            "Verify CUDA PyTorch is installed: "
            "python -c \"import torch; print(torch.cuda.is_available())\""
        )

    # ── Print convergence targets ──────────────────────────────────────────────
    logger.info(
        f"\n  {'='*58}\n"
        f"  V2.1 CONVERGENCE TARGETS (watch in W&B dashboard)\n"
        f"  {'─'*58}\n"
        f"  clip_fraction  target < {ModelConfig.TARGET_CLIP_FRACTION:.2f}  "
        f"(run 1: 0.374 — thrashing)\n"
        f"  approx_kl      target < {ModelConfig.TARGET_KL_DIVERGENCE:.3f}  "
        f"(run 1: 0.051 — 2.6x safe)\n"
        f"  explained_var  target > {ModelConfig.TARGET_EXPL_VARIANCE:.2f}  "
        f"(run 1: 0.852 — already good)\n"
        f"  entropy_loss   target -0.5 to -1.0   (exploration health)\n"
        f"  {'─'*58}\n"
        f"  Entry cost penalty:  {RewardConfig.ENTRY_COST} (was -0.30 — now -0.05)\n"
        f"  LR schedule:         {ModelConfig.LEARNING_RATE:.0e} → {ModelConfig.LR_END:.0e}\n"
        f"  {'='*58}"
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_dir = Paths.best_model_dir(symbol)
    os.makedirs(best_dir, exist_ok=True)

    convergence_cb = ConvergenceMonitorCallback()

    eval_cb = EvalCallback(
        val_env,
        best_model_save_path = best_dir,
        log_path             = best_dir,
        eval_freq            = ModelConfig.N_STEPS * ModelConfig.N_ENVS * 2,
        n_eval_episodes      = 5,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    callback_list = [convergence_cb, eval_cb, ProgressBarCallback()]
    if use_wandb and _WANDB_AVAILABLE:
        callback_list.append(
            WandbCallback(
                gradient_save_freq = WandbConfig.GRADIENT_LOG_FREQ,
                verbose            = 0,
            )
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(
        f"Training started — {ModelConfig.TOTAL_TIMESTEPS:,} timesteps. "
        f"Estimated ~80-95 min on RTX 3050."
    )
    model.learn(
        total_timesteps = ModelConfig.TOTAL_TIMESTEPS,
        callback        = CallbackList(callback_list),
        progress_bar    = True,
    )
    logger.info("Training complete.")

    # ── Save ─────────────────────────────────────────────────────────────────
    env.training    = False
    env.norm_reward = False

    model_path = Paths.model_zip(symbol)
    stats_path = Paths.vec_normalize_pkl(symbol)
    model.save(model_path)
    env.save(stats_path)
    logger.info(f"Model   → {model_path}")
    logger.info(f"VecNorm → {stats_path}")
    logger.info(f"Best checkpoint → {best_dir}/best_model.zip")

    # ── Upload model artifacts to W&B ─────────────────────────────────────────
    if use_wandb and _WANDB_AVAILABLE:
        artifact = wandb.Artifact(
            name     = f"{symbol.lower()}_model",
            type     = "model",
            metadata = {"symbol": symbol, "version": "v2.1"},
        )
        artifact.add_file(model_path)
        artifact.add_file(stats_path)
        wandb.log_artifact(artifact)
        logger.info("Model artifacts uploaded to W&B.")

    # ── Out-of-sample evaluation ──────────────────────────────────────────────
    logger.info("Running out-of-sample evaluation on held-out test set...")
    from engine.backtester import run_backtest
    from tools.tear_sheet import generate_tear_sheet

    equity_curve, trade_log = run_backtest(
        df         = test_df,
        model_path = model_path,
        stats_path = stats_path,
        symbol     = symbol,
        label      = "OOS Test",
    )

    if equity_curve is not None and trade_log is not None:
        metrics = generate_tear_sheet(equity_curve, trade_log, label="Out-of-Sample")

        # Trade frequency diagnostic
        n_trades       = int(metrics.get("n_trades", 0))
        n_test_days    = len(test_df) / 24
        trades_per_day = n_trades / max(n_test_days, 1)
        bars_per_trade = len(test_df) / max(n_trades, 1)

        logger.info(
            f"TRADE FREQUENCY: {trades_per_day:.1f} trades/day "
            f"({bars_per_trade:.0f} bars/trade) "
            f"| Target: 2-3/day (8-12 bars/trade)"
        )

        freq_status = (
            "✅ On target"     if 1.5 <= trades_per_day <= 3.5 else
            "⚠️ Too frequent"  if trades_per_day > 3.5 else
            "⚠️ Too infrequent"
        )
        logger.info(f"Frequency verdict: {freq_status}")

        # W&B final metrics
        if use_wandb and _WANDB_AVAILABLE:
            wandb.log({
                "oos/sharpe_ratio":    metrics.get("sharpe_ratio",     0.0),
                "oos/max_drawdown":    metrics.get("max_drawdown_pct", 0.0),
                "oos/win_rate":        metrics.get("win_rate_pct",     0.0),
                "oos/profit_factor":   metrics.get("profit_factor",    0.0),
                "oos/total_return":    metrics.get("total_return_pct", 0.0),
                "oos/expectancy_usd":  metrics.get("expectancy_usd",   0.0),
                "oos/n_trades":        n_trades,
                "oos/trades_per_day":  trades_per_day,
                "oos/bars_per_trade":  bars_per_trade,
            })

        # Validation gate
        sharpe  = metrics.get("sharpe_ratio",     0.0)
        max_dd  = abs(metrics.get("max_drawdown_pct", 100.0))
        passed  = (sharpe >= ModelConfig.MIN_OOS_SHARPE
                   and max_dd <= ModelConfig.MAX_OOS_DRAWDOWN * 100)

        if passed:
            logger.info(
                f"VALIDATION PASSED ✅  Sharpe: {sharpe:.2f} "
                f"| MaxDD: {max_dd:.1f}%"
            )
        else:
            logger.warning(
                f"VALIDATION FAILED ❌  Sharpe: {sharpe:.2f} "
                f"| MaxDD: {max_dd:.1f}%. "
                f"Try best checkpoint at {best_dir}/best_model.zip, "
                f"or run engine/optuna_search.py for hyperparameter tuning."
            )

        if use_wandb and _WANDB_AVAILABLE:
            wandb.log({"oos/validation_passed": int(passed)})

    if use_wandb and _WANDB_AVAILABLE:
        wandb.finish()
        logger.info(f"W&B run complete — {wandb.run.url if wandb.run else 'run finished'}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    train_agent(symbol=TradingConfig.ACTIVE_SYMBOL)

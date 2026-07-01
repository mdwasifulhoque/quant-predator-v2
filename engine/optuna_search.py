"""
engine/optuna_search.py
=======================
Bayesian hyperparameter search for Quant Predator v2.

HOW IT WORKS:
  1. Loads data once from MT5 (expensive — only done once before the study).
  2. Runs N_TRIALS PPO training runs, each with different hyperparameters
     sampled by Optuna's TPE (Tree-structured Parzen Estimator) algorithm.
  3. Each trial trains for TRIAL_TIMESTEPS (short) then evaluates on the
     validation set. Bad trials are pruned early via MedianPruner.
  4. After all trials, prints the best hyperparameter set and writes it
     to tools/optuna_best_eurusd.json.
  5. Optionally logs every trial to W&B for visual comparison.

SEARCH SPACE covers both training stability and reward shaping:
  Training:  learning_rate, clip_range, ent_coef, n_steps, batch_size
  Reward:    entry_cost, patience_bonus, mtf_gate_threshold, min_rr_ratio

RUNTIME (RTX 3050, TRIAL_TIMESTEPS=400k, N_TRIALS=15):
  ~15-18 min per trial  ×  15 trials  ≈  3.5-4.5 hours (run overnight)

USAGE:
  python engine/optuna_search.py

AFTER SEARCH:
  Update ModelConfig and RewardConfig in core/config.py with the
  values from tools/optuna_best_eurusd.json, then retrain with full
  2.5M timesteps: python engine/trainer.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Dict, Optional, Any

warnings.filterwarnings("ignore")

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig, OptunaConfig, Paths, RewardConfig, TradingConfig, WandbConfig, get_symbol_spec
from core.logger import get_logger
from data.feature_engineer import add_indicators
from data.fetcher import get_historical_data
from engine.policy import LSTMFeatureExtractor  # must import before PPO.load
from engine.rl_env import ForexTradingEnv
from tools.tear_sheet import generate_tear_sheet

logger = get_logger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.error("optuna not installed. Run: pip install optuna")
    sys.exit(1)

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ==============================================================================
# PRUNING CALLBACK
# ==============================================================================

class OptunaPruningCallback(BaseCallback):
    """
    Reports intermediate validation reward to Optuna every EVAL_FREQ steps.
    Allows Optuna to prune (stop early) trials that are clearly underperforming.

    This can save 40-60% of total search compute time by cutting off
    bad hyperparameter configurations halfway through training.
    """

    def __init__(
        self,
        trial: "optuna.Trial",
        val_env: VecNormalize,
        eval_freq: int = OptunaConfig.EVAL_FREQ_STEPS,
        n_eval_episodes: int = OptunaConfig.N_EVAL_EPISODES,
    ) -> None:
        super().__init__(verbose=0)
        self.trial           = trial
        self.val_env         = val_env
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self._report_step    = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, _ = evaluate_policy(
                self.model, self.val_env,
                n_eval_episodes = self.n_eval_episodes,
                deterministic   = True,
                warn            = False,
            )
            self._report_step += 1
            self.trial.report(mean_reward, self._report_step)

            if self.trial.should_prune():
                logger.info(
                    f"  Trial {self.trial.number} pruned at "
                    f"{self.num_timesteps:,} steps "
                    f"(val_reward={mean_reward:.3f})."
                )
                raise optuna.exceptions.TrialPruned()
        return True


# ==============================================================================
# OBJECTIVE FUNCTION
# ==============================================================================

def build_objective(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    symbol:   str,
):
    """
    Returns an Optuna objective callable with data pre-baked in.

    Closing over DataFrames avoids re-fetching MT5 data on each trial.
    """
    spec = get_symbol_spec(symbol)

    def objective(trial: "optuna.Trial") -> float:
        """
        Trains a PPO agent with trial hyperparameters, evaluates on val_df,
        returns OOS Sharpe ratio as the optimisation objective (maximise).
        """
        # ── Sample hyperparameters ────────────────────────────────────────────
        params: Dict[str, Any] = {
            # Training stability
            "learning_rate":    trial.suggest_float("learning_rate",    5e-5, 3e-4, log=True),
            "ent_coef":         trial.suggest_float("ent_coef",         0.02, 0.15),
            "clip_range":       trial.suggest_float("clip_range",       0.10, 0.25),
            "n_steps":          trial.suggest_categorical("n_steps",    [1024, 2048, 4096]),
            "batch_size":       trial.suggest_categorical("batch_size", [128, 256, 512]),
            # Reward shaping
            "entry_cost":       trial.suggest_float("entry_cost",       -0.15, -0.01),
            "patience_bonus":   trial.suggest_float("patience_bonus",    0.001,  0.010),
            "mtf_threshold":    trial.suggest_int("mtf_threshold",       1, 3),
            "min_rr_ratio":     trial.suggest_float("min_rr_ratio",      1.5, 3.0),
        }
        logger.info(f"\nTrial {trial.number}: {params}")

        # Temporarily override global config for this trial
        # (Trials run sequentially — no thread-safety concern)
        RewardConfig.ENTRY_COST          = params["entry_cost"]
        RewardConfig.PATIENCE_BONUS_FLAT = params["patience_bonus"]
        RewardConfig.MTF_GATE_THRESHOLD  = params["mtf_threshold"]
        ModelConfig.MIN_RR_RATIO         = params["min_rr_ratio"]

        # ── Environments ──────────────────────────────────────────────────────
        def make_env():
            return ForexTradingEnv(
                train_df,
                commission  = ModelConfig.COMMISSION,
                spread_pips = spec.spread_pips,
                window_size = ModelConfig.WINDOW_SIZE,
            )
        # DummyVecEnv for Optuna (avoid SubprocVecEnv process overhead per trial)
        train_env = VecNormalize(
            DummyVecEnv([make_env, make_env]),
            norm_obs=True, norm_reward=True, clip_obs=ModelConfig.CLIP_OBS,
        )
        val_env = VecNormalize(
            DummyVecEnv([lambda: ForexTradingEnv(
                val_df, commission=ModelConfig.COMMISSION,
                spread_pips=spec.spread_pips, window_size=ModelConfig.WINDOW_SIZE,
            )]),
            norm_obs=True, norm_reward=False,
            clip_obs=ModelConfig.CLIP_OBS, training=False,
        )

        # ── PPO model ─────────────────────────────────────────────────────────
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
            env           = train_env,
            verbose       = 0,
            learning_rate = params["learning_rate"],
            ent_coef      = params["ent_coef"],
            batch_size    = params["batch_size"],
            n_steps       = params["n_steps"],
            n_epochs      = ModelConfig.N_EPOCHS,
            gamma         = ModelConfig.GAMMA,
            gae_lambda    = ModelConfig.GAE_LAMBDA,
            clip_range    = params["clip_range"],
            max_grad_norm = ModelConfig.MAX_GRAD_NORM,
            policy_kwargs = policy_kwargs,
            device        = "auto",
        )

        # ── Train with pruning ────────────────────────────────────────────────
        pruning_cb = OptunaPruningCallback(trial, val_env)
        try:
            model.learn(
                total_timesteps = OptunaConfig.TRIAL_TIMESTEPS,
                callback        = pruning_cb,
                progress_bar    = False,
            )
        except optuna.exceptions.TrialPruned:
            train_env.close()
            val_env.close()
            raise

        # ── Evaluate on validation set ────────────────────────────────────────
        try:
            from engine.backtester import run_backtest

            # Create safe absolute paths for temporary files within your execution directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            temp_model_path = os.path.join(current_dir, f"temp_model_{trial.number}.zip")
            temp_stats_path = os.path.join(current_dir, f"temp_stats_{trial.number}.pkl")

            # Save the active in-memory trial weights to disk so backtester.py can read them
            model.save(temp_model_path)
            val_env.save(temp_stats_path)

            # Run the deterministic simulation using your exact backtester logic
            equity_curve, trade_log = run_backtest(
                df=val_df,
                model_path=temp_model_path,
                stats_path=temp_stats_path,
                symbol=symbol,
                label=f"Trial-{trial.number}",
            )

            # Clean up the temporary disk assets immediately after evaluation completes
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            if os.path.exists(temp_stats_path):
                os.remove(temp_stats_path)

            if equity_curve is None:
                score = -9.99
            else:
                metrics = generate_tear_sheet(
                    equity_curve, trade_log,
                    label="", print_report=False,
                )
                score = float(metrics.get("sharpe_ratio", -9.99))

        except Exception as exc:
            logger.warning(f"Trial {trial.number} eval failed: {exc}")
            score = -9.99
        finally:
            # Clear gym resource handles safely
            train_env.close()
            val_env.close()

        # Log to W&B if available
        if _WANDB_AVAILABLE and WandbConfig.ENABLED:
            try:
                wandb.log({
                    f"optuna/trial_{trial.number}_sharpe": score,
                    **{f"optuna/{k}": v for k, v in params.items()},
                })
            except Exception:
                pass

        logger.info(
            f"Trial {trial.number} finished — "
            f"Sharpe={score:.3f} | params={params}"
        )
        return score

    return objective


# ==============================================================================
# MAIN SEARCH
# ==============================================================================

def run_optuna_search(symbol: str = TradingConfig.ACTIVE_SYMBOL) -> Optional[Dict]:
    """
    Runs the full Optuna hyperparameter search.

    Returns the best params dict, or None on failure.
    """
    Paths.ensure_dirs()

    # ── One-time data load ────────────────────────────────────────────────────
    load_dotenv()
    logger.info("Connecting to MT5 for data fetch...")
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return None
    if not mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
    ):
        logger.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return None

    raw_df = get_historical_data(symbol, "H1", num_bars=ModelConfig.TRAIN_BARS)
    mt5.shutdown()

    if raw_df is None:
        logger.error("Data fetch failed.")
        return None

    df = add_indicators(raw_df)
    if df is None:
        logger.error("Feature engineering failed.")
        return None

    n       = len(df)
    n_train = int(n * ModelConfig.TRAIN_SPLIT)
    n_val   = int(n * ModelConfig.VAL_SPLIT)
    train_df= df.iloc[:n_train].reset_index(drop=True)
    val_df  = df.iloc[n_train: n_train + n_val].reset_index(drop=True)

    logger.info(
        f"Data ready — train: {n_train:,} | val: {n_val:,} bars | "
        f"{OptunaConfig.N_TRIALS} trials × "
        f"{OptunaConfig.TRIAL_TIMESTEPS:,} steps each"
    )

    # ── Optuna study ──────────────────────────────────────────────────────────
    if OptunaConfig.PRUNER == "median":
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    elif OptunaConfig.PRUNER == "halving":
        pruner = SuccessiveHalvingPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name = OptunaConfig.STUDY_NAME,
        storage    = OptunaConfig.STORAGE,
        sampler    = TPESampler(seed=42, n_startup_trials=5),
        pruner     = pruner,
        direction  = "maximize",    # maximise OOS Sharpe
        load_if_exists = True,
    )

    objective = build_objective(train_df, val_df, symbol)

    logger.info(
        f"Starting Optuna study '{OptunaConfig.STUDY_NAME}'. "
        f"Estimated time: {OptunaConfig.N_TRIALS * 17:.0f}-"
        f"{OptunaConfig.N_TRIALS * 20:.0f} minutes."
    )
    study.optimize(
        objective,
        n_trials  = OptunaConfig.N_TRIALS,
        timeout   = None,
        show_progress_bar = True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_params = best_trial.params
    best_score  = best_trial.value

    print(f"\n{'='*60}")
    print(f"  OPTUNA SEARCH COMPLETE — {len(study.trials)} trials")
    print(f"{'='*60}")
    print(f"  Best Sharpe: {best_score:.4f}  (Trial #{best_trial.number})")
    print(f"{'─'*60}")
    print("  Best hyperparameters:")
    for k, v in best_params.items():
        print(f"    {k:<25} = {v}")
    print(f"{'─'*60}")
    print("  Top 5 trials:")
    for t in sorted(study.trials, key=lambda x: x.value or -99, reverse=True)[:5]:
        print(f"    Trial {t.number:>3}  Sharpe={t.value:.4f}  "
              f"lr={t.params.get('learning_rate', 0):.2e}  "
              f"entry_cost={t.params.get('entry_cost', 0):.3f}")
    print(f"{'='*60}\n")

    # Save to JSON
    output = {
        "best_sharpe":       best_score,
        "best_trial_number": best_trial.number,
        "best_params":       best_params,
        "n_trials_completed":len([t for t in study.trials if t.value is not None]),
        "n_trials_pruned":   len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }
    out_path = Paths.optuna_results(symbol)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Best params saved to {out_path}")

    # W&B summary
    if _WANDB_AVAILABLE and WandbConfig.ENABLED:
        try:
            wandb.init(
                project=WandbConfig.PROJECT, name=f"optuna_summary_{symbol}",
                config=best_params, tags=[symbol, "optuna"],
            )
            wandb.log({"optuna/best_sharpe": best_score})
            wandb.finish()
        except Exception:
            pass

    print("  Next step: update ModelConfig / RewardConfig in core/config.py")
    print(f"  with the values above, then retrain: python engine/trainer.py\n")

    return best_params


if __name__ == "__main__":
    if not _OPTUNA_AVAILABLE:
        print("Install optuna first: pip install optuna")
        sys.exit(1)
    run_optuna_search(symbol=TradingConfig.ACTIVE_SYMBOL)

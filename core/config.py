"""
core/config.py
==============
Centralized configuration for Quant Predator v2.1.

CHANGELOG v2.1 — post training-run-1 diagnostic fixes:
  ENTRY_COST:      -0.30 → -0.05   CRITICAL FIX: run-1 was 1.84 trades/day
                                   (already within 2-3/day target). -0.30 would
                                   have crushed frequency to ~0.4/day. -0.05
                                   creates a quality filter without killing volume.
  LEARNING_RATE:   3e-4  → 1e-4   clip_fraction was 0.374 (thrashing). Target <0.15.
  N_STEPS:         4096  → 2048   More frequent, stable PPO updates.
  CLIP_RANGE:      0.20  → 0.15   Constrain per-update policy shift.
  ENT_COEF:        0.05  → 0.08   Encourage broader exploration.
  BATCH_SIZE:      512   → 256    More stable LSTM gradient updates.
  TOTAL_TIMESTEPS: 1.5M  → 2.5M   More training at the stabilized LR.
  MIN_RR_RATIO:    1.0   → 2.0    Raise structural quality gate.
  LR_SCHEDULE:     constant → linear decay 1e-4 → 1e-5.
  MAX_GRAD_NORM:   default → 0.5  Explicit gradient clipping.

NEW in v2.1:
  RewardConfig   — entry cost, patience bonus, MTF gate, session mult.
  WandbConfig    — experiment tracking configuration.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# PATHS
# ==============================================================================

class Paths:
    ROOT       = PROJECT_ROOT
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    LOGS_DIR   = os.path.join(PROJECT_ROOT, "logs")
    APP_STATE  = os.path.join(PROJECT_ROOT, "app", "state")
    TOOLS_DIR  = os.path.join(PROJECT_ROOT, "tools")

    STATE_FILE   = os.path.join(APP_STATE, "state.json")
    CONFIG_FILE  = os.path.join(APP_STATE, "config.json")
    JOURNAL_FILE = os.path.join(APP_STATE, "trade_journal.csv")
    CHART_FILE   = os.path.join(APP_STATE, "chart_data.csv")

    @staticmethod
    def model_zip(symbol: str) -> str:
        return os.path.join(Paths.MODELS_DIR, symbol.lower(), "ppo_model.zip")

    @staticmethod
    def vec_normalize_pkl(symbol: str) -> str:
        return os.path.join(Paths.MODELS_DIR, symbol.lower(), "vec_normalize.pkl")

    @staticmethod
    def best_model_dir(symbol: str) -> str:
        return os.path.join(Paths.MODELS_DIR, symbol.lower(), "checkpoints/best_model.zip")

    @staticmethod
    def equity_csv(symbol: str) -> str:
        return os.path.join(Paths.TOOLS_DIR, f"equity_{symbol.lower()}.csv")

    @staticmethod
    def optuna_results(symbol: str) -> str:
        return os.path.join(Paths.TOOLS_DIR, f"optuna_best_{symbol.lower()}.json")

    @staticmethod
    def ensure_dirs() -> None:
        for path in [
            Paths.MODELS_DIR, Paths.LOGS_DIR, Paths.APP_STATE, Paths.TOOLS_DIR,
            os.path.join(Paths.MODELS_DIR, "eurusd"),
            os.path.join(Paths.MODELS_DIR, "xauusd"),
        ]:
            os.makedirs(path, exist_ok=True)


# ==============================================================================
# BROKER & EXECUTION
# ==============================================================================

class BrokerConfig:
    """XM broker execution parameters."""
    MAGIC_NUMBER  = 777777
    DEVIATION     = 20
    ORDER_COMMENT = "QuantPredator-v2"


# ==============================================================================
# SYMBOL SPECIFICATIONS
# ==============================================================================

@dataclass(frozen=True)
class SymbolSpec:
    symbol:          str
    lot_unit:        float
    spread_pips:     float
    min_sl_atr_mult: float
    tp_atr_mult:     float
    max_spread_live: float


SYMBOL_SPECS: Dict[str, SymbolSpec] = {
    "EURUSD": SymbolSpec(
        symbol="EURUSD", lot_unit=100_000.0, spread_pips=1.8,
        min_sl_atr_mult=0.5, tp_atr_mult=1.5, max_spread_live=3.0,
    ),
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD", lot_unit=100.0, spread_pips=35.0,
        min_sl_atr_mult=0.8, tp_atr_mult=2.5, max_spread_live=80.0,
    ),
}


def get_symbol_spec(symbol: str) -> SymbolSpec:
    if symbol not in SYMBOL_SPECS:
        raise ValueError(
            f"Symbol '{symbol}' not configured. "
            f"Available: {list(SYMBOL_SPECS.keys())}."
        )
    return SYMBOL_SPECS[symbol]


# ==============================================================================
# LIVE TRADING
# ==============================================================================

class TradingConfig:
    ACTIVE_SYMBOL        = "EURUSD"
    TIMEFRAME            = "H1"
    RISK_PER_TRADE       = 0.01
    MAX_LOT_SIZE         = 5.0
    MAX_DYNAMIC_RISK     = 0.015
    MIN_DYNAMIC_RISK     = 0.005
    BREAKEVEN_R_MULT     = 1.0
    DAILY_LOSS_LIMIT     = 0.03
    MONTHLY_DD_LIMIT     = 0.10
    MAX_OPEN_POSITIONS   = 1
    MAIN_LOOP_SLEEP_SECS = 2
    CHART_BARS           = 500


# ==============================================================================
# MODEL & TRAINING — v2.1 values
# ==============================================================================

class ModelConfig:
    """
    PPO + LSTM hyperparameters.

    OBSERVATION SPACE:
      Market window : 35 features × 24 bars = 840 dimensions
      State scalars : 2 (current_position, floating_profit_normalized)
      Total OBS_DIM : 842

    ARCHITECTURE:
      OBS (842) → LSTM(24×35 → hidden 128) → concat state (2) → (130)
                → MLP [256, 128] → action (3) / value (1)

    v2.1 CHANGES vs v2.0:
      LEARNING_RATE  3e-4 → 1e-4   (clip_fraction=0.374 in run 1 — thrashing)
      LR_END         new  = 1e-5   (linear decay gives fine-tuning at end)
      N_STEPS        4096 → 2048   (more frequent stable updates)
      CLIP_RANGE     0.20 → 0.15   (tighter per-update constraint)
      ENT_COEF       0.05 → 0.08   (more exploration)
      BATCH_SIZE     512  → 256    (stable LSTM gradient updates)
      TOTAL_TS       1.5M → 2.5M   (more training at stable LR)
      MAX_GRAD_NORM  new  = 0.5    (explicit gradient clipping)
      MIN_RR_RATIO   1.0  → 2.0    (raise structural quality gate)
    """
    WINDOW_SIZE  = 24
    N_FEATURES   = 40
    OBS_DIM      = (N_FEATURES * WINDOW_SIZE) + 2   # 962

    LSTM_HIDDEN  = 128
    LSTM_LAYERS  = 2
    LSTM_DROPOUT = 0.1
    POLICY_NET   = [256, 128]

    # PPO — v2.1
    TOTAL_TIMESTEPS = 2_500_000
    LEARNING_RATE   = 1e-4
    LR_END          = 1e-5
    ENT_COEF        = 0.08
    BATCH_SIZE      = 256
    N_STEPS         = 2048
    N_EPOCHS        = 10
    GAMMA           = 0.99
    GAE_LAMBDA      = 0.95
    CLIP_RANGE      = 0.15
    MAX_GRAD_NORM   = 0.5
    CLIP_OBS        = 10.0

    MIN_RR_RATIO = 2.0

    N_ENVS           = 4
    TRAIN_BARS       = 50_000
    TRAIN_SPLIT      = 0.70
    VAL_SPLIT        = 0.15

    INITIAL_BALANCE      = 10_000.0
    COMMISSION           = 0.0
    BANKRUPTCY_FRACTION  = 0.40

    MIN_OOS_SHARPE   = 0.5
    MAX_OOS_DRAWDOWN = 0.25

    # Convergence targets to watch during training (shown at startup)
    TARGET_CLIP_FRACTION  = 0.15   # was 0.374 in run 1
    TARGET_KL_DIVERGENCE  = 0.020  # was 0.051 in run 1
    TARGET_EXPL_VARIANCE  = 0.80   # was 0.852 — already good, keep it


# ==============================================================================
# REWARD FUNCTION — v2.1
# ==============================================================================

class RewardConfig:
    """
    Reward shaping targeting trade quality, not volume suppression.

    CRITICAL LESSON FROM RUN 1:
      Run 1 made 1.84 trades/day — already inside the 2-3/day target.
      The original -0.30 ENTRY_COST would have crushed frequency to ~0.4/day.
      ENTRY_COST is set to -0.05: just large enough to discourage
      low-confidence entries, not large enough to silence the model.

    ENTRY_COST MATH:
      At 2:1 R:R, minimum win rate to justify entry:
        0 = P×2.0 - (1-P)×1.0 - 0.05  →  P > 35%  (vs 33% baseline)
      Reasonable quality gate without destroying frequency.

    PATIENCE_BONUS accumulation:
      8 flat bars between trades: 8 × 0.002 = +0.016
      This offsets the entry cost 3× over a typical inter-trade gap,
      rewarding the model for waiting for genuine setups.
    """
    ENTRY_COST           = -0.05   # ← FIXED from -0.30. See docstring above.
    PATIENCE_BONUS_FLAT  =  0.0001  # Per bar held flat (rewards discipline)
    WEAK_MTF_PENALTY     = -0.20   # |MTF_Alignment| < threshold
    MTF_GATE_THRESHOLD   =  2      # Need ≥2/3 timeframes aligned
    ASIAN_SESSION_MULT   =  0.5    # Reduce Asian-session entry reward
    BANKRUPTCY_PENALTY   = -100.0


# ==============================================================================
# SESSION TIMES (UTC)
# ==============================================================================

class SessionConfig:
    """
    Forex session hours in UTC.

    XM broker offset: typically UTC+2 (winter) / UTC+3 (summer EET).
    If timestamps are in broker time, subtract the offset from all values.
    Verify: compare mt5.symbol_info_tick().time vs datetime.utcnow().
    """
    ASIAN_START   = 0
    ASIAN_END     = 8
    LONDON_START  = 7
    LONDON_END    = 16
    NY_START      = 12
    NY_END        = 21
    OVERLAP_START = 12
    OVERLAP_END   = 16
    TRADE_HOURS: List[int] = list(range(7, 21))


# ==============================================================================
# WEIGHTS & BIASES
# ==============================================================================

class WandbConfig:
    """
    Weights & Biases experiment tracking configuration.

    SETUP (one-time):
      1. Create free account at https://wandb.ai (use student email for Pro tier)
      2. pip install wandb
      3. wandb login   ← paste your API key

    WHAT GETS LOGGED AUTOMATICALLY:
      - All PPO training metrics per rollout iteration:
          clip_fraction, approx_kl, explained_variance,
          entropy_loss, policy_gradient_loss, value_loss, loss
      - Learning rate at each step (shows linear decay)
      - Gradient magnitudes every GRADIENT_LOG_FREQ steps
      - Full config dict at run start (hyperparameter table)
      - OOS Sharpe, drawdown, win rate, trade frequency after training
      - Model zip and VecNormalize pkl as W&B Artifacts

    ENABLED flag: set False to run without W&B (e.g., no internet).
    """
    ENABLED            = True
    PROJECT            = "quant-predator-v2"
    ENTITY             = None      # None = your default W&B username
    GRADIENT_LOG_FREQ  = 200       # Log gradient magnitudes every N steps
    RUN_NAME_PREFIX    = "eurusd"  # Runs appear as eurusd_YYYYMMDD_HHMMSS


# ==============================================================================
# OPTUNA
# ==============================================================================

class OptunaConfig:
    """
    Hyperparameter search configuration.

    HOW TO USE:
      python engine/optuna_search.py

    SEARCH SPACE covers both training stability params and reward shaping:
      - learning_rate, clip_range, ent_coef, n_steps, batch_size
      - entry_cost, patience_bonus, mtf_gate_threshold, min_rr_ratio

    TRIAL BUDGET:
      Each trial runs TRIAL_TIMESTEPS (short) on the train set,
      then evaluates on the validation set.
      ~15-18 min per trial on RTX 3050.
      N_TRIALS=15 → ~4 hour overnight search.

    After search: best params are saved to tools/optuna_best_eurusd.json
    and printed for you to manually update ModelConfig/RewardConfig.
    """
    N_TRIALS          = 15
    TRIAL_TIMESTEPS   = 400_000   # Short runs for search (vs 2.5M for final)
    STUDY_NAME        = "qp_v2_eurusd"
    STORAGE           = None      # None = in-memory. Set to sqlite:///optuna.db to persist.
    PRUNER            = "median"  # "median" or "halving" or None
    EVAL_FREQ_STEPS   = 50_000    # Report intermediate value every N steps for pruning
    N_EVAL_EPISODES   = 3         # Val episodes per intermediate eval


# ==============================================================================
# NEWS FILTER
# ==============================================================================

class NewsConfig:
    HALT_HOURS_AHEAD    = 0.5
    FOREX_FACTORY_URL   = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    CACHE_DURATION_SECS = 3600
    EURUSD_CURRENCIES   = ["USD", "EUR"]
    XAUUSD_CURRENCIES   = ["USD"]


# ==============================================================================
# LOCAL LLM
# ==============================================================================

class LLMConfig:
    BASE_URL    = "http://localhost:11434"
    MODEL_NAME  = "llama3"
    TIMEOUT_SEC = 30
    MAX_TOKENS  = 600


# ==============================================================================
# JOURNAL SCHEMA
# ==============================================================================

class JournalConfig:
    COLUMNS = [
        "Timestamp", "Action", "Price", "Lot_Size",
        "SL", "TP", "PnL",
        "RSI", "EMA_Dist", "ATR", "MTF_Align",
        "BB_Width", "Bull_FVG", "Bear_FVG",
    ]
    BUY        = "BUY"
    SELL       = "SELL"
    BUY_CLOSE  = "BUY_CLOSE"
    SELL_CLOSE = "SELL_CLOSE"

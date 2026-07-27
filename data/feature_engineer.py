"""
data/feature_engineer.py
========================
Multi-Timeframe Feature Engineering Pipeline for Quant Predator v2.

This module is the single source of truth for ALL feature computation.
It is called identically during training, backtesting, and live inference
to guarantee zero train/serve skew.

DESIGN PRINCIPLES:
    1. STATIONARITY — Every feature is price-scale agnostic.
       No absolute EMA/BB/price levels are included. All continuous features
       are expressed as ratios, percentages, or bounded indices.

    2. ZERO LOOKAHEAD BIAS — All computations use only past data.
       rolling() uses center=False. HTF data is shifted 1 period before
       merging. The unmitigated FVG tracker reads only historical bars.

    3. IDENTICAL BEHAVIOUR — The DataFrame output has the same column
       schema and dtypes regardless of whether the input is 50,000 training
       bars or a 6,025-bar live inference window.

OUTPUT FEATURE SET (40 stationary features):
    ┌──────────────────┬───────────────────────────────────────────────────────┐
    │ Category         │ Features                                              │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ H1 Base (10)     │ ATRr_14, RSI_14, macd_norm, macd_signal_norm,        │
    │                  │ macd_hist_norm, bb_pct, BB_Width,                    │
    │                  │ H1_Trend, ema_50_dist, ema_200_dist                  │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ Lagged Memory (9)│ RSI_14_lag{1,2,3}, ATRr_14_lag{1,2,3},               │
    │                  │ macd_norm_lag{1,2,3}                                 │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ Market           │ is_swing_high, is_swing_low,                         │
    │ Structure (4)    │ bull_fvg_dist_atr, bear_fvg_dist_atr                 │
    │                  │ (continuous ATR-normalised zone distance —           │
    │                  │  replaces the earlier binary active-flag features)   │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ Session          │ session_asian, session_london, session_ny,           │
    │ Context (5)      │ session_overlap, day_of_week                         │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ Volume/          │ rel_volume, close_roc_3, close_roc_12                │
    │ Momentum (3)     │                                                       │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ HTF              │ H4_Trend, H4_ema_dist, D1_Trend, D1_ema_dist        │
    │ Projections (4)  │                                                       │
    ├──────────────────┼───────────────────────────────────────────────────────┤
    │ Option A: Price  │ candle_body_ratio, lower_wick_ratio,                 │
    │ Action           │ upper_wick_ratio, session_range_pct, vol_spike       │
    │ Conviction (5)   │                                                       │
    └──────────────────┴───────────────────────────────────────────────────────┘

    NOTE: bull_fvg_active / bear_fvg_active (binary flags) are retained in the
    output DataFrame for debugging/dashboard visibility, but are NOT part of
    FEATURE_COLUMNS — they are superseded by the continuous *_dist_atr features
    above. See _compute_unmitigated_fvg() for both computations.

UTILITY COLUMN (present in DataFrame, excluded from model features):
    BB_Width_rolling_mean — 100-bar rolling mean used for dynamic position
                            sizing. Excluded in rl_env.py's cols_to_drop.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from core.config import SessionConfig
from core.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)   # pandas-ta version warnings
logger = get_logger(__name__)


# ==============================================================================
# PRIVATE HELPERS
# ==============================================================================

def _compute_unmitigated_fvg(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns binary active flags AND continuous ATR-normalised distance to
    the nearest active FVG zone boundary.

    bull_fvg_dist_atr:
        When a bullish FVG is active: (close - zone_bottom) / ATR
        Negative = price below the gap (approaching from below — ideal entry)
        Positive = price inside or above the gap
        Zero     = no active bullish FVG

    bear_fvg_dist_atr:
        When a bearish FVG is active: (zone_top - close) / ATR
        Positive = price above the gap (approaching from above — ideal short entry)
        Zero     = no active bearish FVG
    """
    n = len(df)
    bull_active  = np.zeros(n, dtype=np.float32)
    bear_active  = np.zeros(n, dtype=np.float32)
    bull_dist    = np.zeros(n, dtype=np.float32)
    bear_dist    = np.zeros(n, dtype=np.float32)

    closes     = df["close"].values
    lows       = df["low"].values
    highs      = df["high"].values
    bull_flags = df["bullish_fvg"].values
    bear_flags = df["bearish_fvg"].values
    atr_vals   = df["ATRr_14"].values

    active_bull_bottom: Optional[float] = None
    active_bear_top:    Optional[float] = None

    for i in range(n):
        atr_i = float(atr_vals[i]) if atr_vals[i] > 0 else 0.001

        # ── Bullish FVG ──────────────────────────────────────────────────────
        if bull_flags[i] == 1.0:
            active_bull_bottom = lows[i]

        if active_bull_bottom is not None:
            if closes[i] < active_bull_bottom:
                active_bull_bottom = None
            else:
                bull_active[i] = 1.0
                bull_dist[i]   = (closes[i] - active_bull_bottom) / atr_i

        # ── Bearish FVG ──────────────────────────────────────────────────────
        if bear_flags[i] == 1.0:
            active_bear_top = highs[i]

        if active_bear_top is not None:
            if closes[i] > active_bear_top:
                active_bear_top = None
            else:
                bear_active[i] = 1.0
                bear_dist[i]   = (active_bear_top - closes[i]) / atr_i

    return bull_active, bear_active, bull_dist, bear_dist


def _append_htf_features(
    base_df: pd.DataFrame,
    timeframe_str: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Computes Higher-TimeFrame trend and EMA distance, then merges back to H1.

    ONLY STATIONARY FEATURES are returned (no absolute EMA price levels):
        {prefix}_Trend    : ±1.0 based on EMA50 vs EMA200 crossover
        {prefix}_ema_dist : (close - EMA200) / EMA200 × 100  [percentage]

    BIAS PREVENTION via shift(1):
        After computing indicators on the HTF DataFrame, all values are shifted
        forward by 1 HTF period. This means an H1 bar can only access the HTF
        bar that closed BEFORE the H1 bar's period started — never the
        concurrently-open HTF bar. This eliminates HTF lookahead bias.

    Args:
        base_df:        H1 OHLCV DataFrame with DatetimeIndex.
        timeframe_str:  Pandas resample offset string ('4h' or '1d').
        prefix:         Column name prefix ('H4' or 'D1').

    Returns:
        base_df with two new columns: {prefix}_Trend and {prefix}_ema_dist.
    """
    # Resample H1 candles to the higher timeframe
    htf = base_df.resample(timeframe_str).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    htf.ta.ema(length=50,  append=True)
    htf.ta.ema(length=200, append=True)

    if "EMA_50" in htf.columns and "EMA_200" in htf.columns:
        htf[f"{prefix}_Trend"]    = np.where(htf["EMA_50"] > htf["EMA_200"], 1.0, -1.0)
        htf[f"{prefix}_ema_dist"] = (htf["close"] - htf["EMA_200"]) / htf["EMA_200"] * 100.0
    else:
        htf[f"{prefix}_Trend"]    = 0.0
        htf[f"{prefix}_ema_dist"] = 0.0

    # SHIFT: prevents the current partially-formed HTF bar from leaking into H1
    htf = htf.shift(1)

    features     = [f"{prefix}_Trend", f"{prefix}_ema_dist"]
    htf_features = htf[features].copy()

    # merge_asof: each H1 bar gets the most recent closed HTF bar's data
    merged = pd.merge_asof(
        base_df,
        htf_features,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged


# ==============================================================================
# PUBLIC API
# ==============================================================================

#: Ordered list of all 35 model feature column names.
#: This is the contract between feature_engineer.py and rl_env.py / live_trader.py.
#: Any change here REQUIRES a full model retrain.
FEATURE_COLUMNS: list[str] = [
    # ── H1 Base Indicators ───────────────────────────────────────────────────
    "ATRr_14", "RSI_14",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_pct", "BB_Width",
    "H1_Trend", "ema_50_dist", "ema_200_dist",
    # ── Lagged Memory (short-term trajectory) ────────────────────────────────
    "RSI_14_lag1", "RSI_14_lag2", "RSI_14_lag3",
    "ATRr_14_lag1", "ATRr_14_lag2", "ATRr_14_lag3",
    "macd_norm_lag1", "macd_norm_lag2", "macd_norm_lag3",
    # ── Market Structure ─────────────────────────────────────────────────────
    "is_swing_high", "is_swing_low",
    "bull_fvg_dist_atr", "bear_fvg_dist_atr",   # continuous (replaces binary flags)
    # ── Session & Temporal Context ───────────────────────────────────────────
    "session_asian", "session_london", "session_ny", "session_overlap",
    "day_of_week",
    # ── Volume & Momentum ────────────────────────────────────────────────────
    "rel_volume", "close_roc_3", "close_roc_12",
    # ── HTF Macro Projections ────────────────────────────────────────────────
    "H4_Trend", "H4_ema_dist",
    "D1_Trend", "D1_ema_dist",
    # ── Option A: Price Action Conviction (NEW) ──────────────────────────────
    "candle_body_ratio", "lower_wick_ratio", "upper_wick_ratio",
    "session_range_pct",
    "vol_spike",
]

assert len(FEATURE_COLUMNS) == 40, (
    f"FEATURE_COLUMNS length mismatch: expected 40, got {len(FEATURE_COLUMNS)}. "
    "Update ModelConfig.N_FEATURES in core/config.py if you change this list."
)


def add_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Computes the full Multi-Timeframe Feature Matrix from raw H1 OHLCV data.

    This function is the single entry point for all feature engineering.
    It is called identically in:
        - engine/trainer.py       (training data preparation)
        - engine/backtester.py    (backtest simulation)
        - live_trader.py          (live inference — every H1 bar)

    INPUT REQUIREMENTS:
        df must have a DatetimeIndex and columns: open, high, low, close, volume.
        Minimum length: ~5,000 bars (D1 EMA200 warmup needs ~4,800 H1 bars).
        Recommended: 50,000 bars for training, 6,025+ bars for live inference.

    OUTPUT:
        DataFrame with all 35 FEATURE_COLUMNS plus original OHLCV columns,
        plus utility column BB_Width_rolling_mean (used for position sizing,
        excluded from model observation by rl_env.py).
        NaN rows are dropped (indicator warmup period removed).
        Returns None if computation fails — caller must handle this.

    Args:
        df: Raw H1 OHLCV DataFrame from MT5 with DatetimeIndex.

    Returns:
        Feature-enriched DataFrame with NaN rows removed, or None on error.
    """
    logger.info("Building Multi-Timeframe Feature Matrix (H1 → H4/D1)...")

    try:
        df = df.copy()   # Never mutate the caller's DataFrame

        # ======================================================================
        # SECTION 1: H1 BASE INDICATORS
        # ======================================================================

        # ── Trend: EMA Crossover ──────────────────────────────────────────────
        df.ta.ema(length=50,  append=True)   # → EMA_50  (absolute price level)
        df.ta.ema(length=200, append=True)   # → EMA_200 (absolute price level)

        df["H1_Trend"] = np.where(df["EMA_50"] > df["EMA_200"], 1.0, -1.0)

        # Percentage distance: stationary and scale-agnostic
        df["ema_50_dist"]  = (df["close"] - df["EMA_50"])  / df["EMA_50"]  * 100.0
        df["ema_200_dist"] = (df["close"] - df["EMA_200"]) / df["EMA_200"] * 100.0

        # Drop raw price-level EMA columns — they are non-stationary
        df.drop(columns=["EMA_50", "EMA_200"], inplace=True, errors="ignore")

        # ── Volatility: ATR ───────────────────────────────────────────────────
        # ATRr_14 = ATR(14) / close — already a price-relative ratio (stationary)
        df.ta.atr(length=14, append=True)   # → ATRr_14

        # ── Momentum: RSI ─────────────────────────────────────────────────────
        df.ta.rsi(length=14, append=True)   # → RSI_14 (0–100, stationary)

        # ── Momentum: MACD normalised to ATR ─────────────────────────────────
        # Raw MACD is in price units — non-stationary across price regimes.
        # Dividing by ATR (price units) makes it dimensionless.
        # ATR price units = ATRr_14 * close
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        # → MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9

        _atr_price = (df["ATRr_14"] * df["close"]).replace(0.0, np.nan)
        df["macd_norm"]        = df["MACD_12_26_9"]  / _atr_price
        df["macd_signal_norm"] = df["MACDs_12_26_9"] / _atr_price
        df["macd_hist_norm"]   = df["MACDh_12_26_9"] / _atr_price

        df.drop(
            columns=["MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9"],
            inplace=True, errors="ignore",
        )

        # ── Range: Bollinger Bands ────────────────────────────────────────────
        df.ta.bbands(length=20, std=2, append=True)
        # → BBL_20_2.0 (lower), BBM_20_2.0 (middle), BBU_20_2.0 (upper)
        # → BBB_20_2.0 (bandwidth), BBP_20_2.0 (%B position, 0–1+)

        bbl = next((c for c in df.columns if c.startswith("BBL")), None)
        bbm = next((c for c in df.columns if c.startswith("BBM")), None)
        bbu = next((c for c in df.columns if c.startswith("BBU")), None)
        bbp = next((c for c in df.columns if c.startswith("BBP")), None)

        if bbm and bbu and bbl:
            # BB_Width: ratio of band width to midline (stationary)
            df["BB_Width"] = (df[bbu] - df[bbl]) / df[bbm]
            # Pre-compute 100-bar rolling mean for position sizing (utility, not a model feature)
            df["BB_Width_rolling_mean"] = df["BB_Width"].rolling(window=100).mean()

        if bbp:
            df.rename(columns={bbp: "bb_pct"}, inplace=True)

        # Drop all absolute price-level Bollinger columns
        _bb_drop = (
            [c for c in df.columns if c.startswith("BBU")]
            + [c for c in df.columns if c.startswith("BBL")]
            + [c for c in df.columns if c.startswith("BBM")]
            + [c for c in df.columns if c.startswith("BBB")]
        )
        df.drop(columns=_bb_drop, inplace=True, errors="ignore")

        # Remove any Supertrend columns pandas-ta may have silently added
        df.drop(
            columns=[c for c in df.columns if "SUPERT" in c],
            inplace=True, errors="ignore",
        )

        # ======================================================================
        # SECTION 2: LAGGED FEATURES (Short-Term Temporal Memory)
        # ======================================================================
        # Three lags for RSI, ATR, and normalised MACD.
        # This gives the model a directional trajectory (are things accelerating
        # or decelerating?) without requiring a separate recurrent layer.

        for col in ["RSI_14", "ATRr_14", "macd_norm"]:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # ======================================================================
        # SECTION 3: MARKET STRUCTURE — Swing Points & FVGs
        # ======================================================================

        # ── Swing High / Low ──────────────────────────────────────────────────
        # CRITICAL FIX: center=False ensures no future bars are used.
        # A bar is a swing high if its high equals the rolling max of the
        # PAST 5 bars (including itself). This is confirmed by only past data.
        #
        # Original bug: center=True used bars t+1 and t+2 (lookahead bias).
        df["_rolling_max"] = df["high"].rolling(window=5, min_periods=5).max()
        df["_rolling_min"] = df["low"].rolling(window=5,  min_periods=5).min()
        df["is_swing_high"] = (df["high"] == df["_rolling_max"]).astype(np.float32)
        df["is_swing_low"]  = (df["low"]  == df["_rolling_min"]).astype(np.float32)
        df.drop(columns=["_rolling_max", "_rolling_min"], inplace=True)

        # ── Fair Value Gaps ───────────────────────────────────────────────────
        # Creation flags (point-in-time, no future data used):
        #   Bullish FVG: current low > high 2 bars ago → upward price imbalance
        #   Bearish FVG: current high < low 2 bars ago → downward price imbalance
        df["bullish_fvg"] = (df["low"] > df["high"].shift(2)).astype(np.float32)
        df["bearish_fvg"] = (df["high"] < df["low"].shift(2)).astype(np.float32)

        # Convert single-bar flags → persistent unmitigated zone state
        # (active from creation until price closes through the gap boundary)
        bull_active, bear_active, bull_dist, bear_dist = _compute_unmitigated_fvg(df)
        df["bull_fvg_active"] = bull_active  # kept for reference, excluded from model
        df["bear_fvg_active"] = bear_active  # kept for reference, excluded from model
        df["bull_fvg_dist_atr"] = bull_dist  # continuous replacement — IN model
        df["bear_fvg_dist_atr"] = bear_dist  # continuous replacement — IN model

        # Remove the raw creation flags (replaced by the richer persistent features)
        df.drop(columns=["bullish_fvg", "bearish_fvg"], inplace=True)

        # ======================================================================
        # SECTION 4: SESSION & TEMPORAL CONTEXT
        # ======================================================================
        # Institutional behaviour differs markedly by session. Providing these
        # as features lets the model learn session-specific patterns (e.g., avoid
        # fading moves during the London–NY overlap, expect range during Asia).
        #
        # NOTE: Assumes index is in UTC. Adjust SessionConfig offsets if your
        # broker's server time differs (common: UTC+2 or UTC+3 for XM).

        df["session_asian"]   = (
            (df.index.hour >= SessionConfig.ASIAN_START) &
            (df.index.hour <  SessionConfig.ASIAN_END)
        ).astype(np.float32)

        df["session_london"]  = (
            (df.index.hour >= SessionConfig.LONDON_START) &
            (df.index.hour <  SessionConfig.LONDON_END)
        ).astype(np.float32)

        df["session_ny"]      = (
            (df.index.hour >= SessionConfig.NY_START) &
            (df.index.hour <  SessionConfig.NY_END)
        ).astype(np.float32)

        df["session_overlap"] = (
            (df.index.hour >= SessionConfig.OVERLAP_START) &
            (df.index.hour <  SessionConfig.OVERLAP_END)
        ).astype(np.float32)

        # Day of week normalised to [0, 1] (Monday=0.0, Friday=1.0)
        # Markets behave differently on Mondays (gap fills) vs Fridays (squaring)
        df["day_of_week"] = (df.index.dayofweek / 4.0).astype(np.float32)

        # ======================================================================
        # SECTION 4b: OPTION A — PRICE ACTION CONVICTION FEATURES
        # ======================================================================

        # Candlestick body ratio: signed conviction of the current bar.
        # Range -1.0 to +1.0. +1 = full bullish body, -1 = full bearish body.
        # Near 0 = doji / indecision (high significance at structure levels).
        _bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        df["candle_body_ratio"] = (
                (df["close"] - df["open"]) / _bar_range
        ).fillna(0.0).astype(np.float32)

        # Lower wick ratio: fraction of the bar that is a lower wick.
        # High values = strong rejection of lows = buying pressure present.
        df["lower_wick_ratio"] = (
                (np.minimum(df["open"], df["close"]) - df["low"]) / _bar_range
        ).fillna(0.0).clip(0, 1).astype(np.float32)

        # Upper wick ratio: fraction of the bar that is an upper wick.
        # High values = strong rejection of highs = selling pressure present.
        df["upper_wick_ratio"] = (
                (df["high"] - np.maximum(df["open"], df["close"])) / _bar_range
        ).fillna(0.0).clip(0, 1).astype(np.float32)

        # Session range percentage: where is the current close within the
        # daily session's cumulative high-low range?
        # 0.0 = at the daily low, 1.0 = at the daily high.
        # Requires DatetimeIndex (set in fetcher.py).
        _date_groups = df.index.date
        _daily_high = df.groupby(_date_groups)["high"].cummax()
        _daily_low = df.groupby(_date_groups)["low"].cummin()
        _daily_range = (_daily_high - _daily_low).replace(0, np.nan)
        df["session_range_pct"] = (
                (df["close"] - _daily_low) / _daily_range
        ).fillna(0.5).clip(0, 1).astype(np.float32)

        # Volume spike: current bar volume vs the rolling 3-bar average from
        # the prior bars. Captures sudden institutional participation spikes.
        # Values near 1.0 = normal. Values > 2.0 = significant spike.
        # Different from rel_volume (which uses a 24-bar window).
        _vol_3bar_avg = df["volume"].shift(1).rolling(3).mean().replace(0, np.nan)
        df["vol_spike"] = (
                df["volume"] / _vol_3bar_avg
        ).fillna(1.0).clip(0, 5.0).astype(np.float32)

        # ======================================================================
        # SECTION 5: VOLUME & MOMENTUM
        # ======================================================================

        # Relative volume: ratio vs 24-bar (1-day) rolling mean
        # Values >1 indicate above-average activity; <1 indicates low liquidity
        _vol_mean = df["volume"].rolling(24).mean().replace(0.0, np.nan)
        df["rel_volume"] = (df["volume"] / _vol_mean).astype(np.float32)

        # Rate of Change: percentage price move over N bars (stationary momentum)
        df["close_roc_3"]  = df["close"].pct_change(3)  * 100.0
        df["close_roc_12"] = df["close"].pct_change(12) * 100.0

        # ======================================================================
        # SECTION 6: HIGHER TIMEFRAME PROJECTIONS (H4 & D1)
        # ======================================================================

        # Pandas changed resample frequency strings in v2.2.0 ('H' → 'h', 'D' → 'd')
        h4_str = "4h" if pd.__version__ >= "2.2.0" else "4H"
        d1_str = "1d" if pd.__version__ >= "2.2.0" else "1D"

        df = _append_htf_features(df, h4_str, "H4")
        df = _append_htf_features(df, d1_str, "D1")

        # ======================================================================
        # SECTION 7: FINAL VALIDATION & CLEANUP
        # ======================================================================

        # Drop all NaN rows introduced by:
        #   - EMA200 warmup (200 bars), D1 EMA200 warmup (~4,800 H1 bars)
        #   - Lagged features (3 bars)
        #   - Rolling windows (min 100 bars for BB_Width_rolling_mean)
        initial_len = len(df)
        df.dropna(inplace=True)
        warmup_bars = initial_len - len(df)

        # Validate that all expected feature columns are present
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            logger.error(
                f"Feature engineering produced incomplete output. "
                f"Missing columns: {missing}. "
                f"Check for pandas-ta API changes or indicator computation errors."
            )
            return None

        # Cast all feature columns to float32 for memory efficiency and SB3 compatibility
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(np.float32)

        logger.info(
            f"Feature matrix complete: {len(df):,} bars | "
            f"{len(FEATURE_COLUMNS)} features | "
            f"{warmup_bars:,} warmup bars removed"
        )

        return df

    except Exception as exc:
        logger.error(f"Feature engineering failed: {exc}", exc_info=True)
        return None


def validate_live_observation(df: pd.DataFrame) -> bool:
    """
    Validates that a live DataFrame slice is safe to feed to the model.

    Called in live_trader.py before constructing the observation vector.
    Returns False (and logs the reason) if any check fails, preventing
    the model from receiving a corrupted or NaN observation.

    Args:
        df: The last WINDOW_SIZE bars from the live feature DataFrame.

    Returns:
        True if safe to use; False if the observation should be skipped.
    """
    from core.config import ModelConfig

    if len(df) < ModelConfig.WINDOW_SIZE:
        logger.warning(
            f"Insufficient bars for observation: got {len(df)}, "
            f"need {ModelConfig.WINDOW_SIZE}."
        )
        return False

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.error(f"Live observation missing columns: {missing_cols}")
        return False

    obs_slice = df[FEATURE_COLUMNS].tail(ModelConfig.WINDOW_SIZE)
    if obs_slice.isnull().any().any():
        nan_cols = obs_slice.columns[obs_slice.isnull().any()].tolist()
        logger.warning(f"NaN values in live observation columns: {nan_cols}. Skipping bar.")
        return False

    return True

"""
data/fetcher.py
===============
MT5 historical OHLCV data retrieval.

Assumes MT5 is already initialised and logged in by the caller.
This module never calls mt5.initialize() or mt5.login() — connection
lifecycle is managed by live_trader.py or trainer.py at startup.
"""

from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from core.config import ModelConfig
from core.logger import get_logger

logger = get_logger(__name__)

_TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
}


def get_historical_data(
    symbol:    str,
    timeframe: str    = "H1",
    num_bars:  int    = ModelConfig.TRAIN_BARS,
    start_pos: int    = 0,
) -> Optional[pd.DataFrame]:
    """
    Fetches historical candlestick data from the MT5 terminal.

    MT5's copy_rates_from_pos() returns bars starting from `start_pos`
    bars back from the most recent bar, ordered oldest-first.
    Position 0 = the most recently opened (potentially unclosed) bar.
    Position 1 = the last fully closed bar.

    For live signal generation use start_pos=0 (includes open bar).
    The open bar is excluded from the signal window in live_trader.py
    by using df.iloc[-2] as the most recent confirmed closed bar.

    Args:
        symbol:    MT5 instrument symbol (e.g., "EURUSD").
        timeframe: Bar timeframe string (e.g., "H1"). See _TF_MAP for options.
        num_bars:  Number of bars to fetch.
        start_pos: Offset from the most recent bar (0 = include current open bar).

    Returns:
        DataFrame with DatetimeIndex and columns:
        open, high, low, close, volume, spread, real_volume.
        Returns None on failure.
    """
    tf = _TF_MAP.get(timeframe.upper())
    if tf is None:
        logger.error(f"Unknown timeframe '{timeframe}'. Valid: {list(_TF_MAP.keys())}")
        return None

    logger.info(f"Fetching {num_bars:,} {timeframe} bars for {symbol}...")

    rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, num_bars)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        logger.error(f"MT5 data fetch failed for {symbol} {timeframe}. Error: {err}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    logger.info(f"Fetched {len(df):,} bars ({df.index[0]} → {df.index[-1]})")
    return df

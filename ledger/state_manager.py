"""
ledger/state_manager.py
=======================
Atomic, race-condition-proof state persistence for Quant Predator v2.

DESIGN:
    All file writes use the atomic rename pattern:
        1. Write to a .tmp file in the same directory.
        2. Call os.replace(tmp, target) — atomic on NTFS (Windows).
    This prevents the Streamlit dashboard from ever reading a
    partially-written file and raising JSONDecodeError.

    A threading.Lock guards the trade journal CSV to prevent row
    corruption if future async components share the same process.
"""

from __future__ import annotations

import time
import csv
import json
import os
import tempfile
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.config import JournalConfig, Paths
from core.logger import get_logger

logger = get_logger(__name__)

_journal_lock = threading.Lock()


class StateManager:
    """
    Manages all shared state files accessed by both live_trader.py
    and the Streamlit dashboard (app/dashboard.py).

    Files managed:
        state.json        — Live account metrics (balance, equity, signal, heartbeat)
        config.json       — User preferences (auto_trade, symbol, manual_command)
        trade_journal.csv — Immutable trade ledger (append-only)
        chart_data.csv    — Last N bars for the candlestick chart
    """

    def __init__(self) -> None:
        Paths.ensure_dirs()
        self._ensure_journal()
        self._ensure_config()

    # ==========================================================================
    # STATE.JSON
    # ==========================================================================

    def update_live_state(
        self,
        balance:        float,
        floating_profit: float,
        open_position:  int,
        ai_signal:      str,
        news_status:    str,
    ) -> None:
        """
        Writes the current live metrics to state.json atomically.
        Called by live_trader.py on every main loop iteration.
        """
        pos_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(open_position, "FLAT")
        data = {
            "balance":        round(balance, 2),
            "equity":         round(balance + floating_profit, 2),
            "floating_profit":round(floating_profit, 2),
            "open_position":  pos_str,
            "ai_signal":      ai_signal,
            "news_status":    news_status,
            "last_update":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._atomic_write_json(Paths.STATE_FILE, data)

    def read_live_state(self) -> Dict[str, Any]:
        """Reads and returns state.json. Returns empty dict on read failure."""
        return self._safe_read_json(Paths.STATE_FILE, default={})

    # ==========================================================================
    # CONFIG.JSON
    # ==========================================================================

    def read_config(self) -> Dict[str, Any]:
        """Returns the current trading configuration."""
        return self._safe_read_json(Paths.CONFIG_FILE, default={
            "auto_trade":      True,
            "symbol":          "EURUSD",
            "timeframe":       "H1",
            "manual_command":  "NONE",
        })

    def write_config(self, config: Dict[str, Any]) -> None:
        """Persists updated configuration atomically."""
        self._atomic_write_json(Paths.CONFIG_FILE, config)

    def clear_manual_command(self) -> None:
        """Resets the manual_command field to NONE after it has been consumed."""
        config = self.read_config()
        config["manual_command"] = "NONE"
        self.write_config(config)

    # ==========================================================================
    # TRADE JOURNAL (append-only CSV)
    # ==========================================================================

    def log_entry(
        self,
        action:     str,
        price:      float,
        lot_size:   float,
        sl:         float,
        tp:         float,
        rsi:        float = 0.0,
        ema_dist:   float = 0.0,
        atr:        float = 0.0,
        mtf_align:  int   = 0,
        bb_width:   float = 0.0,
        bull_fvg:   int   = 0,
        bear_fvg:   int   = 0,
    ) -> None:
        """Appends an entry event row to the trade journal."""
        self._append_journal_row(
            action, price, lot_size, sl, tp, pnl=0.0,
            rsi=rsi, ema_dist=ema_dist, atr=atr,
            mtf_align=mtf_align, bb_width=bb_width,
            bull_fvg=bull_fvg, bear_fvg=bear_fvg,
        )

    def log_close(
        self,
        action:     str,
        price:      float,
        lot_size:   float,
        pnl:        float,
        rsi:        float = 0.0,
        ema_dist:   float = 0.0,
        atr:        float = 0.0,
        mtf_align:  int   = 0,
        bb_width:   float = 0.0,
        bull_fvg:   int   = 0,
        bear_fvg:   int   = 0,
    ) -> None:
        """Appends a close event row to the trade journal with true broker P&L."""
        self._append_journal_row(
            action, price, lot_size, sl=0.0, tp=0.0, pnl=pnl,
            rsi=rsi, ema_dist=ema_dist, atr=atr,
            mtf_align=mtf_align, bb_width=bb_width,
            bull_fvg=bull_fvg, bear_fvg=bear_fvg,
        )

    def read_journal(self) -> pd.DataFrame:
        """Returns the full trade journal as a DataFrame. Empty DF on error."""
        try:
            if not os.path.exists(Paths.JOURNAL_FILE):
                return pd.DataFrame(columns=JournalConfig.COLUMNS)
            return pd.read_csv(Paths.JOURNAL_FILE)
        except Exception as exc:
            logger.error(f"Failed to read trade journal: {exc}")
            return pd.DataFrame(columns=JournalConfig.COLUMNS)

    # ==========================================================================
    # CHART DATA
    # ==========================================================================

    def update_chart_data(self, df: pd.DataFrame, n_bars: int = 500) -> None:
        """
        Writes the last n_bars of OHLCV data for the Streamlit candlestick chart.
        Uses a direct (non-atomic) write since stale chart data is acceptable.
        """
        try:
            df.tail(n_bars).to_csv(Paths.CHART_FILE)
        except Exception as exc:
            logger.warning(f"Chart data write failed (non-critical): {exc}")

    # ==========================================================================
    # PRIVATE
    # ==========================================================================

    def _atomic_write_json(self, filepath: str, data: Dict, max_retries: int = 5) -> None:
        """
        Writes data to filepath atomically via tmp file + os.replace.

        Windows-specific: os.replace() can intermittently raise
        PermissionError (WinError 5 — Access is denied) when antivirus
        real-time scanning, the search indexer, or another process (e.g.
        the dashboard reading this same file) briefly holds a lock on the
        tmp file or the destination. This is almost always transient and
        clears within milliseconds, so we retry with a short backoff
        before giving up and logging an error.
        """
        dir_ = os.path.dirname(filepath)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            for attempt in range(max_retries):
                try:
                    os.replace(tmp, filepath)
                    return
                except PermissionError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.05 * (attempt + 1))  # 50ms, 100ms, 150ms...

        except Exception as exc:
            logger.error(f"Atomic JSON write failed for {filepath}: {exc}")
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _safe_read_json(self, filepath: str, default: Any) -> Any:
        """Reads a JSON file, returning `default` on any error."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _ensure_journal(self) -> None:
        """Creates trade_journal.csv with headers if it doesn't exist."""
        if not os.path.exists(Paths.JOURNAL_FILE):
            with open(Paths.JOURNAL_FILE, "w", newline="") as f:
                csv.writer(f).writerow(JournalConfig.COLUMNS)

    def _ensure_config(self) -> None:
        """Creates config.json with defaults if it doesn't exist."""
        if not os.path.exists(Paths.CONFIG_FILE):
            self.write_config({
                "auto_trade":     True,
                "symbol":         "EURUSD",
                "timeframe":      "H1",
                "manual_command": "NONE",
            })

    def _append_journal_row(
        self, action, price, lot_size, sl, tp, pnl,
        rsi, ema_dist, atr, mtf_align, bb_width, bull_fvg, bear_fvg,
    ) -> None:
        """Thread-safe append of one row to trade_journal.csv."""
        with _journal_lock:
            try:
                with open(Paths.JOURNAL_FILE, "a", newline="") as f:
                    csv.writer(f).writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        action, round(price, 5), round(lot_size, 2),
                        round(sl, 5) if sl else 0,
                        round(tp, 5) if tp else 0,
                        round(pnl, 2),
                        round(rsi, 2), round(ema_dist, 4), round(atr, 6),
                        int(mtf_align), round(bb_width, 4),
                        int(bull_fvg), int(bear_fvg),
                    ])
            except Exception as exc:
                logger.error(f"Journal append failed: {exc}")

"""
core/logger.py
==============
Structured logging for Quant Predator v2.

Replaces all bare print() calls throughout the codebase with leveled,
colored, timestamped log entries written to both the console and a
rotating daily log file in logs/.

USAGE:
    from core.logger import get_logger
    logger = get_logger(__name__)

    logger.info("New H1 candle detected — running MTF matrix.")
    logger.warning("Spread unusually wide: 4.2 pips. Entry skipped.")
    logger.error("MT5 connection lost.", exc_info=True)

    # Trade-specific structured event (includes symbol/action context)
    logger.trade("BUY", symbol="EURUSD", price=1.1542, lot=0.05, sl=1.1510, tp=1.1620)
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from core.config import Paths

# ──────────────────────────────────────────────────────────────────────────────
# ANSI colour codes for console output (Windows 10+ supports these natively)
# ──────────────────────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_MAGENTA= "\033[95m"

_LEVEL_COLOURS = {
    "DEBUG":    _GREY,
    "INFO":     _CYAN,
    "TRADE":    _GREEN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _MAGENTA,
}

# Custom numeric level for trade events (between INFO=20 and WARNING=30)
TRADE_LEVEL = 25
logging.addLevelName(TRADE_LEVEL, "TRADE")


class _ColouredFormatter(logging.Formatter):
    """Console formatter with ANSI colours per log level."""

    _FMT = "{colour}{bold}[{level:<8}]{reset} {grey}{time}{reset}  {bold}{name}{reset}  {msg}"

    def format(self, record: logging.LogRecord) -> str:
        level   = record.levelname
        colour  = _LEVEL_COLOURS.get(level, _RESET)
        time    = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        bold    = _BOLD if level in ("TRADE", "WARNING", "ERROR", "CRITICAL") else ""
        return self._FMT.format(
            colour=colour, bold=bold, level=level, reset=_RESET,
            grey=_GREY, time=time, name=record.name,
            msg=record.getMessage(),
        )


class _PlainFormatter(logging.Formatter):
    """File formatter without ANSI codes (clean for log files)."""

    def format(self, record: logging.LogRecord) -> str:
        time = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        return f"[{record.levelname:<8}] {time}  {record.name}  {record.getMessage()}"


class QuantLogger(logging.Logger):
    """
    Extended Logger with a .trade() convenience method for structured trade events.
    """

    def trade(
        self,
        action: str,
        *,
        symbol: str = "",
        price: float = 0.0,
        lot: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        pnl: Optional[float] = None,
    ) -> None:
        """
        Logs a structured trade event at the TRADE level.

        Args:
            action: "BUY", "SELL", "BUY_CLOSE", "SELL_CLOSE", or "REJECTED".
            symbol: Instrument symbol (e.g., "EURUSD").
            price:  Filled execution price.
            lot:    Position size in lots.
            sl:     Stop-loss price.
            tp:     Take-profit price.
            pnl:    Realized P&L (for close events; None for entries).
        """
        pnl_str = f" | PnL: ${pnl:+.2f}" if pnl is not None else ""
        msg = (
            f"{action:<12} {symbol}  "
            f"price={price:.5f}  lot={lot:.2f}  "
            f"SL={sl:.5f}  TP={tp:.5f}{pnl_str}"
        )
        self.log(TRADE_LEVEL, msg)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
logging.setLoggerClass(QuantLogger)

_initialised = False


def _initialise_root_handlers() -> None:
    """
    Attaches console and rotating-file handlers to the root logger once.
    Safe to call multiple times (guarded by _initialised flag).
    """
    global _initialised
    if _initialised:
        return
    _initialised = True

    Paths.ensure_dirs()

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Console handler (coloured, INFO and above) ─────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ColouredFormatter())
    root.addHandler(console)

    # ── Rotating file handler (plain text, DEBUG and above) ────────────────
    log_file = os.path.join(Paths.LOGS_DIR, "quant_predator.log")
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=30,   # Keep 30 days of log files
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_PlainFormatter())
    root.addHandler(file_handler)


def get_logger(name: str) -> QuantLogger:
    """
    Returns a configured QuantLogger for the given module name.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        QuantLogger instance with console + file handlers attached.

    Example:
        logger = get_logger(__name__)
        logger.info("Module loaded.")
    """
    _initialise_root_handlers()
    return logging.getLogger(name)  # type: ignore[return-value]

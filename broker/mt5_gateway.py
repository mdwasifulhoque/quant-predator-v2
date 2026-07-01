"""
broker/mt5_gateway.py
=====================
Thread-safe serialiser for all MetaTrader5 API calls.

The MT5 Python API communicates with the terminal via a Windows named pipe.
Concurrent calls from multiple threads corrupt responses silently.
This gateway funnels every mt5.* call through a single dedicated thread
using a request/response queue pattern.

USAGE:
    gateway = MT5Gateway()
    gateway.start()                          # Initialise + login

    tick = gateway.call(mt5.symbol_info_tick, "EURUSD")
    result = gateway.call(mt5.order_send, request_dict)

    gateway.stop()                           # Clean shutdown
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Optional, Tuple

import MetaTrader5 as mt5
from dotenv import load_dotenv

from core.config import BrokerConfig
from core.logger import get_logger

logger = get_logger(__name__)


class MT5Gateway:
    """
    Single-threaded gateway for serialising all MT5 API calls.

    All public methods are thread-safe and can be called from any thread.
    Internally, each call is submitted to a queue and processed by the
    dedicated gateway thread. The calling thread blocks until the result
    is ready (or a timeout occurs).
    """

    _CALL_TIMEOUT = 15.0   # Seconds to wait for an MT5 response

    def __init__(self) -> None:
        self._queue:   queue.Queue = queue.Queue()
        self._thread:  threading.Thread = threading.Thread(
            target=self._run, daemon=True, name="MT5-Gateway"
        )
        self._running: bool = False

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def start(self) -> bool:
        """
        Initialises the MT5 terminal connection and starts the gateway thread.
        Reads credentials from environment variables (MT5_LOGIN, MT5_PASSWORD,
        MT5_SERVER) loaded from .env.

        Returns:
            True if connection succeeded; False otherwise.
        """
        load_dotenv()
        login    = int(os.getenv("MT5_LOGIN", "0"))
        password = os.getenv("MT5_PASSWORD", "")
        server   = os.getenv("MT5_SERVER", "")

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        if not mt5.login(login, password=password, server=server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        acc = mt5.account_info()
        logger.info(
            f"MT5 connected — {acc.company} | "
            f"Account: {acc.login} | "
            f"Balance: ${acc.balance:,.2f}"
        )

        self._running = True
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stops the gateway thread and shuts down the MT5 connection cleanly."""
        self._running = False
        mt5.shutdown()
        logger.info("MT5 gateway stopped.")

    def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Submits an MT5 API call and blocks until the result is returned.

        Args:
            fn:      Any mt5.* function (e.g., mt5.order_send, mt5.account_info).
            *args:   Positional arguments forwarded to fn.
            **kwargs:Keyword arguments forwarded to fn.

        Returns:
            The return value of fn(*args, **kwargs).

        Raises:
            RuntimeError:  If the gateway thread is not running.
            Exception:     Any exception raised inside fn is re-raised here.
        """
        if not self._running:
            raise RuntimeError("MT5Gateway is not running. Call start() first.")

        event      = threading.Event()
        result_box = [None, None]    # [result, exception]
        self._queue.put((fn, args, kwargs, event, result_box))

        completed = event.wait(timeout=self._CALL_TIMEOUT)
        if not completed:
            raise TimeoutError(
                f"MT5 call {fn.__name__} timed out after {self._CALL_TIMEOUT}s. "
                f"Check MT5 terminal connection."
            )

        if result_box[1] is not None:
            raise result_box[1]

        return result_box[0]

    # ==========================================================================
    # CONVENIENCE WRAPPERS (commonly used calls with type hints)
    # ==========================================================================

    def account_info(self):
        """Returns mt5.AccountInfo namedtuple for the connected account."""
        return self.call(mt5.account_info)

    def positions_get(self, symbol: str):
        """Returns open positions filtered by symbol."""
        return self.call(mt5.positions_get, symbol=symbol)

    def symbol_info_tick(self, symbol: str):
        """Returns the latest tick for a symbol."""
        return self.call(mt5.symbol_info_tick, symbol)

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start: int, count: int):
        """Returns OHLCV rates as a numpy structured array."""
        return self.call(mt5.copy_rates_from_pos, symbol, timeframe, start, count)

    def order_send(self, request: dict):
        """Sends a trading order and returns the OrderSendResult."""
        return self.call(mt5.order_send, request)

    def history_deals_get(self, from_date, to_date):
        """Returns completed deal history in the given date range."""
        return self.call(mt5.history_deals_get, from_date, to_date)

    # ==========================================================================
    # PRIVATE
    # ==========================================================================

    def _run(self) -> None:
        """
        Gateway thread main loop. Processes MT5 requests serially, one at a time.
        This is the ONLY thread that executes mt5.* calls after startup.
        """
        while self._running:
            try:
                fn, args, kwargs, event, result_box = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result_box[0] = fn(*args, **kwargs)
            except Exception as exc:
                result_box[1] = exc
            finally:
                event.set()

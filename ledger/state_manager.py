"""
ledger/state_manager.py
=======================
Postgres-backed state persistence for Quant Predator v2 (Supabase-hosted).

Replaces the earlier JSON/CSV file-based approach. Concurrency and
atomicity (previously hand-rolled via os.replace + retry-with-backoff for
Windows WinError 5) are now handled natively by Postgres transactions —
the dashboard and live_trader.py can read/write concurrently without
either process risking a torn read.

chart_data.csv is intentionally NOT migrated — it's a rolling OHLCV cache
for the candlestick chart, not transactional state, and stays as a flat
file for simplicity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import select

from core.db import get_session, init_db
from core.db_models import AppConfig, EvalLogEntry, LiveState, TradeJournalEntry
from core.logger import get_logger

logger = get_logger(__name__)


class StateManager:
    """
    Manages all shared state accessed by both live_trader.py and the
    Streamlit dashboard (app/dashboard.py), now backed by Postgres.
    """

    def __init__(self) -> None:
        init_db()
        self._ensure_singletons()

    # ==========================================================================
    # LIVE STATE
    # ==========================================================================

    def update_live_state(
            self,
            balance: float,
            floating_profit: float,
            open_position: int,
            ai_signal: str,
            news_status: str,
            mt5_ok: bool = True,
            ollama_ok: bool = False,
            last_bar_time: Optional[datetime] = None,  # new param
    ) -> None:
        ...
        with get_session() as session:
            row = session.get(LiveState, 1)
            if row is None:
                row = LiveState(id=1)
                session.add(row)
            ...
            if last_bar_time is not None:
                row.last_bar_time = last_bar_time
            row.last_update = datetime.now(timezone.utc)

    def get_last_bar_time(self):
        """Returns the last processed H1 bar's timestamp, or None if never set."""
        try:
            with get_session() as session:
                row = session.get(LiveState, 1)
                return row.last_bar_time if row else None
        except Exception as exc:
            logger.error(f"Failed to read last_bar_time: {exc}")
            return None

    def read_live_state(self) -> Dict[str, Any]:
        """Reads the current live_state row. Returns empty dict on failure.
        Numeric/string fields are coalesced from None to sensible defaults —
        Postgres NULL (unset column) needs explicit handling here, unlike
        the old JSON file where an absent key naturally fell through to
        dict.get()'s default in dashboard.py."""
        try:
            with get_session() as session:
                row = session.get(LiveState, 1)
                if row is None:
                    return {}
                return {
                    "balance": row.balance if row.balance is not None else 0.0,
                    "equity": row.equity if row.equity is not None else 0.0,
                    "floating_profit": row.floating_profit if row.floating_profit is not None else 0.0,
                    "open_position": row.open_position or "FLAT",
                    "ai_signal": row.ai_signal or "HOLD",
                    "news_status": row.news_status or "",
                    "mt5_ok": row.mt5_ok if row.mt5_ok is not None else False,
                    "ollama_ok": row.ollama_ok if row.ollama_ok is not None else False,
                    "last_update": row.last_update.strftime("%Y-%m-%d %H:%M:%S") if row.last_update else "",
                }
        except Exception as exc:
            logger.error(f"Failed to read live state: {exc}")
            return {}

    # ==========================================================================
    # APP CONFIG
    # ==========================================================================

    def read_config(self) -> Dict[str, Any]:
        try:
            with get_session() as session:
                row = session.get(AppConfig, 1)
                if row is None:
                    return {
                        "auto_trade": True, "symbol": "EURUSD",
                        "timeframe": "H1", "manual_command": "NONE",
                    }
                return {
                    "auto_trade": row.auto_trade, "symbol": row.symbol,
                    "timeframe": row.timeframe, "manual_command": row.manual_command,
                }
        except Exception as exc:
            logger.error(f"Failed to read config: {exc}")
            return {"auto_trade": True, "symbol": "EURUSD", "timeframe": "H1", "manual_command": "NONE"}

    def write_config(self, config: Dict[str, Any]) -> None:
        try:
            with get_session() as session:
                row = session.get(AppConfig, 1)
                if row is None:
                    row = AppConfig(id=1)
                    session.add(row)
                row.auto_trade = config.get("auto_trade", True)
                row.symbol = config.get("symbol", "EURUSD")
                row.timeframe = config.get("timeframe", "H1")
                row.manual_command = config.get("manual_command", "NONE")
        except Exception as exc:
            logger.error(f"Failed to write config: {exc}")

    def clear_manual_command(self) -> None:
        config = self.read_config()
        config["manual_command"] = "NONE"
        self.write_config(config)

    # ==========================================================================
    # TRADE JOURNAL (append-only)
    # ==========================================================================

    def log_entry(
        self,
        action: str, price: float, lot_size: float, sl: float, tp: float,
        rsi: float = 0.0, ema_dist: float = 0.0, atr: float = 0.0,
        mtf_align: int = 0, bb_width: float = 0.0,
        bull_fvg_dist_atr: float = 0.0, bear_fvg_dist_atr: float = 0.0,
        run_id: Optional[int] = None,
    ) -> None:
        """Appends an entry event row. NOTE: now logs the continuous FVG
        distance features actually fed to the model, not the old binary
        flags — fixes the journal/model-input mismatch found during the
        portfolio audit."""
        self._append_journal_row(
            action, price, lot_size, sl, tp, pnl=0.0,
            rsi=rsi, ema_dist=ema_dist, atr=atr, mtf_align=mtf_align, bb_width=bb_width,
            bull_fvg_dist_atr=bull_fvg_dist_atr, bear_fvg_dist_atr=bear_fvg_dist_atr,
            run_id=run_id,
        )

    def log_close(
        self,
        action: str, price: float, lot_size: float, pnl: float,
        rsi: float = 0.0, ema_dist: float = 0.0, atr: float = 0.0,
        mtf_align: int = 0, bb_width: float = 0.0,
        bull_fvg_dist_atr: float = 0.0, bear_fvg_dist_atr: float = 0.0,
        run_id: Optional[int] = None,
    ) -> None:
        """Appends a close event row with true broker P&L."""
        self._append_journal_row(
            action, price, lot_size, sl=0.0, tp=0.0, pnl=pnl,
            rsi=rsi, ema_dist=ema_dist, atr=atr, mtf_align=mtf_align, bb_width=bb_width,
            bull_fvg_dist_atr=bull_fvg_dist_atr, bear_fvg_dist_atr=bear_fvg_dist_atr,
            run_id=run_id,
        )

    def _append_journal_row(
        self, action, price, lot_size, sl, tp, pnl,
        rsi, ema_dist, atr, mtf_align, bb_width,
        bull_fvg_dist_atr, bear_fvg_dist_atr, run_id,
    ) -> None:
        try:
            with get_session() as session:
                session.add(TradeJournalEntry(
                    timestamp=datetime.now(timezone.utc),
                    run_id=run_id,
                    action=action, price=round(price, 5), lot_size=round(lot_size, 2),
                    sl=round(sl, 5) if sl else 0, tp=round(tp, 5) if tp else 0,
                    pnl=round(pnl, 2),
                    rsi=round(rsi, 2), ema_dist=round(ema_dist, 4), atr=round(atr, 6),
                    mtf_align=int(mtf_align), bb_width=round(bb_width, 4),
                    bull_fvg_dist_atr=round(bull_fvg_dist_atr, 4),
                    bear_fvg_dist_atr=round(bear_fvg_dist_atr, 4),
                ))
        except Exception as exc:
            logger.error(f"Journal append failed: {exc}")

    def read_journal(self) -> pd.DataFrame:
        """Returns the full trade journal as a DataFrame. Empty DF on error."""
        try:
            with get_session() as session:
                rows = session.execute(select(TradeJournalEntry)).scalars().all()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame([{
                    "Timestamp": r.timestamp, "Action": r.action, "Price": r.price,
                    "Lot_Size": r.lot_size, "SL": r.sl, "TP": r.tp, "PnL": r.pnl,
                    "RSI": r.rsi, "EMA_Dist": r.ema_dist, "ATR": r.atr,
                    "MTF_Align": r.mtf_align, "BB_Width": r.bb_width,
                    "Bull_FVG_Dist_ATR": r.bull_fvg_dist_atr,
                    "Bear_FVG_Dist_ATR": r.bear_fvg_dist_atr,
                    "run_id": r.run_id,
                } for r in rows])
        except Exception as exc:
            logger.error(f"Failed to read trade journal: {exc}")
            return pd.DataFrame()

    def read_journal(self) -> pd.DataFrame:
        """Returns the full trade journal as a DataFrame. Empty DF on error."""
        try:
            with get_session() as session:
                rows = session.execute(select(TradeJournalEntry)).scalars().all()
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame([{
                    "Timestamp": r.timestamp, "Action": r.action, "Price": r.price,
                    "Lot_Size": r.lot_size, "SL": r.sl, "TP": r.tp, "PnL": r.pnl,
                    "RSI": r.rsi, "EMA_Dist": r.ema_dist, "ATR": r.atr,
                    "MTF_Align": r.mtf_align, "BB_Width": r.bb_width,
                    "Bull_FVG_Dist_ATR": r.bull_fvg_dist_atr,
                    "Bear_FVG_Dist_ATR": r.bear_fvg_dist_atr,
                    "run_id": r.run_id,
                } for r in rows])
        except Exception as exc:
            logger.error(f"Failed to read trade journal: {exc}")
            return pd.DataFrame()

    def overwrite_journal(self, records: list) -> bool:
        """
        Replaces the ENTIRE trade_journal table with the given records, in a
        single transaction. Used by ledger/reconciler.py to rebuild the
        journal from MT5 broker history when it may have drifted out of sync.

        NOTE: wipes the whole table, not just rows for one symbol — the
        current schema has no per-symbol column (single-symbol bot). If/when
        multi-symbol support is added, this needs a WHERE-scoped delete
        instead of a full wipe, matching rebuild_journal()'s symbol parameter.

        `records` is a list of dicts using the same Title-Case keys as the
        old CSV/JournalConfig.COLUMNS format (Timestamp, Action, Price,
        Lot_Size, SL, TP, PnL, RSI, EMA_Dist, ATR, MTF_Align, BB_Width,
        Bull_FVG_Dist_ATR, Bear_FVG_Dist_ATR) — mapped here to the actual
        ORM column names.

        Returns True on success, False on failure (existing data is
        preserved on failure — delete + inserts happen in one transaction).
        """
        try:
            with get_session() as session:
                session.query(TradeJournalEntry).delete()
                for r in records:
                    session.add(TradeJournalEntry(
                        timestamp=datetime.strptime(
                            r["Timestamp"], "%Y-%m-%d %H:%M:%S"
                        ).replace(tzinfo=timezone.utc),
                        action=r.get("Action"),
                        price=r.get("Price"),
                        lot_size=r.get("Lot_Size"),
                        sl=r.get("SL", 0),
                        tp=r.get("TP", 0),
                        pnl=r.get("PnL", 0.0),
                        rsi=r.get("RSI", 0.0),
                        ema_dist=r.get("EMA_Dist", 0.0),
                        atr=r.get("ATR", 0.0),
                        mtf_align=r.get("MTF_Align", 0),
                        bb_width=r.get("BB_Width", 0.0),
                        bull_fvg_dist_atr=r.get("Bull_FVG_Dist_ATR", 0.0),
                        bear_fvg_dist_atr=r.get("Bear_FVG_Dist_ATR", 0.0),
                    ))
            return True
        except Exception as exc:
            logger.error(f"Failed to overwrite trade journal: {exc}")
            return False

    # ==========================================================================
    # EVAL LOG (rolling live feed)
    # ==========================================================================
    # ==========================================================================
    # EVAL LOG (rolling live feed)
    # ==========================================================================

    def log_eval(self, mtf: int, rsi: float, fvg_bull: float, fvg_bear: float,
                 signal: str, reason: str = "", max_rows: int = 50) -> None:
        """Appends one eval-feed row and prunes to the last `max_rows`.
        Session is derived from the current UTC hour, matching the
        classification live_trader.py previously computed inline."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 12 <= hour < 16:
            session = "OVERLAP"
        elif 7 <= hour < 16:
            session = "LDN"
        elif 12 <= hour < 21:
            session = "NY"
        elif 0 <= hour < 8:
            session = "ASIAN"
        else:
            session = "OFF"

        try:
            with get_session() as session_db:
                session_db.add(EvalLogEntry(
                    timestamp=now, session=session, mtf=mtf, rsi=round(rsi, 1),
                    fvg_bull=round(fvg_bull, 2), fvg_bear=round(fvg_bear, 2),
                    signal=signal, reason=reason,
                ))
                session_db.flush()
                all_ids = session_db.execute(
                    select(EvalLogEntry.id).order_by(EvalLogEntry.timestamp.desc())
                ).scalars().all()
                if len(all_ids) > max_rows:
                    excess_ids = all_ids[max_rows:]
                    session_db.query(EvalLogEntry).filter(EvalLogEntry.id.in_(excess_ids)).delete(
                        synchronize_session=False
                    )
        except Exception as exc:
            logger.error(f"Eval log append failed: {exc}")

    def read_eval_log(self) -> list:
        """Returns eval log entries newest-first as a list of dicts, in the
        same shape dashboard.py's _load_eval_log() previously read from
        eval_log.json — so dashboard code needs no field-name changes."""
        try:
            with get_session() as session:
                rows = session.execute(
                    select(EvalLogEntry).order_by(EvalLogEntry.timestamp.desc())
                ).scalars().all()
                return [{
                    "t": r.timestamp.strftime("%H:%M"),
                    "date": r.timestamp.strftime("%d %b"),
                    "session": r.session,
                    "mtf": r.mtf,
                    "rsi": r.rsi,
                    "fvg_bull": r.fvg_bull,
                    "fvg_bear": r.fvg_bear,
                    "signal": r.signal,
                    "reason": r.reason,
                } for r in rows]
        except Exception as exc:
            logger.error(f"Failed to read eval log: {exc}")
            return []

    # ==========================================================================
    # CHART DATA — unchanged, stays as flat file (not transactional state)
    # ==========================================================================

    def update_chart_data(self, df: pd.DataFrame, n_bars: int = 500) -> None:
        from core.config import Paths
        try:
            df.tail(n_bars).to_csv(Paths.CHART_FILE)
        except Exception as exc:
            logger.warning(f"Chart data write failed (non-critical): {exc}")

    # ==========================================================================
    # PRIVATE
    # ==========================================================================

    def _ensure_singletons(self) -> None:
        """Creates the id=1 rows for live_state/app_config if they don't exist."""
        try:
            with get_session() as session:
                if session.get(LiveState, 1) is None:
                    session.add(LiveState(id=1))
                if session.get(AppConfig, 1) is None:
                    session.add(AppConfig(
                        id=1, auto_trade=True, symbol="EURUSD",
                        timeframe="H1", manual_command="NONE",
                    ))
        except Exception as exc:
            logger.error(f"Failed to ensure singleton rows: {exc}")
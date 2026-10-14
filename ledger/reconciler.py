"""
ledger/reconciler.py
====================
Broker-side trade journal reconstruction from MT5 history.

Rebuilds the trade_journal table (Postgres) entirely from
mt5.history_deals_get(). Run this when the journal is out-of-sync due to
connectivity dropout, bot restart, or manual trades placed outside the bot.

RUN:
    python ledger/reconciler.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import MetaTrader5 as mt5
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import BrokerConfig, JournalConfig, Paths, TradingConfig
from core.logger import get_logger
from ledger.state_manager import StateManager

logger = get_logger(__name__)


def rebuild_journal(
    symbol:        str = TradingConfig.ACTIVE_SYMBOL,
    lookback_days: int = 90,
) -> bool:
    """
    Downloads full deal history from MT5 and rebuilds the trade_journal table.

    PROCESS:
        1. Fetch all deals for the bot's magic number in the lookback window.
        2. Group by position ticket (each round-trip = one position).
        3. Reconstruct entry and exit events with true net P&L.
        4. Back up the CURRENT journal (from Postgres) to a timestamped CSV
           before wiping — preserves a rollback point without touching
           the live data source.
        5. Overwrite the journal via StateManager.overwrite_journal()
           (single transaction — no partial-write risk).

    Returns:
        True on success; False on failure.
    """
    Paths.ensure_dirs()
    state = StateManager()

    logger.info(f"Starting ledger reconciliation — {symbol} | last {lookback_days} days")

    # ── Fetch deal history ─────────────────────────────────────────────────────
    from_date = datetime.now() - timedelta(days=lookback_days)
    to_date   = datetime.now() + timedelta(days=1)
    deals     = mt5.history_deals_get(from_date, to_date)

    if deals is None:
        logger.error(f"MT5 history_deals_get failed: {mt5.last_error()}")
        return False

    bot_deals = [d for d in deals if d.magic == BrokerConfig.MAGIC_NUMBER and d.symbol == symbol]

    if not bot_deals:
        logger.warning(
            f"No deals found for magic {BrokerConfig.MAGIC_NUMBER} / {symbol} "
            f"in the last {lookback_days} days. Journal not modified."
        )
        return True   # Not an error — no trades were made

    logger.info(f"Found {len(bot_deals)} deals across {symbol}.")

    # ── Group by position ticket ──────────────────────────────────────────────
    positions: dict = {}
    for d in bot_deals:
        positions.setdefault(d.position_id, []).append(d)

    records = []
    for pos_id, pos_deals in positions.items():
        pos_deals.sort(key=lambda x: x.time)
        entry = pos_deals[0]

        entry_action = JournalConfig.BUY if entry.type == mt5.DEAL_TYPE_BUY else JournalConfig.SELL
        records.append({
            "Timestamp": datetime.fromtimestamp(entry.time).strftime("%Y-%m-%d %H:%M:%S"),
            "Action":    entry_action,
            "Price":     entry.price,
            "Lot_Size":  entry.volume,
            "SL": 0, "TP": 0, "PnL": 0.0,
            # Indicator values unknown at reconciliation time — MT5 doesn't
            # record them; they only exist at signal-time in live_trader.py.
            "RSI": 0, "EMA_Dist": 0, "ATR": 0, "MTF_Align": 0,
            "BB_Width": 0, "Bull_FVG_Dist_ATR": 0, "Bear_FVG_Dist_ATR": 0,
        })

        if len(pos_deals) > 1:
            exit_deal   = pos_deals[-1]
            exit_action = (
                JournalConfig.BUY_CLOSE if exit_deal.type == mt5.DEAL_TYPE_SELL
                else JournalConfig.SELL_CLOSE
            )
            true_comm  = sum(d.commission for d in pos_deals)
            true_swap  = sum(d.swap       for d in pos_deals)
            raw_profit = sum(d.profit     for d in pos_deals)
            net_pnl    = raw_profit + true_comm + true_swap

            records.append({
                "Timestamp": datetime.fromtimestamp(exit_deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                "Action":    exit_action,
                "Price":     exit_deal.price,
                "Lot_Size":  exit_deal.volume,
                "SL": 0, "TP": 0,
                "PnL":       round(net_pnl, 2),
                "RSI": 0, "EMA_Dist": 0, "ATR": 0, "MTF_Align": 0,
                "BB_Width": 0, "Bull_FVG_Dist_ATR": 0, "Bear_FVG_Dist_ATR": 0,
            })

    if not records:
        logger.warning("No records to write after processing deals.")
        return True

    # ── Backup current journal state before wiping ────────────────────────────
    try:
        current_df = state.read_journal()
        if not current_df.empty:
            backup_path = os.path.join(
                Paths.LOGS_DIR, f"journal_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
            )
            current_df.to_csv(backup_path, index=False)
            logger.info(f"Existing journal backed up to {backup_path}")
    except Exception as exc:
        logger.warning(f"Journal backup failed (non-fatal, continuing): {exc}")

    # ── Overwrite journal (single transaction) ────────────────────────────────
    success = state.overwrite_journal(records)
    if success:
        logger.info(
            f"Journal rebuilt successfully — "
            f"{len(records)} records written ({len(positions)} positions reconstructed)."
        )
    else:
        logger.error("Journal overwrite failed — existing data preserved.")
    return success


if __name__ == "__main__":
    load_dotenv()
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)
    if not mt5.login(
        int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
    ):
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    success = rebuild_journal(symbol=TradingConfig.ACTIVE_SYMBOL)
    mt5.shutdown()
    sys.exit(0 if success else 1)
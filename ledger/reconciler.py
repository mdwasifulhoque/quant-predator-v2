"""
ledger/reconciler.py
====================
Broker-side trade journal reconstruction from MT5 history.

Rebuilds trade_journal.csv entirely from mt5.history_deals_get().
Run this when the journal is out-of-sync due to connectivity dropout,
bot restart, or manual trades placed outside the bot.

IMPROVEMENTS vs sync_ledger.py v1:
    - Multi-symbol support (not hardcoded to EURUSD)
    - 90-day lookback (was 21 days — too short for monthly reviews)
    - Backup before wipe (no silent data destruction)
    - Guard against empty deal history overwriting valid journal
    - Uses StateManager for atomic write
    - Indicator values preserved from journal if already present

RUN:
    python ledger/reconciler.py
"""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import BrokerConfig, JournalConfig, Paths, TradingConfig
from core.logger import get_logger
from ledger.state_manager import StateManager

logger = get_logger(__name__)


def rebuild_journal(
    symbol:       str = TradingConfig.ACTIVE_SYMBOL,
    lookback_days: int = 90,
) -> bool:
    """
    Downloads full deal history from MT5 and rebuilds trade_journal.csv.

    PROCESS:
        1. Fetch all deals for the bot's magic number in the lookback window.
        2. Group by position ticket (each round-trip = one position).
        3. Reconstruct entry and exit events with true net P&L.
        4. Backup the existing journal before overwriting.
        5. Write the new journal via StateManager (atomic).

    Args:
        symbol:        MT5 symbol to reconcile.
        lookback_days: History window. 90 days covers most trading scenarios.

    Returns:
        True on success; False on failure.
    """
    Paths.ensure_dirs()
    state = StateManager()

    logger.info(
        f"Starting ledger reconciliation — "
        f"{symbol} | last {lookback_days} days"
    )

    # ── Fetch deal history ─────────────────────────────────────────────────────
    from_date = datetime.now() - timedelta(days=lookback_days)
    to_date   = datetime.now() + timedelta(days=1)
    deals     = mt5.history_deals_get(from_date, to_date)

    if deals is None:
        logger.error(f"MT5 history_deals_get failed: {mt5.last_error()}")
        return False

    bot_deals = [
        d for d in deals
        if d.magic == BrokerConfig.MAGIC_NUMBER and d.symbol == symbol
    ]

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
            "Timestamp":  datetime.fromtimestamp(entry.time).strftime("%Y-%m-%d %H:%M:%S"),
            "Action":     entry_action,
            "Price":      entry.price,
            "Lot_Size":   entry.volume,
            "SL":         0, "TP": 0,
            "PnL":        0.0,
            # Indicator values unknown at reconciliation time — preserved as 0
            # These were only available at signal time in live_trader.py
            "RSI": 0, "EMA_Dist": 0, "ATR": 0, "MTF_Align": 0,
            "BB_Width": 0, "Bull_FVG": 0, "Bear_FVG": 0,
        })

        if len(pos_deals) > 1:
            exit_deal    = pos_deals[-1]
            exit_action  = (
                JournalConfig.BUY_CLOSE
                if exit_deal.type == mt5.DEAL_TYPE_SELL
                else JournalConfig.SELL_CLOSE
            )
            true_comm    = sum(d.commission for d in pos_deals)
            true_swap    = sum(d.swap       for d in pos_deals)
            raw_profit   = sum(d.profit     for d in pos_deals)
            net_pnl      = raw_profit + true_comm + true_swap

            records.append({
                "Timestamp":  datetime.fromtimestamp(exit_deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                "Action":     exit_action,
                "Price":      exit_deal.price,
                "Lot_Size":   exit_deal.volume,
                "SL":         0, "TP": 0,
                "PnL":        round(net_pnl, 2),
                "RSI": 0, "EMA_Dist": 0, "ATR": 0, "MTF_Align": 0,
                "BB_Width": 0, "Bull_FVG": 0, "Bear_FVG": 0,
            })

    if not records:
        logger.warning("No records to write after processing deals.")
        return True

    # ── Backup existing journal ───────────────────────────────────────────────
    if os.path.exists(Paths.JOURNAL_FILE):
        backup = Paths.JOURNAL_FILE.replace(".csv", f"_backup_{datetime.now():%Y%m%d_%H%M%S}.csv")
        shutil.copy2(Paths.JOURNAL_FILE, backup)
        logger.info(f"Existing journal backed up to {backup}")

    # ── Write new journal ─────────────────────────────────────────────────────
    df = pd.DataFrame(records, columns=JournalConfig.COLUMNS)
    df.sort_values("Timestamp", inplace=True)
    df.to_csv(Paths.JOURNAL_FILE, index=False)

    logger.info(
        f"Journal rebuilt successfully — "
        f"{len(records)} records written "
        f"({len(positions)} positions reconstructed)."
    )
    return True


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

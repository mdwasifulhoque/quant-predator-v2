"""
core/db_models.py
==================
SQLAlchemy ORM models — replaces state.json, config.json, trade_journal.csv,
eval_log.json with a single Postgres database (Supabase-hosted).

Design notes:
    - live_state / app_config are singleton tables (id=1 enforced via CHECK).
    - trade_journal is append-only, optionally linked to the training_runs
      row that produced the model making that trade (NULL for live trades
      not yet attributed, though in practice you'll usually know the run).
    - training_runs.config_snapshot is JSONB — dumps the full ModelConfig/
      RewardConfig state at train time rather than needing a rigid column
      per hyperparameter. Queryable via config_snapshot->>'ENTRY_COST' etc.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey,
    Integer, String, Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def utcnow():
    return datetime.now(timezone.utc)


class LiveState(Base):
    """Singleton row — current live account metrics. Replaces state.json."""
    __tablename__ = "live_state"

    id = Column(Integer, primary_key=True, default=1)
    balance = Column(Float)
    equity = Column(Float)
    floating_profit = Column(Float)
    open_position = Column(String)   # "LONG" / "SHORT" / "FLAT"
    ai_signal = Column(String)
    news_status = Column(String)
    mt5_ok = Column(Boolean, default=True)
    ollama_ok = Column(Boolean, default=False)
    last_update = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (CheckConstraint("id = 1", name="live_state_singleton"),)


class AppConfig(Base):
    """Singleton row — user preferences. Replaces config.json."""
    __tablename__ = "app_config"

    id = Column(Integer, primary_key=True, default=1)
    auto_trade = Column(Boolean, default=True)
    symbol = Column(String, default="EURUSD")
    timeframe = Column(String, default="H1")
    manual_command = Column(String, default="NONE")

    __table_args__ = (CheckConstraint("id = 1", name="app_config_singleton"),)


class TrainingRun(Base):
    """
    One row per training run (Run 1, Run 2, Run 3, Run 4, ...).
    Replaces the hand-written run-history table in PROJECT_BLUEPRINT.md.
    """
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True)
    run_label = Column(String, nullable=False)       # e.g. "run_4_option_a"
    git_commit = Column(String)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    config_snapshot = Column(JSONB)                    # full ModelConfig/RewardConfig dump

    total_timesteps = Column(Integer)
    sharpe_final = Column(Float)
    sharpe_best_checkpoint = Column(Float)
    sortino = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    trades_per_day = Column(Float)
    profit_factor = Column(Float)
    expectancy = Column(Float)
    starting_balance = Column(Float)
    ending_balance = Column(Float)
    wandb_run_url = Column(String)
    notes = Column(Text)
    verdict = Column(String)                            # 'PASS' / 'FAIL'

    trades = relationship("TradeJournalEntry", back_populates="run")


class TradeJournalEntry(Base):
    """Append-only trade ledger. Replaces trade_journal.csv."""
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)

    action = Column(String)
    price = Column(Float)
    lot_size = Column(Float)
    sl = Column(Float)
    tp = Column(Float)
    pnl = Column(Float)
    rsi = Column(Float)
    ema_dist = Column(Float)
    atr = Column(Float)
    mtf_align = Column(Integer)
    bb_width = Column(Float)
    bull_fvg_dist_atr = Column(Float)
    bear_fvg_dist_atr = Column(Float)

    run = relationship("TrainingRun", back_populates="trades")


class EvalLogEntry(Base):
    """Rolling live evaluation feed (dashboard's per-H1-bar log). Replaces eval_log.json."""
    __tablename__ = "eval_log"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    session = Column(String)          # "ASIAN" / "LDN" / "NY" / "OVERLAP" / "OFF"
    mtf = Column(Integer)
    rsi = Column(Float)
    fvg_bull = Column(Float)          # bull_fvg_dist_atr at eval time (0 = no active zone)
    fvg_bear = Column(Float)          # bear_fvg_dist_atr at eval time
    signal = Column(String)           # "BUY" / "SELL" / "HOLD"
    reason = Column(Text)             # human-readable HOLD explanation
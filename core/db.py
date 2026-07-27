"""
core/db.py
==========
SQLAlchemy engine + session factory. Reads connection string from the
DATABASE_URL environment variable (never hardcode credentials).

Usage:
    from core.db import get_session, init_db

    init_db()  # run once at startup — creates tables if they don't exist

    with get_session() as session:
        session.add(some_row)
        session.commit()
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.db_models import Base

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable not set. "
        "Set it to your Supabase connection string, e.g. via a .env file "
        "(never commit credentials to git)."
    )

# pool_pre_ping avoids stale-connection errors after idle periods —
# relevant since this bot only writes ~0.6 times/day, so connections
# will sit idle for long stretches between trades.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Creates all tables if they don't already exist. Safe to call repeatedly."""
    Base.metadata.create_all(engine)


@contextmanager
def get_session():
    """Context-managed session — commits on success, rolls back on exception."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
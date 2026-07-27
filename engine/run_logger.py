"""
engine/run_logger.py
=====================
Helper for trainer.py to record training run results into the
training_runs table. Call log_run() once at the end of training.

Usage in trainer.py:

    from engine.run_logger import log_run

    log_run(
        run_label="run_5_optuna_tuned",
        config_snapshot={
            "ModelConfig": {k: v for k, v in vars(ModelConfig).items() if not k.startswith("_")},
            "RewardConfig": {k: v for k, v in vars(RewardConfig).items() if not k.startswith("_")},
        },
        started_at=run_start_time,
        sharpe_final=..., sharpe_best_checkpoint=..., ...
    )
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.db import get_session
from core.db_models import TrainingRun
from core.logger import get_logger

logger = get_logger(__name__)


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def log_run(
    run_label: str,
    config_snapshot: Dict[str, Any],
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    total_timesteps: Optional[int] = None,
    sharpe_final: Optional[float] = None,
    sharpe_best_checkpoint: Optional[float] = None,
    sortino: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    win_rate: Optional[float] = None,
    total_trades: Optional[int] = None,
    trades_per_day: Optional[float] = None,
    profit_factor: Optional[float] = None,
    expectancy: Optional[float] = None,
    starting_balance: Optional[float] = None,
    ending_balance: Optional[float] = None,
    wandb_run_url: Optional[str] = None,
    notes: Optional[str] = None,
    verdict: Optional[str] = None,
) -> Optional[int]:
    """Writes one row to training_runs. Returns the new run's id, or None on failure."""
    try:
        with get_session() as session:
            run = TrainingRun(
                run_label=run_label,
                git_commit=_get_git_commit(),
                started_at=started_at or datetime.now(timezone.utc),
                completed_at=completed_at or datetime.now(timezone.utc),
                config_snapshot=config_snapshot,
                total_timesteps=total_timesteps,
                sharpe_final=sharpe_final,
                sharpe_best_checkpoint=sharpe_best_checkpoint,
                sortino=sortino,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                trades_per_day=trades_per_day,
                profit_factor=profit_factor,
                expectancy=expectancy,
                starting_balance=starting_balance,
                ending_balance=ending_balance,
                wandb_run_url=wandb_run_url,
                notes=notes,
                verdict=verdict,
            )
            session.add(run)
            session.flush()
            run_id = run.id
        logger.info(f"Logged training run '{run_label}' as run_id={run_id}")
        return run_id
    except Exception as exc:
        logger.error(f"Failed to log training run: {exc}")
        return None

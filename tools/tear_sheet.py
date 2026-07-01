"""
tools/tear_sheet.py
===================
Institutional performance metrics for Quant Predator v2.

CRITICAL FIX vs v1:
    Sharpe ratio annualisation factor corrected from sqrt(252) to sqrt(6240).
    H1 data produces ~6,240 bars per year (24h × 5 days × 52 weeks).
    The original sqrt(252) assumed DAILY returns, overstating Sharpe by ~5x.
    A backtest Sharpe of 1.0 in v1 = a real Sharpe of ~0.20.

RUN STANDALONE:
    python tools/tear_sheet.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import ModelConfig
from core.logger import get_logger

logger = get_logger(__name__)

# H1 bars per trading year: 24 hours * 5 days * 52 weeks
_H1_ANNUAL_FACTOR = np.sqrt(24 * 5 * 52)   # ≈ 78.99


def generate_tear_sheet(
    equity_df:  pd.DataFrame,
    trades_df:  Optional[pd.DataFrame] = None,
    label:      str = "Backtest",
    print_report: bool = True,
) -> Dict[str, float]:
    """
    Computes and prints institutional-grade performance metrics.

    Args:
        equity_df:    DataFrame with a single 'balance' column containing the
                      account equity at each H1 bar step.
        trades_df:    Optional trade log DataFrame with a 'pnl' column.
                      If provided, trade-level win rate and profit factor
                      are computed separately from the equity curve.
        label:        Title shown in the printed report.
        print_report: If True, prints the formatted report to stdout.

    Returns:
        Dictionary of metric names to float values for programmatic access.
    """
    if equity_df is None or len(equity_df) == 0:
        logger.error("generate_tear_sheet: empty equity_df provided.")
        return {}

    df = equity_df.copy()
    df["return"] = df["balance"].pct_change().fillna(0.0)

    start_balance = float(df["balance"].iloc[0])
    end_balance   = float(df["balance"].iloc[-1])
    total_return  = (end_balance / start_balance - 1.0) * 100.0

    # ── Drawdown ──────────────────────────────────────────────────────────────
    df["peak"]     = df["balance"].cummax()
    df["drawdown"] = (df["balance"] - df["peak"]) / df["peak"]
    max_dd_pct     = float(df["drawdown"].min() * 100.0)

    # ── Sharpe (H1-annualised) ────────────────────────────────────────────────
    mean_ret = float(df["return"].mean())
    std_ret  = float(df["return"].std())
    sharpe   = (mean_ret / std_ret * _H1_ANNUAL_FACTOR) if std_ret > 0 else 0.0

    # ── Sortino (downside risk only) ──────────────────────────────────────────
    downside = df.loc[df["return"] < 0, "return"]
    down_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino  = (mean_ret / down_std * _H1_ANNUAL_FACTOR) if down_std > 0 else 0.0

    # ── Calmar (return / max drawdown) ───────────────────────────────────────
    annual_return = ((1 + total_return / 100) ** (6240 / len(df)) - 1) * 100.0
    calmar = (annual_return / abs(max_dd_pct)) if max_dd_pct != 0.0 else 0.0

    # ── Trade statistics ──────────────────────────────────────────────────────
    n_trades    = 0
    win_rate    = 0.0
    profit_factor = 0.0
    avg_win     = 0.0
    avg_loss    = 0.0
    expectancy  = 0.0

    if trades_df is not None and len(trades_df) > 0:
        closed = trades_df[trades_df["pnl"] != 0.0]["pnl"].dropna()
        if len(closed) > 0:
            n_trades = len(closed)
            wins     = closed[closed > 0]
            losses   = closed[closed < 0]
            win_rate = len(wins) / n_trades * 100.0
            avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
            gross_profit = float(wins.sum())
            gross_loss   = abs(float(losses.sum()))
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
            # Expectancy = (WR * avg_win) + (LR * avg_loss)
            lr = 1.0 - win_rate / 100.0
            expectancy = (win_rate / 100.0 * avg_win) + (lr * avg_loss)

    metrics = {
        "start_balance":   start_balance,
        "end_balance":     end_balance,
        "total_return_pct":total_return,
        "annual_return_pct":annual_return,
        "max_drawdown_pct":max_dd_pct,
        "sharpe_ratio":    sharpe,
        "sortino_ratio":   sortino,
        "calmar_ratio":    calmar,
        "n_trades":        n_trades,
        "win_rate_pct":    win_rate,
        "profit_factor":   profit_factor,
        "avg_win_usd":     avg_win,
        "avg_loss_usd":    avg_loss,
        "expectancy_usd":  expectancy,
    }

    if print_report:
        _print_report(label, metrics)

    return metrics


def _print_report(label: str, m: Dict[str, float]) -> None:
    """Prints a formatted institutional-style performance report."""
    div   = "═" * 50
    thin  = "─" * 50

    def _verdict(sharpe: float) -> str:
        if sharpe > 2.0: return "ELITE — Institutional grade."
        if sharpe > 1.5: return "STRONG — Deployable with monitoring."
        if sharpe > 1.0: return "ADEQUATE — Passes validation gate."
        if sharpe > 0.5: return "WEAK — Profitable but too risky."
        return "FAILING — Not suitable for deployment."

    print(f"\n{div}")
    print(f"  QUANT PREDATOR v2 — {label.upper()}")
    print(div)
    print(f"  Starting Balance : ${m['start_balance']:>12,.2f}")
    print(f"  Ending Balance   : ${m['end_balance']:>12,.2f}")
    print(f"  Total Return     : {m['total_return_pct']:>+11.2f}%")
    print(f"  Annual Return    : {m['annual_return_pct']:>+11.2f}%")
    print(thin)
    print(f"  Max Drawdown     : {m['max_drawdown_pct']:>11.2f}%")
    print(f"  Sharpe Ratio     : {m['sharpe_ratio']:>12.3f}  (H1-annualised)")
    print(f"  Sortino Ratio    : {m['sortino_ratio']:>12.3f}")
    print(f"  Calmar Ratio     : {m['calmar_ratio']:>12.3f}")
    print(thin)
    if m["n_trades"] > 0:
        print(f"  Total Trades     : {m['n_trades']:>12.0f}")
        print(f"  Win Rate         : {m['win_rate_pct']:>11.1f}%")
        print(f"  Profit Factor    : {m['profit_factor']:>12.3f}")
        print(f"  Avg Win          : ${m['avg_win_usd']:>+11.2f}")
        print(f"  Avg Loss         : ${m['avg_loss_usd']:>+11.2f}")
        print(f"  Expectancy       : ${m['expectancy_usd']:>+11.2f}")
        print(thin)
    print(f"  Verdict: {_verdict(m['sharpe_ratio'])}")
    print(f"{div}\n")


if __name__ == "__main__":
    from core.config import TradingConfig
    sym        = TradingConfig.ACTIVE_SYMBOL
    equity_path= os.path.join(os.path.dirname(__file__), f"equity_{sym.lower()}.csv")
    if not os.path.exists(equity_path):
        print(f"No equity file found at {equity_path}. Run a backtest first.")
    else:
        eq_df = pd.read_csv(equity_path)
        generate_tear_sheet(eq_df, label=f"{sym} Backtest")

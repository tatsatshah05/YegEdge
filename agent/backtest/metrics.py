# agent/backtest/metrics.py
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

_RISK_FREE_RATE = 0.07  # India 10Y gilt ≈ 7%
_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class SessionResult:
    """Outcome of one backtest trading session."""

    session_date: date
    bars_processed: int
    fills: int
    gross_pnl: Decimal
    costs: Decimal  # transaction costs (IndianCostModel)
    net_pnl: Decimal  # gross_pnl - costs
    final_nav: Decimal  # running NAV after this session's net_pnl


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate performance metrics across all backtest sessions."""

    total_sessions: int
    winning_sessions: int
    win_rate: float
    total_gross_pnl: Decimal
    total_costs: Decimal
    total_net_pnl: Decimal
    sharpe_ratio: float  # annualized, net of costs, excess over risk-free rate
    max_drawdown: float  # peak-to-trough fraction of peak NAV
    cagr: float  # annualized net return (assumes 252 trading days/year)
    initial_nav: Decimal
    final_nav: Decimal


@dataclass(frozen=True)
class BacktestReport:
    """Full backtest output: per-session detail + aggregate metrics."""

    sessions: list[SessionResult]
    metrics: BacktestMetrics


def compute_metrics(
    sessions: Sequence[SessionResult],
    initial_nav: Decimal,
) -> BacktestMetrics:
    """Compute aggregate performance metrics from session results.

    All return calculations use net_pnl (after transaction costs).
    Sharpe uses daily excess returns over the Indian risk-free rate (7%).
    CAGR assumes 252 trading days per calendar year.
    """
    if not sessions:
        return BacktestMetrics(
            total_sessions=0,
            winning_sessions=0,
            win_rate=0.0,
            total_gross_pnl=Decimal("0"),
            total_costs=Decimal("0"),
            total_net_pnl=Decimal("0"),
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            initial_nav=initial_nav,
            final_nav=initial_nav,
        )

    total_sessions = len(sessions)
    winning_sessions = sum(1 for s in sessions if s.net_pnl > 0)
    win_rate = winning_sessions / total_sessions
    total_gross_pnl = sum((s.gross_pnl for s in sessions), Decimal("0"))
    total_costs = sum((s.costs for s in sessions), Decimal("0"))
    total_net_pnl = sum((s.net_pnl for s in sessions), Decimal("0"))
    final_nav = sessions[-1].final_nav

    # Daily net returns as fraction of initial NAV (consistent denominator)
    initial = float(initial_nav)
    daily_returns = [float(s.net_pnl) / initial for s in sessions]

    # Annualized Sharpe ratio (excess over risk-free rate)
    if len(daily_returns) > 1:
        daily_rf = _RISK_FREE_RATE / _TRADING_DAYS_PER_YEAR
        excess = [r - daily_rf for r in daily_returns]
        mean_excess = sum(excess) / len(excess)
        variance = sum((r - mean_excess) ** 2 for r in excess) / (len(excess) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        if std_dev > 0:
            sharpe = mean_excess / std_dev * math.sqrt(_TRADING_DAYS_PER_YEAR)
        elif mean_excess > 0:
            # Zero volatility but positive excess return = infinite Sharpe
            sharpe = 1.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Max peak-to-trough drawdown
    peak = initial
    max_dd = 0.0
    running = initial
    for s in sessions:
        running += float(s.net_pnl)
        if running > peak:
            peak = running
        dd = (peak - running) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # CAGR (annualized net return)
    n_years = total_sessions / _TRADING_DAYS_PER_YEAR
    total_return = float(final_nav) / initial if initial > 0 else 0.0
    cagr = (total_return ** (1.0 / n_years) - 1.0) if n_years > 0 and total_return > 0 else 0.0

    return BacktestMetrics(
        total_sessions=total_sessions,
        winning_sessions=winning_sessions,
        win_rate=round(win_rate, 4),
        total_gross_pnl=total_gross_pnl,
        total_costs=total_costs,
        total_net_pnl=total_net_pnl,
        sharpe_ratio=round(sharpe, 3),
        max_drawdown=round(max_dd, 4),
        cagr=round(cagr, 4),
        initial_nav=initial_nav,
        final_nav=final_nav,
    )

# tests/backtest/test_metrics.py
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from agent.backtest.metrics import BacktestMetrics, BacktestReport, SessionResult, compute_metrics


def _sess(net_pnl: float, nav: float, costs: float = 0.0, fills: int = 2) -> SessionResult:
    """Build a SessionResult with session_date=2024-01-02 (date doesn't affect metrics)."""
    gross = net_pnl + costs
    return SessionResult(
        session_date=date(2024, 1, 2),
        bars_processed=6,
        fills=fills,
        gross_pnl=Decimal(str(gross)),
        costs=Decimal(str(costs)),
        net_pnl=Decimal(str(net_pnl)),
        final_nav=Decimal(str(nav)),
    )


def test_empty_sessions_returns_zero_metrics() -> None:
    m = compute_metrics([], Decimal("83000"))
    assert m.total_sessions == 0
    assert m.win_rate == 0.0
    assert m.sharpe_ratio == 0.0
    assert m.max_drawdown == 0.0
    assert m.cagr == 0.0
    assert m.final_nav == Decimal("83000")
    assert m.total_costs == Decimal("0")


def test_win_rate_two_wins_one_loss() -> None:
    sessions = [
        _sess(net_pnl=1000.0, nav=84000.0),
        _sess(net_pnl=-300.0, nav=83700.0),
        _sess(net_pnl=500.0, nav=84200.0),
    ]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.total_sessions == 3
    assert m.winning_sessions == 2
    assert m.win_rate == pytest.approx(2 / 3, rel=0.001)


def test_max_drawdown_peak_then_trough() -> None:
    # NAV path: 83000 → 85000 (+2000) → 83000 (-2000) → 84000 (+1000)
    # Peak = 85000, trough after peak = 83000
    # drawdown = 2000 / 85000
    sessions = [
        _sess(net_pnl=2000.0, nav=85000.0),
        _sess(net_pnl=-2000.0, nav=83000.0),
        _sess(net_pnl=1000.0, nav=84000.0),
    ]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.max_drawdown == pytest.approx(2000 / 85000, rel=0.01)


def test_consistent_positive_returns_give_positive_sharpe() -> None:
    sessions = [_sess(net_pnl=200.0, nav=83000.0 + 200.0 * (i + 1)) for i in range(30)]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.sharpe_ratio > 0.0


def test_total_costs_and_pnl_aggregation() -> None:
    sessions = [
        _sess(net_pnl=900.0, nav=83900.0, costs=50.0),
        _sess(net_pnl=450.0, nav=84350.0, costs=30.0),
    ]
    m = compute_metrics(sessions, Decimal("83000"))
    assert m.total_costs == Decimal("80.0")
    assert m.total_gross_pnl == Decimal("1430.0")
    assert m.total_net_pnl == Decimal("1350.0")


def test_cagr_positive_for_consistently_profitable_run() -> None:
    initial = Decimal("83000")
    # 60 sessions each netting +₹200
    sessions = [
        SessionResult(
            session_date=date(2024, 1, 2),
            bars_processed=6,
            fills=2,
            gross_pnl=Decimal("200"),
            costs=Decimal("0"),
            net_pnl=Decimal("200"),
            final_nav=Decimal(str(float(initial) + 200.0 * (i + 1))),
        )
        for i in range(60)
    ]
    m = compute_metrics(sessions, initial)
    assert m.cagr > 0.0
    assert m.final_nav == Decimal(str(float(initial) + 200.0 * 60))

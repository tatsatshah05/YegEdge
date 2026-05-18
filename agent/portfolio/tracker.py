from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import structlog

from agent.data.types import Position
from agent.execution.types import Fill
from agent.risk.types import PortfolioState
from agent.strategies.types import Action

logger = structlog.get_logger()


class PortfolioTracker:
    """Mutable tracker that produces immutable PortfolioState snapshots.

    apply_fill() updates internal state and returns a snapshot.
    mark_to_market() revalues open positions at current prices and returns a snapshot.
    state property returns the latest snapshot without modifying internal state.

    NOTE: `nav` in snapshots returned by `state` is book-cost NAV (positions valued at
    average_price). Snapshots returned by `mark_to_market()` carry live-price NAV.
    `peak_nav` may therefore exceed `nav` between mark-to-market calls — this is expected,
    not a drawdown.
    """

    def __init__(
        self,
        initial_nav: Decimal,
        initial_cash: Decimal,
        start_time: datetime,
    ) -> None:
        self._cash = initial_cash
        self._initial_nav = initial_nav
        self._peak_nav = initial_nav
        self._daily_pnl = Decimal("0")
        self._weekly_pnl = Decimal("0")
        self._positions: dict[str, Position] = {}
        self._orders_today: int = 0
        self._last_order_time: dict[str, datetime] = {}
        self._kill_switch_active: bool = False
        self._evaluation_time: datetime = start_time

    def apply_fill(self, fill: Fill, *, evaluation_time: datetime) -> PortfolioState:
        """Apply a fill to internal state and return a PortfolioState snapshot.

        ENTER_LONG: deduct cash, create/increase position at fill_price.
        EXIT_LONG:  add cash proceeds, realize P&L, remove/reduce position.
        State mutations (orders_today, last_order_time) happen inside each branch so
        that ghost exits against non-existent positions do not consume an order slot.
        """
        self._evaluation_time = evaluation_time

        if fill.action == Action.ENTER_LONG:
            self._orders_today += 1
            self._last_order_time[fill.symbol] = fill.timestamp
            cost = fill.fill_price * fill.quantity
            self._cash -= cost
            existing = self._positions.get(fill.symbol)
            if existing is None:
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=fill.quantity,
                    average_price=fill.fill_price,
                    product="MIS",
                )
            else:
                # Weighted average for add-on entries
                total_qty = existing.quantity + fill.quantity
                avg_price = (
                    existing.average_price * existing.quantity + fill.fill_price * fill.quantity
                ) / Decimal(str(total_qty))
                self._positions[fill.symbol] = Position(
                    symbol=fill.symbol,
                    quantity=total_qty,
                    average_price=avg_price.quantize(Decimal("0.01")),
                    product="MIS",
                )

        elif fill.action == Action.EXIT_LONG:
            existing = self._positions.get(fill.symbol)
            if existing is not None:
                self._orders_today += 1
                self._last_order_time[fill.symbol] = fill.timestamp
                exit_qty = min(fill.quantity, existing.quantity)
                if exit_qty < fill.quantity:
                    logger.warning(
                        "portfolio_tracker.oversized_exit_clamped",
                        symbol=fill.symbol,
                        fill_qty=fill.quantity,
                        held_qty=existing.quantity,
                        clamped_qty=exit_qty,
                    )
                realized_pnl = (fill.fill_price - existing.average_price) * exit_qty
                self._daily_pnl += realized_pnl
                self._weekly_pnl += realized_pnl
                proceeds = fill.fill_price * exit_qty
                self._cash += proceeds
                remaining_qty = existing.quantity - exit_qty
                if remaining_qty <= 0:
                    del self._positions[fill.symbol]
                else:
                    self._positions[fill.symbol] = Position(
                        symbol=fill.symbol,
                        quantity=remaining_qty,
                        average_price=existing.average_price,
                        product="MIS",
                    )

        nav = self._compute_nav()
        if nav > self._peak_nav:
            self._peak_nav = nav

        logger.debug(
            "portfolio_tracker.apply_fill",
            symbol=fill.symbol,
            action=str(fill.action),
            quantity=fill.quantity,
            nav=str(nav),
            cash=str(self._cash),
        )
        return self._snapshot(nav)

    def mark_to_market(
        self,
        prices: dict[str, Decimal],
        *,
        evaluation_time: datetime,
    ) -> PortfolioState:
        """Revalue open positions at current prices. Does not modify positions."""
        self._evaluation_time = evaluation_time
        nav = self._cash + sum(
            prices.get(pos.symbol, pos.average_price) * pos.quantity
            for pos in self._positions.values()
        )
        if nav > self._peak_nav:
            self._peak_nav = nav
        return self._snapshot(nav)

    @property
    def state(self) -> PortfolioState:
        return self._snapshot(self._compute_nav())

    def activate_kill_switch(self) -> PortfolioState:
        """Activate the kill switch. All subsequent RiskManager calls will be rejected."""
        self._kill_switch_active = True
        logger.warning("portfolio_tracker.kill_switch_activated")
        return self._snapshot(self._compute_nav())

    def _compute_nav(self) -> Decimal:
        return self._cash + sum(
            pos.average_price * pos.quantity for pos in self._positions.values()
        )

    def _snapshot(self, nav: Decimal) -> PortfolioState:
        return PortfolioState(
            nav=nav,
            cash=self._cash,
            positions=dict(self._positions),
            daily_pnl=self._daily_pnl,
            weekly_pnl=self._weekly_pnl,
            peak_nav=self._peak_nav,
            orders_today=self._orders_today,
            last_order_time=dict(self._last_order_time),
            kill_switch_active=self._kill_switch_active,
            evaluation_time=self._evaluation_time,
        )

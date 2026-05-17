from __future__ import annotations

from datetime import time
from decimal import Decimal
from zoneinfo import ZoneInfo

import structlog

from agent.data.types import DataQuality
from agent.risk.rules import RiskRules
from agent.risk.types import (
    PortfolioState,
    RejectionReason,
    RiskDecision,
    RiskVerdict,
)
from agent.strategies.types import Action, Signal

logger = structlog.get_logger()

IST = ZoneInfo("Asia/Kolkata")

_ALLOWED_QUALITIES = {DataQuality.OK, DataQuality.PARTIAL}


class RiskManager:
    """Evaluate a Signal against all risk rules and return a RiskDecision.

    evaluate() is a pure function of (signal, portfolio, entry_price).
    Rules are checked in priority order — first rejection wins.
    """

    def __init__(self, rules: RiskRules) -> None:
        self._rules = rules
        self._window_start = _parse_time(rules.windows.trade_start_ist)
        self._window_end = _parse_time(rules.windows.trade_end_ist)

    def evaluate(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        entry_price: Decimal,
    ) -> RiskDecision:
        """Apply all risk rules to the signal. Returns first rejection encountered.

        EXIT_LONG bypasses all entry rules except the kill switch.
        """
        # 1. Kill switch — blocks everything including exits
        if portfolio.kill_switch_active:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.KILL_SWITCH_ACTIVE,
                "Kill switch is active — manual restart required",
            )

        # 2. EXIT_LONG bypasses all entry-specific checks
        if signal.action == Action.EXIT_LONG:
            return self._approve_exit(signal, portfolio, entry_price)

        # 3. Data quality gate
        if signal.data_quality not in _ALLOWED_QUALITIES:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.SUSPECT_DATA_QUALITY,
                f"Data quality is {signal.data_quality.value!r} — skipping",
            )

        # 4. Reward/risk from actual entry vs signal stop/target
        risk = entry_price - signal.suggested_stop
        reward = signal.suggested_target - entry_price
        min_rr = Decimal(str(self._rules.per_trade.min_reward_risk))
        if risk <= 0 or reward / risk < min_rr:
            actual_rr = (reward / risk).quantize(Decimal("0.01")) if risk > 0 else Decimal("0")
            return self._reject(
                signal,
                entry_price,
                RejectionReason.INSUFFICIENT_REWARD_RISK,
                f"R/R={actual_rr} below minimum {min_rr}",
            )

        # 5. Trading window
        eval_ist = portfolio.evaluation_time.astimezone(IST)
        t = eval_ist.time().replace(second=0, microsecond=0)
        if not (self._window_start <= t <= self._window_end):
            return self._reject(
                signal,
                entry_price,
                RejectionReason.OUTSIDE_TRADING_WINDOW,
                f"Current time {t} outside [{self._window_start}, {self._window_end}] IST",
            )

        # 6. Max concurrent positions
        if len(portfolio.positions) >= self._rules.portfolio.max_concurrent_positions:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.MAX_POSITIONS_REACHED,
                f"Already at max {self._rules.portfolio.max_concurrent_positions} positions",
            )

        # 7. Minimum cash buffer (strict <, not <=: cash == min_cash is allowed)
        min_cash = portfolio.nav * Decimal(str(self._rules.portfolio.min_cash_fraction))
        if portfolio.cash < min_cash:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.INSUFFICIENT_CASH,
                f"Cash {portfolio.cash} below minimum buffer {min_cash}",
            )

        # 8. Daily loss cap (loss at-or-beyond cap → reject)
        max_daily_loss = portfolio.nav * Decimal(str(self._rules.loss_caps.max_daily_loss_fraction))
        if portfolio.daily_pnl <= -max_daily_loss:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.DAILY_LOSS_CAP,
                f"Daily P&L {portfolio.daily_pnl} hit cap -{max_daily_loss}",
            )

        # 9. Weekly loss cap
        max_weekly_loss = portfolio.nav * Decimal(
            str(self._rules.loss_caps.max_weekly_loss_fraction)
        )
        if portfolio.weekly_pnl <= -max_weekly_loss:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.WEEKLY_LOSS_CAP,
                f"Weekly P&L {portfolio.weekly_pnl} hit cap -{max_weekly_loss}",
            )

        # 10. Max drawdown (at-threshold = reject)
        if portfolio.peak_nav > 0:
            drawdown = (portfolio.peak_nav - portfolio.nav) / portfolio.peak_nav
            max_dd = Decimal(str(self._rules.loss_caps.max_drawdown_fraction))
            if drawdown >= max_dd:
                return self._reject(
                    signal,
                    entry_price,
                    RejectionReason.DRAWDOWN_BREACH,
                    f"Drawdown {drawdown:.2%} >= max {max_dd:.2%} — kill switch required",
                )

        # 11. Max orders per day
        if portfolio.orders_today >= self._rules.frequency.max_new_orders_per_day:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.MAX_ORDERS_TODAY,
                f"Already placed {portfolio.orders_today} orders today "
                f"(max {self._rules.frequency.max_new_orders_per_day})",
            )

        # 12. Symbol cooldown
        last_time = portfolio.last_order_time.get(signal.symbol)
        if last_time is not None:
            elapsed_secs = (portfolio.evaluation_time - last_time).total_seconds()
            cooldown_secs = self._rules.frequency.symbol_cooldown_minutes * 60
            if elapsed_secs < cooldown_secs:
                remaining = int((cooldown_secs - elapsed_secs) / 60)
                return self._reject(
                    signal,
                    entry_price,
                    RejectionReason.SYMBOL_COOLDOWN,
                    f"{signal.symbol} in cooldown — {remaining}m remaining",
                )

        # 13. Position sizing — computes and returns approved quantity
        return self._size_position(signal, portfolio, entry_price)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reject(
        self,
        signal: Signal,
        entry_price: Decimal,
        reason: RejectionReason,
        detail: str,
    ) -> RiskDecision:
        logger.info(
            "risk_manager.rejected",
            symbol=signal.symbol,
            reason=reason.value,
            detail=detail,
        )
        return RiskDecision(
            verdict=RiskVerdict.REJECTED,
            quantity=0,
            entry_price=entry_price,
            stop_price=signal.suggested_stop,
            target_price=signal.suggested_target,
            risk_per_share=Decimal("0"),
            position_value=Decimal("0"),
            rejection_reason=reason,
            rejection_detail=detail,
            signal=signal,
        )

    def _approve_exit(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        entry_price: Decimal,
    ) -> RiskDecision:
        existing = portfolio.positions.get(signal.symbol)
        quantity = existing.quantity if existing is not None else 0
        risk_per_share = max(entry_price - signal.suggested_stop, Decimal("0.01"))
        logger.info(
            "risk_manager.exit_approved",
            symbol=signal.symbol,
            quantity=quantity,
        )
        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=signal.suggested_stop,
            target_price=signal.suggested_target,
            risk_per_share=risk_per_share,
            position_value=Decimal(str(quantity)) * entry_price,
            rejection_reason=RejectionReason.NONE,
            rejection_detail="",
            signal=signal,
        )

    def _size_position(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        entry_price: Decimal,
    ) -> RiskDecision:
        risk_per_share = entry_price - signal.suggested_stop
        if risk_per_share <= 0:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.ZERO_QUANTITY,
                "risk_per_share <= 0 (entry_price <= stop_price)",
            )

        max_risk_amount = portfolio.nav * Decimal(str(self._rules.per_trade.max_risk_fraction))
        qty_by_risk = int(max_risk_amount / risk_per_share)

        max_position_value = portfolio.nav * Decimal(
            str(self._rules.per_trade.max_position_fraction)
        )
        qty_by_size = int(max_position_value / entry_price)

        quantity = min(qty_by_risk, qty_by_size)

        if quantity == 0:
            return self._reject(
                signal,
                entry_price,
                RejectionReason.ZERO_QUANTITY,
                f"Position sizing produced 0 shares "
                f"(max_risk={max_risk_amount:.2f}, risk_per_share={risk_per_share})",
            )

        position_value = Decimal(str(quantity)) * entry_price
        logger.info(
            "risk_manager.approved",
            symbol=signal.symbol,
            quantity=quantity,
            entry_price=str(entry_price),
            risk_per_share=str(risk_per_share),
            position_value=str(position_value),
        )
        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=signal.suggested_stop,
            target_price=signal.suggested_target,
            risk_per_share=risk_per_share,
            position_value=position_value,
            rejection_reason=RejectionReason.NONE,
            rejection_detail="",
            signal=signal,
        )


def _parse_time(hhmm: str) -> time:
    """Parse 'HH:MM' string into a datetime.time object."""
    h, m = hhmm.split(":")
    return time(int(h), int(m))

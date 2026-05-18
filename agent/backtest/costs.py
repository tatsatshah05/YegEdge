# agent/backtest/costs.py
from __future__ import annotations

from decimal import Decimal

from agent.execution.types import Fill
from agent.strategies.types import Action

# Upstox/Zerodha discount broker: ₹20 flat per order
_BROKERAGE_PER_ORDER = Decimal("20")

# NSE equity rates (as fractions of trade value)
_EXCHANGE_RATE = Decimal("0.0000325")  # 0.00325% NSE transaction charge
_SEBI_RATE = Decimal("0.000001")  # 0.0001% SEBI charge
_STT_SELL_RATE = Decimal("0.00025")  # 0.025% STT on intraday sell only
_STAMP_DUTY_RATE = Decimal("0.00003")  # 0.003% stamp duty on buy only
_GST_RATE = Decimal("0.18")  # 18% GST on brokerage + exchange charges


class IndianCostModel:
    """Compute realistic NSE equity intraday (MIS) transaction costs per fill.

    Rates per SEBI/NSE schedule. STT applies to sell side only (intraday MIS).
    Stamp duty applies to buy side only. Round-trip ~6-10 bps for typical lots.
    """

    def compute_cost(self, fill: Fill) -> Decimal:
        """Return total transaction cost for one fill, in INR, rounded to paise.

        ENTER_LONG: stamp duty + exchange + SEBI + brokerage + GST.
        EXIT_LONG:  STT + exchange + SEBI + brokerage + GST.
        """
        trade_value = fill.fill_price * Decimal(str(fill.quantity))

        exchange_inr = (trade_value * _EXCHANGE_RATE).quantize(Decimal("0.01"))
        sebi_inr = (trade_value * _SEBI_RATE).quantize(Decimal("0.01"))
        brokerage_inr = min(_BROKERAGE_PER_ORDER, trade_value * Decimal("0.0003"))
        gst_inr = ((brokerage_inr + exchange_inr) * _GST_RATE).quantize(Decimal("0.01"))

        if fill.action == Action.ENTER_LONG:
            side_charge = (trade_value * _STAMP_DUTY_RATE).quantize(Decimal("0.01"))
        else:  # EXIT_LONG
            side_charge = (trade_value * _STT_SELL_RATE).quantize(Decimal("0.01"))

        return (side_charge + exchange_inr + sebi_inr + brokerage_inr + gst_inr).quantize(
            Decimal("0.01")
        )

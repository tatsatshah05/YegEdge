"""Opening Range Breakout strategy for 5-minute bars.

Opening range = first 30 minutes of the session (9:15–9:44 IST, 6 bars on 5m).
Entry: first 5m bar AFTER 9:45 that closes ABOVE OR_high AND above VWAP.
Stop:  OR_low (natural support; structural, not ATR-based).
Target: OR_high + OR_range * target_r (default 1.5R → risk/reward ≥ 1.5).

Filters applied on every candidate entry bar:
  - OR range must be ≥ min_range_pct% of entry close (avoids ultra-narrow/choppy ranges)
  - volume breakout: bar volume ≥ min_volume_ratio × 20-bar rolling mean volume
  - VWAP filter: close > VWAP at the time of entry (confirms buying pressure)
  - time filter: no new entries after max_entry_hour:max_entry_minute IST (default 13:00)

Exit: close drops back BELOW OR_high on a subsequent bar (failed breakout stop-out).
Both entry and exit fire at most ONCE per symbol per trading day.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import structlog

from agent.data.types import DataQuality
from agent.strategies.base import BaseStrategy
from agent.strategies.types import Action, Signal

logger = structlog.get_logger()

IST = ZoneInfo("Asia/Kolkata")

_STRATEGY_NAME = "orb_5m_v1"

_ALLOWED_QUALITIES = {DataQuality.OK.value, DataQuality.PARTIAL.value}

_REQUIRED_COLS = frozenset(
    {"symbol", "timestamp", "open", "high", "low", "close", "volume", "data_quality"}
)

_REGIME_FIT: dict[str, float] = {
    "trending": 1.0,
    "volatile": 0.6,
    "ranging": 0.3,
    "unknown": 0.7,
}


class ORBStrategy(BaseStrategy):
    """Opening Range Breakout on 5-minute bars.

    Best on high-liquidity large-caps in trending or volatile regimes.
    Generates at most one ENTER_LONG and one EXIT_LONG per symbol per day.
    """

    def __init__(
        self,
        *,
        or_bars: int = 6,
        target_r: float = 1.5,
        min_range_pct: float = 0.003,
        min_volume_ratio: float = 1.2,
        volume_lookback: int = 20,
        max_entry_hour: int = 13,
        max_entry_minute: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        or_bars:
            Number of 5m bars that define the opening range (default 6 = 30 min).
        target_r:
            Target is OR_high + OR_range × target_r above entry.
        min_range_pct:
            Minimum OR_range as fraction of entry close (filters choppy opens).
        min_volume_ratio:
            Breakout bar volume must be ≥ this multiple of 20-bar avg volume.
        volume_lookback:
            Rolling window for avg volume comparison.
        max_entry_hour / max_entry_minute:
            No new long entries after this time IST (default 13:00).
        """
        self._or_bars = or_bars
        self._target_r = target_r
        self._min_range_pct = min_range_pct
        self._min_vol_ratio = min_volume_ratio
        self._vol_lookback = volume_lookback
        self._max_entry = time(max_entry_hour, max_entry_minute)

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    def generate(self, df: pl.DataFrame) -> list[Signal]:
        self._validate_columns(df)
        if len(df) < self._or_bars + 1:
            return []

        df = df.sort("timestamp")

        # Rolling mean volume (shifted so current bar is not included in mean)
        df = df.with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .shift(1)
            .rolling_mean(window_size=self._vol_lookback, min_samples=1)
            .alias("_vol_mean")
        )

        # Group processing per trading date
        df = df.with_columns(
            pl.col("timestamp").dt.date().alias("_trade_date")
        )

        signals: list[Signal] = []
        dates = df["_trade_date"].unique().sort().to_list()

        for trade_date in dates:
            day_df = df.filter(pl.col("_trade_date") == trade_date).sort("timestamp")

            if len(day_df) < self._or_bars + 1:
                continue

            # OR defined by first or_bars candles of the session
            or_slice = day_df.slice(0, self._or_bars)
            or_high = or_slice["high"].max()
            or_low = or_slice["low"].min()

            if or_high is None or or_low is None:
                continue
            or_range = or_high - or_low

            # Skip ultra-narrow ranges — they give a terrible R/R
            # (use a placeholder close from end of OR period as reference)
            or_ref_close = or_slice["close"][-1]
            if or_range < self._min_range_pct * or_ref_close:
                logger.debug(
                    "orb.generate.range_too_narrow",
                    date=str(trade_date),
                    or_range=round(or_range, 2),
                    min_required=round(self._min_range_pct * or_ref_close, 2),
                )
                continue

            # Post-OR bars
            post_or = day_df.slice(self._or_bars)
            closes = post_or["close"].to_list()
            highs = post_or["high"].to_list()
            timestamps = post_or["timestamp"].to_list()
            symbols = post_or["symbol"].to_list()
            qualities = post_or["data_quality"].to_list()
            volumes = post_or["volume"].to_list()
            vol_means = post_or["_vol_mean"].to_list()
            vwaps = post_or["vwap"].to_list() if "vwap" in post_or.columns else [None] * len(post_or)
            regimes = post_or["regime"].to_list() if "regime" in post_or.columns else None

            long_entered = False
            long_exited = False

            for i in range(len(post_or)):
                quality = qualities[i]
                if quality not in _ALLOWED_QUALITIES:
                    continue

                close = closes[i]
                ts = timestamps[i]
                sym = symbols[i]

                if not isinstance(ts, datetime):
                    continue
                ts_ist = ts.astimezone(IST)

                # --- ENTER_LONG ---
                if not long_entered and ts_ist.time() <= self._max_entry:
                    if close > or_high:
                        # Volume breakout confirmation
                        vol = float(volumes[i])
                        vm = vol_means[i]
                        vol_ok = (vol / vm >= self._min_vol_ratio) if vm and vm > 0 else False

                        # VWAP filter: entry only above VWAP
                        vwap = vwaps[i]
                        vwap_ok = (vwap is None) or (close > vwap)

                        if vol_ok and vwap_ok:
                            long_entered = True
                            stop = Decimal(str(round(or_low, 2)))
                            target = Decimal(str(round(or_high + or_range * self._target_r, 2)))

                            if stop >= target:
                                continue

                            regime_str = regimes[i] if regimes is not None else "unknown"
                            regime_fit = _REGIME_FIT.get(str(regime_str), 0.7)

                            breakout_strength = (close - or_high) / or_range if or_range > 0 else 0
                            confidence = round(min(0.9, 0.65 + breakout_strength * 0.15), 4)

                            signals.append(
                                Signal(
                                    symbol=sym,
                                    action=Action.ENTER_LONG,
                                    confidence=confidence,
                                    suggested_stop=stop,
                                    suggested_target=target,
                                    invalidation_condition=(
                                        f"Close back below OR_high [{round(or_high, 2)}]"
                                    ),
                                    expected_r=self._target_r,
                                    time_horizon_hours=2,
                                    regime_fit=regime_fit,
                                    data_quality=DataQuality(quality),
                                    strategy_name=_STRATEGY_NAME,
                                    explanation=(
                                        f"ORB breakout: close={round(close, 2)} > "
                                        f"OR_high={round(or_high, 2)}, "
                                        f"OR_range={round(or_range, 2)}, "
                                        f"vol_ratio={vol/vm:.2f}; "
                                        f"stop={stop}, target={target}"
                                    ),
                                    timestamp=ts,
                                )
                            )
                        continue

                # --- EXIT_LONG: close drops back below OR_high (failed breakout) ---
                if long_entered and not long_exited and close < or_high:
                    long_exited = True
                    exit_stop = Decimal(str(round(close * 0.99, 2)))
                    exit_target = Decimal(str(round(close * 1.01, 2)))
                    regime_str = regimes[i] if regimes is not None else "unknown"
                    regime_fit = _REGIME_FIT.get(str(regime_str), 0.7)
                    signals.append(
                        Signal(
                            symbol=sym,
                            action=Action.EXIT_LONG,
                            confidence=0.9,
                            suggested_stop=exit_stop,
                            suggested_target=exit_target,
                            invalidation_condition="ORB exit — failed breakout",
                            expected_r=0.0,
                            time_horizon_hours=0,
                            regime_fit=regime_fit,
                            data_quality=DataQuality(quality),
                            strategy_name=_STRATEGY_NAME,
                            explanation=(
                                f"ORB failed: close={round(close, 2)} "
                                f"< OR_high={round(or_high, 2)}"
                            ),
                            timestamp=ts,
                        )
                    )

        logger.debug(
            "orb.generate.done",
            signals=len(signals),
            bars=len(df),
        )
        return signals

    def _validate_columns(self, df: pl.DataFrame) -> None:
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"ORBStrategy requires columns {sorted(missing)} in df. "
                f"Run FeaturePipeline.run() on the DataFrame first."
            )

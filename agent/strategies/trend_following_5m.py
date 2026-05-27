"""5-minute EMA9 × EMA21 trend-following strategy.

Faster variant of the 15m EMA21 × EMA50 strategy, tuned for intraday 5m bars:
  - EMA9 × EMA21 crossover (responds to trends within 45–105 minutes)
  - ADX14 ≥ 18 (slightly relaxed — 5m noise means strong trends read lower ADX)
  - RSI14 between 45–72 on entry (avoids overbought exhaustion)
  - Close > VWAP on entry (demand above fair value)
  - Stop: 1.5 × ATR14 below entry close
  - Target: 3.0 × ATR14 above entry close (2R)
  - Time horizon: 2 hours
  - No new entries after 13:30 IST (avoid afternoon chop)

The strategy reuses TrendFollowingStrategy's crossover logic via subclassing and
adds RSI + VWAP filters by overriding generate().
"""
from __future__ import annotations

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

_STRATEGY_NAME = "trend_following_5m_v1"

_REGIME_FIT: dict[str, float] = {
    "trending": 1.0,
    "volatile": 0.5,
    "ranging": 0.15,
    "unknown": 0.5,
}

_ALLOWED_QUALITIES = {DataQuality.OK.value, DataQuality.PARTIAL.value}

_REQUIRED_COLS = frozenset(
    {
        "symbol", "timestamp", "close", "volume",
        "ema_9", "ema_21", "adx_14", "atr_14",
        "rsi_14", "data_quality",
    }
)

_NO_ENTRY_AFTER = time(13, 30)


class TrendFollowing5mStrategy(BaseStrategy):
    """EMA9 × EMA21 crossover for 5-minute bars.

    Adds RSI14 and VWAP filters on top of the base crossover logic to reduce
    false entries during exhausted moves.
    """

    def __init__(
        self,
        *,
        min_adx: float = 18.0,
        min_volume_ratio: float = 1.0,
        volume_lookback: int = 20,
        stop_atr_multiple: float = 1.5,
        target_r_multiple: float = 2.0,
        rsi_max_entry: float = 72.0,
        rsi_min_entry: float = 45.0,
    ) -> None:
        self._min_adx = min_adx
        self._min_vol_ratio = min_volume_ratio
        self._vol_lookback = volume_lookback
        self._stop_mult = stop_atr_multiple
        self._target_mult = stop_atr_multiple * target_r_multiple
        self._expected_r = target_r_multiple
        self._rsi_max = rsi_max_entry
        self._rsi_min = rsi_min_entry

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    def generate(self, df: pl.DataFrame) -> list[Signal]:
        self._validate_columns(df)
        if len(df) < 2:
            return []

        df = df.sort("timestamp")

        df = df.with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .shift(1)
            .rolling_mean(window_size=self._vol_lookback, min_samples=1)
            .alias("_vol_mean")
        )

        fast = df["ema_9"].to_list()
        slow = df["ema_21"].to_list()
        adx = df["adx_14"].to_list()
        atr = df["atr_14"].to_list()
        rsi = df["rsi_14"].to_list()
        closes = df["close"].to_list()
        volumes = df["volume"].to_list()
        vol_means = df["_vol_mean"].to_list()
        timestamps = df["timestamp"].to_list()
        qualities = df["data_quality"].to_list()
        symbols = df["symbol"].to_list()
        vwaps = df["vwap"].to_list() if "vwap" in df.columns else [None] * len(df)
        regimes = df["regime"].to_list() if "regime" in df.columns else None

        signals: list[Signal] = []

        for i in range(1, len(df)):
            quality = qualities[i]
            if quality not in _ALLOWED_QUALITIES:
                continue

            prev_fast = fast[i - 1]
            prev_slow = slow[i - 1]
            curr_fast = fast[i]
            curr_slow = slow[i]
            curr_adx = adx[i]
            curr_atr = atr[i]
            curr_close = closes[i]
            curr_rsi = rsi[i]
            curr_vol = float(volumes[i])
            curr_vol_mean = vol_means[i]
            vwap = vwaps[i]
            ts = timestamps[i]
            symbol = symbols[i]

            if any(
                v is None
                for v in [prev_fast, prev_slow, curr_fast, curr_slow,
                          curr_adx, curr_atr, curr_rsi]
            ):
                continue

            if not isinstance(ts, datetime):
                continue
            ts_ist = ts.astimezone(IST)

            regime_str = regimes[i] if regimes is not None else "unknown"
            regime_fit = _REGIME_FIT.get(str(regime_str), 0.5)

            # --- ENTER_LONG ---
            golden_cross = prev_fast < prev_slow and curr_fast >= curr_slow
            adx_ok = curr_adx >= self._min_adx
            vol_ratio = (curr_vol / curr_vol_mean) if curr_vol_mean and curr_vol_mean > 0 else 0.0
            volume_ok = vol_ratio >= self._min_vol_ratio
            rsi_ok = self._rsi_min <= curr_rsi <= self._rsi_max
            vwap_ok = (vwap is None) or (curr_close > vwap)
            time_ok = ts_ist.time() <= _NO_ENTRY_AFTER

            if golden_cross and adx_ok and volume_ok and rsi_ok and vwap_ok and time_ok:
                stop = Decimal(str(round(curr_close - self._stop_mult * curr_atr, 2)))
                target = Decimal(str(round(curr_close + self._target_mult * curr_atr, 2)))
                if stop >= target:
                    logger.warning(
                        "tf5m.enter_long.invalid_stop_target",
                        symbol=symbol,
                        stop=str(stop),
                        target=str(target),
                    )
                    continue
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=Action.ENTER_LONG,
                        confidence=self._confidence(curr_adx),
                        suggested_stop=stop,
                        suggested_target=target,
                        invalidation_condition=(
                            f"Close below EMA9 [{round(curr_fast, 2)}] or stop [{stop}]"
                        ),
                        expected_r=self._expected_r,
                        time_horizon_hours=2,
                        regime_fit=regime_fit,
                        data_quality=DataQuality(quality),
                        strategy_name=_STRATEGY_NAME,
                        explanation=(
                            f"EMA9 crossed above EMA21 "
                            f"(ADX={curr_adx:.1f}, RSI={curr_rsi:.1f}, vol_ratio={vol_ratio:.2f}); "
                            f"stop={stop}, target={target}"
                        ),
                        timestamp=ts,
                    )
                )
                continue

            # --- EXIT_LONG ---
            death_cross = prev_fast > prev_slow and curr_fast <= curr_slow
            prev_close = closes[i - 1]
            in_uptrend_prev = prev_fast > prev_slow
            close_below_fast = (
                in_uptrend_prev
                and prev_close >= prev_fast
                and curr_close < curr_fast
            )
            rsi_overbought_exit = curr_rsi > 78.0

            if death_cross or close_below_fast or rsi_overbought_exit:
                exit_stop = Decimal(str(round(curr_close * 0.99, 2)))
                exit_target = Decimal(str(round(curr_close * 1.01, 2)))
                if death_cross:
                    reason = "EMA9 crossed below EMA21 (death cross)"
                    conf = 0.85
                elif rsi_overbought_exit:
                    reason = f"RSI overbought exit ({curr_rsi:.1f} > 78)"
                    conf = 0.7
                else:
                    reason = f"Close ({curr_close}) fell below EMA9 ({curr_fast:.2f})"
                    conf = 0.65
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=Action.EXIT_LONG,
                        confidence=conf,
                        suggested_stop=exit_stop,
                        suggested_target=exit_target,
                        invalidation_condition="Exit signal — close position",
                        expected_r=0.0,
                        time_horizon_hours=0,
                        regime_fit=regime_fit,
                        data_quality=DataQuality(quality),
                        strategy_name=_STRATEGY_NAME,
                        explanation=reason,
                        timestamp=ts,
                    )
                )

        logger.debug(
            "tf5m.generate.done",
            symbol=df["symbol"][0] if len(df) > 0 else "?",
            signals=len(signals),
            bars=len(df),
        )
        return signals

    def _confidence(self, adx: float) -> float:
        adx_range = 60.0 - self._min_adx
        if adx_range <= 0:
            return 0.5
        normalized = (adx - self._min_adx) / adx_range
        return round(max(0.5, min(0.88, 0.5 + normalized * 0.38)), 4)

    def _validate_columns(self, df: pl.DataFrame) -> None:
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"TrendFollowing5mStrategy requires columns {sorted(missing)} in df. "
                f"Run FeaturePipeline.run() on the DataFrame first."
            )

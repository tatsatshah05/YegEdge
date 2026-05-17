from __future__ import annotations

from decimal import Decimal

import polars as pl
import structlog

from agent.data.types import DataQuality
from agent.strategies.base import BaseStrategy
from agent.strategies.types import Action, Signal

logger = structlog.get_logger()

_STRATEGY_NAME = "trend_following_v1"

_REGIME_FIT: dict[str, float] = {
    "trending": 1.0,
    "volatile": 0.4,
    "ranging": 0.1,
    "unknown": 0.5,
}

_ALLOWED_QUALITIES = {DataQuality.OK.value, DataQuality.PARTIAL.value}

# Columns the strategy requires in the input DataFrame.
_REQUIRED_COLS = frozenset(
    {
        "symbol",
        "timestamp",
        "close",
        "volume",
        "ema_21",
        "ema_50",
        "adx_14",
        "atr_14",
        "data_quality",
    }
)


class TrendFollowingStrategy(BaseStrategy):
    """EMA21 x EMA50 crossover with ADX and volume confirmation.

    Entry (ENTER_LONG):
    - EMA21 crosses above EMA50
    - ADX14 >= min_adx (default 20)
    - volume >= min_volume_ratio x 20-bar rolling mean volume (default 1.1x)
    - data_quality in {ok, partial}

    Exit (EXIT_LONG):
    - EMA21 crosses below EMA50 (death cross), OR
    - close drops below EMA21

    Stop: close - stop_atr_multiple * ATR14 (default 2.0x)
    Target: close + stop_atr_multiple * target_r_multiple * ATR14 (default 4.0x)
    Expected R: target_r_multiple (default 2.0)

    Parameters match config/strategies.yaml defaults.
    """

    def __init__(
        self,
        *,
        fast_ema_col: str = "ema_21",
        slow_ema_col: str = "ema_50",
        adx_col: str = "adx_14",
        atr_col: str = "atr_14",
        min_adx: float = 20.0,
        min_volume_ratio: float = 1.0,
        volume_lookback: int = 20,
        stop_atr_multiple: float = 2.0,
        target_r_multiple: float = 2.0,
        time_horizon_hours: int = 4,
    ) -> None:
        self._fast = fast_ema_col
        self._slow = slow_ema_col
        self._adx_col = adx_col
        self._atr_col = atr_col
        self._min_adx = min_adx
        self._min_vol_ratio = min_volume_ratio
        self._vol_lookback = volume_lookback
        self._stop_mult = stop_atr_multiple
        self._target_mult = stop_atr_multiple * target_r_multiple
        self._expected_r = target_r_multiple
        self._horizon = time_horizon_hours

    @property
    def name(self) -> str:
        return _STRATEGY_NAME

    def generate(self, df: pl.DataFrame) -> list[Signal]:
        """Scan *df* and return a Signal for every bar where entry/exit conditions are met.

        Callers should pass the full history for backtesting.
        For live trading, pass the last N bars (at minimum volume_lookback + 2).
        """
        self._validate_columns(df)

        if len(df) < 2:
            return []

        # Sort so crossover detection uses chronological order.
        df = df.sort("timestamp")

        # Rolling mean of the PRECEDING volume_lookback bars (not including the current bar).
        # We shift by 1 first so bar[i]'s mean is computed from bars [i-lookback, i-1].
        # This avoids look-ahead: the current bar's volume is what we compare the mean against.
        df = df.with_columns(
            pl.col("volume")
            .cast(pl.Float64)
            .shift(1)
            .rolling_mean(window_size=self._vol_lookback, min_samples=1)
            .alias("_vol_mean")
        )

        fast = df[self._fast].to_list()
        slow = df[self._slow].to_list()
        adx = df[self._adx_col].to_list()
        atr = df[self._atr_col].to_list()
        closes = df["close"].to_list()
        volumes = df["volume"].to_list()
        vol_means = df["_vol_mean"].to_list()
        timestamps = df["timestamp"].to_list()
        qualities = df["data_quality"].to_list()
        symbols = df["symbol"].to_list()
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
            curr_vol = float(volumes[i])
            curr_vol_mean = vol_means[i]
            ts = timestamps[i]
            symbol = symbols[i]

            # Skip warm-up bars where any indicator is None
            if any(
                v is None for v in [prev_fast, prev_slow, curr_fast, curr_slow, curr_adx, curr_atr]
            ):
                continue

            regime_str = regimes[i] if regimes is not None else "unknown"
            regime_fit = _REGIME_FIT.get(str(regime_str), 0.5)

            # -----------------------------------------------------------
            # ENTER_LONG: golden cross + ADX filter + volume confirmation
            # -----------------------------------------------------------
            golden_cross = prev_fast < prev_slow and curr_fast >= curr_slow
            adx_ok = curr_adx >= self._min_adx
            vol_ratio = (curr_vol / curr_vol_mean) if curr_vol_mean and curr_vol_mean > 0 else 0.0
            volume_ok = vol_ratio >= self._min_vol_ratio

            if golden_cross and adx_ok and volume_ok:
                stop = Decimal(str(round(curr_close - self._stop_mult * curr_atr, 2)))
                target = Decimal(str(round(curr_close + self._target_mult * curr_atr, 2)))
                if stop >= target:
                    logger.warning(
                        "trend_following.enter_long.invalid_stop_target",
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
                            f"Close below EMA21 [{round(curr_fast, 2)}] or stop [{stop}]"
                        ),
                        expected_r=self._expected_r,
                        time_horizon_hours=self._horizon,
                        regime_fit=regime_fit,
                        data_quality=DataQuality(quality),
                        strategy_name=_STRATEGY_NAME,
                        explanation=(
                            f"EMA21 crossed above EMA50 "
                            f"(ADX={curr_adx:.1f}, vol_ratio={vol_ratio:.2f}); "
                            f"stop={stop}, target={target}"
                        ),
                        timestamp=ts,
                    )
                )
                continue  # don't also check exit on the same bar as entry

            # -----------------------------------------------------------
            # EXIT_LONG: death cross OR close crossing below EMA21
            # close_below_fast only fires on a NEW transition (prev close >= EMA21,
            # curr close < EMA21) while already in an uptrend (EMA21 > EMA50 prev bar).
            # This prevents spurious exits when close was already below EMA21 without
            # an active long position.
            # -----------------------------------------------------------
            death_cross = prev_fast > prev_slow and curr_fast <= curr_slow
            prev_close = closes[i - 1]
            in_uptrend_prev = prev_fast > prev_slow
            close_below_fast = (
                in_uptrend_prev
                and prev_close >= prev_fast  # was above or at EMA21
                and curr_close < curr_fast  # now dropped below EMA21
            )

            if death_cross or close_below_fast:
                # Dummy stop/target to satisfy Signal invariant (stop < target).
                # The decision engine treats EXIT_LONG as "close the position immediately".
                exit_stop = Decimal(str(round(curr_close * 0.99, 2)))
                exit_target = Decimal(str(round(curr_close * 1.01, 2)))
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=Action.EXIT_LONG,
                        confidence=0.8 if death_cross else 0.6,
                        suggested_stop=exit_stop,
                        suggested_target=exit_target,
                        invalidation_condition="Exit signal — close position",
                        expected_r=0.0,
                        time_horizon_hours=0,
                        regime_fit=regime_fit,
                        data_quality=DataQuality(quality),
                        strategy_name=_STRATEGY_NAME,
                        explanation=(
                            "EMA21 crossed below EMA50 (death cross)"
                            if death_cross
                            else f"Close ({curr_close}) fell below EMA21 ({curr_fast:.2f})"
                        ),
                        timestamp=ts,
                    )
                )

        logger.debug(
            "trend_following.generate.done",
            symbol=df["symbol"][0] if len(df) > 0 else "?",
            signals=len(signals),
            bars=len(df),
        )
        return signals

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pl.DataFrame) -> None:
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"TrendFollowingStrategy requires columns {sorted(missing)} in df. "
                f"Run FeaturePipeline.run() on the DataFrame first."
            )

    def _confidence(self, adx: float) -> float:
        """Map ADX to confidence in [0.5, 0.9]. Linear between min_adx and 60."""
        adx_range = 60.0 - self._min_adx
        if adx_range <= 0:
            return 0.5
        normalized = (adx - self._min_adx) / adx_range
        return round(max(0.5, min(0.9, 0.5 + normalized * 0.4)), 4)

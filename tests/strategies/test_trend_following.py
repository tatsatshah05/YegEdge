from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.data.types import DataQuality
from agent.strategies.trend_following import TrendFollowingStrategy
from agent.strategies.types import Action

IST = ZoneInfo("Asia/Kolkata")


def _ts(i: int) -> datetime:
    return datetime(2024, 1, 2, 9, 15, tzinfo=IST) + timedelta(hours=i)


def _make_df(rows: list[dict]) -> pl.DataFrame:
    """Build an enriched-style DataFrame from a list of row dicts.

    Defaults: data_quality="ok", regime="trending", volume=100_000,
    atr_14=10.0.  Pass explicit values to override.
    """
    n = len(rows)
    filled: list[dict] = []
    for i, r in enumerate(rows):
        filled.append(
            {
                "symbol": r.get("symbol", "TEST"),
                "timeframe": "60m",
                "timestamp": r.get("timestamp", _ts(i)),
                "open": float(r.get("close", 1000.0)) * 0.999,
                "high": float(r.get("close", 1000.0)) * 1.005,
                "low": float(r.get("close", 1000.0)) * 0.995,
                "close": float(r.get("close", 1000.0)),
                "volume": int(r.get("volume", 100_000)),
                "value": float(r.get("close", 1000.0)) * float(r.get("volume", 100_000)),
                "ema_21": float(r["ema_21"]),
                "ema_50": float(r["ema_50"]),
                "adx_14": float(r.get("adx_14", 25.0)),
                "atr_14": float(r.get("atr_14", 10.0)),
                "data_quality": str(r.get("data_quality", "ok")),
                "regime": str(r.get("regime", "trending")),
            }
        )
    return pl.DataFrame(filled)


def _golden_cross_df(*, adx: float = 25.0, volume: int = 120_000) -> pl.DataFrame:
    """Two-bar DataFrame where bar[1] is a golden cross (EMA21 crosses above EMA50)."""
    return _make_df(
        [
            {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": adx, "volume": volume},
            {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": adx, "volume": volume},
        ]
    )


def _death_cross_df() -> pl.DataFrame:
    """Two-bar DataFrame where bar[1] is a death cross (EMA21 crosses below EMA50)."""
    return _make_df(
        [
            {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
            {"ema_21": 1005.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
        ]
    )


def test_strategy_name() -> None:
    assert TrendFollowingStrategy().name == "trend_following_v1"


def test_generate_raises_on_missing_required_columns() -> None:
    df = pl.DataFrame({"close": [1000.0]})
    with pytest.raises(ValueError, match="ema_21"):
        TrendFollowingStrategy().generate(df)


def test_generate_returns_empty_for_single_bar() -> None:
    df = _make_df([{"ema_21": 1000.0, "ema_50": 1010.0}])
    assert TrendFollowingStrategy().generate(df) == []


def test_generate_returns_empty_for_empty_df() -> None:
    df = _make_df([{"ema_21": 1000.0, "ema_50": 1010.0}]).clear()
    assert TrendFollowingStrategy().generate(df) == []


def test_generate_enter_long_on_golden_cross() -> None:
    df = _golden_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    assert len(signals) == 1
    assert signals[0].action == Action.ENTER_LONG
    assert signals[0].symbol == "TEST"


def test_generate_no_enter_long_when_adx_too_low() -> None:
    df = _golden_cross_df(adx=15.0)
    signals = TrendFollowingStrategy().generate(df)
    assert signals == []


def test_generate_no_enter_long_when_volume_too_low() -> None:
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 100_000},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 50_000},
    ]
    df = _make_df(rows)
    signals = TrendFollowingStrategy().generate(df)
    enter_signals = [s for s in signals if s.action == Action.ENTER_LONG]
    assert enter_signals == [], f"Expected no ENTER_LONG, got {enter_signals}"


def test_generate_no_enter_long_when_ema21_already_above_ema50() -> None:
    df = _make_df(
        [
            {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
            {"ema_21": 1020.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
        ]
    )
    signals = TrendFollowingStrategy().generate(df)
    assert signals == []


def test_generate_enter_long_stop_less_than_target() -> None:
    df = _golden_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    for sig in signals:
        assert sig.suggested_stop < sig.suggested_target


def test_generate_enter_long_expected_r_equals_target_r_multiple() -> None:
    strategy = TrendFollowingStrategy(target_r_multiple=2.0)
    df = _golden_cross_df()
    signals = strategy.generate(df)
    enter = [s for s in signals if s.action == Action.ENTER_LONG]
    assert len(enter) == 1
    assert enter[0].expected_r == 2.0


def test_generate_enter_long_strategy_name() -> None:
    signals = TrendFollowingStrategy().generate(_golden_cross_df())
    assert signals[0].strategy_name == "trend_following_v1"


def test_generate_confidence_scales_with_adx() -> None:
    low_adx_df = _golden_cross_df(adx=21.0)
    high_adx_df = _golden_cross_df(adx=55.0)
    strategy = TrendFollowingStrategy()
    low_sig = strategy.generate(low_adx_df)
    high_sig = strategy.generate(high_adx_df)
    assert len(low_sig) == 1 and len(high_sig) == 1
    assert low_sig[0].confidence < high_sig[0].confidence


def test_generate_confidence_capped_at_0_9() -> None:
    df = _golden_cross_df(adx=200.0)
    signals = TrendFollowingStrategy().generate(df)
    assert signals[0].confidence <= 0.9


def test_generate_confidence_floor_at_0_5() -> None:
    df = _golden_cross_df(adx=20.0)
    signals = TrendFollowingStrategy().generate(df)
    assert signals[0].confidence >= 0.5


def test_generate_regime_fit_trending_is_1() -> None:
    df = _golden_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    assert signals[0].regime_fit == 1.0


def test_generate_regime_fit_volatile() -> None:
    rows = [
        {
            "ema_21": 1000.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "regime": "volatile",
        },
        {
            "ema_21": 1015.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "regime": "volatile",
        },
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals[0].regime_fit == 0.4


def test_generate_regime_fit_ranging_is_low() -> None:
    rows = [
        {
            "ema_21": 1000.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "regime": "ranging",
        },
        {
            "ema_21": 1015.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "regime": "ranging",
        },
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals[0].regime_fit == 0.1


def test_generate_works_without_regime_column() -> None:
    rows = [
        {"ema_21": 1000.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
        {"ema_21": 1015.0, "ema_50": 1010.0, "adx_14": 25.0, "volume": 120_000},
    ]
    df = _make_df(rows).drop("regime")
    signals = TrendFollowingStrategy().generate(df)
    assert len(signals) == 1
    assert signals[0].regime_fit == 0.5


def test_generate_skips_suspect_bars() -> None:
    rows = [
        {
            "ema_21": 1000.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "data_quality": "suspect",
        },
        {
            "ema_21": 1015.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "data_quality": "suspect",
        },
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals == []


def test_generate_skips_missing_bars() -> None:
    rows = [
        {
            "ema_21": 1000.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "data_quality": "missing",
        },
        {
            "ema_21": 1015.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "data_quality": "missing",
        },
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert signals == []


def test_generate_accepts_partial_quality_bars() -> None:
    rows = [
        {
            "ema_21": 1000.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "data_quality": "partial",
        },
        {
            "ema_21": 1015.0,
            "ema_50": 1010.0,
            "adx_14": 25.0,
            "volume": 120_000,
            "data_quality": "partial",
        },
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    assert len(signals) == 1
    assert signals[0].data_quality == DataQuality.PARTIAL


def test_generate_exit_long_on_death_cross() -> None:
    df = _death_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    assert len(signals) == 1
    assert signals[0].action == Action.EXIT_LONG


def test_generate_exit_long_when_close_below_ema21() -> None:
    rows = [
        {"ema_21": 1010.0, "ema_50": 1005.0, "close": 1012.0, "adx_14": 25.0, "volume": 120_000},
        {"ema_21": 1010.0, "ema_50": 1005.0, "close": 1008.0, "adx_14": 25.0, "volume": 120_000},
    ]
    signals = TrendFollowingStrategy().generate(_make_df(rows))
    exits = [s for s in signals if s.action == Action.EXIT_LONG]
    assert len(exits) == 1


def test_generate_exit_long_stop_less_than_target() -> None:
    df = _death_cross_df()
    signals = TrendFollowingStrategy().generate(df)
    for sig in signals:
        if sig.action == Action.EXIT_LONG:
            assert sig.suggested_stop < sig.suggested_target

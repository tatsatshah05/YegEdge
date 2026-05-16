from __future__ import annotations

from zoneinfo import ZoneInfo

import polars as pl
import pytest

from agent.features.indicators import add_adx, add_atr
from agent.features.regime import Regime, RegimeDetector
from tests.features.conftest import make_ohlcv_df

IST = ZoneInfo("Asia/Kolkata")


def _enriched(closes: list[float]) -> pl.DataFrame:
    """Build a feature-enriched DataFrame with adx_14 and atr_14 columns."""
    df = make_ohlcv_df(closes)
    df = add_atr(df, period=14)
    df = add_adx(df, period=14)
    return df


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------


def test_regime_values_are_strings() -> None:
    assert Regime.TRENDING == "trending"
    assert Regime.RANGING == "ranging"
    assert Regime.VOLATILE == "volatile"
    assert Regime.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# RegimeDetector — before fit
# ---------------------------------------------------------------------------


def test_regime_detector_not_fit_by_default() -> None:
    rd = RegimeDetector()
    assert rd.is_fit is False


def test_predict_before_fit_returns_unknown() -> None:
    """predict() before fit() must tag every row as UNKNOWN."""
    rd = RegimeDetector()
    df = _enriched([1000.0 + i * 5 for i in range(40)])
    result = rd.predict(df)
    assert "regime" in result.columns
    assert all(v == Regime.UNKNOWN.value for v in result["regime"].to_list())


# ---------------------------------------------------------------------------
# RegimeDetector — fit
# ---------------------------------------------------------------------------


def test_fit_with_enough_data_marks_is_fit() -> None:
    rd = RegimeDetector()
    df = _enriched([1000.0 + i * 5 for i in range(100)])
    rd.fit(df)
    assert rd.is_fit is True


def test_fit_with_insufficient_data_leaves_unfit() -> None:
    """fit() with fewer than 60 non-null rows must leave the detector unfit."""
    rd = RegimeDetector()
    # Only 30 bars total — all 30 are non-null after EWM smoothing (EWM produces no
    # leading nulls), but 30 < _MIN_FIT_ROWS=60 so the detector stays unfit.
    df = _enriched([1000.0 + i for i in range(30)])
    rd.fit(df)
    assert rd.is_fit is False


def test_fit_invalid_n_regimes_raises() -> None:
    with pytest.raises(ValueError, match="n_regimes"):
        RegimeDetector(n_regimes=1)
    with pytest.raises(ValueError, match="n_regimes"):
        RegimeDetector(n_regimes=5)


# ---------------------------------------------------------------------------
# RegimeDetector — predict after fit
# ---------------------------------------------------------------------------


def test_predict_after_fit_returns_valid_regime_values() -> None:
    """After fit(), every row must have a valid Regime value."""
    closes = [1000.0 + i * 10 for i in range(120)]
    df = _enriched(closes)
    rd = RegimeDetector()
    rd.fit(df)
    result = rd.predict(df)
    valid_values = {r.value for r in Regime}
    for v in result["regime"].to_list():
        assert v in valid_values, f"Unexpected regime value: {v!r}"


def test_predict_adds_regime_column_without_modifying_input() -> None:
    """predict() must not modify the input DataFrame."""
    df = _enriched([1000.0 + i * 5 for i in range(100)])
    rd = RegimeDetector()
    rd.fit(df)
    original_cols = set(df.columns)
    result = rd.predict(df)
    assert set(df.columns) == original_cols, "Input DataFrame was modified"
    assert "regime" in result.columns
    assert "regime" not in df.columns


def test_predict_trending_label_on_strong_uptrend() -> None:
    """A strong, sustained uptrend should be classified as TRENDING."""
    closes = [1000.0 + i * 20 for i in range(200)]
    df = _enriched(closes)
    rd = RegimeDetector()
    rd.fit(df)
    result = rd.predict(df)
    tail_regimes = result["regime"].tail(50).to_list()
    trending_count = sum(1 for r in tail_regimes if r == Regime.TRENDING.value)
    assert trending_count > 30, f"Expected majority TRENDING on uptrend, got {tail_regimes[-10:]}"


def test_regime_column_dtype_is_string() -> None:
    df = _enriched([1000.0 + i * 5 for i in range(100)])
    rd = RegimeDetector()
    rd.fit(df)
    result = rd.predict(df)
    assert result["regime"].dtype == pl.Utf8, f"Expected Utf8, got {result['regime'].dtype}"


# ---------------------------------------------------------------------------
# Edge cases: NaN safety and empty DataFrame
# ---------------------------------------------------------------------------


def test_predict_empty_dataframe_returns_unknown() -> None:
    """predict() on an empty DataFrame must return an empty DataFrame with regime=UNKNOWN dtype."""
    df = _enriched([1000.0 + i * 5 for i in range(100)])
    rd = RegimeDetector()
    rd.fit(df)
    empty = df.clear()  # same schema, zero rows
    result = rd.predict(empty)
    assert len(result) == 0
    assert "regime" in result.columns


def test_fit_and_predict_tolerates_nan_in_features() -> None:
    """NaN values in adx_14/atr_14 must not crash fit() or predict()."""
    closes = [1000.0 + i * 5 for i in range(100)]
    df = _enriched(closes)
    # Inject NaN directly into the adx_14 column for a few rows
    adx_with_nan = df["adx_14"].to_list()
    adx_with_nan[0] = float("nan")
    adx_with_nan[10] = float("nan")
    df = df.with_columns(pl.Series("adx_14", adx_with_nan, dtype=pl.Float64))

    rd = RegimeDetector()
    rd.fit(df)   # must not raise
    assert rd.is_fit is True
    result = rd.predict(df)  # must not raise
    valid_values = {r.value for r in Regime}
    for v in result["regime"].to_list():
        assert v in valid_values

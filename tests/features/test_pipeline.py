from __future__ import annotations

import polars as pl

from agent.features.pipeline import FeaturePipeline
from agent.features.regime import Regime, RegimeDetector
from tests.features.conftest import make_ohlcv_df

_EXPECTED_INDICATOR_COLS = {
    "ema_9",
    "ema_21",
    "ema_50",
    "rsi_14",
    "atr_14",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "vwap",
}


def _large_df(n: int = 150) -> pl.DataFrame:
    """150 bars of rising price — enough for all indicators to stabilize."""
    return make_ohlcv_df([1000.0 + i * 5 for i in range(n)])


# ---------------------------------------------------------------------------
# FeaturePipeline — without regime
# ---------------------------------------------------------------------------


def test_pipeline_adds_all_indicator_columns() -> None:
    df = _large_df()
    result = FeaturePipeline().run(df)
    for col in _EXPECTED_INDICATOR_COLS:
        assert col in result.columns, f"Missing column: {col}"


def test_pipeline_does_not_modify_input() -> None:
    df = _large_df()
    original_cols = set(df.columns)
    _ = FeaturePipeline().run(df)
    assert set(df.columns) == original_cols


def test_pipeline_no_temp_columns_leaked() -> None:
    df = _large_df()
    result = FeaturePipeline().run(df)
    leaked = [c for c in result.columns if c.startswith("_")]
    assert leaked == [], f"Leaked temp columns: {leaked}"


def test_pipeline_preserves_row_count() -> None:
    df = _large_df(n=80)
    result = FeaturePipeline().run(df)
    assert len(result) == len(df)


def test_pipeline_without_regime_has_no_regime_column() -> None:
    df = _large_df()
    result = FeaturePipeline().run(df)
    assert "regime" not in result.columns


# ---------------------------------------------------------------------------
# FeaturePipeline — with regime
# ---------------------------------------------------------------------------


def test_pipeline_with_regime_detector_adds_regime_column() -> None:
    df = _large_df(n=150)
    rd = RegimeDetector()
    # fit() requires adx_14/atr_14 — must enrich first
    enriched = FeaturePipeline().run(df)
    rd.fit(enriched)
    pipeline = FeaturePipeline(regime_detector=rd)
    result = pipeline.run(df)
    assert "regime" in result.columns


def test_pipeline_regime_values_are_valid() -> None:
    df = _large_df(n=150)
    rd = RegimeDetector()
    pipeline = FeaturePipeline(regime_detector=rd)
    enriched = FeaturePipeline().run(df)
    rd.fit(enriched)
    result = pipeline.run(df)
    valid_values = {r.value for r in Regime}
    for v in result["regime"].to_list():
        assert v in valid_values, f"Invalid regime: {v!r}"


def test_pipeline_with_unfit_regime_detector_returns_unknown() -> None:
    """An unfit RegimeDetector attached to the pipeline should produce UNKNOWN for all rows."""
    df = _large_df()
    rd = RegimeDetector()  # not fit
    pipeline = FeaturePipeline(regime_detector=rd)
    result = pipeline.run(df)
    assert "regime" in result.columns
    assert all(v == Regime.UNKNOWN.value for v in result["regime"].to_list())


# ---------------------------------------------------------------------------
# FeaturePipeline — edge cases
# ---------------------------------------------------------------------------


def test_pipeline_on_minimum_viable_df() -> None:
    """Pipeline should not raise on a very small DataFrame (just produce many nulls)."""
    df = make_ohlcv_df([1000.0, 1001.0, 1002.0])
    result = FeaturePipeline().run(df)
    for col in _EXPECTED_INDICATOR_COLS:
        assert col in result.columns

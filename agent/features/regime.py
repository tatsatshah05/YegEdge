from __future__ import annotations

from enum import StrEnum

import numpy as np
import polars as pl
import structlog
from sklearn.cluster import KMeans

logger = structlog.get_logger()

_MIN_FIT_ROWS: int = 60


class Regime(StrEnum):
    """Market regime label assigned by :class:`RegimeDetector`."""

    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class RegimeDetector:
    """Classify market regime using KMeans clustering.

    Features used (all from the output of ``add_atr`` + ``add_adx``):

    - ``adx_14``: trend strength (0-100)
    - ``atr_14 / close * 100``: normalised volatility (ATR as % of price)
    - ``close.pct_change(20) * 100``: 20-bar momentum

    Cluster → Regime mapping (post-hoc, based on centroid values):

    - Highest ADX centroid → ``TRENDING``
    - Of the remainder, highest ATR% centroid → ``VOLATILE``
    - Remaining cluster(s) → ``RANGING``

    Call :meth:`fit` on historical data before calling :meth:`predict`.
    ``predict`` returns ``UNKNOWN`` for every row when called before ``fit``.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        if not (2 <= n_regimes <= 4):
            raise ValueError(f"n_regimes must be between 2 and 4, got {n_regimes}")
        self._n = n_regimes
        self._random_state = random_state
        self._model: KMeans | None = None
        self._label_map: dict[int, Regime] = {}

    @property
    def is_fit(self) -> bool:
        return self._model is not None

    def fit(self, df: pl.DataFrame) -> None:
        """Fit KMeans on *df*.  Requires ``adx_14``, ``atr_14``, ``close`` columns.

        Drops rows with null values in any feature column before fitting.
        Does nothing (leaves detector unfit) when fewer than
        ``_MIN_FIT_ROWS`` clean rows are available.
        """
        required = ["adx_14", "atr_14", "close"]
        clean = df.drop_nulls(subset=required)
        if len(clean) < _MIN_FIT_ROWS:
            logger.warning(
                "regime_detector.fit.insufficient_data",
                rows=len(clean),
                minimum=_MIN_FIT_ROWS,
            )
            return
        x = self._build_features(clean)
        model = KMeans(n_clusters=self._n, random_state=self._random_state, n_init=10)
        model.fit(x)
        self._label_map = self._assign_labels(model.cluster_centers_)
        self._model = model
        logger.info("regime_detector.fit.done", n_regimes=self._n, rows=len(clean))

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add a ``regime`` column to *df*.

        Returns ``UNKNOWN`` for every row when the detector has not been fit or
        when *df* is empty.  Null-feature rows are filled with zeros before
        prediction (the strategy layer should discard early-bar rows with null
        indicators anyway).  Overwrites any existing ``regime`` column.
        """
        if self._model is None or len(df) == 0:
            return df.with_columns(pl.lit(Regime.UNKNOWN.value).alias("regime"))
        x = self._build_features(df)
        clusters = self._model.predict(x)
        regime_values = [self._label_map[int(c)].value for c in clusters]
        return df.with_columns(pl.Series("regime", regime_values, dtype=pl.Utf8))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_features(self, df: pl.DataFrame) -> np.ndarray:
        """Return a (n_rows, 3) float64 array: [adx, atr_pct, momentum_20].

        NaN guard: fill_nan().fill_null() covers both Polars NaN and null so
        sklearn never receives NaN/inf from zero-close or zero-prior-close rows.
        """
        adx = df["adx_14"].fill_nan(0.0).fill_null(0.0).to_numpy().astype(np.float64)
        atr_pct = (
            (df["atr_14"] / df["close"] * 100.0)
            .fill_nan(0.0)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float64)
        )
        momentum = (
            (df["close"] / df["close"].shift(20) - 1.0)
            .fill_nan(0.0)
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float64)
        ) * 100.0
        # Clip inf produced by division by zero (e.g. zero prior-close)
        adx = np.clip(adx, -1e9, 1e9)
        atr_pct = np.clip(atr_pct, -1e9, 1e9)
        momentum = np.clip(momentum, -1e9, 1e9)
        return np.column_stack([adx, atr_pct, momentum])

    def _assign_labels(self, centers: np.ndarray) -> dict[int, Regime]:
        """Map cluster indices to Regime enum values based on centroid characteristics.

        centers shape: (n_clusters, 3) = [adx, atr_pct, momentum]
        """
        adx_col = centers[:, 0]
        atr_col = centers[:, 1]
        n = len(centers)

        trending_idx = int(np.argmax(adx_col))
        remaining = [i for i in range(n) if i != trending_idx]

        if len(remaining) == 1:
            # n_regimes == 2: only trending + ranging
            return {trending_idx: Regime.TRENDING, remaining[0]: Regime.RANGING}

        volatile_idx = remaining[int(np.argmax(atr_col[remaining]))]
        ranging_indices = [i for i in remaining if i != volatile_idx]

        label_map: dict[int, Regime] = {
            trending_idx: Regime.TRENDING,
            volatile_idx: Regime.VOLATILE,
        }
        for idx in ranging_indices:
            label_map[idx] = Regime.RANGING
        return label_map

from __future__ import annotations

from decimal import Decimal

import polars as pl

from agent.data.types import DataQuality

# Single-bar price move exceeding this fraction of open is suspicious.
_OUTLIER_MOVE_THRESHOLD = Decimal("0.50")


class DataValidator:
    """Annotate each bar in a Polars DataFrame with a DataQuality tag.

    The returned DataFrame has all input columns plus a ``data_quality`` column.
    If a ``data_quality`` column is already present it is replaced.

    Rules applied in priority order — the worst quality tag wins.

    Priority (highest wins): SUSPECT > PARTIAL > OK
    """

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        n = len(df)
        qualities: list[str] = [DataQuality.OK.value] * n

        for i in range(n):
            row = df.row(i, named=True)
            quality = self._check_row(row)
            qualities[i] = quality.value

        return df.with_columns(pl.Series("data_quality", qualities))

    def _check_row(self, row: dict[str, object]) -> DataQuality:
        o = Decimal(str(row["open"]))
        h = Decimal(str(row["high"]))
        lo = Decimal(str(row["low"]))
        c = Decimal(str(row["close"]))
        vol = int(str(row["volume"]))

        # Zero or negative prices — data corruption
        if any(p <= 0 for p in (o, h, lo, c)):
            return DataQuality.SUSPECT

        # OHLC sanity: high must be >= all price values; low must be <= all price values
        if h < max(o, c) or h < lo:
            return DataQuality.SUSPECT
        if lo > min(o, c) or lo > h:
            return DataQuality.SUSPECT

        # Outlier: close moved more than 50% from open in a single bar
        if o > 0 and abs(c - o) / o > _OUTLIER_MOVE_THRESHOLD:
            return DataQuality.SUSPECT

        # Zero volume — bar exists but no trading activity
        if vol == 0:
            return DataQuality.PARTIAL

        return DataQuality.OK

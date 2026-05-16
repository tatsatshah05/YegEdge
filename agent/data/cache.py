from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import structlog

log = structlog.get_logger()
_IST = ZoneInfo("Asia/Kolkata")


class ParquetCache:
    """Year-partitioned Parquet cache for OHLCV bars.

    Layout: {root}/{timeframe}/{year}/{SYMBOL}.parquet

    Each file holds all bars for a single symbol within a single calendar year.
    Incremental writes merge new bars with existing data, deduplicating by
    (symbol, timeframe, timestamp) and keeping the latest version of any duplicate.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, df: pl.DataFrame, *, symbol: str, timeframe: str) -> None:
        """Write (or merge) bars for a symbol+timeframe, partitioned by calendar year.

        If a year file already exists, the new bars are merged with the existing data:
        duplicates (matched on symbol, timeframe, timestamp) are resolved by keeping
        the last-seen row, and the result is sorted by timestamp before writing.
        Writes are atomic: data is written to a temp file then renamed to prevent
        corrupt Parquet files on crash.
        """
        with_year = df.with_columns(pl.col("timestamp").dt.year().alias("_year"))
        for year_val in with_year["_year"].unique().sort().to_list():
            year_df = with_year.filter(pl.col("_year") == year_val).drop("_year")
            path = self._year_path(timeframe, int(year_val), symbol)
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                existing = pl.read_parquet(path)
                merged = (
                    pl.concat([existing, year_df])
                    .unique(
                        subset=["symbol", "timeframe", "timestamp"],
                        keep="last",
                        maintain_order=True,
                    )
                    .sort("timestamp")
                )
                tmp = path.with_suffix(".tmp.parquet")
                merged.write_parquet(tmp)
                tmp.replace(path)
            else:
                tmp = path.with_suffix(".tmp.parquet")
                year_df.sort("timestamp").write_parquet(tmp)
                tmp.replace(path)

        log.info("cache.write", symbol=symbol, timeframe=timeframe, rows=len(df))

    def read(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """Read bars in the closed interval [start, end] for a symbol+timeframe.

        Scans all year directories under the timeframe directory, reads files
        matching the symbol name, filters by date range, and concatenates results.
        Returns an empty DataFrame when no matching data is found.
        """
        tf_dir = self._root / timeframe
        if not tf_dir.exists():
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        for year_dir in sorted(tf_dir.iterdir(), key=lambda p: p.name):
            if not year_dir.is_dir():
                continue
            path = year_dir / f"{symbol}.parquet"
            if not path.exists():
                continue
            df = pl.read_parquet(path)
            filtered = df.filter(
                (pl.col("timestamp") >= start) & (pl.col("timestamp") <= end)
            )
            if len(filtered) > 0:
                frames.append(filtered)

        return pl.concat(frames).sort("timestamp") if frames else pl.DataFrame()

    def last_timestamp(self, *, symbol: str, timeframe: str) -> datetime | None:
        """Return the most recent bar timestamp for this symbol+timeframe, or None.

        Reads only the timestamp column from each year file for efficiency.
        Attaches IST timezone if Polars returns a naive datetime after Parquet
        round-trip.
        """
        tf_dir = self._root / timeframe
        if not tf_dir.exists():
            return None

        latest: datetime | None = None
        for year_dir in sorted(tf_dir.iterdir(), key=lambda p: p.name):
            if not year_dir.is_dir():
                continue
            path = year_dir / f"{symbol}.parquet"
            if not path.exists():
                continue
            df = pl.read_parquet(path, columns=["timestamp"])
            if len(df) == 0:
                continue
            ts: datetime | None = df["timestamp"].max()  # type: ignore[assignment]
            if ts is not None:
                ts_aware = ts if ts.tzinfo else ts.replace(tzinfo=_IST)
                if latest is None or ts_aware > latest:
                    latest = ts_aware

        return latest

    def coverage_report(self) -> dict[str, dict[str, tuple[datetime, datetime]]]:
        """Return {symbol: {timeframe: (min_ts, max_ts)}} for all cached data.

        Aggregates min/max across all year files for each (symbol, timeframe) pair.
        Useful for identifying gaps before requesting incremental history from the
        broker. Returns an empty dict when the cache root does not exist yet.
        """
        # Accumulate per-(symbol, timeframe): running min and max across all years
        acc: dict[tuple[str, str], tuple[datetime, datetime]] = {}

        if not self._root.exists():
            return {}

        for tf_dir in sorted(self._root.iterdir(), key=lambda p: p.name):
            if not tf_dir.is_dir():
                continue
            timeframe = tf_dir.name
            for year_dir in sorted(tf_dir.iterdir(), key=lambda p: p.name):
                if not year_dir.is_dir():
                    continue
                for parquet_file in year_dir.glob("*.parquet"):
                    symbol = parquet_file.stem
                    df = pl.read_parquet(parquet_file, columns=["timestamp"])
                    if len(df) == 0:
                        continue
                    mn_raw: datetime | None = df["timestamp"].min()  # type: ignore[assignment]
                    mx_raw: datetime | None = df["timestamp"].max()  # type: ignore[assignment]
                    if mn_raw is None or mx_raw is None:
                        continue
                    mn = mn_raw if mn_raw.tzinfo else mn_raw.replace(tzinfo=_IST)
                    mx = mx_raw if mx_raw.tzinfo else mx_raw.replace(tzinfo=_IST)
                    key = (symbol, timeframe)
                    if key not in acc:
                        acc[key] = (mn, mx)
                    else:
                        prev_mn, prev_mx = acc[key]
                        acc[key] = (min(prev_mn, mn), max(prev_mx, mx))

        report: dict[str, dict[str, tuple[datetime, datetime]]] = {}
        for (symbol, timeframe), (mn, mx) in acc.items():
            report.setdefault(symbol, {})[timeframe] = (mn, mx)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _year_path(self, timeframe: str, year: int, symbol: str) -> Path:
        return self._root / timeframe / str(year) / f"{symbol}.parquet"

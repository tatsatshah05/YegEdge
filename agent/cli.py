from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import click
import structlog
from rich.console import Console
from rich.table import Table

from config.settings import AppSettings

log = structlog.get_logger()
console = Console()

IST = ZoneInfo("Asia/Kolkata")

_TIMEFRAMES: tuple[str, ...] = ("15m", "60m", "1d")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """YegEdge — research-driven, risk-first algo trading agent for NSE equities."""


# ---------------------------------------------------------------------------
# refresh command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--symbol", default=None, help="Refresh a single symbol only.")
@click.option(
    "--timeframe",
    default=None,
    type=click.Choice(["15m", "60m", "1d"]),
    help="Refresh a single timeframe only.",
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Ignore existing cache; fetch full history from scratch.",
)
def refresh(symbol: str | None, timeframe: str | None, full: bool) -> None:
    """Fetch and cache historical OHLCV bars from Upstox."""
    settings = AppSettings()

    if not settings.upstox_access_token:
        console.print("[red]Error: UPSTOX_ACCESS_TOKEN is not set. Cannot connect to broker.[/red]")
        sys.exit(1)

    # Import here to avoid circular issues and heavy imports at module level
    from agent.data.cache import ParquetCache
    from agent.data.universe import UniverseLoader
    from agent.data.upstox_adapter import UpstoxAdapter
    from agent.data.validator import DataValidator

    universe = UniverseLoader(Path("config/universe.yaml"))
    adapter = UpstoxAdapter(access_token=settings.upstox_access_token)
    cache = ParquetCache(root=settings.parquet_cache_dir)
    validator = DataValidator()

    symbols: list[str] = [symbol] if symbol else universe.all_symbols()
    timeframes: list[str] = [timeframe] if timeframe else list(_TIMEFRAMES)

    today = datetime.now(tz=IST).date()
    two_years_ago = datetime.now(tz=IST) - timedelta(days=730)

    errors: list[str] = []

    for sym in symbols:
        for tf in timeframes:
            try:
                if full:
                    start_dt = two_years_ago
                else:
                    last_ts = cache.last_timestamp(symbol=sym, timeframe=tf)
                    if last_ts is not None:
                        start_dt = last_ts + timedelta(hours=1)
                    else:
                        start_dt = two_years_ago

                if start_dt.date() > today:
                    console.print(f"[dim]{sym} [{tf}] — already up to date[/dim]")
                    continue

                end_dt = datetime.now(tz=IST)

                log.info(
                    "cli.refresh.fetching",
                    symbol=sym,
                    timeframe=tf,
                    start=str(start_dt.date()),
                    end=str(end_dt.date()),
                )

                df = adapter.fetch_historical(sym, tf, start_dt, end_dt)

                if len(df) == 0:
                    console.print(f"[yellow]{sym} [{tf}] — no new bars[/yellow]")
                    continue

                validated_df = validator.validate(df)
                cache.write(validated_df, symbol=sym, timeframe=tf)

                console.print(f"[green]{sym} [{tf}] — +{len(df)} bars[/green]")

            except Exception as exc:
                log.error(
                    "cli.refresh.error",
                    symbol=sym,
                    timeframe=tf,
                    error=str(exc),
                )
                console.print(f"[red]Error fetching {sym} [{tf}]: {exc}[/red]")
                errors.append(f"{sym}/{tf}: {exc}")

    if errors:
        console.print(f"\n[red]{len(errors)} error(s) occurred during refresh.[/red]")
        sys.exit(1)
    else:
        console.print("\n[bold green]Refresh complete.[/bold green]")


# ---------------------------------------------------------------------------
# verify command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--symbol", default=None, help="Verify a single symbol.")
@click.option(
    "--timeframe",
    default=None,
    type=click.Choice(["15m", "60m", "1d"]),
    help="Verify a single timeframe.",
)
def verify(symbol: str | None, timeframe: str | None) -> None:
    """Show cache coverage and data quality for the trading universe."""
    settings = AppSettings()

    from agent.data.cache import ParquetCache
    from agent.data.universe import UniverseLoader
    from agent.data.validator import DataValidator

    cache = ParquetCache(root=settings.parquet_cache_dir)
    universe = UniverseLoader(Path("config/universe.yaml"))
    validator = DataValidator()

    report = cache.coverage_report()

    if not report:
        console.print(
            "[yellow]No cached data found. Run `python -m agent refresh` first.[/yellow]"
        )
        sys.exit(1)

    symbols: list[str] = [symbol] if symbol else universe.all_symbols()
    timeframes: list[str] = [timeframe] if timeframe else list(_TIMEFRAMES)

    table = Table(
        title="Cache Coverage Report",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Symbol", style="bold")
    table.add_column("Timeframe")
    table.add_column("From")
    table.add_column("To")
    table.add_column("Bars", justify="right")
    table.add_column("OK", justify="right", style="green")
    table.add_column("Partial", justify="right", style="yellow")
    table.add_column("Suspect", justify="right", style="red")

    for sym in symbols:
        for tf in timeframes:
            sym_report = report.get(sym, {})

            if tf not in sym_report:
                table.add_row(
                    sym,
                    tf,
                    "[red]NO DATA[/red]",
                    "[red]NO DATA[/red]",
                    "-",
                    "-",
                    "-",
                    "-",
                )
                continue

            from_ts, to_ts = sym_report[tf]

            df = cache.read(symbol=sym, timeframe=tf, start=from_ts, end=to_ts)

            if len(df) == 0:
                table.add_row(
                    sym,
                    tf,
                    str(from_ts.date()),
                    str(to_ts.date()),
                    "0",
                    "-",
                    "-",
                    "-",
                )
                continue

            validated_df = validator.validate(df)
            quality_counts = (
                validated_df["data_quality"].value_counts().to_dicts()
            )
            count_map: dict[str, int] = {
                row["data_quality"]: row["count"] for row in quality_counts
            }

            ok_count = count_map.get("ok", 0)
            partial_count = count_map.get("partial", 0)
            suspect_count = count_map.get("suspect", 0)

            table.add_row(
                sym,
                tf,
                str(from_ts.date()),
                str(to_ts.date()),
                str(len(df)),
                str(ok_count),
                str(partial_count),
                str(suspect_count),
            )

    console.print(table)

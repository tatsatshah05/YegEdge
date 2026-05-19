from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import click
import structlog
from rich.console import Console
from rich.table import Table

from agent.backtest.runner import BacktestRunner
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
        console.print("[yellow]No cached data found. Run `python -m agent refresh` first.[/yellow]")
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
            quality_counts = validated_df["data_quality"].value_counts().to_dicts()
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


# ---------------------------------------------------------------------------
# run-paper command
# ---------------------------------------------------------------------------


@cli.command("run-paper")
@click.option(
    "--date",
    "session_date_str",
    default=None,
    help="Session date as YYYY-MM-DD (default: today)",
)
@click.option(
    "--warmup-bars",
    default=200,
    show_default=True,
    help="Number of historical bars to load for indicator warm-up.",
)
def run_paper(session_date_str: str | None, warmup_bars: int) -> None:
    """Run one paper trading session by replaying cached bars."""
    from datetime import date as date_type
    from decimal import Decimal

    import polars as pl

    from agent.data.cache import ParquetCache
    from agent.data.universe import UniverseLoader
    from agent.execution.paper import PaperExecution
    from agent.features.pipeline import FeaturePipeline
    from agent.journal.store import JournalStore
    from agent.monitoring.alerter import TelegramAlerter
    from agent.monitoring.heartbeat import Heartbeat
    from agent.monitoring.kill_switch import KillSwitch
    from agent.portfolio.tracker import PortfolioTracker
    from agent.risk.manager import RiskManager
    from agent.risk.rules import load_risk_rules
    from agent.runner.daily_loop import DailyLoop
    from agent.runner.session_counter import PaperSessionCounter
    from agent.strategies.trend_following import TrendFollowingStrategy

    settings = AppSettings()

    session_date = (
        date_type.fromisoformat(session_date_str) if session_date_str else date_type.today()
    )

    console.print(f"[bold]YegEdge Paper Trading — {session_date}[/bold]")

    cache = ParquetCache(root=settings.parquet_cache_dir)
    report = cache.coverage_report()
    if not report:
        console.print("[yellow]No cached data. Run `refresh` first.[/yellow]")
        sys.exit(1)

    universe = UniverseLoader(Path("config/universe.yaml"))
    timeframe = universe.primary_timeframe

    example_sym = universe.symbols()[0]
    if example_sym not in report or timeframe not in report.get(example_sym, {}):
        console.print(f"[red]No cached data for {example_sym}/{timeframe}[/red]")
        sys.exit(1)

    session_start = datetime(
        session_date.year, session_date.month, session_date.day, 9, 15, tzinfo=IST
    )
    session_end = datetime(
        session_date.year, session_date.month, session_date.day, 15, 30, tzinfo=IST
    )

    warmup_frames = []
    session_frames = []
    pipeline = FeaturePipeline()
    for sym in universe.symbols():
        if sym not in report or timeframe not in report.get(sym, {}):
            continue
        sym_earliest, _ = report[sym][timeframe]
        # Load all history for this symbol so rolling indicators are correctly seeded
        all_sym = cache.read(symbol=sym, timeframe=timeframe, start=sym_earliest, end=session_end)
        if len(all_sym) == 0:
            continue
        # Enrich per-symbol — rolling windows must not cross symbol boundaries
        enriched = pipeline.run(all_sym)
        wdf = enriched.filter(pl.col("timestamp") < session_start).tail(warmup_bars)
        sdf = enriched.filter(
            (pl.col("timestamp") >= session_start) & (pl.col("timestamp") <= session_end)
        )
        if len(wdf) > 0:
            warmup_frames.append(wdf)
        if len(sdf) > 0:
            session_frames.append(sdf)

    if not session_frames:
        console.print(f"[yellow]No session bars for {session_date}. Try a different date.[/yellow]")
        sys.exit(1)

    warmup_df = pl.concat(warmup_frames).sort("timestamp") if warmup_frames else pl.DataFrame()
    session_df = pl.concat(session_frames).sort("timestamp")

    console.print(f"Warmup bars: {len(warmup_df)}  Session bars: {len(session_df)}")

    alerter = TelegramAlerter(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )
    kill_switch = KillSwitch()
    heartbeat = Heartbeat(alerter=alerter, alert_every_n_beats=4)
    portfolio = PortfolioTracker(
        initial_nav=Decimal(str(settings.paper_starting_capital)),
        initial_cash=Decimal(str(settings.paper_starting_capital)),
        start_time=session_start,
    )
    journal = JournalStore(db_path=settings.journal_db_path)
    strategy = TrendFollowingStrategy()
    risk_rules = load_risk_rules(Path("config/risk_rules.yaml"))
    risk_manager = RiskManager(rules=risk_rules)
    executor = PaperExecution()

    loop = DailyLoop(
        strategy=strategy,
        risk_manager=risk_manager,
        executor=executor,
        portfolio=portfolio,
        journal=journal,
        analyst=None,
        kill_switch=kill_switch,
        heartbeat=heartbeat,
        alerter=alerter,
    )

    result = loop.run(
        session_date=session_date,
        warmup_df=warmup_df,
        session_df=session_df,
    )

    counter = PaperSessionCounter(path=Path("data/paper_sessions.json"))
    new_count = counter.increment()

    console.print("\n[bold green]Session complete.[/bold green]")
    console.print(f"Bars processed: {result.bars_processed}")
    console.print(f"Fills: {len(result.fills)}")
    console.print(f"Final NAV: ₹{result.final_nav:,.2f}")
    console.print(f"Daily P&L: ₹{result.daily_pnl:,.2f}")
    console.print(f"Paper sessions completed: {new_count}/60")

    if counter.is_ready_for_live():
        _msg = (
            "[bold yellow]60 sessions complete — "
            "review results before enabling live trading.[/bold yellow]"
        )
        console.print(_msg)


# ---------------------------------------------------------------------------
# backtest command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--symbol", required=True, help="NSE symbol to backtest (e.g. HDFCBANK)")
@click.option(
    "--timeframe",
    default="60m",
    show_default=True,
    type=click.Choice(["15m", "60m", "1d"]),
    help="Bar timeframe",
)
@click.option("--start", "start_str", required=True, help="Start date YYYY-MM-DD (inclusive)")
@click.option("--end", "end_str", required=True, help="End date YYYY-MM-DD (inclusive)")
@click.option(
    "--warmup",
    default=100,
    show_default=True,
    help="Warmup bars before each session for indicator seeding",
)
def backtest(symbol: str, timeframe: str, start_str: str, end_str: str, warmup: int) -> None:
    """Replay historical bars through the strategy and report net-of-cost performance."""
    from datetime import date as date_type
    from decimal import Decimal

    from agent.data.cache import ParquetCache
    from agent.risk.manager import RiskManager
    from agent.risk.rules import load_risk_rules
    from agent.strategies.trend_following import TrendFollowingStrategy

    settings = AppSettings()
    start_date = date_type.fromisoformat(start_str)
    end_date = date_type.fromisoformat(end_str)

    console.print(f"[bold]YegEdge Backtest — {symbol} {timeframe} {start_date} → {end_date}[/bold]")

    cache = ParquetCache(root=settings.parquet_cache_dir)
    strategy = TrendFollowingStrategy()
    risk_rules = load_risk_rules(Path("config/risk_rules.yaml"))
    risk_manager = RiskManager(rules=risk_rules)

    runner = BacktestRunner(
        strategy=strategy,
        risk_manager=risk_manager,
        cache=cache,
        initial_nav=Decimal(str(settings.paper_starting_capital)),
    )

    report = runner.run(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        warmup_bars=warmup,
    )

    if not report.sessions:
        console.print(
            f"[yellow]No cached data or sessions for {symbol}/{timeframe} in range. "
            "Run `refresh` first if cache is empty.[/yellow]"
        )
        sys.exit(1)

    m = report.metrics

    console.print()
    table = Table(title="Backtest Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Sessions", str(m.total_sessions))
    table.add_row("Win Rate", f"{m.win_rate:.1%}")
    table.add_row("Gross P&L", f"₹{m.total_gross_pnl:,.2f}")
    table.add_row("Total Costs", f"₹{m.total_costs:,.2f}")
    table.add_row("Net P&L", f"₹{m.total_net_pnl:+,.2f}")
    table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.3f}")
    table.add_row("Max Drawdown", f"{m.max_drawdown:.2%}")
    table.add_row("CAGR", f"{m.cagr:.2%}")
    table.add_row("Initial NAV", f"₹{m.initial_nav:,.2f}")
    table.add_row("Final NAV", f"₹{m.final_nav:,.2f}")
    console.print(table)

    if len(report.sessions) > 0:
        console.print()
        detail = Table(title="Last 10 Sessions", show_header=True, header_style="bold")
        detail.add_column("Date")
        detail.add_column("Bars", justify="right")
        detail.add_column("Fills", justify="right")
        detail.add_column("Gross P&L", justify="right")
        detail.add_column("Costs", justify="right")
        detail.add_column("Net P&L", justify="right")
        detail.add_column("NAV", justify="right")
        for s in report.sessions[-10:]:
            pnl_color = "green" if s.net_pnl >= 0 else "red"
            detail.add_row(
                str(s.session_date),
                str(s.bars_processed),
                str(s.fills),
                f"₹{s.gross_pnl:,.2f}",
                f"₹{s.costs:,.2f}",
                f"[{pnl_color}]₹{s.net_pnl:+,.2f}[/{pnl_color}]",
                f"₹{s.final_nav:,.2f}",
            )
        console.print(detail)


# ---------------------------------------------------------------------------
# live-paper command
# ---------------------------------------------------------------------------


@cli.command("live-paper")
@click.option(
    "--timeframe",
    default="60m",
    type=click.Choice(["15m", "60m"]),
    show_default=True,
    help="Bar timeframe for live aggregation.",
)
@click.option(
    "--warmup-bars",
    default=100,
    show_default=True,
    help="Number of historical bars to prepend for indicator warm-up.",
)
def live_paper(timeframe: str, warmup_bars: int) -> None:
    """Paper-trade in real time using live Upstox WebSocket ticks."""
    import asyncio
    import threading
    from decimal import Decimal

    import polars as pl
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text

    from agent.data.bar_builder import ClosedBar
    from agent.data.cache import ParquetCache
    from agent.data.universe import UniverseLoader
    from agent.data.upstox_adapter import UpstoxAdapter
    from agent.execution.types import Fill
    from agent.features.pipeline import FeaturePipeline
    from agent.monitoring.alerter import TelegramAlerter
    from agent.monitoring.kill_switch import KillSwitch
    from agent.portfolio.tracker import PortfolioTracker
    from agent.runner.live_session import LiveSession

    settings = AppSettings()

    if not settings.upstox_access_token:
        console.print("[red]UPSTOX_ACCESS_TOKEN not set. Run your daily login first.[/red]")
        sys.exit(1)

    cache = ParquetCache(root=settings.parquet_cache_dir)
    report = cache.coverage_report()

    if not report:
        console.print("[red]No cached data found. Run `refresh` first to load warmup bars.[/red]")
        sys.exit(1)

    universe = UniverseLoader(Path("config/universe.yaml"))
    symbols = universe.symbols()

    today = datetime.now(tz=IST).date()
    session_start = datetime(today.year, today.month, today.day, 9, 15, tzinfo=IST)

    pipeline = FeaturePipeline()
    warmup_frames: list[pl.DataFrame] = []
    for sym in symbols:
        if sym not in report or timeframe not in report.get(sym, {}):
            continue
        sym_earliest, _ = report[sym][timeframe]
        all_sym = cache.read(symbol=sym, timeframe=timeframe, start=sym_earliest, end=session_start)
        if len(all_sym) == 0:
            continue
        enriched = pipeline.run(all_sym)
        warmup_frames.append(enriched.tail(warmup_bars))

    warmup_df = pl.concat(warmup_frames) if warmup_frames else pl.DataFrame()

    portfolio = PortfolioTracker(
        initial_nav=Decimal(str(settings.paper_starting_capital)),
        initial_cash=Decimal(str(settings.paper_starting_capital)),
        start_time=session_start,
    )

    alerter = TelegramAlerter(
        bot_token=settings.telegram_bot_token,
        chat_id=settings.telegram_chat_id,
    )

    kill_switch = KillSwitch(flag_path=Path("./data/.kill_switch"))

    # --- Live display state (mutated from async callback, read by Rich renderer) ---
    _display_lock = threading.Lock()
    # Each entry: (symbol, bar_open_str, open, high, low, close, tick_count)
    _last_bars: list[tuple[str, str, float, float, float, float, int]] = []
    _event_log: list[str] = []  # most recent events, newest first
    _bars_processed = 0

    def _build_display() -> Layout:
        with _display_lock:
            state = portfolio.state
            pnl_color = "green" if state.daily_pnl >= 0 else "red"
            now_ist = datetime.now(tz=IST).strftime("%H:%M:%S IST")

            header = Text.assemble(
                ("YegEdge  ", "bold cyan"),
                (f"{timeframe} bars  ", "dim"),
                (f"{len(symbols)} symbols  ", "dim"),
                (now_ist, "bold"),
            )

            nav_text = Text.assemble(
                ("NAV  ", "dim"),
                (f"₹{state.nav:,.0f}   ", "bold"),
                ("P&L  ", "dim"),
                (f"₹{state.daily_pnl:+,.0f}   ", f"bold {pnl_color}"),
                ("Cash  ", "dim"),
                (f"₹{state.cash:,.0f}   ", "bold"),
                ("Orders  ", "dim"),
                (f"{state.orders_today}   ", "bold"),
                ("Bars  ", "dim"),
                (str(_bars_processed), "bold"),
            )

            bars_table = Table(show_header=True, header_style="bold blue", box=None, padding=(0, 1))
            bars_table.add_column("Symbol", width=12)
            bars_table.add_column("Bar open", width=8)
            bars_table.add_column("O", justify="right", width=8)
            bars_table.add_column("H", justify="right", width=8)
            bars_table.add_column("L", justify="right", width=8)
            bars_table.add_column("C", justify="right", width=8)
            bars_table.add_column("Ticks", justify="right", width=6)
            for sym, bar_ts, o, h, lo, c, ticks in _last_bars[-10:]:
                bars_table.add_row(
                    sym,
                    bar_ts,
                    f"{o:.2f}",
                    f"{h:.2f}",
                    f"{lo:.2f}",
                    f"{c:.2f}",
                    str(ticks),
                )

            events_text = Text()
            for line in _event_log[:12]:
                events_text.append(line + "\n")

            layout = Layout()
            layout.split_column(
                Layout(Panel(header, style="on grey11"), size=3),
                Layout(Panel(nav_text, title="Portfolio", border_style="cyan"), size=4),
                Layout(Panel(bars_table, title="Last closed bars", border_style="blue"), size=14),
                Layout(Panel(events_text, title="Events", border_style="yellow")),
            )
        return layout

    def on_bar_closed(bar: object, fills: list[object]) -> None:
        nonlocal _bars_processed
        assert isinstance(bar, ClosedBar)
        with _display_lock:
            _bars_processed += 1
            _last_bars.append(
                (
                    bar.symbol,
                    bar.bar_open.strftime("%H:%M"),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.tick_count,
                )
            )
            if len(_last_bars) > 20:
                _last_bars.pop(0)

            for fill in fills:
                assert isinstance(fill, Fill)
                _event_log.insert(
                    0,
                    f"[green]FILL[/green]  {fill.symbol}  {fill.action}  "
                    f"qty={fill.quantity}  @₹{fill.fill_price:.2f}  "
                    f"{bar.bar_open.strftime('%H:%M')}",
                )
            if not fills:
                _event_log.insert(
                    0,
                    f"[dim]BAR[/dim]   {bar.symbol}  C={bar.close:.2f}  "
                    f"ticks={bar.tick_count}  {bar.bar_open.strftime('%H:%M')}",
                )
            if len(_event_log) > 50:
                _event_log.pop()

    live_session = LiveSession(
        symbols=symbols,
        timeframe=timeframe,
        portfolio=portfolio,
        warmup_df=warmup_df,
        alerter=alerter,
        kill_switch=kill_switch,
        on_bar_closed=on_bar_closed,
    )

    adapter = UpstoxAdapter(access_token=settings.upstox_access_token)

    def on_tick_df(df: pl.DataFrame) -> None:
        if len(df) == 0:
            return
        sym = str(df["symbol"][0])
        ltp = float(df["ltp"][0])
        ts = df["timestamp"][0]
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        live_session.put_tick(sym, ltp, ts)

    async def _run() -> None:
        stream_task = asyncio.create_task(adapter.stream_live(symbols, callback=on_tick_df))
        session_task = asyncio.create_task(live_session.run())
        await session_task
        stream_task.cancel()
        try:
            await stream_task
        except asyncio.CancelledError:
            pass

    try:
        with Live(_build_display(), refresh_per_second=2, screen=True) as live:

            async def _run_with_refresh() -> None:
                stream_task = asyncio.create_task(adapter.stream_live(symbols, callback=on_tick_df))
                session_task = asyncio.create_task(live_session.run())

                async def _refresh_loop() -> None:
                    while not session_task.done():
                        live.update(_build_display())
                        await asyncio.sleep(0.5)
                    live.update(_build_display())

                refresh_task = asyncio.create_task(_refresh_loop())
                await session_task
                stream_task.cancel()
                refresh_task.cancel()
                await asyncio.gather(stream_task, refresh_task, return_exceptions=True)

            asyncio.run(_run_with_refresh())
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted by user.[/yellow]")

    final_state = portfolio.state
    pnl_sign = "+" if final_state.daily_pnl >= 0 else ""
    console.print(
        f"\n[bold]Session complete.[/bold] "
        f"NAV: ₹{final_state.nav:,.0f} | "
        f"P&L: {pnl_sign}₹{final_state.daily_pnl:,.0f} | "
        f"Orders: {final_state.orders_today} | "
        f"Bars processed: {_bars_processed}"
    )

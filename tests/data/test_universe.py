from __future__ import annotations

from pathlib import Path

import pytest

from agent.data.universe import UniverseLoader

UNIVERSE_YAML = Path("config/universe.yaml")


@pytest.fixture
def loader() -> UniverseLoader:
    return UniverseLoader(UNIVERSE_YAML)


def test_symbols_returns_list(loader: UniverseLoader) -> None:
    syms = loader.symbols()
    assert isinstance(syms, list)
    assert len(syms) > 0


def test_known_symbol_present(loader: UniverseLoader) -> None:
    assert "HDFCBANK" in loader.symbols()


def test_benchmarks_present(loader: UniverseLoader) -> None:
    bmarks = loader.benchmarks()
    assert "NIFTY50" in bmarks


def test_sector_lookup(loader: UniverseLoader) -> None:
    assert loader.sector("HDFCBANK") == "financials"
    assert loader.sector("TCS") == "it"


def test_unknown_symbol_raises(loader: UniverseLoader) -> None:
    with pytest.raises(KeyError):
        loader.sector("UNKNOWNSYMBOL")


def test_all_symbols_includes_benchmarks(loader: UniverseLoader) -> None:
    all_syms = loader.all_symbols()
    assert "HDFCBANK" in all_syms
    assert "NIFTY50" in all_syms


def test_exchange(loader: UniverseLoader) -> None:
    assert loader.exchange == "NSE"


def test_primary_timeframe(loader: UniverseLoader) -> None:
    assert loader.primary_timeframe == "60m"

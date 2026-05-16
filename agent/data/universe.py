from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class UniverseLoader:
    """Load trading universe from YAML. Designed for future point-in-time filtering."""

    def __init__(self, path: Path) -> None:
        with path.open() as f:
            self._cfg: dict[str, Any] = yaml.safe_load(f)

    @property
    def exchange(self) -> str:
        return self._cfg["exchange"]  # type: ignore[no-any-return]

    @property
    def primary_timeframe(self) -> str:
        return self._cfg["timeframes"]["primary"]  # type: ignore[no-any-return]

    def symbols(self) -> list[str]:
        return list(self._cfg["symbols"])

    def benchmarks(self) -> list[str]:
        return list(self._cfg["benchmarks"])

    def all_symbols(self) -> list[str]:
        return self.symbols() + self.benchmarks()

    def sector(self, symbol: str) -> str:
        sectors: dict[str, str] = self._cfg["sectors"]
        if symbol not in sectors:
            raise KeyError(f"Symbol {symbol!r} not in universe sectors")
        return sectors[symbol]

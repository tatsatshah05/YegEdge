from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()

_DEFAULT_FLAG_PATH = Path(".kill_switch")


class KillSwitch:
    """File-based kill switch. When the flag file exists, all trading is halted.

    The flag file path defaults to `.kill_switch` in the working directory.
    Write any text to the file to explain the reason; the content is logged on read.
    """

    def __init__(self, flag_path: Path = _DEFAULT_FLAG_PATH) -> None:
        self._path = flag_path

    def is_active(self) -> bool:
        return self._path.exists()

    def activate(self, reason: str = "") -> None:
        self._path.write_text(reason or "Kill switch activated.")
        logger.warning("kill_switch.activated", reason=reason, path=str(self._path))

    def deactivate(self) -> None:
        if self._path.exists():
            self._path.unlink()
        logger.info("kill_switch.deactivated", path=str(self._path))

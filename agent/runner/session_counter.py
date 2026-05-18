from __future__ import annotations

import json
from pathlib import Path

import structlog

logger = structlog.get_logger()

_LIVE_THRESHOLD = 60


class PaperSessionCounter:
    """JSON-backed counter for completed paper trading sessions.

    Counts toward the mandatory 60-session threshold before live trading.
    The count file is created on first increment.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def count(self) -> int:
        if not self._path.exists():
            return 0
        return json.loads(self._path.read_text()).get("sessions_completed", 0)

    def increment(self) -> int:
        new_count = self.count() + 1
        self._path.write_text(json.dumps({"sessions_completed": new_count}))
        logger.info(
            "session_counter.incremented",
            sessions_completed=new_count,
            threshold=_LIVE_THRESHOLD,
            ready_for_live=new_count >= _LIVE_THRESHOLD,
        )
        return new_count

    def is_ready_for_live(self) -> bool:
        return self.count() >= _LIVE_THRESHOLD

"""Server settings. Tests inject a fake clock and disable the background tick."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kuulo_protocol.models import Track


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Settings:
    db_path: Path = Path("data/kuulo.db")
    clock: Callable[[], datetime] = utc_now
    tick_interval_s: float | None = 1.0
    future_tolerance_s: float = 30.0
    late_after_s: float = 60.0
    fusion_engine: str = "kuulo_server.fusion.basic:BasicFusion"
    on_track_update: Callable[[Track], None] | None = None
    # Browsers send Origin on WebSocket upgrades and do not apply CORS to them, so any web page
    # could otherwise read the live feed from a local server. Non-browser clients send none.
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173", "http://localhost:5173",
        "http://127.0.0.1:8000", "http://localhost:8000",
    )
    max_batch: int = 100

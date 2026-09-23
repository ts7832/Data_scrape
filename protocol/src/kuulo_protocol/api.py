"""Server API views: what the dashboard reads."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from .models import Observation, SensorLocation, TimeQuality, Track, WireModel


class NodeStatus(StrEnum):
    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"


class NodeView(WireModel):
    node_id: str
    location: SensorLocation
    time_quality: TimeQuality
    status: NodeStatus
    last_heartbeat_at: datetime | None = None
    mic_ok: bool | None = None
    software_version: str | None = None
    uncorroborated_rate_24h: float | None = None


class TrackDetail(WireModel):
    track: Track
    observations: list[Observation]
    silent_neighbours: list[NodeView]


class LiveEvent(WireModel):
    """One message on the /v1/live WebSocket. `resync` tells the client to reload its snapshot."""

    type: Literal["observation", "track", "node_status", "resync"]
    data: Observation | Track | NodeView | None = None


class IngestResult(WireModel):
    status: Literal["accepted", "duplicate"]
    late: bool = False

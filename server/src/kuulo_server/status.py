"""Node health from heartbeat age."""

from __future__ import annotations

from datetime import datetime

from kuulo_protocol.api import NodeStatus, NodeView
from kuulo_protocol.models import SensorLocation, TimeQuality

from .db import NodeRow, as_utc

ONLINE_S = 120
STALE_S = 300


def node_status(last_heartbeat_at: datetime | None, now: datetime) -> NodeStatus:
    if last_heartbeat_at is None:
        return NodeStatus.OFFLINE
    age = (now - last_heartbeat_at).total_seconds()
    if age < ONLINE_S:
        return NodeStatus.ONLINE
    if age < STALE_S:
        return NodeStatus.STALE
    return NodeStatus.OFFLINE


def node_view(row: NodeRow, now: datetime, uncorroborated_rate: float | None = None) -> NodeView:
    last = as_utc(row.last_heartbeat_at)
    return NodeView(
        node_id=row.node_id,
        location=SensorLocation(lat=row.lat, lon=row.lon, accuracy_m=row.accuracy_m),
        time_quality=TimeQuality(row.time_quality),
        status=node_status(last, now),
        last_heartbeat_at=last,
        mic_ok=row.mic_ok,
        software_version=row.software_version,
        uncorroborated_rate_24h=uncorroborated_rate,
    )

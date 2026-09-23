"""FusionContext backed by the server database."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from kuulo_protocol.models import DRONE_LABELS, GeoPoint, Observation

from ..db import NodeRow, ObservationRow, as_utc
from ..status import node_status
from .base import NodeInfo


class DbFusionContext:
    def __init__(self, sessions: sessionmaker, now: datetime):
        self._sessions = sessions
        self._now = now

    def nodes(self) -> list[NodeInfo]:
        with self._sessions() as session:
            return [
                NodeInfo(row.node_id, GeoPoint(lat=row.lat, lon=row.lon),
                         node_status(as_utc(row.last_heartbeat_at), self._now))
                for row in session.scalars(select(NodeRow))
            ]

    def recent_observations(self, since: datetime) -> list[Observation]:
        labels = [label.value for label in DRONE_LABELS]
        with self._sessions() as session:
            rows = session.scalars(
                select(ObservationRow)
                .where(ObservationRow.observed_at >= since)
                .where(ObservationRow.late.is_(False))
                .where(ObservationRow.label.in_(labels))
                .order_by(ObservationRow.observed_at)
            )
            return [Observation.model_validate_json(r.raw_json) for r in rows]

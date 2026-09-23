"""Ingest rules: validate identity, verify signatures, deduplicate, check time, store."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from kuulo_protocol.api import IngestResult
from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation
from kuulo_protocol.signing import verify

from .config import Settings
from .db import NodeRow, ObservationRow


class IngestError(Exception):
    def __init__(self, status_code: int, reason: str):
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


def register_node(session: Session, reg: NodeRegistration, now: datetime) -> None:
    row = session.get(NodeRow, reg.node_id)
    if row is not None and row.public_key != reg.public_key:
        raise IngestError(409, "node_id already registered with a different key")
    if row is None:
        row = NodeRow(node_id=reg.node_id, public_key=reg.public_key, registered_at=now)
        session.add(row)
    row.lat = reg.location.lat
    row.lon = reg.location.lon
    row.accuracy_m = reg.location.accuracy_m
    row.time_quality = reg.time_quality.value
    session.commit()


def _node_for(session: Session, node_id: str) -> NodeRow:
    row = session.get(NodeRow, node_id)
    if row is None:
        raise IngestError(401, f"unknown node: {node_id}")
    return row


def _check_not_future(ts: datetime, now: datetime, settings: Settings, field: str) -> None:
    if ts > now + timedelta(seconds=settings.future_tolerance_s):
        raise IngestError(400, f"{field} is in the future")


def ingest_observation(
    session: Session, obs: Observation, now: datetime, settings: Settings
) -> IngestResult:
    node = _node_for(session, obs.source.id)
    if not verify(obs, node.public_key):
        raise IngestError(401, "bad signature")
    _check_not_future(obs.observed_at, now, settings, "observed_at")
    if session.get(ObservationRow, str(obs.observation_id)) is not None:
        return IngestResult(status="duplicate")
    late = obs.observed_at < now - timedelta(seconds=settings.late_after_s)
    session.add(
        ObservationRow(
            observation_id=str(obs.observation_id),
            node_id=obs.source.id,
            observed_at=obs.observed_at,
            received_at=now,
            late=late,
            label=obs.detection.label.value,
            detection_id=str(obs.event.detection_id),
            raw_json=obs.model_dump_json(),
        )
    )
    session.commit()
    return IngestResult(status="accepted", late=late)


def ingest_heartbeat(session: Session, hb: Heartbeat, now: datetime, settings: Settings) -> NodeRow:
    node = _node_for(session, hb.node_id)
    if not verify(hb, node.public_key):
        raise IngestError(401, "bad signature")
    _check_not_future(hb.sent_at, now, settings, "sent_at")
    node.last_heartbeat_at = now
    node.software_version = hb.software_version
    node.mic_ok = hb.mic_ok
    node.queue_depth = hb.queue_depth
    session.commit()
    return node

"""FeatureTrace storage and server pull: requests are created when a track confirms."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from kuulo_protocol.models import Track
from kuulo_protocol.signing import verify
from kuulo_protocol.traces import (
    FeatureTraceHeader,
    TraceRequest,
    TraceUnavailable,
    decode_body,
)

from .db import NodeRow, ObservationRow, TraceRequestRow, TraceRow, as_utc
from .ingest import IngestError


def _node_key(session: Session, node_id: str) -> str:
    row = session.get(NodeRow, node_id)
    if row is None:
        raise IngestError(401, f"unknown node: {node_id}")
    return row.public_key


def _detections(session: Session, observation_ids) -> set[tuple[str, str]]:
    rows = session.execute(
        select(ObservationRow.node_id, ObservationRow.detection_id)
        .where(ObservationRow.observation_id.in_([str(i) for i in observation_ids]))
    )
    return {(node_id, detection_id) for node_id, detection_id in rows}


def create_requests_for_track(session: Session, track: Track, now: datetime) -> int:
    """One open request per contributing (node, detection); idempotent. Returns how many are new."""
    created = 0
    for node_id, detection_id in sorted(_detections(session, track.observation_ids)):
        exists = session.scalar(
            select(TraceRequestRow.request_id)
            .where(TraceRequestRow.node_id == node_id)
            .where(TraceRequestRow.detection_id == detection_id)
        )
        if exists is None:
            session.add(TraceRequestRow(
                request_id=str(uuid4()), node_id=node_id, detection_id=detection_id,
                created_at=now,
            ))
            created += 1
    session.commit()
    return created


def _close(session: Session, node_id: str, detection_id: str, reason: str, now: datetime) -> None:
    row = session.scalar(
        select(TraceRequestRow)
        .where(TraceRequestRow.node_id == node_id)
        .where(TraceRequestRow.detection_id == detection_id)
        .where(TraceRequestRow.closed_at.is_(None))
    )
    if row is not None:
        row.closed_at = now
        row.close_reason = reason


def store_trace(
    session: Session, header: FeatureTraceHeader, body: bytes, traces_dir: Path, now: datetime
) -> str:
    """Validate, verify and store one segment. Returns "stored" or "duplicate"."""
    if not verify(header, _node_key(session, header.node_id)):
        raise IngestError(401, "bad signature")
    if hashlib.sha256(body).hexdigest() != header.body_sha256:
        raise IngestError(400, "body does not match body_sha256")
    try:
        decode_body(body, header.frame_count)
    except ValueError as exc:
        raise IngestError(400, f"invalid trace body: {exc}") from exc
    if session.get(TraceRow, str(header.trace_id)) is not None:
        return "duplicate"
    folder = traces_dir / header.node_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{header.trace_id}.npz"
    tmp = path.with_suffix(".part")
    tmp.write_bytes(body)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written segment behind
    session.add(TraceRow(
        trace_id=str(header.trace_id), node_id=header.node_id,
        detection_id=str(header.detection_id), segment_index=header.segment_index,
        final=header.final, start_at=header.start_at, frame_count=header.frame_count,
        size_bytes=len(body), received_at=now, header_json=header.model_dump_json(),
    ))
    if header.final:
        _close(session, header.node_id, str(header.detection_id), "fulfilled", now)
    session.commit()
    return "stored"


def mark_unavailable(session: Session, msg: TraceUnavailable, now: datetime) -> None:
    if not verify(msg, _node_key(session, msg.node_id)):
        raise IngestError(401, "bad signature")
    _close(session, msg.node_id, str(msg.detection_id), "unavailable", now)
    session.commit()


def open_requests(session: Session, node_id: str) -> list[TraceRequest]:
    rows = session.scalars(
        select(TraceRequestRow)
        .where(TraceRequestRow.node_id == node_id)
        .where(TraceRequestRow.closed_at.is_(None))
        .order_by(TraceRequestRow.created_at)
    )
    return [
        TraceRequest(request_id=r.request_id, node_id=r.node_id, detection_id=r.detection_id,
                     created_at=as_utc(r.created_at))
        for r in rows
    ]


def segment_count(session: Session, track: Track) -> int:
    pairs = _detections(session, track.observation_ids)
    if not pairs:
        return 0
    return session.scalar(
        select(func.count()).select_from(TraceRow)
        .where(tuple_(TraceRow.node_id, TraceRow.detection_id).in_(sorted(pairs)))
    ) or 0

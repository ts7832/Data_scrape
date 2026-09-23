"""Track persistence and evidence queries."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from kuulo_protocol.api import TrackDetail
from kuulo_protocol.models import Observation, Track, TrackStatus

from .db import NodeRow, ObservationRow, TrackObservationRow, TrackRow
from .status import node_view

RECENT_CLOSED = timedelta(minutes=10)


def open_tracks(session: Session, now: datetime) -> list[Track]:
    not_closed = TrackRow.status != TrackStatus.CLOSED.value
    recently_closed = TrackRow.last_seen >= now - RECENT_CLOSED
    rows = session.scalars(select(TrackRow).where(or_(not_closed, recently_closed)))
    return [Track.model_validate_json(row.raw_json) for row in rows]


def track_detail(session: Session, track_id: str, now: datetime) -> TrackDetail | None:
    row = session.get(TrackRow, track_id)
    if row is None:
        return None
    track = Track.model_validate_json(row.raw_json)
    same_obs_id = TrackObservationRow.observation_id == ObservationRow.observation_id
    obs_rows = session.scalars(
        select(ObservationRow)
        .join(TrackObservationRow, same_obs_id)
        .where(TrackObservationRow.track_id == track_id)
        .order_by(ObservationRow.observed_at)
    )
    observations = [Observation.model_validate_json(r.raw_json) for r in obs_rows]
    silent_rows = session.scalars(
        select(NodeRow).where(NodeRow.node_id.in_(track.silent_neighbour_ids))
    )
    silent = [node_view(n, now) for n in silent_rows]
    return TrackDetail(track=track, observations=observations, silent_neighbours=silent)


def persist_track(session: Session, track: Track) -> None:
    track_id = str(track.track_id)
    row = session.get(TrackRow, track_id)
    if row is None:
        row = TrackRow(track_id=track_id, ever_confirmed=False)
        session.add(row)
    row.status = track.status.value
    row.ever_confirmed = bool(row.ever_confirmed) or track.status is TrackStatus.CONFIRMED
    row.last_seen = track.last_seen
    row.raw_json = track.model_dump_json()
    known = set(session.scalars(
        select(TrackObservationRow.observation_id).where(TrackObservationRow.track_id == track_id)
    ))
    for obs_id in map(str, track.observation_ids):
        if obs_id not in known:
            session.add(TrackObservationRow(track_id=track_id, observation_id=obs_id))
    session.commit()


def close_open_tracks(session: Session) -> int:
    """Tracks cannot survive a restart (fusion state is in memory), so close them explicitly."""
    open_rows = select(TrackRow).where(TrackRow.status != TrackStatus.CLOSED.value)
    rows = list(session.scalars(open_rows))
    for row in rows:
        track = Track.model_validate_json(row.raw_json).model_copy(
            update={"status": TrackStatus.CLOSED}
        )
        row.status = TrackStatus.CLOSED.value
        row.raw_json = track.model_dump_json()
    session.commit()
    return len(rows)


def uncorroborated_rate(session: Session, node_id: str, now: datetime) -> float | None:
    """Share of this node's drone detections in 24 h that never joined a confirmed track."""
    since = now - timedelta(hours=24)
    drone = ["drone_multirotor", "drone_fixedwing_engine"]
    detections = set(session.scalars(
        select(ObservationRow.detection_id)
        .where(ObservationRow.node_id == node_id)
        .where(ObservationRow.observed_at >= since)
        .where(ObservationRow.label.in_(drone))
    ))
    if not detections:
        return None
    same_obs_id = TrackObservationRow.observation_id == ObservationRow.observation_id
    corroborated = set(session.scalars(
        select(ObservationRow.detection_id)
        .join(TrackObservationRow, same_obs_id)
        .join(TrackRow, TrackRow.track_id == TrackObservationRow.track_id)
        .where(ObservationRow.node_id == node_id)
        .where(ObservationRow.detection_id.in_(detections))
        .where(TrackRow.ever_confirmed.is_(True))
    ))
    return 1 - len(corroborated) / len(detections)

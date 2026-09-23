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

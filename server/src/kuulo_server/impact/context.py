"""ImpulseContext backed by the server database."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from kuulo_protocol.impulses import ImpulseReport
from kuulo_protocol.models import Track

from ..db import ImpulseEventReportRow, ImpulseReportRow, TrackRow


class DbImpulseContext:
    def __init__(self, sessions: sessionmaker, now: datetime):
        self._sessions = sessions
        self.now = now

    def reports_between(self, start: datetime, end: datetime) -> list[ImpulseReport]:
        with self._sessions() as session:
            rows = session.scalars(
                select(ImpulseReportRow)
                .where(ImpulseReportRow.onset_at >= start)
                .where(ImpulseReportRow.onset_at <= end)
            )
            return [ImpulseReport.model_validate_json(r.raw_json) for r in rows]

    def event_ids_for(self, report_ids: list[UUID]) -> dict[UUID, UUID]:
        with self._sessions() as session:
            rows = session.scalars(
                select(ImpulseEventReportRow)
                .where(ImpulseEventReportRow.report_id.in_([str(i) for i in report_ids]))
            )
            return {UUID(r.report_id): UUID(r.event_id) for r in rows}

    def tracks_seen_between(self, start: datetime, end: datetime) -> list[Track]:
        with self._sessions() as session:
            rows = session.scalars(
                select(TrackRow)
                .where(TrackRow.last_seen >= start)
                .where(TrackRow.last_seen <= end)
            )
            return [Track.model_validate_json(r.raw_json) for r in rows]

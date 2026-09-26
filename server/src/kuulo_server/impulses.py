"""Impulse event persistence and queries (mirrors tracks.py)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from kuulo_protocol.api import ImpulseEventDetail
from kuulo_protocol.impulses import ImpulseEvent, ImpulseReport

from .db import ImpulseEventReportRow, ImpulseEventRow, ImpulseReportRow


def persist_impulse_event(session: Session, event: ImpulseEvent) -> None:
    event_id = str(event.event_id)
    row = session.get(ImpulseEventRow, event_id)
    if row is None:
        row = ImpulseEventRow(event_id=event_id)
        session.add(row)
    row.occurred_at = event.occurred_at
    row.updated_at = event.updated_at
    row.raw_json = event.model_dump_json()
    known = set(session.scalars(
        select(ImpulseEventReportRow.report_id)
        .where(ImpulseEventReportRow.event_id == event_id)
    ))
    for report_id in map(str, event.report_ids):
        if report_id not in known:
            session.add(ImpulseEventReportRow(event_id=event_id, report_id=report_id))
    session.commit()


def recent_impulse_events(session: Session, since: datetime) -> list[ImpulseEvent]:
    rows = session.scalars(
        select(ImpulseEventRow)
        .where(ImpulseEventRow.updated_at >= since)
        .order_by(ImpulseEventRow.updated_at.desc())
    )
    return [ImpulseEvent.model_validate_json(row.raw_json) for row in rows]


def impulse_event_detail(session: Session, event_id: str) -> ImpulseEventDetail | None:
    row = session.get(ImpulseEventRow, event_id)
    if row is None:
        return None
    event = ImpulseEvent.model_validate_json(row.raw_json)
    same_report_id = ImpulseEventReportRow.report_id == ImpulseReportRow.report_id
    report_rows = session.scalars(
        select(ImpulseReportRow)
        .join(ImpulseEventReportRow, same_report_id)
        .where(ImpulseEventReportRow.event_id == event_id)
        .order_by(ImpulseReportRow.onset_at)
    )
    reports = [ImpulseReport.model_validate_json(r.raw_json) for r in report_rows]
    return ImpulseEventDetail(event=event, reports=reports)

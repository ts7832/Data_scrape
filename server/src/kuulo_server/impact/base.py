"""The locator plug-in interface, plus grouping helpers any locator can use (spec §3, §5)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from kuulo_protocol.geo import distance_m, from_local, to_local
from kuulo_protocol.impulses import (
    CLOCK_SIGMA_S,
    Ellipse,
    ImpulseEvent,
    ImpulseKind,
    ImpulseReport,
    LocationQuality,
)
from kuulo_protocol.models import Track


@dataclass(frozen=True)
class LocatorConfig:
    air_temperature_c: float = 10.0
    model_sigma_s: float = 0.005
    max_array_m: float = 5000.0
    coarse_radius_m: float = 1500.0
    confidence: float = 0.95
    lookback_s: float = 60.0
    max_extrapolation_s: float = 30.0
    margin_m: float = 300.0
    max_track_speed_mps: float = 80.0


class ImpulseContext(Protocol):
    now: datetime

    def reports_between(self, start: datetime, end: datetime) -> list[ImpulseReport]: ...

    def event_ids_for(self, report_ids: list[UUID]) -> dict[UUID, UUID]: ...

    def tracks_seen_between(self, start: datetime, end: datetime) -> list[Track]: ...


class ImpulseLocator(Protocol):
    def on_report(self, report: ImpulseReport, ctx: ImpulseContext) -> ImpulseEvent | None: ...


def report_sigma(r: ImpulseReport, cfg: LocatorConfig) -> float:
    """1-sigma timing uncertainty: node clock, the node's own onset picking, propagation floor."""
    return math.sqrt(CLOCK_SIGMA_S[r.time_quality] ** 2 + r.onset_sigma_s ** 2
                     + cfg.model_sigma_s ** 2)


def consistent(a: ImpulseReport, b: ImpulseReport, c: float, cfg: LocatorConfig) -> bool:
    """Could one impulse have reached both nodes with this onset difference?"""
    d = distance_m(a.sensor_location, b.sensor_location)
    dt = abs((a.onset_at - b.onset_at).total_seconds())
    return dt <= d / c + 3 * math.hypot(report_sigma(a, cfg), report_sigma(b, cfg)) + 0.05


def nearby_reports(report, ctx, c, cfg) -> list[ImpulseReport]:
    window = timedelta(seconds=cfg.max_array_m / c + 1.0)
    return [r for r in ctx.reports_between(report.onset_at - window, report.onset_at + window)
            if r.report_id != report.report_id]


def group_reports(report, nearby, c, cfg) -> list[ImpulseReport]:
    group = [report]
    for cand in sorted(nearby, key=lambda r: abs((r.onset_at - report.onset_at).total_seconds())):
        if any(m.source.id == cand.source.id for m in group):
            continue  # at most one report per node: the nearest in time wins
        if all(consistent(cand, m, c, cfg) for m in group):
            group.append(cand)
    return sorted(group, key=lambda r: r.onset_at)


def event_id_for(group, ctx) -> UUID:
    known = ctx.event_ids_for([r.report_id for r in group])
    return Counter(known.values()).most_common(1)[0][0] if known else uuid4()


def coarse_event(event_id, group, cfg, now) -> ImpulseEvent:
    origin = group[0].sensor_location
    local = [to_local(origin, r.sensor_location) for r in group]
    cx = sum(x for x, _ in local) / len(local)
    cy = sum(y for _, y in local) / len(local)
    spread = max(math.hypot(x - cx, y - cy) for x, y in local)
    radius = min(cfg.coarse_radius_m + spread, 100_000.0)
    return ImpulseEvent(
        event_id=event_id, kind=ImpulseKind.UNASSOCIATED, quality=LocationQuality.COARSE,
        position=from_local(origin, cx, cy),
        ellipse=Ellipse(semi_major_m=radius, semi_minor_m=radius, bearing_deg=0.0,
                        confidence=cfg.confidence),
        occurred_at=group[0].onset_at, node_ids=[r.source.id for r in group],
        report_ids=[r.report_id for r in group], updated_at=now,
    )

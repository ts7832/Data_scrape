"""Public baseline locator: which nodes heard the same impulse, and a circle around them.

A precise location and drone-impact attribution come from a private locator loaded through
Settings.impulse_locator (spec §5).
"""

from __future__ import annotations

from kuulo_protocol.impulses import ImpulseEvent, ImpulseReport, speed_of_sound

from .base import (
    ImpulseContext,
    LocatorConfig,
    coarse_event,
    event_id_for,
    group_reports,
    nearby_reports,
)


class CoarseLocator:
    def __init__(self, config: LocatorConfig | None = None):
        self.config = config or LocatorConfig()
        self.c = speed_of_sound(self.config.air_temperature_c)

    def on_report(self, report: ImpulseReport, ctx: ImpulseContext) -> ImpulseEvent | None:
        group = group_reports(report, nearby_reports(report, ctx, self.c, self.config), self.c,
                              self.config)
        return coarse_event(event_id_for(group, ctx), group, self.config, ctx.now)

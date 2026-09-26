"""Impulse wire format: what a node reports when it hears a bang, and what the server makes of it.

A node only says "I heard a sharp broadband event at this instant". The server decides where it
happened (an ImpulseEvent) and whether a drone track explains it (kind = drone_impact).
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from .features import N_BANDS
from .models import (
    SCHEMA_VERSION,
    GeoPoint,
    SensorLocation,
    Source,
    TimeQuality,
    WireModel,
    to_utc_us,
)

# 1-sigma clock error per time_quality (same values as the parent spec §9).
CLOCK_SIGMA_S = {TimeQuality.GPS: 1e-6, TimeQuality.NTP: 0.02, TimeQuality.MANUAL: 0.5}


def speed_of_sound(temp_c: float) -> float:
    """Speed of sound in dry air, m/s."""
    return 331.3 * math.sqrt(1.0 + temp_c / 273.15)


class ImpulseFeatures(WireModel):
    peak_dbfs: float = Field(ge=-200, le=0)
    snr_db: float = Field(ge=-50, le=150)
    rise_time_ms: float = Field(ge=0, le=1000)
    duration_ms: float = Field(ge=0, le=60_000)
    clipped: bool
    band_db: list[float] = Field(min_length=N_BANDS, max_length=N_BANDS)  # 20 ms at the peak


class ImpulseReport(WireModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    report_id: UUID = Field(default_factory=uuid4)
    source: Source
    onset_at: datetime  # UTC, microsecond precision
    onset_sigma_s: float = Field(gt=0, le=5)  # the node's own onset-picking uncertainty (1 sigma)
    time_quality: TimeQuality
    sensor_location: SensorLocation
    features: ImpulseFeatures
    signature: str = ""

    @field_validator("onset_at")
    @classmethod
    def _onset_utc(cls, value: datetime) -> datetime:
        return to_utc_us(value)


class Ellipse(WireModel):
    semi_major_m: float = Field(ge=0, le=100_000)
    semi_minor_m: float = Field(ge=0, le=100_000)
    bearing_deg: float = Field(ge=0, lt=180)  # major-axis direction, clockwise from north
    confidence: float = Field(gt=0, lt=1)


class ImpulseKind(StrEnum):
    DRONE_IMPACT = "drone_impact"
    UNASSOCIATED = "unassociated"


class LocationQuality(StrEnum):
    MULTILATERATED = "multilaterated"  # >= 3 nodes, one clear solution
    AMBIGUOUS = "ambiguous"  # an equally good second solution exists (see alternatives)
    COARSE = "coarse"  # 1-2 nodes: "heard near these sensors"


class ImpulseEvent(WireModel):
    event_id: UUID
    kind: ImpulseKind
    quality: LocationQuality
    position: GeoPoint
    ellipse: Ellipse
    alternatives: list[GeoPoint] = []
    occurred_at: datetime  # estimated emission time
    node_ids: list[str]
    excluded_node_ids: list[str] = []  # dropped as inconsistent (e.g. sound path blocked)
    report_ids: list[UUID]
    residuals_ms: dict[str, float] = {}
    rms_residual_ms: float | None = None
    associated_track_id: UUID | None = None
    updated_at: datetime

    @field_validator("occurred_at", "updated_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return to_utc_us(value)

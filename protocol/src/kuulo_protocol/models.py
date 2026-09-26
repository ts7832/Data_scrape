"""Wire-format messages shared by every Kuulo component."""

from __future__ import annotations

import base64
import binascii
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


class SourceType(StrEnum):
    ACOUSTIC_NODE = "acoustic_node"
    SIMULATED_NODE = "simulated_node"
    CITIZEN_REPORT = "citizen_report"
    REMOTE_ID = "remote_id"
    ADSB = "adsb"
    EXTERNAL_NETWORK = "external_network"


class TimeQuality(StrEnum):
    GPS = "gps"
    NTP = "ntp"
    MANUAL = "manual"


class Label(StrEnum):
    DRONE_MULTIROTOR = "drone_multirotor"
    DRONE_FIXEDWING_ENGINE = "drone_fixedwing_engine"
    AIRCRAFT = "aircraft"
    BIRD = "bird"
    UNKNOWN = "unknown"


DRONE_LABELS = frozenset({Label.DRONE_MULTIROTOR, Label.DRONE_FIXEDWING_ENGINE})


class Phase(StrEnum):
    START = "start"
    UPDATE = "update"
    END = "end"


class TrackStatus(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DOWNGRADED = "downgraded"
    CLOSED = "closed"


def to_utc_ms(value: datetime) -> datetime:
    """Require a timezone-aware timestamp; return it in UTC with millisecond precision."""
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    value = value.astimezone(UTC)
    return value.replace(microsecond=value.microsecond // 1000 * 1000)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Source(WireModel):
    type: SourceType
    id: str = Field(min_length=1, max_length=64)


class GeoPoint(WireModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class SensorLocation(GeoPoint):
    accuracy_m: float = Field(ge=0)


class Detection(WireModel):
    label: Label
    confidence: float = Field(ge=0, le=1)
    bearing_deg: float | None = Field(default=None, ge=0, lt=360)


class EventRef(WireModel):
    detection_id: UUID
    phase: Phase


class Acoustic(WireModel):
    # Bounded so a bad value can't make BasicFusion's 10 ** (snr_db / 20) weighting
    # overflow (see the "extreme snr_db disables fusion" finding): -50..150 covers
    # every realistic reading with wide margin on both sides.
    snr_db: float = Field(ge=-50, le=150)
    peak_freq_hz: float = Field(ge=0)


class Observation(WireModel):
    """A source noticed something. Used by every source type."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    observation_id: UUID = Field(default_factory=uuid4)
    source: Source
    observed_at: datetime
    time_quality: TimeQuality
    sensor_location: SensorLocation
    detection: Detection
    event: EventRef
    acoustic: Acoustic | None = None
    signature: str = ""

    @field_validator("observed_at")
    @classmethod
    def _observed_at_utc(cls, value: datetime) -> datetime:
        return to_utc_ms(value)


class Heartbeat(WireModel):
    """A node saying it is alive, so silence can be told apart from failure."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    node_id: str = Field(min_length=1, max_length=64)
    sent_at: datetime
    software_version: str
    mic_ok: bool
    cpu_temp_c: float | None = None
    queue_depth: int = Field(ge=0)
    signature: str = ""

    @field_validator("sent_at")
    @classmethod
    def _sent_at_utc(cls, value: datetime) -> datetime:
        return to_utc_ms(value)


class Velocity(WireModel):
    speed_mps: float = Field(ge=0)
    heading_deg: float = Field(ge=0, lt=360)


class Track(WireModel):
    """The server's belief that a drone is at a place, with its evidence."""

    track_id: UUID
    status: TrackStatus
    label: Label
    confidence: float = Field(ge=0, le=1)
    position: GeoPoint
    uncertainty_m: float = Field(ge=0)
    velocity: Velocity | None = None
    first_seen: datetime
    last_seen: datetime
    observation_ids: list[UUID]
    silent_neighbour_ids: list[str]


class NodeRegistration(WireModel):
    node_id: str = Field(min_length=1, max_length=64)
    public_key: str = Field(min_length=1)
    location: SensorLocation
    time_quality: TimeQuality

    @field_validator("public_key")
    @classmethod
    def _ed25519_public_key(cls, value: str) -> str:
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("public_key must be base64") from exc
        if len(raw) != 32:
            raise ValueError("public_key must be a 32-byte Ed25519 key")
        return value

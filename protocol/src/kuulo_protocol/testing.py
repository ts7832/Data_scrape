"""Factories for building valid messages in tests (unsigned unless a key is given)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from .models import (
    Acoustic,
    Detection,
    EventRef,
    Heartbeat,
    Label,
    Observation,
    Phase,
    SensorLocation,
    Source,
    SourceType,
    TimeQuality,
)


def make_observation(
    node_id: str = "n1",
    *,
    observed_at: datetime,
    label: Label = Label.DRONE_MULTIROTOR,
    confidence: float = 0.9,
    lat: float = 60.1699,
    lon: float = 24.9384,
    snr_db: float | None = 12.0,
    phase: Phase = Phase.START,
    detection_id: UUID | None = None,
    time_quality: TimeQuality = TimeQuality.NTP,
) -> Observation:
    return Observation(
        source=Source(type=SourceType.SIMULATED_NODE, id=node_id),
        observed_at=observed_at,
        time_quality=time_quality,
        sensor_location=SensorLocation(lat=lat, lon=lon, accuracy_m=10.0),
        detection=Detection(label=label, confidence=confidence),
        event=EventRef(detection_id=detection_id or uuid4(), phase=phase),
        acoustic=None if snr_db is None else Acoustic(snr_db=snr_db, peak_freq_hz=180.0),
    )


def make_heartbeat(node_id: str = "n1", *, sent_at: datetime, mic_ok: bool = True) -> Heartbeat:
    return Heartbeat(
        node_id=node_id,
        sent_at=sent_at,
        software_version="0.1.0",
        mic_ok=mic_ok,
        queue_depth=0,
    )

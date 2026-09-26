from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kuulo_protocol.api import LiveEvent
from kuulo_protocol.geo import distance_m, ellipse_ring
from kuulo_protocol.impulses import (
    Ellipse,
    ImpulseEvent,
    ImpulseKind,
    LocationQuality,
    speed_of_sound,
)
from kuulo_protocol.models import GeoPoint
from kuulo_protocol.signing import generate_keypair, sign, verify
from kuulo_protocol.testing import make_impulse_report

T = datetime(2026, 9, 26, 12, 0, 0, 123456, tzinfo=UTC)


def test_onset_keeps_microseconds_and_converts_to_utc():
    helsinki = timezone(timedelta(hours=3))
    r = make_impulse_report("n1", onset_at=T.astimezone(helsinki), lat=60.17, lon=24.94)
    assert r.onset_at == T and r.onset_at.tzinfo == UTC


def test_naive_onset_is_rejected():
    with pytest.raises(ValueError):
        make_impulse_report("n1", onset_at=T.replace(tzinfo=None), lat=60.17, lon=24.94)


def test_report_signature_round_trip():
    priv, pub = generate_keypair()
    signed = sign(make_impulse_report("n1", onset_at=T, lat=60.17, lon=24.94), priv)
    assert verify(signed, pub)
    assert not verify(signed.model_copy(update={"onset_sigma_s": 0.5}), pub)


def test_features_need_32_bands():
    r = make_impulse_report("n1", onset_at=T, lat=60.17, lon=24.94)
    with pytest.raises(ValueError):
        r.features.model_validate({**r.features.model_dump(), "band_db": [0.0] * 31})


@pytest.mark.parametrize(("temp", "expected"), [(-20, 319.1), (0, 331.3), (20, 343.2)])
def test_speed_of_sound(temp, expected):
    assert speed_of_sound(temp) == pytest.approx(expected, abs=0.2)


def test_ellipse_ring_axes_and_orientation():
    c = GeoPoint(lat=60.17, lon=24.94)
    ring = ellipse_ring(c, 400.0, 100.0, 90.0, steps=64)  # major axis east-west
    assert len(ring) == 65 and ring[0] == ring[-1]
    assert distance_m(c, ring[0]) == pytest.approx(400, rel=1e-3)
    assert ring[0].lon > c.lon and ring[0].lat == pytest.approx(c.lat, abs=1e-6)
    assert distance_m(c, ring[16]) == pytest.approx(100, rel=1e-3)  # quarter turn: minor axis


def test_ellipse_bearing_must_be_half_turn():
    with pytest.raises(ValueError):
        Ellipse(semi_major_m=1, semi_minor_m=1, bearing_deg=180, confidence=0.95)


def test_live_event_carries_an_impulse_event():
    event = ImpulseEvent(
        event_id=uuid4(), kind=ImpulseKind.UNASSOCIATED, quality=LocationQuality.COARSE,
        position=GeoPoint(lat=60.17, lon=24.94),
        ellipse=Ellipse(semi_major_m=1500, semi_minor_m=1500, bearing_deg=0, confidence=0.95),
        occurred_at=T, node_ids=["n1"], report_ids=[uuid4()], updated_at=T,
    )
    parsed = LiveEvent.model_validate_json(
        LiveEvent(type="impulse_event", data=event).model_dump_json())
    assert parsed.data == event

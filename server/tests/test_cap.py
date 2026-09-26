import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kuulo_protocol.geo import distance_m
from kuulo_protocol.impulses import Ellipse, ImpulseEvent, ImpulseKind, LocationQuality
from kuulo_protocol.models import GeoPoint
from kuulo_server.impact.cap import NS, to_cap_xml

T = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def event(kind=ImpulseKind.DRONE_IMPACT):
    return ImpulseEvent(
        event_id=uuid4(), kind=kind, quality=LocationQuality.MULTILATERATED,
        position=GeoPoint(lat=60.17, lon=24.94),
        ellipse=Ellipse(semi_major_m=120, semi_minor_m=40, bearing_deg=30, confidence=0.95),
        occurred_at=T, node_ids=["a", "b", "c", "d"], report_ids=[uuid4()], updated_at=T,
        associated_track_id=uuid4() if kind is ImpulseKind.DRONE_IMPACT else None,
    )


def parse(xml: bytes):
    return ET.fromstring(xml)


def text(root, path):
    return root.find(path, {"cap": NS}).text


def test_required_cap_elements_are_present():
    root = parse(to_cap_xml(event(), sender="kuulo@localhost", status="Test"))
    assert root.tag == f"{{{NS}}}alert"
    for path in ("cap:identifier", "cap:sender", "cap:sent", "cap:status", "cap:msgType",
                 "cap:scope", "cap:info/cap:category", "cap:info/cap:event",
                 "cap:info/cap:urgency", "cap:info/cap:severity", "cap:info/cap:certainty",
                 "cap:info/cap:area/cap:areaDesc", "cap:info/cap:area/cap:polygon"):
        assert root.find(path, {"cap": NS}) is not None, path
    assert text(root, "cap:status") == "Test"
    assert text(root, "cap:info/cap:category") == "Security"


def test_sent_uses_a_numeric_offset_not_z():
    root = parse(to_cap_xml(event(), sender="s", status="Test"))
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", text(root, "cap:sent"))


def test_polygon_is_closed_and_follows_the_ellipse():
    e = event()
    root = parse(to_cap_xml(e, sender="s", status="Test"))
    pairs = [tuple(map(float, p.split(","))) for p in
             text(root, "cap:info/cap:area/cap:polygon").split()]
    assert len(pairs) >= 4 and pairs[0] == pairs[-1]
    for lat, lon in pairs:
        assert 39 <= distance_m(e.position, GeoPoint(lat=lat, lon=lon)) <= 121


def test_severity_and_event_name_follow_the_kind():
    impact = parse(to_cap_xml(event(), sender="s", status="Test"))
    other = parse(to_cap_xml(event(ImpulseKind.UNASSOCIATED), sender="s", status="Test"))
    assert text(impact, "cap:info/cap:severity") == "Severe"
    assert text(other, "cap:info/cap:severity") == "Moderate"
    assert "impact" in text(impact, "cap:info/cap:event").lower()


def test_invalid_status_is_rejected():
    with pytest.raises(ValueError):
        to_cap_xml(event(), sender="s", status="Real")


def test_invalid_cap_status_setting_fails_at_startup(tmp_path):
    from kuulo_server.app import create_app
    from kuulo_server.config import Settings

    with pytest.raises(ValueError, match="cap_status"):
        create_app(Settings(db_path=tmp_path / "k.db", tick_interval_s=None, cap_status="Real"))


def test_endpoint_serves_cap_and_404s_unknown_events(client):
    from kuulo_server.testing import IMPULSE_ORIGIN, T0, post_bang, register_ring

    post_bang(client, register_ring(client), IMPULSE_ORIGIN, T0)
    [e] = client.get("/v1/impulse-events").json()
    response = client.get(f"/v1/impulse-events/{e['event_id']}/cap")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/cap+xml")
    assert parse(response.content).tag == f"{{{NS}}}alert"
    assert client.get(f"/v1/impulse-events/{uuid4()}/cap").status_code == 404

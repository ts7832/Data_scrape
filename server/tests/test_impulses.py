from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from kuulo_protocol.geo import distance_m, from_local
from kuulo_protocol.models import GeoPoint
from kuulo_protocol.signing import generate_keypair
from kuulo_protocol.testing import make_impulse_report
from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.testing import IMPULSE_ORIGIN as ORIGIN
from kuulo_server.testing import T0, FakeClock, post_bang, post_signed, register, register_ring


def events(client):
    return client.get("/v1/impulse-events").json()


def test_six_nodes_hearing_one_bang_make_one_coarse_event(client):
    ring = register_ring(client)
    source = from_local(ORIGIN, 150, -220)
    post_bang(client, ring, source, T0)
    [e] = events(client)
    assert e["quality"] == "coarse" and e["kind"] == "unassociated"
    assert set(e["node_ids"]) == {n.node_id for n in ring}
    assert distance_m(GeoPoint(**e["position"]), source) <= e["ellipse"]["semi_major_m"]


def test_the_same_event_grows_as_reports_arrive(client):
    ring = register_ring(client)
    post_bang(client, ring, ORIGIN, T0, only=[0])
    [first] = events(client)
    post_bang(client, ring, ORIGIN, T0, only=[1, 2, 3])
    [later] = events(client)
    assert later["event_id"] == first["event_id"] and len(later["report_ids"]) == 4


def test_late_report_refines_the_same_event(client, clock):
    ring = register_ring(client)
    post_bang(client, ring, from_local(ORIGIN, 100, 100), T0, only=[0, 1, 2, 3])
    clock.advance(600)  # the fifth node's report arrives ten minutes late
    post_bang(client, ring, from_local(ORIGIN, 100, 100), T0, only=[4])
    [e] = events(client)
    assert len(e["report_ids"]) == 5


def test_two_bangs_seconds_apart_in_different_places_are_two_events(client):
    ring = register_ring(client)
    post_bang(client, ring, from_local(ORIGIN, -400, 0), T0)
    post_bang(client, ring, from_local(ORIGIN, 3000, 3000), T0 + timedelta(seconds=3))
    assert len(events(client)) == 2


def test_duplicate_report_is_idempotent(client):
    ring = register_ring(client, 3)
    r = make_impulse_report("s0", onset_at=T0, lat=ring[0].location.lat, lon=ring[0].location.lon)
    post_signed(client, "/v1/impulses", r, ring[0].private_key)
    again = post_signed(client, "/v1/impulses", r, ring[0].private_key)
    assert again.json()["status"] == "duplicate" and len(events(client)) == 1


def test_bad_signature_unknown_node_and_future_are_rejected(client):
    ring = register_ring(client, 1)
    other, _ = generate_keypair()
    r = make_impulse_report("s0", onset_at=T0, lat=60.17, lon=24.94)
    assert post_signed(client, "/v1/impulses", r, other).status_code == 401
    ghost = make_impulse_report("ghost", onset_at=T0, lat=60.17, lon=24.94)
    assert post_signed(client, "/v1/impulses", ghost, other).status_code == 401
    future = make_impulse_report("s0", onset_at=T0 + timedelta(minutes=5), lat=60.17, lon=24.94)
    assert post_signed(client, "/v1/impulses", future, ring[0].private_key).status_code == 400


def _plugin(name: str, cls) -> str:
    import sys
    import types

    mod = types.ModuleType(name)
    mod.Locator = cls
    sys.modules[name] = mod
    return f"{name}:Locator"


def test_a_crashing_locator_does_not_break_ingest(tmp_path):
    class Broken:
        def __init__(self, *_):
            pass

        def on_report(self, *_):
            raise RuntimeError("boom")

    app = create_app(Settings(db_path=tmp_path / "k.db", clock=FakeClock(T0), tick_interval_s=None,
                              impulse_locator=_plugin("broken_locator", Broken)))
    with TestClient(app) as c:
        keys = register(c, "s0")
        r = make_impulse_report("s0", onset_at=T0, lat=60.17, lon=24.94)
        assert post_signed(c, "/v1/impulses", r, keys.private_key).status_code == 200
        assert c.get("/v1/impulse-events").json() == []


def test_a_locator_returning_a_malformed_event_does_not_break_ingest(tmp_path):
    class Malformed:
        def __init__(self, *_):
            pass

        def on_report(self, *_):
            return "not an ImpulseEvent"  # e.g. a buggy plugin's wrong return type

    app = create_app(Settings(db_path=tmp_path / "k.db", clock=FakeClock(T0), tick_interval_s=None,
                              impulse_locator=_plugin("malformed_locator", Malformed)))
    with TestClient(app) as c:
        keys = register(c, "s0")
        r = make_impulse_report("s0", onset_at=T0, lat=60.17, lon=24.94)
        assert post_signed(c, "/v1/impulses", r, keys.private_key).status_code == 200
        assert c.get("/v1/impulse-events").json() == []


def test_the_plugin_receives_the_locator_config(tmp_path):
    seen = []

    class Spy:
        def __init__(self, config):
            seen.append(config)

        def on_report(self, *_):
            return None

    create_app(Settings(db_path=tmp_path / "k.db", tick_interval_s=None, air_temperature_c=-15.0,
                        impulse_locator=_plugin("spy_locator", Spy)))
    assert seen[0].air_temperature_c == -15.0


def test_unknown_locator_fails_at_startup_naming_the_setting(tmp_path):
    with pytest.raises(ImportError, match="impulse_locator"):
        create_app(Settings(db_path=tmp_path / "k.db", tick_interval_s=None,
                            impulse_locator="no_such_module:Locator"))


def test_settings_from_environment():
    s = Settings.from_env({"KUULO_IMPULSE_LOCATOR": "a.b:C", "KUULO_FUSION_ENGINE": "d.e:F",
                           "KUULO_AIR_TEMPERATURE_C": "-5"})
    assert (s.impulse_locator, s.fusion_engine, s.air_temperature_c) == ("a.b:C", "d.e:F", -5.0)
    assert Settings.from_env({}).impulse_locator == "kuulo_server.impact.coarse:CoarseLocator"


def test_events_survive_restart_and_detail_lists_reports(tmp_path):
    settings = Settings(db_path=tmp_path / "k.db", clock=FakeClock(T0), tick_interval_s=None)
    with TestClient(create_app(settings)) as c:
        post_bang(c, register_ring(c), ORIGIN, T0)
    with TestClient(create_app(settings)) as c:
        [e] = events(c)
        detail = c.get(f"/v1/impulse-events/{e['event_id']}").json()
        assert len(detail["reports"]) == 6
        assert c.get(f"/v1/impulse-events/{uuid4()}").status_code == 404


def test_impulse_events_are_published_live(client):
    ring = register_ring(client, 3)
    with client.websocket_connect("/v1/live") as ws:
        post_bang(client, ring, ORIGIN, T0, only=[0])
        assert '"impulse_event"' in ws.receive_text()

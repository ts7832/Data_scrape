from fastapi.testclient import TestClient

from kuulo_protocol.models import Track, TrackStatus
from kuulo_protocol.testing import make_heartbeat, make_observation
from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.testing import T0, FakeClock, post_signed, register


def test_observations_create_confirmed_track_with_evidence(client, node, clock):
    n2 = register(client, "n2", lon=24.9438)
    post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0),
                node.private_key)
    post_signed(client, "/v1/observations",
                make_observation("n2", observed_at=T0, lon=24.9438), n2.private_key)
    [track] = client.get("/v1/tracks").json()
    assert track["status"] == "confirmed"
    detail = client.get(f"/v1/tracks/{track['track_id']}").json()
    assert {o["source"]["id"] for o in detail["observations"]} == {"n1", "n2"}


def test_track_published_live_and_closed_by_tick(app, client, node, clock):
    with client.websocket_connect("/v1/live") as ws:
        post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0),
                    node.private_key)
        assert ws.receive_json()["type"] == "observation"
        track_event = ws.receive_json()
        assert track_event["type"] == "track" and track_event["data"]["status"] == "tentative"
        clock.advance(40)
        app.state.run_tick()
        closed = ws.receive_json()
        assert closed["type"] == "track" and closed["data"]["status"] == "closed"


def test_fusion_crash_does_not_break_ingest(tmp_path):
    clock = FakeClock(T0)
    settings = Settings(db_path=tmp_path / "k.db", clock=clock, tick_interval_s=None,
                        fusion_engine="tests_support_broken:Broken")
    import sys
    import types

    module = types.ModuleType("tests_support_broken")

    class Broken:
        def on_observation(self, obs, ctx):
            raise RuntimeError("boom")

        def on_tick(self, now, ctx):
            raise RuntimeError("boom")

    module.Broken = Broken
    sys.modules["tests_support_broken"] = module
    app = create_app(settings)
    with TestClient(app) as c:
        n1 = register(c, "n1")
        obs = make_observation("n1", observed_at=T0)
        r = post_signed(c, "/v1/observations", obs, n1.private_key)
        assert r.status_code == 200 and r.json()["status"] == "accepted"
        app.state.run_tick()  # must not raise


def test_restart_closes_open_tracks(tmp_path):
    # Review Focus 3.
    clock = FakeClock(T0)
    settings = Settings(db_path=tmp_path / "k.db", clock=clock, tick_interval_s=None)
    with TestClient(create_app(settings)) as c:
        n1 = register(c, "n1")
        post_signed(c, "/v1/observations", make_observation("n1", observed_at=T0), n1.private_key)
        assert c.get("/v1/tracks").json()[0]["status"] == "tentative"
    with TestClient(create_app(settings)) as c:
        [track] = c.get("/v1/tracks").json()
        assert Track.model_validate(track).status is TrackStatus.CLOSED


def test_uncorroborated_rate(client, node, clock):
    n2 = register(client, "n2", lon=24.9438)
    post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0), node.private_key)
    post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0),
                node.private_key)
    views = {n["node_id"]: n for n in client.get("/v1/nodes").json()}
    assert views["n1"]["uncorroborated_rate_24h"] == 1.0
    assert views["n2"]["uncorroborated_rate_24h"] is None
    clock.advance(2)
    post_signed(client, "/v1/observations",
                make_observation("n2", observed_at=clock(), lon=24.9438), n2.private_key)
    views = {n["node_id"]: n for n in client.get("/v1/nodes").json()}
    assert views["n1"]["uncorroborated_rate_24h"] == 0.0

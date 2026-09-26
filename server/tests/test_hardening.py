"""Deferred review findings from Part 1, plus batch ingest (spec §6: 'one or a batch')."""

import base64
from datetime import timedelta

import pytest
from starlette.websockets import WebSocketDisconnect

import kuulo_server.app as app_module
from kuulo_protocol.models import SensorLocation, TimeQuality
from kuulo_protocol.signing import generate_keypair, sign
from kuulo_protocol.testing import make_heartbeat, make_observation
from kuulo_server.testing import T0, post_signed, register


def _registration(public_key: str) -> dict:
    # A raw body: the model itself now refuses to build with a malformed key.
    location = SensorLocation(lat=60.17, lon=24.94, accuracy_m=10).model_dump(mode="json")
    return {"node_id": "bad-key", "public_key": public_key, "location": location,
            "time_quality": TimeQuality.NTP.value}


@pytest.mark.parametrize("key", ["not-base64!", base64.b64encode(b"x" * 31).decode()])
def test_registration_rejects_malformed_public_key(client, key):
    response = client.post("/v1/nodes/register", json=_registration(key))
    assert response.status_code == 400


def test_replayed_older_heartbeat_is_rejected(client, node, clock):
    fresh = make_heartbeat(node.node_id, sent_at=T0)
    assert post_signed(client, "/v1/heartbeats", fresh, node.private_key).status_code == 200
    clock.advance(200)  # node would now be stale without a new heartbeat
    old = make_heartbeat(node.node_id, sent_at=T0 - timedelta(seconds=10))
    response = post_signed(client, "/v1/heartbeats", old, node.private_key)
    assert response.status_code == 409
    [view] = client.get("/v1/nodes").json()
    assert view["status"] == "stale"


def test_same_heartbeat_replayed_is_rejected(client, node):
    hb = make_heartbeat(node.node_id, sent_at=T0)
    assert post_signed(client, "/v1/heartbeats", hb, node.private_key).status_code == 200
    assert post_signed(client, "/v1/heartbeats", hb, node.private_key).status_code == 409


def test_websocket_rejects_foreign_origin(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/v1/live", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


@pytest.mark.parametrize("headers", [{"origin": "http://127.0.0.1:5173"},
                                     {"origin": "http://localhost:5173"}, {}])
def test_websocket_accepts_dashboard_and_non_browser(client, node, headers):
    with client.websocket_connect("/v1/live", headers=headers) as ws:
        post_signed(client, "/v1/observations", make_observation(node.node_id, observed_at=T0),
                    node.private_key)
        assert '"observation"' in ws.receive_text()


def _signed_json(msg, key) -> dict:
    return sign(msg, key).model_dump(mode="json")


def test_batch_ingest_reports_each_item(client, node):
    other_priv, _ = generate_keypair()
    items = [
        _signed_json(make_observation(node.node_id, observed_at=T0), node.private_key),
        _signed_json(make_observation(node.node_id, observed_at=T0), other_priv),
        _signed_json(make_observation(node.node_id, observed_at=T0), node.private_key),
    ]
    response = client.post("/v1/observations", json=items)
    assert response.status_code == 200
    results = response.json()
    assert [r["status"] for r in results] == ["accepted", "rejected", "accepted"]
    assert "signature" in results[1]["reason"]
    tracks = client.get("/v1/tracks").json()
    assert len(tracks) == 1 and len(tracks[0]["observation_ids"]) == 2  # both good items fused


def test_batch_item_that_fails_validation_is_rejected_not_fatal(client, node):
    good = _signed_json(make_observation(node.node_id, observed_at=T0), node.private_key)
    response = client.post("/v1/observations", json=[{"nonsense": True}, good])
    assert response.status_code == 200
    assert [r["status"] for r in response.json()] == ["rejected", "accepted"]


@pytest.mark.parametrize("count", [0, 101])
def test_batch_size_limits(client, node, count):
    item = _signed_json(make_observation(node.node_id, observed_at=T0), node.private_key)
    response = client.post("/v1/observations", json=[item] * count)
    assert response.status_code == 400


def test_tick_only_computes_noisy_rate_for_changed_nodes(app, client, clock, monkeypatch):
    for i in range(5):
        keys = register(client, f"n{i}")
        post_signed(client, "/v1/heartbeats", make_heartbeat(keys.node_id, sent_at=T0),
                    keys.private_key)
    calls = []
    real = app_module.uncorroborated_rate
    monkeypatch.setattr(app_module, "uncorroborated_rate",
                        lambda *a: calls.append(a) or real(*a))
    app.state.run_tick()  # first tick learns statuses; nothing changes
    app.state.run_tick()
    assert calls == []
    clock.advance(150)  # all five go stale
    app.state.run_tick()
    assert len(calls) == 5


def test_existing_database_gains_new_columns(tmp_path):
    import sqlite3

    from kuulo_server.db import make_session_factory

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:  # a Part 1 database: nodes without the new column
        conn.execute("CREATE TABLE nodes (node_id VARCHAR(64) PRIMARY KEY, public_key VARCHAR(64),"
                     " lat FLOAT, lon FLOAT, accuracy_m FLOAT, time_quality VARCHAR(16),"
                     " registered_at DATETIME, last_heartbeat_at DATETIME,"
                     " software_version VARCHAR(32), mic_ok BOOLEAN, queue_depth INTEGER)")
    make_session_factory(path)
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(nodes)")}
    assert "last_heartbeat_sent_at" in columns

"""FeatureTrace upload and server pull (spec §5.1, §6, §7 step 5)."""

import io
from datetime import timedelta
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from kuulo_protocol.models import Label
from kuulo_protocol.signing import generate_keypair, sign
from kuulo_protocol.testing import make_heartbeat, make_observation, make_trace
from kuulo_protocol.traces import TraceUnavailable
from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.testing import T0, FakeClock, post_signed, register


def upload(client, header, body: bytes):
    return client.post(
        "/v1/traces", data={"header": header.model_dump_json()},
        files={"body": ("trace.npz", body, "application/octet-stream")},
    )


def detect(client, keys, detection_id, at=T0, label=Label.DRONE_MULTIROTOR):
    obs = make_observation(keys.node_id, observed_at=at, detection_id=detection_id, label=label)
    assert post_signed(client, "/v1/observations", obs, keys.private_key).status_code == 200


def requests_for(client, node_id):
    response = client.get("/v1/traces/requests", params={"node_id": node_id})
    assert response.status_code == 200
    return response.json()


def confirmed_pair(client):
    a, b = register(client, "a"), register(client, "b")
    da, db = uuid4(), uuid4()
    detect(client, a, da)
    detect(client, b, db)
    return a, b, da, db


def test_confirmed_track_requests_every_contributing_detection(client):
    a, b, da, db = confirmed_pair(client)
    assert [r["detection_id"] for r in requests_for(client, "a")] == [str(da)]
    assert [r["detection_id"] for r in requests_for(client, "b")] == [str(db)]


def test_tentative_track_requests_nothing(client, node):
    detect(client, node, uuid4())
    assert requests_for(client, node.node_id) == []


def test_repeated_confirmation_does_not_duplicate_requests(client):
    a, b, da, db = confirmed_pair(client)
    detect(client, a, da, at=T0 + timedelta(seconds=2))
    assert len(requests_for(client, "a")) == 1


def test_segments_stored_and_final_closes_request(app, client):
    a, _, da, _ = confirmed_pair(client)
    first, body0 = make_trace("a", a.private_key, da, start_at=T0, segment_index=0, final=False)
    last, body1 = make_trace("a", a.private_key, da, start_at=T0 + timedelta(seconds=60),
                             segment_index=1, final=True)
    assert upload(client, first, body0).json() == {"status": "stored"}
    assert len(requests_for(client, "a")) == 1  # not final yet
    assert upload(client, last, body1).json() == {"status": "stored"}
    assert requests_for(client, "a") == []
    stored = app.state.settings.traces_dir / "a" / f"{first.trace_id}.npz"
    assert stored.read_bytes() == body0


def test_duplicate_trace_is_idempotent(client, node):
    header, body = make_trace(node.node_id, node.private_key, uuid4(), start_at=T0)
    upload(client, header, body)
    assert upload(client, header, body).json() == {"status": "duplicate"}


def test_unrequested_trace_is_accepted(client, node):
    # Auto-push and hard-negative samples arrive without a request.
    header, body = make_trace(node.node_id, node.private_key, uuid4(), start_at=T0)
    assert upload(client, header, body).status_code == 200


def test_body_not_matching_hash_is_400(client, node):
    header, body = make_trace(node.node_id, node.private_key, uuid4(), start_at=T0)
    _, other = make_trace(node.node_id, node.private_key, uuid4(), start_at=T0, frames=49)
    assert upload(client, header, other).status_code == 400


def test_pickled_body_is_400_and_never_loaded(client, node, monkeypatch):
    import hashlib

    buf = io.BytesIO()
    np.savez(buf, t_offset_ms=np.empty(50, dtype=object), band_db=np.zeros((50, 32)),
             rms_db=np.zeros(50), peak_freq_hz=np.zeros(50))
    body = buf.getvalue()
    header, _ = make_trace(node.node_id, node.private_key, uuid4(), start_at=T0)
    header = sign(header.model_copy(update={"body_sha256": hashlib.sha256(body).hexdigest()}),
                  node.private_key)
    import pickle

    monkeypatch.setattr(pickle, "loads", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert upload(client, header, body).status_code == 400


def test_oversized_body_is_400(client, node):
    header, _ = make_trace(node.node_id, node.private_key, uuid4(), start_at=T0)
    assert upload(client, header, b"\0" * (5 * 1024 * 1024)).status_code == 400


def test_malformed_header_is_400(client, node):
    _, body = make_trace(node.node_id, node.private_key, uuid4(), start_at=T0)
    response = client.post("/v1/traces", data={"header": "{not json"},
                           files={"body": ("t.npz", body, "application/octet-stream")})
    assert response.status_code == 400


def test_bad_signature_is_401(client, node):
    other, _ = generate_keypair()
    header, body = make_trace(node.node_id, other, uuid4(), start_at=T0)
    assert upload(client, header, body).status_code == 401


def test_unknown_node_is_401(client):
    priv, _ = generate_keypair()
    header, body = make_trace("ghost", priv, uuid4(), start_at=T0)
    assert upload(client, header, body).status_code == 401


def test_unavailable_closes_request(client):
    a, _, da, _ = confirmed_pair(client)
    msg = TraceUnavailable(node_id="a", detection_id=da, sent_at=T0)
    assert post_signed(client, "/v1/traces/unavailable", msg, a.private_key).status_code == 200
    assert requests_for(client, "a") == []


def test_unavailable_with_bad_signature_is_401(client):
    confirmed_pair(client)
    other, _ = generate_keypair()
    msg = TraceUnavailable(node_id="a", detection_id=uuid4(), sent_at=T0)
    assert post_signed(client, "/v1/traces/unavailable", msg, other).status_code == 401


def test_requests_survive_server_restart(tmp_path):
    settings = Settings(db_path=tmp_path / "k.db", clock=FakeClock(T0), tick_interval_s=None)
    with TestClient(create_app(settings)) as client:
        confirmed_pair(client)
    with TestClient(create_app(settings)) as client:
        assert len(requests_for(client, "a")) == 1


def test_track_detail_counts_trace_segments(client):
    a, _, da, _ = confirmed_pair(client)
    for i in range(2):
        header, body = make_trace("a", a.private_key, da, start_at=T0, segment_index=i,
                                  final=i == 1)
        upload(client, header, body)
    [track] = client.get("/v1/tracks").json()
    detail = client.get(f"/v1/tracks/{track['track_id']}").json()
    assert detail["trace_segments"] == 2


def test_heartbeat_fixture_still_works(client, node):
    # Guard: the new routes must not shadow existing ones.
    hb = make_heartbeat(node.node_id, sent_at=T0)
    assert post_signed(client, "/v1/heartbeats", hb, node.private_key).status_code == 200

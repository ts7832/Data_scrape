from datetime import timedelta

from kuulo_protocol.models import NodeRegistration, SensorLocation, TimeQuality
from kuulo_protocol.signing import generate_keypair
from kuulo_protocol.testing import make_observation
from kuulo_server.testing import T0, post_signed


def test_signed_observation_accepted(client, node):
    obs = make_observation(node.node_id, observed_at=T0)
    response = post_signed(client, "/v1/observations", obs, node.private_key)
    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "late": False}


def test_extreme_snr_db_rejected_over_http(client, node):
    # Review finding: reproduced via a raw HTTP body (not just Python model
    # construction), the same path a real attacker or buggy node would use.
    obs = make_observation(node.node_id, observed_at=T0)
    body = obs.model_dump(mode="json")
    body["acoustic"]["snr_db"] = 1e4
    response = client.post("/v1/observations", json=body)
    assert response.status_code == 400


def test_infinite_snr_db_rejected_over_http(client, node):
    # httpx's own `json=` kwarg refuses to encode inf/nan client-side, so this
    # sends a hand-built body over the wire the way a non-httpx caller could:
    # a bare `Infinity` JSON literal, which Python's stdlib json module accepts
    # by default (json.dumps(..., allow_nan=True) is the default).
    import json as jsonlib

    obs = make_observation(node.node_id, observed_at=T0)
    body = obs.model_dump(mode="json")
    body["acoustic"]["snr_db"] = float("inf")
    response = client.post(
        "/v1/observations", content=jsonlib.dumps(body),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


def test_retry_is_duplicate_not_double(client, node):
    obs = make_observation(node.node_id, observed_at=T0)
    post_signed(client, "/v1/observations", obs, node.private_key)
    again = post_signed(client, "/v1/observations", obs, node.private_key)
    assert again.status_code == 200
    assert again.json()["status"] == "duplicate"


def test_bad_signature_rejected(client, node):
    other_priv, _ = generate_keypair()
    obs = make_observation(node.node_id, observed_at=T0)
    response = post_signed(client, "/v1/observations", obs, other_priv)
    assert response.status_code == 401
    assert "signature" in response.json()["detail"]


def test_unsigned_rejected(client, node):
    obs = make_observation(node.node_id, observed_at=T0)
    response = client.post("/v1/observations", content=obs.model_dump_json(),
                           headers={"content-type": "application/json"})
    assert response.status_code == 401


def test_unknown_node_rejected(client):
    priv, _ = generate_keypair()
    obs = make_observation("ghost", observed_at=T0)
    response = post_signed(client, "/v1/observations", obs, priv)
    assert response.status_code == 401
    assert "unknown node" in response.json()["detail"]


def test_malformed_body_is_400(client, node):
    response = client.post("/v1/observations", json={"hello": "world"})
    assert response.status_code == 400


def test_future_timestamp_rejected(client, node, clock):
    obs = make_observation(node.node_id, observed_at=T0)
    clock.set(T0 - timedelta(seconds=31))
    response = post_signed(client, "/v1/observations", obs, node.private_key)
    assert response.status_code == 400
    assert "future" in response.json()["detail"]


def test_old_observation_accepted_but_late(client, node, clock):
    obs = make_observation(node.node_id, observed_at=T0)
    clock.advance(120)
    response = post_signed(client, "/v1/observations", obs, node.private_key)
    assert response.json() == {"status": "accepted", "late": True}


def test_reregister_with_different_key_is_conflict(client, node):
    # Review Focus 2: a node identity cannot be taken over by re-registering.
    _, attacker_pub = generate_keypair()
    reg = NodeRegistration(
        node_id=node.node_id, public_key=attacker_pub,
        location=SensorLocation(lat=60.0, lon=25.0, accuracy_m=5), time_quality=TimeQuality.NTP,
    )
    response = client.post("/v1/nodes/register", json=reg.model_dump(mode="json"))
    assert response.status_code == 409


def test_reregister_same_key_updates_location(client, node):
    reg = NodeRegistration(
        node_id=node.node_id, public_key=node.public_key,
        location=SensorLocation(lat=60.2, lon=25.0, accuracy_m=5), time_quality=TimeQuality.GPS,
    )
    assert client.post("/v1/nodes/register", json=reg.model_dump(mode="json")).status_code == 200

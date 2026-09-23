from datetime import UTC, datetime, timedelta, timezone

from kuulo_protocol.api import LiveEvent, NodeStatus
from kuulo_protocol.models import Observation
from kuulo_protocol.schema import export_schema
from kuulo_protocol.signing import (
    canonical_bytes,
    generate_keypair,
    keypair_from_seed,
    sign,
    verify,
)
from kuulo_protocol.testing import make_heartbeat, make_observation

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_sign_and_verify():
    priv, pub = generate_keypair()
    obs = sign(make_observation(observed_at=T0), priv)
    assert obs.signature
    assert verify(obs, pub)


def test_tampered_message_fails():
    priv, pub = generate_keypair()
    obs = sign(make_observation(observed_at=T0, confidence=0.4), priv)
    forged_detection = obs.detection.model_copy(update={"confidence": 0.99})
    forged = obs.model_copy(update={"detection": forged_detection})
    assert not verify(forged, pub)


def test_wrong_key_fails():
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    assert not verify(sign(make_observation(observed_at=T0), priv), other_pub)


def test_garbage_signature_returns_false_not_exception():
    _, pub = generate_keypair()
    obs = make_observation(observed_at=T0).model_copy(update={"signature": "not base64!!"})
    assert verify(obs, pub) is False
    assert verify(make_observation(observed_at=T0), pub) is False  # empty signature


def test_signature_survives_json_round_trip_with_awkward_inputs():
    # Review Focus 1: sub-millisecond, non-UTC timestamp and a non-ASCII node id.
    priv, pub = generate_keypair()
    ts = datetime(2026, 9, 24, 15, 0, 0, 987654, tzinfo=timezone(timedelta(hours=3)))
    obs = sign(make_observation("solmu-ÄÖ-1", observed_at=ts), priv)
    received = Observation.model_validate_json(obs.model_dump_json().encode("utf-8"))
    assert verify(received, pub)


def test_canonical_bytes_exclude_signature_and_sort_keys():
    obs = make_observation(observed_at=T0)
    a = canonical_bytes(obs)
    b = canonical_bytes(obs.model_copy(update={"signature": "xyz"}))
    assert a == b
    assert b'"signature"' not in a
    assert a.index(b'"acoustic"') < a.index(b'"detection"')


def test_heartbeat_signing():
    priv, pub = generate_keypair()
    assert verify(sign(make_heartbeat(sent_at=T0), priv), pub)


def test_keypair_from_seed_is_deterministic():
    assert keypair_from_seed(b"a" * 32) == keypair_from_seed(b"a" * 32)
    assert keypair_from_seed(b"a" * 32) != keypair_from_seed(b"b" * 32)


def test_live_event_resync_has_no_data():
    ev = LiveEvent(type="resync")
    assert ev.data is None
    assert NodeStatus.ONLINE == "online"


def test_schema_export_contains_all_models():
    schema = export_schema()
    for name in [
        "Observation", "Heartbeat", "Track", "NodeRegistration",
        "NodeView", "TrackDetail", "LiveEvent", "IngestResult",
    ]:
        assert name in schema["$defs"]
        assert schema["properties"][name] == {"$ref": f"#/$defs/{name}"}

import json

import httpx
import pytest

from kuulo_node.uplink import RegistrationConflict, Uplink
from kuulo_protocol.models import Heartbeat, NodeRegistration, SensorLocation, TimeQuality
from kuulo_protocol.signing import generate_keypair

REG = NodeRegistration(
    node_id="demo-laptop", public_key=generate_keypair()[1],
    location=SensorLocation(lat=60.1694, lon=24.949, accuracy_m=50), time_quality=TimeQuality.NTP,
)


def hb(i=0):
    from datetime import UTC, datetime
    return Heartbeat(node_id="demo-laptop", sent_at=datetime(2026, 9, 24, tzinfo=UTC),
                     software_version=str(i), mic_ok=True, queue_depth=0)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def client(handler):
    return httpx.Client(base_url="http://node.test", transport=httpx.MockTransport(handler))


def test_server_down_at_start_is_not_fatal_and_backs_off():
    calls = []

    def down(request):
        calls.append(request.url.path)
        raise httpx.ConnectError("refused")

    clock = Clock()
    up = Uplink(client(down), monotonic=clock)
    assert up.ensure_registered(REG) is False
    assert up.ensure_registered(REG) is False  # within backoff: no second attempt
    assert calls == ["/v1/nodes/register"]
    up.send("/v1/heartbeats", hb())
    assert up.pending_count == 1
    clock.t = 1.5
    up.ensure_registered(REG)
    assert len(calls) == 2


def test_registers_then_flushes_in_order():
    seen = []

    def ok(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    up = Uplink(client(ok))
    up.send("/v1/heartbeats", hb(1))
    up.flush()
    assert seen == []  # not registered yet: nothing sent
    assert up.ensure_registered(REG)
    up.flush()
    assert seen == ["/v1/nodes/register", "/v1/heartbeats"] and up.pending_count == 0


def test_key_conflict_names_the_node_and_the_fix():
    up = Uplink(client(lambda r: httpx.Response(409, json={"detail": "different key"})))
    with pytest.raises(RegistrationConflict, match="demo-laptop") as err:
        up.ensure_registered(REG)
    assert "node_id" in str(err.value)


def test_unknown_node_after_server_reset_triggers_reregistration():
    responses = iter([httpx.Response(200), httpx.Response(401, json={"detail": "unknown node"})])
    up = Uplink(client(lambda r: next(responses)))
    up.ensure_registered(REG)
    up.send("/v1/heartbeats", hb())
    up.flush()
    assert up.registered is False and up.pending_count == 1  # kept, not dropped


def test_rejected_message_is_dropped_and_counted():
    responses = iter([httpx.Response(200), httpx.Response(400, json={"detail": "bad"})])
    up = Uplink(client(lambda r: next(responses)))
    up.ensure_registered(REG)
    up.send("/v1/heartbeats", hb())
    up.flush()
    assert up.pending_count == 0 and up.dropped == 1


def test_full_queue_drops_heartbeats_before_observations():
    up = Uplink(client(lambda r: httpx.Response(500)), max_pending=3)
    up.send("/v1/observations", obs(1))
    up.send("/v1/heartbeats", hb(2))
    up.send("/v1/observations", obs(3))
    up.send("/v1/observations", obs(4))
    kept = [json.loads(body)["detection"]["confidence"] for _, _, body in up.outbox.peek(10)]
    assert kept == [0.01, 0.03, 0.04] and up.dropped == 1


def obs(i=0):
    from datetime import UTC, datetime

    from kuulo_protocol.testing import make_observation
    return make_observation("demo-laptop", observed_at=datetime(2026, 9, 24, tzinfo=UTC),
                            confidence=i / 100)


def test_consecutive_observations_go_as_one_batch():
    seen = []

    def server(request):
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/v1/observations":
            return httpx.Response(200, json=[{"status": "accepted", "late": False}] * 3)
        return httpx.Response(200, json={"status": "ok"})

    up = Uplink(client(server))
    up.ensure_registered(REG)
    for i in range(3):
        up.send("/v1/observations", obs(i))
    up.send("/v1/heartbeats", hb())
    up.flush()
    paths = [p for p, _ in seen]
    assert paths == ["/v1/nodes/register", "/v1/observations", "/v1/heartbeats"]
    assert isinstance(seen[1][1], list) and len(seen[1][1]) == 3
    assert up.pending_count == 0


def test_batch_item_rejections_are_dropped_and_counted():
    def server(request):
        if request.url.path == "/v1/observations":
            return httpx.Response(200, json=[{"status": "accepted"},
                                             {"status": "rejected", "reason": "bad signature"}])
        return httpx.Response(200)

    up = Uplink(client(server))
    up.ensure_registered(REG)
    up.send("/v1/observations", obs(1))
    up.send("/v1/observations", obs(2))
    up.flush()
    assert up.pending_count == 0 and up.dropped == 1


def test_batch_unknown_node_keeps_messages_and_reregisters():
    def server(request):
        if request.url.path == "/v1/observations":
            return httpx.Response(200, json=[{"status": "rejected", "reason": "unknown node: x"}])
        return httpx.Response(200)

    up = Uplink(client(server))
    up.ensure_registered(REG)
    up.send("/v1/observations", obs(1))
    up.flush()
    assert up.registered is False and up.pending_count == 1 and up.dropped == 0


def test_transport_error_keeps_the_batch():
    calls = iter([httpx.Response(200)])

    def server(request):
        if request.url.path == "/v1/nodes/register":
            return next(calls)
        raise httpx.ConnectError("down")

    up = Uplink(client(server))
    up.ensure_registered(REG)
    up.send("/v1/observations", obs(1))
    up.flush()
    assert up.pending_count == 1


def test_pending_messages_survive_a_node_restart(tmp_path):
    from kuulo_node.outbox import Outbox

    path = tmp_path / "outbox.db"
    down = Uplink(client(lambda r: httpx.Response(503)), outbox=Outbox(path))
    down.send("/v1/observations", obs(7))
    down.outbox.close()
    seen = []

    def ok(request):
        seen.append(json.loads(request.content))
        if request.url.path == "/v1/observations":
            return httpx.Response(200, json=[{"status": "accepted"}])
        return httpx.Response(200)

    up = Uplink(client(ok), outbox=Outbox(path))
    up.ensure_registered(REG)
    up.flush()
    assert seen[-1][0]["detection"]["confidence"] == 0.07 and up.pending_count == 0

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
    up.send("/v1/observations", hb(1))
    up.send("/v1/heartbeats", hb(2))
    up.send("/v1/observations", hb(3))
    up.send("/v1/observations", hb(4))
    kept = [m.software_version for _, m in up._pending]
    assert kept == ["1", "3", "4"] and up.dropped == 1

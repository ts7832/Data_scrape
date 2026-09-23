import asyncio
from datetime import timedelta

from kuulo_protocol.api import LiveEvent, NodeStatus
from kuulo_protocol.testing import make_heartbeat, make_observation
from kuulo_server.live import LiveHub
from kuulo_server.status import node_status
from kuulo_server.testing import T0, post_signed, register


def test_node_status_thresholds():
    assert node_status(None, T0) is NodeStatus.OFFLINE
    assert node_status(T0, T0 + timedelta(seconds=119)) is NodeStatus.ONLINE
    assert node_status(T0, T0 + timedelta(seconds=120)) is NodeStatus.STALE
    assert node_status(T0, T0 + timedelta(seconds=299)) is NodeStatus.STALE
    assert node_status(T0, T0 + timedelta(seconds=300)) is NodeStatus.OFFLINE


def test_registered_node_without_heartbeat_is_offline(client, node):
    nodes = client.get("/v1/nodes").json()
    assert nodes[0]["node_id"] == "n1"
    assert nodes[0]["status"] == "offline"


def test_heartbeat_makes_node_online_then_it_ages(client, node, clock):
    response = post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0),
                           node.private_key)
    assert response.status_code == 200
    assert client.get("/v1/nodes").json()[0]["status"] == "online"
    clock.advance(180)
    assert client.get("/v1/nodes").json()[0]["status"] == "stale"
    clock.advance(180)
    assert client.get("/v1/nodes").json()[0]["status"] == "offline"


def test_heartbeat_bad_signature(client, node):
    other = register(client, "n2")
    response = post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0),
                           other.private_key)
    assert response.status_code == 401


def test_websocket_receives_observation(client, node):
    with client.websocket_connect("/v1/live") as ws:
        obs = make_observation("n1", observed_at=T0)
        post_signed(client, "/v1/observations", obs, node.private_key)
        event = ws.receive_json()
        assert event["type"] == "observation"
        assert event["data"]["observation_id"] == str(obs.observation_id)


def test_late_observation_not_published_live(client, node, clock):
    with client.websocket_connect("/v1/live") as ws:
        clock.advance(120)
        post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0),
                    node.private_key)
        post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=clock()),
                    node.private_key)
        assert ws.receive_json()["type"] == "node_status"  # the late observation was skipped


def test_tick_publishes_status_transition(app, client, node, clock):
    post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0), node.private_key)
    app.state.run_tick()  # records "online"
    with client.websocket_connect("/v1/live") as ws:
        clock.advance(150)
        app.state.run_tick()
        event = ws.receive_json()
        assert event == {"type": "node_status", "data": event["data"]}
        assert event["data"]["status"] == "stale"


def test_unknown_track_is_404(client):
    assert client.get("/v1/tracks/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/v1/tracks").json() == []


def test_stalled_client_gets_resync_and_others_unaffected():
    # Review Focus 4.
    async def scenario():
        hub = LiveHub(max_queue=2)
        slow, fast = hub.subscribe(), hub.subscribe()
        for _ in range(2):
            hub.publish(LiveEvent(type="resync"))
            await fast.get()
        hub.publish(LiveEvent(type="resync"))  # slow queue is full now
        assert slow.qsize() == 1
        assert '"resync"' in slow.get_nowait()
        assert fast.qsize() == 1
        assert hub.client_count == 2

    asyncio.run(scenario())

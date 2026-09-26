"""Helpers for server tests and scenario harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import BaseModel

from kuulo_protocol.geo import distance_m, from_local
from kuulo_protocol.impulses import speed_of_sound
from kuulo_protocol.models import GeoPoint, NodeRegistration, SensorLocation, TimeQuality
from kuulo_protocol.signing import generate_keypair, sign
from kuulo_protocol.testing import make_impulse_report

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def set(self, now: datetime) -> None:
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass(frozen=True)
class NodeKeys:
    node_id: str
    private_key: str
    public_key: str


def register(
    client: TestClient, node_id: str, lat: float = 60.1699, lon: float = 24.9384
) -> NodeKeys:
    priv, pub = generate_keypair()
    reg = NodeRegistration(
        node_id=node_id,
        public_key=pub,
        location=SensorLocation(lat=lat, lon=lon, accuracy_m=10),
        time_quality=TimeQuality.NTP,
    )
    response = client.post("/v1/nodes/register", json=reg.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    return NodeKeys(node_id, priv, pub)


def post_signed(client: TestClient, path: str, msg: BaseModel, private_key: str):
    return client.post(
        path,
        content=sign(msg, private_key).model_dump_json(),
        headers={"content-type": "application/json"},
    )


IMPULSE_ORIGIN = GeoPoint(lat=60.17, lon=24.94)
RING_OFFSETS = [(-800, -600), (900, -500), (700, 800), (-600, 900), (0, -1000), (1000, 200)]


@dataclass(frozen=True)
class RingNode:
    keys: NodeKeys
    location: GeoPoint

    @property
    def node_id(self) -> str:
        return self.keys.node_id

    @property
    def private_key(self) -> str:
        return self.keys.private_key


def register_ring(client: TestClient, n: int = 6, prefix: str = "s") -> list[RingNode]:
    """n nodes at fixed offsets (metres) around IMPULSE_ORIGIN."""
    out = []
    for i, (x, y) in enumerate(RING_OFFSETS[:n]):
        p = from_local(IMPULSE_ORIGIN, x, y)
        out.append(RingNode(register(client, f"{prefix}{i}", lat=p.lat, lon=p.lon), p))
    return out


def post_bang(client: TestClient, ring: list[RingNode], source: GeoPoint, at: datetime, *,
              only: list[int] | None = None, temp_c: float = 10.0) -> list[UUID]:
    """Post every (or `only` the listed) node's signed, GPS-timed report of an impulse."""
    c = speed_of_sound(temp_c)
    ids = []
    for node in ring if only is None else [ring[i] for i in only]:
        onset = at + timedelta(seconds=distance_m(node.location, source) / c)
        r = make_impulse_report(node.node_id, onset_at=onset, lat=node.location.lat,
                                lon=node.location.lon)
        response = post_signed(client, "/v1/impulses", r, node.private_key)
        assert response.status_code == 200, response.text
        ids.append(r.report_id)
    return ids

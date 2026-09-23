"""Helpers for server tests and scenario harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from pydantic import BaseModel

from kuulo_protocol.models import NodeRegistration, SensorLocation, TimeQuality
from kuulo_protocol.signing import generate_keypair, sign

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

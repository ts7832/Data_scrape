"""Scenario files: nodes, drones, false alarms and failures, validated with Pydantic."""

from __future__ import annotations

from importlib import resources
from itertools import pairwise
from math import hypot
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kuulo_protocol.geo import from_local, to_local
from kuulo_protocol.models import GeoPoint, Label, NodeId, TimeQuality


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeSpec(_Spec):
    id: NodeId
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    time_quality: TimeQuality = TimeQuality.NTP
    noise_floor_db: float = 35.0


class ImpactSpec(_Spec):
    """A drone that detonates or crashes hard at the end of its route."""

    source_db: float = 150.0


class ImpulseSpec(_Spec):
    """A standalone acoustic impulse (a firework, a bang) not caused by any drone."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    at_s: float = Field(ge=0)
    source_db: float = 150.0
    label: str = "impulse"


class DroneSpec(_Spec):
    id: str
    label: Label = Label.DRONE_MULTIROTOR
    source_db: float = 100.0
    speed_mps: float = Field(gt=0)
    start_s: float = Field(default=0.0, ge=0)
    waypoints: list[tuple[float, float]] = Field(min_length=2)  # (lat, lon)
    # >1: fly the route as a closed loop (back to the first waypoint) this many times.
    repeat: int = Field(default=1, ge=1)
    impact: ImpactSpec | None = None  # detonates at route_end_s / route_end_point


class FalseAlarmSpec(_Spec):
    """A sound that fools the classifier (leaf blower, two-stroke scooter) near some nodes."""

    lat: float
    lon: float
    radius_m: float = Field(default=100.0, gt=0)
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    confidence: float = Field(default=0.9, ge=0, le=1)
    label: Label = Label.DRONE_MULTIROTOR


class NodeFailureSpec(_Spec):
    node_id: str
    offline_from_s: float = Field(default=0.0, ge=0)
    offline_to_s: float | None = None  # None = until the end


class Scenario(_Spec):
    name: str
    seed: int = 0
    duration_s: float = Field(gt=0)
    tick_s: float = Field(default=1.0, gt=0)
    heartbeat_every_s: float = Field(default=30.0, gt=0)
    air_temperature_c: float = 10.0
    nodes: list[NodeSpec] = Field(min_length=1)
    drones: list[DroneSpec] = []
    false_alarms: list[FalseAlarmSpec] = []
    node_failures: list[NodeFailureSpec] = []
    impulses: list[ImpulseSpec] = []
    # Extra delay (seconds) a node's sound path adds, e.g. a building blocking a direct path.
    nlos_delay_s: dict[str, float] = {}

    @model_validator(mode="after")
    def _check_references(self) -> Scenario:
        ids = [n.id for n in self.nodes]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate node id(s): {sorted(duplicates)}")
        for failure in self.node_failures:
            if failure.node_id not in ids:
                raise ValueError(f"node_failures refers to unknown node {failure.node_id!r}")
        for node_id, delay in self.nlos_delay_s.items():
            if node_id not in ids:
                raise ValueError(f"nlos_delay_s refers to unknown node {node_id!r}")
            if not 0 <= delay <= 5:
                raise ValueError(f"nlos_delay_s[{node_id!r}] must be in [0, 5], got {delay}")
        return self


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(Path(path).read_text()))


def _scenario_dir() -> Path:
    return Path(str(resources.files("kuulo_sim") / "scenarios"))


def bundled_scenarios() -> list[str]:
    return sorted(p.stem for p in _scenario_dir().glob("*.yaml"))


def resolve_scenario(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    bundled = _scenario_dir() / f"{name_or_path}.yaml"
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"no scenario file or bundled scenario named {name_or_path!r}")


def _route(drone: DroneSpec) -> tuple[GeoPoint, list[tuple[float, float]]]:
    """The drone's local-metre route, repeated as a closed loop when drone.repeat > 1."""
    origin = GeoPoint(lat=drone.waypoints[0][0], lon=drone.waypoints[0][1])
    route = list(drone.waypoints)
    if drone.repeat > 1:
        loop = [*route, route[0]]
        route = loop + [p for _ in range(drone.repeat - 1) for p in loop[1:]]
    points = [to_local(origin, GeoPoint(lat=lat, lon=lon)) for lat, lon in route]
    return origin, points


def _route_length_m(points: list[tuple[float, float]]) -> float:
    return sum(hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in pairwise(points))


def drone_position(drone: DroneSpec, t: float) -> GeoPoint | None:
    """Where the drone is at scenario time t, or None before start / after its route ends."""
    if t < drone.start_s:
        return None
    origin, points = _route(drone)
    remaining = (t - drone.start_s) * drone.speed_mps
    for (x0, y0), (x1, y1) in pairwise(points):
        length = hypot(x1 - x0, y1 - y0)
        if remaining <= length:
            f = remaining / length if length else 0.0
            return from_local(origin, x0 + f * (x1 - x0), y0 + f * (y1 - y0))
        remaining -= length
    return None


def route_end_s(drone: DroneSpec) -> float:
    """Scenario time at which the drone reaches the end of its route."""
    _, points = _route(drone)
    return drone.start_s + _route_length_m(points) / drone.speed_mps


def route_end_point(drone: DroneSpec) -> GeoPoint:
    """Where the drone's route ends."""
    origin, points = _route(drone)
    return from_local(origin, *points[-1])


def node_offline(scenario: Scenario, node_id: str, t: float) -> bool:
    for failure in scenario.node_failures:
        if failure.node_id != node_id:
            continue
        end = failure.offline_to_s if failure.offline_to_s is not None else float("inf")
        if failure.offline_from_s <= t < end:
            return True
    return False

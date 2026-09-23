"""Scenario files: nodes, drones, false alarms and failures, validated with Pydantic."""

from __future__ import annotations

from importlib import resources
from itertools import pairwise
from math import hypot
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kuulo_protocol.geo import from_local, to_local
from kuulo_protocol.models import GeoPoint, Label, TimeQuality


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeSpec(_Spec):
    id: str = Field(min_length=1, max_length=64)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    time_quality: TimeQuality = TimeQuality.NTP
    noise_floor_db: float = 35.0


class DroneSpec(_Spec):
    id: str
    label: Label = Label.DRONE_MULTIROTOR
    source_db: float = 100.0
    speed_mps: float = Field(gt=0)
    start_s: float = Field(default=0.0, ge=0)
    waypoints: list[tuple[float, float]] = Field(min_length=2)  # (lat, lon)


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
    nodes: list[NodeSpec] = Field(min_length=1)
    drones: list[DroneSpec] = []
    false_alarms: list[FalseAlarmSpec] = []
    node_failures: list[NodeFailureSpec] = []

    @model_validator(mode="after")
    def _check_references(self) -> Scenario:
        ids = [n.id for n in self.nodes]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate node id(s): {sorted(duplicates)}")
        for failure in self.node_failures:
            if failure.node_id not in ids:
                raise ValueError(f"node_failures refers to unknown node {failure.node_id!r}")
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


def drone_position(drone: DroneSpec, t: float) -> GeoPoint | None:
    """Where the drone is at scenario time t, or None before start / after its route ends."""
    if t < drone.start_s:
        return None
    origin = GeoPoint(lat=drone.waypoints[0][0], lon=drone.waypoints[0][1])
    points = [to_local(origin, GeoPoint(lat=lat, lon=lon)) for lat, lon in drone.waypoints]
    remaining = (t - drone.start_s) * drone.speed_mps
    for (x0, y0), (x1, y1) in pairwise(points):
        length = hypot(x1 - x0, y1 - y0)
        if remaining <= length:
            f = remaining / length if length else 0.0
            return from_local(origin, x0 + f * (x1 - x0), y0 + f * (y1 - y0))
        remaining -= length
    return None


def node_offline(scenario: Scenario, node_id: str, t: float) -> bool:
    for failure in scenario.node_failures:
        if failure.node_id != node_id:
            continue
        end = failure.offline_to_s if failure.offline_to_s is not None else float("inf")
        if failure.offline_from_s <= t < end:
            return True
    return False

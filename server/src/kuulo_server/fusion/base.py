"""The plugin interface every fusion engine implements (the private engine included)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from kuulo_protocol.api import NodeStatus
from kuulo_protocol.models import GeoPoint, Observation, Track


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    location: GeoPoint
    status: NodeStatus


@dataclass(frozen=True)
class TrackUpdate:
    track: Track


class FusionContext(Protocol):
    def nodes(self) -> list[NodeInfo]: ...

    def recent_observations(self, since: datetime) -> list[Observation]:
        """Non-late, drone-labelled observations with observed_at >= since."""
        ...


class FusionEngine(Protocol):
    def on_observation(self, obs: Observation, ctx: FusionContext) -> list[TrackUpdate]: ...

    def on_tick(self, now: datetime, ctx: FusionContext) -> list[TrackUpdate]: ...

"""Baseline fusion: group nearby detections, weighted-centroid location, silent-neighbour logic.

Coarse by design (hundreds of metres). Precise localisation (TDOA, bearings, tracking filters)
belongs to the private engine behind the same interface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import atan2, degrees, hypot
from uuid import uuid4

from kuulo_protocol.api import NodeStatus
from kuulo_protocol.geo import distance_m, from_local, to_local
from kuulo_protocol.models import (
    DRONE_LABELS,
    GeoPoint,
    Observation,
    Phase,
    Track,
    TrackStatus,
    Velocity,
)

from .base import FusionContext, TrackUpdate


@dataclass(frozen=True)
class FusionConfig:
    group_radius_m: float = 2000.0
    group_window_s: float = 10.0
    base_range_m: float = 300.0
    gate_m: float = 1000.0
    gate_s: float = 10.0
    close_after_s: float = 30.0
    expected_hearing_m: float = 500.0
    silent_penalty: float = 0.7
    smoothing_alpha: float = 0.5


def _weight(obs: Observation) -> float:
    snr = obs.acoustic.snr_db if obs.acoustic else 0.0
    return obs.detection.confidence * 10 ** (snr / 20)


class BasicFusion:
    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()
        self.tracks: dict[str, Track] = {}

    def on_observation(self, obs: Observation, ctx: FusionContext) -> list[TrackUpdate]:
        cfg = self.config
        if obs.detection.label not in DRONE_LABELS or obs.event.phase is Phase.END:
            return []
        since = obs.observed_at - timedelta(seconds=cfg.group_window_s)
        candidates = [
            o for o in ctx.recent_observations(since)
            if o.detection.label in DRONE_LABELS
            and o.event.phase is not Phase.END
            and o.observed_at <= obs.observed_at
            and distance_m(o.sensor_location, obs.sensor_location) <= cfg.group_radius_m
        ]
        if all(o.observation_id != obs.observation_id for o in candidates):
            candidates.append(obs)
        latest: dict[str, Observation] = {}
        for o in sorted(candidates, key=lambda o: o.observed_at):
            latest[o.source.id] = o
        group = list(latest.values())

        centroid = self._centroid(group)
        spread = max(distance_m(centroid, o.sensor_location) for o in group)
        uncertainty = max(spread, cfg.base_range_m)
        detecting = set(latest)

        def in_hearing_range(n) -> bool:
            hearing = cfg.expected_hearing_m
            return any(distance_m(n.location, o.sensor_location) <= hearing for o in group)

        silent = sorted(
            n.node_id for n in ctx.nodes()
            if n.status is NodeStatus.ONLINE and n.node_id not in detecting and in_hearing_range(n)
        )
        if len(group) >= 2:
            status = TrackStatus.CONFIRMED
            confidence = 1.0
            for o in group:
                confidence *= 1 - o.detection.confidence
            confidence = 1 - confidence
            silent = []
        else:
            status = TrackStatus.DOWNGRADED if silent else TrackStatus.TENTATIVE
            confidence = group[0].detection.confidence * cfg.silent_penalty ** len(silent)
        label = Counter(o.detection.label for o in group).most_common(1)[0][0]

        existing = self._associate(centroid, obs.observed_at)
        if existing is None:
            track = Track(
                track_id=uuid4(), status=status, label=label, confidence=confidence,
                position=centroid, uncertainty_m=uncertainty, velocity=None,
                first_seen=obs.observed_at, last_seen=obs.observed_at,
                observation_ids=[o.observation_id for o in group],
                silent_neighbour_ids=silent,
            )
        else:
            track = self._update(existing, centroid, status, label, confidence, uncertainty, group,
                                 silent, obs.observed_at)
        self.tracks[str(track.track_id)] = track
        return [TrackUpdate(track)]

    def on_tick(self, now: datetime, ctx: FusionContext) -> list[TrackUpdate]:
        updates = []
        for key, track in list(self.tracks.items()):
            if (now - track.last_seen).total_seconds() > self.config.close_after_s:
                closed = track.model_copy(update={"status": TrackStatus.CLOSED})
                del self.tracks[key]
                updates.append(TrackUpdate(closed))
        return updates

    def _centroid(self, group: list[Observation]) -> GeoPoint:
        origin = group[0].sensor_location
        total = sum(_weight(o) for o in group) or 1.0
        x = y = 0.0
        for o in group:
            ox, oy = to_local(origin, o.sensor_location)
            x += ox * _weight(o) / total
            y += oy * _weight(o) / total
        return from_local(origin, x, y)

    def _associate(self, position: GeoPoint, at: datetime) -> Track | None:
        best, best_d = None, None
        for track in self.tracks.values():
            if abs((at - track.last_seen).total_seconds()) > self.config.gate_s:
                continue
            d = distance_m(track.position, position)
            if d <= self.config.gate_m and (best_d is None or d < best_d):
                best, best_d = track, d
        return best

    def _update(self, old: Track, centroid: GeoPoint, status: TrackStatus, label, confidence: float,
                uncertainty: float, group: list[Observation], silent: list[str],
                at: datetime) -> Track:
        a = self.config.smoothing_alpha
        dx, dy = to_local(old.position, centroid)
        position = from_local(old.position, a * dx, a * dy)
        dt = (at - old.last_seen).total_seconds()
        velocity = old.velocity
        if dt > 0:
            mx, my = to_local(old.position, position)
            velocity = Velocity(
                speed_mps=hypot(mx, my) / dt, heading_deg=degrees(atan2(mx, my)) % 360
            )
        if old.status is TrackStatus.CONFIRMED and status is not TrackStatus.CONFIRMED:
            status, silent = TrackStatus.CONFIRMED, []  # a confirmed track stays confirmed
            confidence = max(confidence, old.confidence)
        ids = list(dict.fromkeys([*old.observation_ids, *(o.observation_id for o in group)]))
        return old.model_copy(update={
            "status": status, "label": label, "confidence": confidence, "position": position,
            "uncertainty_m": uncertainty, "velocity": velocity, "last_seen": max(at, old.last_seen),
            "observation_ids": ids, "silent_neighbour_ids": silent,
        })

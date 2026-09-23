"""Turns a scenario into the exact signed messages real nodes would send."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from kuulo_protocol.geo import distance_m
from kuulo_protocol.models import (
    Acoustic,
    Detection,
    EventRef,
    GeoPoint,
    Heartbeat,
    Label,
    NodeRegistration,
    Observation,
    SensorLocation,
    Source,
    SourceType,
)
from kuulo_protocol.signing import keypair_from_seed, sign
from kuulo_protocol.smoothing import DetectionSmoother

from .physics import (
    clock_offset_s,
    confidence_from_snr,
    detection_probability,
    propagation_delay_s,
    received_level_db,
)
from .scenario import Scenario, drone_position, node_offline

SOFTWARE_VERSION = "sim-0.1.0"
FALSE_ALARM_SNR_DB = 20.0


@dataclass(frozen=True)
class SimMessage:
    at: datetime
    kind: Literal["observation", "heartbeat"]
    payload: Observation | Heartbeat


@dataclass(frozen=True)
class TruthPoint:
    at: datetime
    drone_id: str
    position: GeoPoint


@dataclass(frozen=True)
class _Heard:
    confidence: float
    snr_db: float
    label: Label
    delay_s: float


class SimulationRun:
    def __init__(self, scenario: Scenario, t0: datetime, time_scale: float = 1.0):
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self.scenario = scenario
        self.t0 = t0
        self.time_scale = time_scale
        self.node_keys: dict[str, tuple[str, str]] = {
            n.id: keypair_from_seed(hashlib.sha256(f"{scenario.seed}:{n.id}".encode()).digest())
            for n in scenario.nodes
        }
        self._messages: list[SimMessage] | None = None

    @property
    def end_at(self) -> datetime:
        return self._at(self.scenario.duration_s)

    def _at(self, sim_seconds: float) -> datetime:
        return self.t0 + timedelta(seconds=sim_seconds / self.time_scale)

    def registrations(self) -> list[NodeRegistration]:
        return [
            NodeRegistration(
                node_id=n.id,
                public_key=self.node_keys[n.id][1],
                location=SensorLocation(lat=n.lat, lon=n.lon, accuracy_m=10.0),
                time_quality=n.time_quality,
            )
            for n in self.scenario.nodes
        ]

    def truth(self) -> list[TruthPoint]:
        points = []
        steps = int(self.scenario.duration_s / self.scenario.tick_s)
        for step in range(steps + 1):
            t = step * self.scenario.tick_s
            for drone in self.scenario.drones:
                pos = drone_position(drone, t)
                if pos is not None:
                    points.append(TruthPoint(self._at(t), drone.id, pos))
        return points

    def messages(self) -> list[SimMessage]:
        if self._messages is None:
            self._messages = self._build()
        return self._messages

    def _build(self) -> list[SimMessage]:
        sc = self.scenario
        rng = random.Random(sc.seed)

        def new_id() -> UUID:
            return UUID(int=rng.getrandbits(128), version=4)

        nodes = sorted(sc.nodes, key=lambda n: n.id)
        offsets = {n.id: clock_offset_s(n.time_quality, rng) for n in nodes}
        smoothers = {n.id: DetectionSmoother(id_factory=new_id) for n in nodes}
        last_heard: dict[str, _Heard] = {}
        heartbeat_every = max(1, round(sc.heartbeat_every_s / sc.tick_s))
        out: list[SimMessage] = []

        steps = int(sc.duration_s / sc.tick_s)
        for step in range(steps + 1):
            t = step * sc.tick_s
            for node in nodes:
                if node_offline(sc, node.id, t):
                    continue
                if step % heartbeat_every == 0:
                    hb = Heartbeat(
                        node_id=node.id, sent_at=self._at(t), software_version=SOFTWARE_VERSION,
                        mic_ok=True, queue_depth=0,
                    )
                    hb_signed = sign(hb, self.node_keys[node.id][0])
                    out.append(SimMessage(self._at(t), "heartbeat", hb_signed))

                heard: _Heard | None = None
                for drone in sc.drones:
                    pos = drone_position(drone, t)
                    if pos is None:
                        continue
                    r = distance_m(node, pos)
                    snr = received_level_db(drone.source_db, r) - node.noise_floor_db
                    if rng.random() < detection_probability(snr):
                        candidate = _Heard(
                            confidence_from_snr(snr, rng), snr, drone.label, propagation_delay_s(r)
                        )
                        if heard is None or candidate.confidence > heard.confidence:
                            heard = candidate
                for alarm in sc.false_alarms:
                    active = alarm.start_s <= t < alarm.start_s + alarm.duration_s
                    if active and distance_m(node, alarm) <= alarm.radius_m:
                        candidate = _Heard(alarm.confidence, FALSE_ALARM_SNR_DB, alarm.label, 0.0)
                        if heard is None or candidate.confidence > heard.confidence:
                            heard = candidate

                smoother = smoothers[node.id]
                phase = smoother.push(t, heard.confidence if heard else 0.0)
                if heard:
                    last_heard[node.id] = heard
                if phase is None:
                    continue
                basis = heard or last_heard[node.id]
                sim_observed = t + basis.delay_s + offsets[node.id]
                obs = Observation(
                    observation_id=new_id(),
                    source=Source(type=SourceType.SIMULATED_NODE, id=node.id),
                    observed_at=self._at(sim_observed),
                    time_quality=node.time_quality,
                    sensor_location=SensorLocation(lat=node.lat, lon=node.lon, accuracy_m=10.0),
                    detection=Detection(label=basis.label, confidence=round(basis.confidence, 3)),
                    event=EventRef(detection_id=smoother.detection_id, phase=phase),
                    acoustic=Acoustic(snr_db=round(basis.snr_db, 2), peak_freq_hz=180.0),
                )
                send_at = self._at(t + basis.delay_s)
                obs_signed = sign(obs, self.node_keys[node.id][0])
                out.append(SimMessage(send_at, "observation", obs_signed))

        out.sort(key=lambda m: m.at)
        return out

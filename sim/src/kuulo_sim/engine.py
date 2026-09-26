"""Turns a scenario into the exact signed messages real nodes would send."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

import numpy as np

from kuulo_protocol.features import BAND_EDGES_HZ, FRAME_PERIOD_MS
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
    Phase,
    SensorLocation,
    Source,
    SourceType,
)
from kuulo_protocol.signing import keypair_from_seed, sign
from kuulo_protocol.smoothing import DetectionSmoother
from kuulo_protocol.traces import SEGMENT_FRAMES, FeatureTraceHeader, encode_body

from .physics import (
    SPEED_OF_SOUND_MPS,
    clock_offset_s,
    confidence_from_snr,
    detection_probability,
    propagation_delay_s,
    received_level_db,
)
from .scenario import Scenario, drone_position, node_offline
from .traces import synth_frames

SOFTWARE_VERSION = "sim-0.1.0"
FALSE_ALARM_SNR_DB = 20.0
ROTOR_F0_HZ = 180.0
FALSE_ALARM_F0_HZ = 110.0  # a two-stroke engine's firing frequency, no Doppler
FRAMES_PER_S = 1000 // FRAME_PERIOD_MS


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
class SimTraceSegment:
    header: FeatureTraceHeader
    body: bytes
    available_at: datetime  # when a real node would have closed this segment


@dataclass(frozen=True)
class _Sound:
    """The loudest source a node hears at one tick, for trace synthesis."""

    t: float
    level_db: float
    f0_hz: float
    doppler: float
    noise_floor_db: float


@dataclass
class _DetectionLog:
    node_id: str
    samples: list[_Sound] = field(default_factory=list)


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
        # Keyed by node id alone (not scenario.seed): several bundled scenarios reuse
        # ids like "n01", and running them one after another against the same server
        # (as the README does) must not re-register the same node with a new key.
        self.node_keys: dict[str, tuple[str, str]] = {
            n.id: keypair_from_seed(hashlib.sha256(n.id.encode()).digest())
            for n in scenario.nodes
        }
        self._messages: list[SimMessage] | None = None
        self._logs: dict[tuple[str, UUID], _DetectionLog] = {}
        self._segments: dict[tuple[str, UUID], list[SimTraceSegment]] = {}

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
                if smoother.active or phase is Phase.END:
                    key = (node.id, smoother.detection_id)
                    log = self._logs.setdefault(key, _DetectionLog(node.id))
                    log.samples.append(self._loudest(node, t))
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

    def _loudest(self, node, t: float) -> _Sound:
        sc = self.scenario
        best = _Sound(t, node.noise_floor_db, ROTOR_F0_HZ, 1.0, node.noise_floor_db)
        for drone in sc.drones:
            pos = drone_position(drone, t)
            if pos is None:
                continue
            r = distance_m(node, pos)
            before = drone_position(drone, max(t - sc.tick_s, drone.start_s))
            v_recede = (r - distance_m(node, before)) / sc.tick_s if before else 0.0
            level = received_level_db(drone.source_db, r)
            if level > best.level_db:
                doppler = SPEED_OF_SOUND_MPS / (SPEED_OF_SOUND_MPS + v_recede)
                best = _Sound(t, level, ROTOR_F0_HZ, doppler, node.noise_floor_db)
        for alarm in sc.false_alarms:
            active = alarm.start_s <= t < alarm.start_s + alarm.duration_s
            if active and distance_m(node, alarm) <= alarm.radius_m:
                level = node.noise_floor_db + FALSE_ALARM_SNR_DB
                if level > best.level_db:
                    best = _Sound(t, level, FALSE_ALARM_F0_HZ, 1.0, node.noise_floor_db)
        return best

    def detections(self) -> list[tuple[str, UUID]]:
        """Every (node_id, detection_id) the scenario produced, in order of first appearance."""
        self.messages()
        return list(self._logs)

    def trace_segments(self, node_id: str, detection_id: UUID) -> list[SimTraceSegment]:
        """The signed 60 s FeatureTrace segments this node would hold for this detection."""
        key = (node_id, detection_id)
        if key not in self._segments:
            self._segments[key] = self._synthesise(key)
        return self._segments[key]

    def _synthesise(self, key: tuple[str, UUID]) -> list[SimTraceSegment]:
        self.messages()
        log = self._logs.get(key)
        if log is None:
            return []
        node = next(n for n in self.scenario.nodes if n.id == log.node_id)
        ticks = np.array([s.t for s in log.samples])
        start = ticks[0]
        frames = max(1, round((ticks[-1] - start + self.scenario.tick_s) * FRAMES_PER_S))
        times = start + np.arange(frames) / FRAMES_PER_S

        def interp(attr: str) -> np.ndarray:
            return np.interp(times, ticks, [getattr(s, attr) for s in log.samples])

        digest = hashlib.sha256(f"{self.scenario.seed}:{key[1]}".encode()).digest()
        seed = int.from_bytes(digest[:8])
        block = synth_frames(
            level_db=interp("level_db"), noise_floor_db=interp("noise_floor_db"),
            f0_hz=interp("f0_hz"), doppler=interp("doppler"), frames=frames,
            rng=np.random.default_rng(seed),
        )
        private_key = self.node_keys[node.id][0]
        out = []
        for index, first in enumerate(range(0, frames, SEGMENT_FRAMES)):
            last = min(first + SEGMENT_FRAMES, frames)
            body = encode_body(
                t_offset_ms=np.arange(last - first) * FRAME_PERIOD_MS,
                band_db=block.band_db[first:last], rms_db=block.rms_db[first:last],
                peak_freq_hz=block.peak_freq_hz[first:last],
            )
            header = FeatureTraceHeader(
                node_id=node.id, detection_id=key[1], segment_index=index, final=last == frames,
                time_quality=node.time_quality, start_at=self._at(times[first]),
                frame_count=last - first, band_edges_hz=list(BAND_EDGES_HZ),
                body_sha256=hashlib.sha256(body).hexdigest(),
            )
            end_s = start + last / FRAMES_PER_S
            out.append(SimTraceSegment(sign(header, private_key), body, self._at(end_s)))
        return out

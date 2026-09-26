"""Wires audio -> windows -> classifier -> detector -> uplink, plus heartbeats and mic health."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

import numpy as np

from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation, Phase
from kuulo_protocol.signing import sign

from . import __version__
from .acoustic import AcousticMeter
from .audio import HOP_S, MIC_HELP, SAMPLE_RATE, AudioBlock, MicHealth, Windower
from .classify import Classifier, drone_score
from .config import NodeConfig
from .detector import Detector, utc_now
from .keys import NodeKeys
from .tracestore import TraceStore

PUSH_CONFIDENCE = 0.8

log = logging.getLogger("kuulo.node")


@dataclass
class RunStats:
    windows: int = 0
    observations: int = 0
    heartbeats: int = 0


def format_scores(t: float, drone: float, scores: dict[str, float]) -> str:
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return f"{t:8.2f}s  drone={drone:.2f}  | " + ", ".join(f"{n} {p:.2f}" for n, p in top)


class NodeRunner:
    def __init__(
        self,
        cfg: NodeConfig,
        keys: NodeKeys,
        classifier: Classifier,
        uplink,
        *,
        now: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        heartbeat_every_s: float = 60.0,
        print_scores: bool = False,
        out: Callable[[str], None] = print,
        traces: TraceStore | None = None,
        trace_uploader=None,
        debug_clip_dir: Path | None = None,
    ) -> None:
        self.cfg, self.keys, self.classifier, self.uplink = cfg, keys, classifier, uplink
        self._now, self._monotonic = now, monotonic
        self._heartbeat_every_s = heartbeat_every_s
        self._print_scores, self._out = print_scores, out
        self.health = MicHealth()
        self._windower = Windower()
        self._meter = AcousticMeter()
        self._detector = Detector(cfg, keys, now=now)
        self._last_heartbeat: float | None = None
        self.stats = RunStats()
        self._traces, self._trace_uploader = traces, trace_uploader
        self._high_run_s = 0.0  # current run of windows at >= 0.8 in this detection
        self._max_high_run_s = 0.0
        self._ended: list[tuple] = []
        # Opt-in only (--debug-save-clips): raw detection audio, written locally, never uploaded.
        self._clip_dir = debug_clip_dir
        self._clip: list[np.ndarray] = []
        self._clip_id: UUID | None = None

    def _registration(self) -> NodeRegistration:
        return NodeRegistration(
            node_id=self.cfg.node_id, public_key=self.keys.public_key,
            location=self.cfg.location, time_quality=self.cfg.time_quality,
        )

    def _emit(self, obs: Observation | None) -> None:
        if obs is not None and obs.event.phase is Phase.END:
            self._ended.append((obs.event.detection_id, self._max_high_run_s))
            self._high_run_s = self._max_high_run_s = 0.0
        if obs is not None:
            self.uplink.send("/v1/observations", obs)
            self.stats.observations += 1
            log.info("detection %s %s conf=%.2f", obs.event.phase.value.upper(),
                     str(obs.event.detection_id)[:8], obs.detection.confidence)

    def _heartbeat(self) -> None:
        hb = Heartbeat(
            node_id=self.cfg.node_id, sent_at=self._now(), software_version=__version__,
            mic_ok=self.health.ok, queue_depth=self.uplink.pending_count,
        )
        self.uplink.send("/v1/heartbeats", sign(hb, self.keys.private_key))
        self.stats.heartbeats += 1
        self._last_heartbeat = self._monotonic()

    def _maybe_heartbeat(self) -> None:
        if self._last_heartbeat is None or (
            self._monotonic() - self._last_heartbeat >= self._heartbeat_every_s
        ):
            self._heartbeat()

    def _process(self, block: AudioBlock) -> None:
        if self.health.update(block):
            log.warning(MIC_HELP)
        for window in self._windower.push(block):
            if window.gap_before:
                self._emit(self._detector.reset())
            scores = self.classifier.score(window.samples)
            score = drone_score(scores, self.cfg.weights)
            if self._print_scores:
                self._out(format_scores(window.t, score, scores))
            acoustic = self._meter.measure(window.samples)
            self._emit(self._detector.process(window.t, score, acoustic))
            self._track_confidence(score)
            self.stats.windows += 1
        self._record(block)

    def _track_confidence(self, score: float) -> None:
        if self._detector.active_detection_id is None:
            self._high_run_s = self._max_high_run_s = 0.0
            return
        self._high_run_s = self._high_run_s + HOP_S if score >= PUSH_CONFIDENCE else 0.0
        self._max_high_run_s = max(self._max_high_run_s, self._high_run_s)

    def _save_clip(self, block: AudioBlock) -> None:
        active = self._detector.active_detection_id
        if self._clip_id is not None and active != self._clip_id:
            import soundfile as sf

            self._clip_dir.mkdir(parents=True, exist_ok=True)
            sf.write(self._clip_dir / f"{self._clip_id}.wav", np.concatenate(self._clip),
                     SAMPLE_RATE)
            self._clip, self._clip_id = [], None
        if active is not None:
            self._clip_id = active
            self._clip.append(block.samples)

    def _record(self, block: AudioBlock) -> None:
        """Feed traces after the detector has seen this block, then report ended detections."""
        if self._traces is not None:
            self._traces.feed(block.samples, block.t, self._detector.active_detection_id)
        if self._clip_dir is not None:
            self._save_clip(block)
        ended, self._ended = self._ended, []
        if self._trace_uploader is not None:
            for detection_id, sustained in ended:
                self._trace_uploader.on_detection_end(detection_id, sustained)

    def _service_traces(self) -> None:
        if self._trace_uploader is not None:
            self._trace_uploader.poll()
            self._trace_uploader.pump()

    def run(self, blocks: Iterable[AudioBlock]) -> RunStats:
        reg = self._registration()
        self.uplink.ensure_registered(reg)
        try:
            for block in blocks:
                self._process(block)
                self._maybe_heartbeat()
                self.uplink.ensure_registered(reg)
                self.uplink.flush()
                self._service_traces()
        except KeyboardInterrupt:
            log.info("stopping")
        self._emit(self._detector.reset())
        self._record(AudioBlock(np.zeros(0, np.float32), 0.0))
        self._heartbeat()
        self.uplink.ensure_registered(reg)
        self.uplink.flush(force=True)
        if self.uplink.pending_count:
            log.warning("%d messages were not delivered", self.uplink.pending_count)
        return self.stats

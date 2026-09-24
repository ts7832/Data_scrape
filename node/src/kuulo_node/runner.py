"""Wires audio -> windows -> classifier -> detector -> uplink, plus heartbeats and mic health."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation
from kuulo_protocol.signing import sign

from . import __version__
from .acoustic import AcousticMeter
from .audio import MIC_HELP, AudioBlock, MicHealth, Windower
from .classify import Classifier, drone_score
from .config import NodeConfig
from .detector import Detector, utc_now
from .keys import NodeKeys

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

    def _registration(self) -> NodeRegistration:
        return NodeRegistration(
            node_id=self.cfg.node_id, public_key=self.keys.public_key,
            location=self.cfg.location, time_quality=self.cfg.time_quality,
        )

    def _emit(self, obs: Observation | None) -> None:
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
            self.stats.windows += 1

    def run(self, blocks: Iterable[AudioBlock]) -> RunStats:
        reg = self._registration()
        self.uplink.ensure_registered(reg)
        try:
            for block in blocks:
                self._process(block)
                self._maybe_heartbeat()
                self.uplink.ensure_registered(reg)
                self.uplink.flush()
        except KeyboardInterrupt:
            log.info("stopping")
        self._emit(self._detector.reset())
        self._heartbeat()
        self.uplink.ensure_registered(reg)
        self.uplink.flush(force=True)
        if self.uplink.pending_count:
            log.warning("%d messages were not delivered", self.uplink.pending_count)
        return self.stats

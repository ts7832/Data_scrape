"""Per-window drone scores -> smoothed detections -> signed Observations."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from kuulo_protocol.models import (
    Acoustic,
    Detection,
    EventRef,
    Label,
    Observation,
    Phase,
    Source,
    SourceType,
)
from kuulo_protocol.signing import sign
from kuulo_protocol.smoothing import DetectionSmoother

from .config import NodeConfig
from .keys import NodeKeys


def utc_now() -> datetime:
    return datetime.now(UTC)


class Detector:
    def __init__(
        self,
        cfg: NodeConfig,
        keys: NodeKeys,
        *,
        now: Callable[[], datetime] = utc_now,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self.cfg = cfg
        self.keys = keys
        self._now = now
        self._id_factory = id_factory
        self._smoother = self._new_smoother()
        self._last_score = 0.0
        self._last_acoustic: Acoustic | None = None

    def _new_smoother(self) -> DetectionSmoother:
        return DetectionSmoother(self.cfg.smoother, id_factory=self._id_factory)

    @property
    def active_detection_id(self) -> UUID | None:
        return self._smoother.detection_id if self._smoother.active else None

    def process(self, t: float, score: float, acoustic: Acoustic) -> Observation | None:
        self._last_score, self._last_acoustic = score, acoustic
        phase = self._smoother.push(t, score)
        if phase is None:
            return None
        return self._observation(phase, self._smoother.detection_id, score, acoustic)

    def reset(self) -> Observation | None:
        """Audio was interrupted: close any open detection and start from a clean slate."""
        obs = None
        if self._smoother.active:
            obs = self._observation(
                Phase.END, self._smoother.detection_id, self._last_score, self._last_acoustic
            )
        self._smoother = self._new_smoother()
        return obs

    def _observation(
        self, phase: Phase, detection_id: UUID | None, score: float, acoustic: Acoustic | None
    ) -> Observation:
        assert detection_id is not None
        obs = Observation(
            source=Source(type=SourceType.ACOUSTIC_NODE, id=self.cfg.node_id),
            observed_at=self._now(),
            time_quality=self.cfg.time_quality,
            sensor_location=self.cfg.location,
            detection=Detection(
                label=Label.DRONE_MULTIROTOR, confidence=round(min(1.0, max(0.0, score)), 4)
            ),
            event=EventRef(detection_id=detection_id, phase=phase),
            acoustic=acoustic,
        )
        return sign(obs, self.keys.private_key)

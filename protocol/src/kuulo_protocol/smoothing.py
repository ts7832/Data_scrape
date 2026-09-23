"""Turns a stream of per-window drone scores into start / update / end events.

Shared by the simulator now and the real node later, so both behave identically.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

from .models import Phase


@dataclass(frozen=True)
class SmootherConfig:
    threshold: float = 0.5
    k: int = 3
    n: int = 5
    end_after_s: float = 5.0
    update_every_s: float = 5.0


class DetectionSmoother:
    def __init__(
        self, config: SmootherConfig | None = None, id_factory: Callable[[], UUID] = uuid4
    ):
        self.config = config or SmootherConfig()
        self._id_factory = id_factory
        self._recent: deque[bool] = deque(maxlen=self.config.n)
        self._last_above: float | None = None
        self._last_emit: float | None = None
        self.active = False
        self.detection_id: UUID | None = None

    def push(self, t: float, score: float) -> Phase | None:
        above = score >= self.config.threshold
        self._recent.append(above)
        if above:
            self._last_above = t
        if not self.active:
            if sum(self._recent) >= self.config.k:
                self.active = True
                self.detection_id = self._id_factory()
                self._last_emit = t
                return Phase.START
            return None
        if self._last_above is not None and t - self._last_above >= self.config.end_after_s:
            self.active = False
            self._recent.clear()
            self._last_emit = None
            return Phase.END
        if self._last_emit is not None and t - self._last_emit >= self.config.update_every_s:
            self._last_emit = t
            return Phase.UPDATE
        return None

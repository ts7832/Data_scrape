"""Rough acoustics for an observation: SNR against a running noise floor, and peak frequency."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from kuulo_protocol.models import Acoustic

from .audio import HOP_S, SAMPLE_RATE


class AcousticMeter:
    """Noise floor = EMA of the quietest window RMS seen in the last floor_window_s."""

    def __init__(self, floor_window_s: float = 10.0, alpha: float = 0.2) -> None:
        self._recent: deque[float] = deque(maxlen=max(1, round(floor_window_s / HOP_S)))
        self._alpha = alpha
        self._floor: float | None = None

    def measure(self, samples: np.ndarray) -> Acoustic:
        x = np.asarray(samples, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        self._recent.append(rms)
        quietest = min(self._recent)
        if self._floor is None:
            self._floor = quietest
        else:
            self._floor = (1 - self._alpha) * self._floor + self._alpha * quietest
        snr = 20 * math.log10(max(rms, 1e-9) / max(self._floor, 1e-9))
        peak_hz = 0.0
        if x.size:
            spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
            peak_hz = float(np.argmax(spectrum)) * SAMPLE_RATE / x.size
        return Acoustic(
            snr_db=round(min(150.0, max(-50.0, snr)), 1), peak_freq_hz=round(peak_hz, 1)
        )
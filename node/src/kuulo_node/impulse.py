"""Impulse detector: STA/LTA trigger plus AIC onset picker, both borrowed from seismology.

Runs on the node's 16 kHz audio beside the drone classifier and reports only an onset time,
its uncertainty and summary features: no waveform leaves the node (spec §4).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from kuulo_protocol.features import extract_frames
from kuulo_protocol.impulses import ImpulseFeatures

from .audio import SAMPLE_RATE as SR
from .audio import AudioBlock

FRAME = 32  # 2 ms energy frames
PRE_S = 0.25  # audio kept from before the trigger, for onset picking
WARM_S = 0.5
SIGMA_RISE_FACTOR = 0.25


@dataclass(frozen=True)
class ImpulseConfig:
    enabled: bool = True
    sta_ms: float = 2.0
    lta_s: float = 5.0
    trigger_db: float = 20.0
    min_peak_dbfs: float = -40.0
    min_active_bands: int = 6
    max_rise_ms: float = 50.0
    min_decay_db: float = 10.0
    post_s: float = 1.0
    dead_time_s: float = 2.0
    highpass_hz: float = 100.0


@dataclass(frozen=True)
class ImpulseCandidate:
    onset_utc: datetime
    onset_sigma_s: float
    features: ImpulseFeatures


def aic_pick(x: np.ndarray) -> int:
    """Onset index in x (Maeda 1985): argmin of k*log var(x[:k]) + (n-k-1)*log var(x[k:])."""
    n = x.size
    if n < 16:
        return 0
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    k = np.arange(4, n - 4)
    v1 = c2[k - 1] / k - (c1[k - 1] / k) ** 2
    v2 = (c2[-1] - c2[k - 1]) / (n - k) - ((c1[-1] - c1[k - 1]) / (n - k)) ** 2
    aic = k * np.log(np.maximum(v1, 1e-20)) + (n - k - 1) * np.log(np.maximum(v2, 1e-20))
    return int(k[np.argmin(aic)])


def _db(x: float) -> float:
    return 10 * math.log10(max(x, 1e-20))


class ImpulseDetector:
    def __init__(self, cfg: ImpulseConfig | None = None) -> None:
        self.cfg = cfg or ImpulseConfig()
        self._sos = butter(2, self.cfg.highpass_hz, btype="highpass", fs=SR, output="sos")
        self._n = 0  # absolute index of the next incoming sample
        self._anchors: deque[tuple[int, datetime]] = deque(maxlen=256)
        self.reset()

    def reset(self) -> None:
        self._zi = sosfilt_zi(self._sos) * 0.0
        self._raw = np.zeros(0, np.float32)
        self._hp = np.zeros(0, np.float64)
        self._start = self._n  # absolute index of _raw[0]
        self._next_frame = self._n
        self._lta: float | None = None
        self._warm = 0
        self._dead_until = -1
        self._pending: int | None = None

    def _utc_of(self, idx: int) -> datetime:
        for a_idx, a_utc in reversed(self._anchors):
            if a_idx <= idx:
                return a_utc + timedelta(seconds=(idx - a_idx) / SR)
        a_idx, a_utc = self._anchors[0]
        return a_utc + timedelta(seconds=(idx - a_idx) / SR)

    def push(self, block: AudioBlock, utc: datetime) -> list[ImpulseCandidate]:
        if block.restart:
            self.reset()
        x = np.asarray(block.samples, np.float32)
        self._anchors.append((self._n, utc))
        hp, self._zi = sosfilt(self._sos, x.astype(np.float64), zi=self._zi)
        self._raw = np.concatenate([self._raw, x])
        self._hp = np.concatenate([self._hp, hp])
        self._n += x.size
        self._scan()
        out = []
        if self._pending is not None and self._n >= self._pending + int(self.cfg.post_s * SR):
            cand = self._analyse(self._pending)
            self._pending = None
            if cand is not None:
                out.append(cand)
        self._trim()
        return out

    def _scan(self) -> None:
        cfg = self.cfg
        warm_frames = int(WARM_S * SR / FRAME)
        alpha = FRAME / (cfg.lta_s * SR)
        while self._next_frame + FRAME <= self._n:
            i = self._next_frame - self._start
            e = float(np.mean(self._hp[i:i + FRAME] ** 2)) + 1e-20
            s = self._next_frame
            self._next_frame += FRAME
            if self._lta is None:
                self._lta = e
            if (self._warm >= warm_frames and self._pending is None and s >= self._dead_until
                    and _db(e / self._lta) >= cfg.trigger_db):
                self._pending = s
                self._dead_until = s + int(cfg.dead_time_s * SR)
            elif s >= self._dead_until:
                rate = 1.0 / (self._warm + 1) if self._warm < warm_frames else alpha
                self._lta += (e - self._lta) * rate
            self._warm += 1

    def _trim(self) -> None:
        keep_from = self._n - int((PRE_S + self.cfg.post_s + 0.1) * SR)
        if self._pending is not None:
            keep_from = min(keep_from, self._pending - int(PRE_S * SR))
        keep_from = min(keep_from, self._next_frame)
        cut = keep_from - self._start
        if cut > 0:
            self._raw, self._hp = self._raw[cut:], self._hp[cut:]
            self._start = keep_from

    def _reject(self, tail_energy: float) -> None:
        self._lta = max(tail_energy, 1e-20)  # re-baseline: sustained sound is the new normal

    def _analyse(self, trigger: int) -> ImpulseCandidate | None:
        cfg = self.cfg
        lo = max(trigger - int(PRE_S * SR), self._start)
        hi = min(trigger + int(cfg.post_s * SR), self._n)
        raw = self._raw[lo - self._start: hi - self._start]
        hp = self._hp[lo - self._start: hi - self._start]
        pk_lo = max(trigger - int(0.01 * SR), lo) - lo
        peak = pk_lo + int(np.argmax(np.abs(hp[pk_lo:])))
        tail = float(np.mean(hp[-int(0.1 * SR):] ** 2))
        peak_amp = float(np.max(np.abs(raw[pk_lo:])))
        peak_dbfs = min(0.0, 20 * math.log10(max(peak_amp, 1e-9)))
        if peak_dbfs < cfg.min_peak_dbfs:
            self._reject(tail)
            return None
        a_lo = max(0, (trigger - lo) - int(0.1 * SR))
        onset = a_lo + aic_pick(hp[a_lo: peak + 1])
        env = np.sqrt(np.convolve(hp ** 2, np.ones(16) / 16, mode="same"))
        top = float(env[onset: peak + 16].max())
        rise_idx = np.nonzero(env[onset:] >= 0.9 * top)[0]
        start_idx = np.nonzero(env[onset:] >= 0.1 * top)[0]
        rise_s = ((rise_idx[0] - start_idx[0]) / SR) if rise_idx.size and start_idx.size else 0.0
        frames = hp[onset: onset + (hp.size - onset) // FRAME * FRAME].reshape(-1, FRAME)
        energies = np.mean(frames ** 2, axis=1) if frames.size else np.array([1e-20])
        decay_db = _db(float(energies.max()) / max(tail, 1e-20))
        env10 = np.sqrt(np.convolve(hp ** 2, np.ones(160) / 160, mode="same"))
        above = np.nonzero(env10[peak:] >= math.sqrt(self._lta) * 10 ** (10 / 20))[0]
        duration_s = ((peak + (above[-1] if above.size else 0)) - onset) / SR
        centre = max(0, min(peak - 160, raw.size - 320))
        bands = extract_frames(raw[centre: centre + 320]).band_db[0]
        active = int(np.sum(bands >= bands.max() - 20))
        if (rise_s * 1000 > cfg.max_rise_ms or decay_db < cfg.min_decay_db
                or active < cfg.min_active_bands):
            self._reject(tail)
            return None
        sigma = min(0.02, max(2 / SR, SIGMA_RISE_FACTOR * rise_s))
        features = ImpulseFeatures(
            peak_dbfs=round(peak_dbfs, 2),
            snr_db=round(min(150.0, max(-50.0, _db(float(energies.max()) / self._lta))), 2),
            rise_time_ms=round(rise_s * 1000, 3), duration_ms=round(max(duration_s, 0) * 1000, 1),
            clipped=bool(np.sum(np.abs(raw) >= 0.99) >= 3),
            band_db=[round(float(b), 2) for b in bands],
        )
        return ImpulseCandidate(self._utc_of(lo + onset), sigma, features)

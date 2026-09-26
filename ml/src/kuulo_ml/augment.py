"""Training-only augmentation (spec §10): make close, clean drone recordings look like a distant
drone heard by a city node. Evaluation always uses clean audio.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

SR = 16_000


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """signal + noise scaled so that 20*log10(rms(signal)/rms(scaled noise)) == snr_db."""
    noise = np.resize(noise, signal.shape)
    scale = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return (signal + scale * noise).astype(np.float32)


def _distance(x, rng):
    """Gain loss plus a first-order low-pass: air absorbs high frequencies over distance."""
    gain = 10 ** (-rng.uniform(0, 30) / 20)
    cutoff = rng.uniform(2000, 8000)
    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x) * gain


def _reverb(x, rng):
    """Convolve with a synthetic exponentially decaying noise impulse response."""
    rt60 = rng.uniform(0.2, 1.0)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir**2))
    wet = np.convolve(x, ir)[: x.size]
    return 0.6 * x + 0.4 * wet


def _pitch_drift(x, rng):
    """Time-varying resampling of up to +/-3 %: a slow Doppler-like glide."""
    rate = 1 + rng.uniform(-0.03, 0.03) * np.linspace(-1, 1, x.size) * rng.choice([-1, 1])
    positions = np.clip(np.cumsum(rate) - rate[0], 0, x.size - 1)
    return np.interp(positions, np.arange(x.size), x)


def augment(x: np.ndarray, rng: np.random.Generator, backgrounds: list[np.ndarray]) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    y = _distance(y, rng)
    if rng.random() < 0.5:
        y = _reverb(y, rng)
    y = _pitch_drift(y, rng)
    if backgrounds:
        bg = backgrounds[rng.integers(len(backgrounds))]
        start = rng.integers(max(1, bg.size - y.size + 1))
        y = mix_at_snr(y, bg[start:start + y.size], rng.uniform(-5, 20))
    y = np.roll(y * 10 ** (rng.uniform(-6, 6) / 20), rng.integers(y.size))
    peak = np.max(np.abs(y))
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)

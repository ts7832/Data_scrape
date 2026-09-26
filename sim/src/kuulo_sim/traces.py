"""Synthetic FeatureTraces: what a node's 32-band spectrum would look like for a simulated source.

A multirotor is modelled as a harmonic comb (blade-pass fundamental plus 9 overtones, each 3 dB
weaker) at the received level, shifted by Doppler, over a pink-ish background at the node's noise
floor. Deliberately simple, but the band layout and units match a real node's traces exactly.
"""

from __future__ import annotations

import numpy as np

from kuulo_protocol.features import BAND_EDGES_HZ, N_BANDS, FrameBlock

HARMONICS = 10
HARMONIC_STEP_DB = 3.0
DB_SPL_TO_DBFS = -110.0  # nominal microphone calibration: 110 dB SPL = 0 dBFS
_EDGES = np.asarray(BAND_EDGES_HZ)
_CENTRES = np.sqrt(_EDGES[:-1] * _EDGES[1:])
# Pink-ish background: -3 dB per octave relative to the lowest band, normalised to 0 dB total.
_TILT = -3.0 * np.log2(_CENTRES / _CENTRES[0])
_TILT -= 10 * np.log10(np.sum(10 ** (_TILT / 10)))


def synth_frames(
    *, level_db, noise_floor_db, f0_hz, doppler, frames: int, rng: np.random.Generator
) -> FrameBlock:
    """level_db/noise_floor_db in dB SPL; every argument may be a scalar or a per-frame array."""
    level = np.broadcast_to(np.asarray(level_db, float), (frames,))
    floor = np.broadcast_to(np.asarray(noise_floor_db, float), (frames,))
    f0 = np.broadcast_to(np.asarray(f0_hz, float), (frames,)) * np.broadcast_to(
        np.asarray(doppler, float), (frames,)
    )
    power = 10 ** ((floor[:, None] + DB_SPL_TO_DBFS + _TILT[None, :]) / 10)
    for k in range(1, HARMONICS + 1):
        freq = f0 * k
        band = np.searchsorted(_EDGES, freq, side="right") - 1
        ok = (band >= 0) & (band < N_BANDS)
        amp = 10 ** ((level + DB_SPL_TO_DBFS - HARMONIC_STEP_DB * (k - 1)) / 10)
        np.add.at(power, (np.nonzero(ok)[0], band[ok]), amp[ok])
    band_db = 10 * np.log10(power) + rng.normal(0, 0.5, power.shape)
    rms_db = 10 * np.log10(power.sum(axis=1)) + rng.normal(0, 0.2, frames)
    return FrameBlock(
        band_db=band_db.astype(np.float32),
        rms_db=rms_db.astype(np.float32),
        peak_freq_hz=f0.astype(np.float32),
    )

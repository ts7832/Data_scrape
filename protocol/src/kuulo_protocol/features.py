"""20 ms log-mel frames: the FeatureTrace body. Shared by the node and the simulator.

32 mel-spaced bands between 50 Hz and 8 kHz are enough to see a drone's rotor harmonics and
their Doppler shift, and far too coarse to reconstruct intelligible speech.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
FRAME_PERIOD_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_PERIOD_MS // 1000  # 320
N_BANDS = 32
F_MIN_HZ = 50.0
F_MAX_HZ = 8000.0
N_FFT = 1024  # zero-padded: 15.6 Hz bins, so even the narrow low bands hold several bins
FLOOR = 1e-10


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


_MEL_EDGES = np.linspace(_hz_to_mel(F_MIN_HZ), _hz_to_mel(F_MAX_HZ), N_BANDS + 1)
BAND_EDGES_HZ: tuple[float, ...] = tuple(float(f) for f in _mel_to_hz(_MEL_EDGES))

_FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
_WINDOW = np.hanning(FRAME_SAMPLES).astype(np.float64)
# Rectangular bands: bin k belongs to band b when edge[b] <= f < edge[b + 1] (last band closed).
_BAND_OF_BIN = np.searchsorted(np.asarray(BAND_EDGES_HZ), _FREQS, side="right") - 1
_BAND_OF_BIN[_FREQS == F_MAX_HZ] = N_BANDS - 1
_IN_RANGE = (_BAND_OF_BIN >= 0) & (_BAND_OF_BIN < N_BANDS)
_PEAK_BINS = np.nonzero(_FREQS >= F_MIN_HZ)[0]


@dataclass(frozen=True)
class FrameBlock:
    band_db: np.ndarray  # float32 [F, 32]
    rms_db: np.ndarray  # float32 [F], dB relative to full scale
    peak_freq_hz: np.ndarray  # float32 [F]

    @property
    def frames(self) -> int:
        return int(self.rms_db.shape[0])


def extract_frames(samples: np.ndarray) -> FrameBlock:
    """Every complete 320-sample frame of 16 kHz mono audio; a trailing partial frame is dropped."""
    x = np.asarray(samples, dtype=np.float64)
    count = x.size // FRAME_SAMPLES
    frames = x[: count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=N_FFT, axis=1)) ** 2
    bands = np.zeros((count, N_BANDS))
    np.add.at(bands.T, _BAND_OF_BIN[_IN_RANGE], power[:, _IN_RANGE].T)
    peak = _FREQS[_PEAK_BINS[np.argmax(power[:, _PEAK_BINS], axis=1)]] if count else np.zeros(0)
    return FrameBlock(
        band_db=(10 * np.log10(bands + FLOOR)).astype(np.float32),
        rms_db=(20 * np.log10(rms + FLOOR)).astype(np.float32),
        peak_freq_hz=np.asarray(peak, dtype=np.float32),
    )

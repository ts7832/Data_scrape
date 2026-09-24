"""Audio in: 16 kHz mono blocks from a wav file or the microphone, cut into YAMNet windows."""

from __future__ import annotations

import logging
import math
import queue
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

log = logging.getLogger("kuulo.node")

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_600  # YAMNet's fixed input: 0.975 s
HOP_SAMPLES = 7_800  # 50 % overlap
HOP_S = HOP_SAMPLES / SAMPLE_RATE

MIC_HELP = (
    "No audio from the microphone. On macOS, allow your terminal app in System Settings > "
    "Privacy & Security > Microphone, then restart it. On Linux, install libportaudio2 and "
    "check the input device (`python -m sounddevice`)."
)


@dataclass(frozen=True)
class AudioBlock:
    samples: np.ndarray  # float32, mono, 16 kHz
    t: float  # audio time of the first sample, seconds
    restart: bool = False  # the source reopened; earlier audio is not contiguous


@dataclass(frozen=True)
class Window:
    samples: np.ndarray
    t: float  # time of the window's first sample
    gap_before: bool  # audio before this window was lost: detection state must reset


class Windower:
    """Ring buffer that emits 15 600-sample windows every 7 800 samples."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, np.float32)
        self._buf_t = 0.0
        self._expected: float | None = None
        self._gap = False

    def push(self, block: AudioBlock) -> list[Window]:
        jumped = self._expected is not None and abs(block.t - self._expected) > HOP_S
        if block.restart or jumped:
            self._buf = np.zeros(0, np.float32)
            self._gap = True
        if self._buf.size == 0:
            self._buf_t = block.t
        self._buf = np.concatenate([self._buf, block.samples.astype(np.float32, copy=False)])
        self._expected = block.t + block.samples.size / SAMPLE_RATE
        out: list[Window] = []
        while self._buf.size >= WINDOW_SAMPLES:
            out.append(Window(self._buf[:WINDOW_SAMPLES].copy(), self._buf_t, self._gap))
            self._gap = False
            self._buf = self._buf[HOP_SAMPLES:]
            self._buf_t += HOP_S
        return out


def to_mono_16k(data: np.ndarray, rate: int) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = math.gcd(int(rate), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
    return x


def wav_source(
    path: Path,
    *,
    speed: float = 1.0,
    block_s: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[AudioBlock]:
    """Replay a file as if it were live. speed=1 is real time; speed<=0 is as fast as possible."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    x = to_mono_16k(data, rate)
    n = int(round(block_s * SAMPLE_RATE))
    for i in range(0, x.size, n):
        chunk = x[i : i + n]
        yield AudioBlock(chunk, round(i / SAMPLE_RATE, 6))
        if speed > 0:
            sleep(chunk.size / SAMPLE_RATE / speed)


def _open_stream(sd, block_s: float, q: queue.Queue):
    """Open at 16 kHz mono; if the device refuses, open at its native rate.

    Returns (stream, rate).
    """

    def callback(indata, _frames, _time, status):
        if status:
            log.debug("audio status: %s", status)
        q.put(indata[:, 0].copy())

    for rate in (SAMPLE_RATE, None):
        try:
            if rate is None:
                rate = int(sd.query_devices(kind="input")["default_samplerate"])
            stream = sd.InputStream(
                samplerate=rate, channels=1, dtype="float32",
                blocksize=int(block_s * rate), callback=callback,
            )
            stream.start()
            return stream, rate
        except sd.PortAudioError:
            if rate != SAMPLE_RATE:
                raise
    raise RuntimeError("unreachable")


def mic_source(
    *,
    block_s: float = 0.1,
    retry_s: float = 10.0,
    on_error: Callable[[], None] | None = None,
) -> Iterator[AudioBlock]:
    """Live microphone blocks, timed on the monotonic clock; reopens every retry_s on failure."""
    import sounddevice as sd  # imported here so tests and CI never need PortAudio

    t0 = time.monotonic()
    restart = False
    while True:
        q: queue.Queue = queue.Queue()
        stream = None
        try:
            stream, rate = _open_stream(sd, block_s, q)
            log.info("microphone open at %d Hz", rate)
            while True:
                chunk = q.get(timeout=2.0)
                samples = chunk if rate == SAMPLE_RATE else to_mono_16k(chunk, rate)
                yield AudioBlock(samples, time.monotonic() - t0, restart)
                restart = False
        except (sd.PortAudioError, queue.Empty, OSError) as exc:
            log.error("%s (%s). Retrying in %.0f s.", MIC_HELP, exc, retry_s)
            if on_error is not None:
                on_error()
            restart = True
            time.sleep(retry_s)
        finally:
            if stream is not None:
                stream.close()


class MicHealth:
    """Tracks digital silence: a blocked mic delivers exact zeros, a quiet room does not."""

    SILENCE = 1e-6

    def __init__(self, silent_after_s: float = 10.0) -> None:
        self.silent_after_s = silent_after_s
        self.ok = True
        self._silent_since: float | None = None

    def update(self, block: AudioBlock) -> bool:
        silent = block.samples.size == 0 or float(np.max(np.abs(block.samples))) < self.SILENCE
        if not silent:
            self._silent_since = None
            self.ok = True
            return False
        if self._silent_since is None:
            self._silent_since = block.t
        end = block.t + block.samples.size / SAMPLE_RATE
        if self.ok and end - self._silent_since >= self.silent_after_s:
            self.ok = False
            return True
        return False

    def mark_failed(self) -> None:
        self.ok = False

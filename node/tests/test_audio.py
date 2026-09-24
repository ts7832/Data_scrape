import sys
import time

import numpy as np
import pytest
import soundfile as sf

from kuulo_node.audio import (
    HOP_S,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    AudioBlock,
    MicHealth,
    Windower,
    mic_source,
    to_mono_16k,
    wav_source,
)


def blocks(n_blocks, block=1600, start=0.0, value=0.1):
    for i in range(n_blocks):
        yield AudioBlock(np.full(block, value, np.float32), start + i * block / SAMPLE_RATE)


def test_windower_emits_overlapping_windows_with_times():
    w = Windower()
    windows = [win for b in blocks(20) for win in w.push(b)]  # 2.0 s of audio
    assert [round(x.t, 4) for x in windows] == [0.0, round(HOP_S, 4), round(2 * HOP_S, 4)]
    assert all(x.samples.shape == (WINDOW_SAMPLES,) for x in windows)
    assert not any(x.gap_before for x in windows)


def test_windower_gap_clears_buffer_and_flags_next_window():
    w = Windower()
    before = [x for b in blocks(20) for x in w.push(b)]
    after = [x for b in blocks(20, start=5.0) for x in w.push(b)]  # 3 s jump
    assert before and after
    assert after[0].gap_before and after[0].t == 5.0
    assert not any(x.gap_before for x in after[1:])


def test_windower_restart_flag_is_a_gap_even_without_time_jump():
    w = Windower()
    list(w.push(b) for b in blocks(20))
    restarted = AudioBlock(np.zeros(16000, np.float32), 2.0, restart=True)
    out = w.push(restarted)
    assert out and out[0].gap_before


def test_to_mono_16k_downmixes_and_resamples_stereo_44k():
    stereo = np.zeros((44100, 2), np.float32)
    stereo[:, 0] = 0.5
    x = to_mono_16k(stereo, 44100)
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(x.size - 16000) <= 1
    assert abs(float(np.median(x)) - 0.25) < 1e-3


def test_wav_source_replays_48k_stereo_file_at_16k(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros((48000 * 2, 2), np.float32), 48000)
    slept = []
    got = list(wav_source(path, speed=2.0, sleep=slept.append))
    assert abs(sum(b.samples.size for b in got) - 32000) <= 2
    assert got[0].t == 0.0 and got[1].t == 0.1
    assert abs(sum(slept) - 1.0) < 0.01  # 2 s of audio at 2x speed


def test_wav_source_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, np.float32), 16000)
    assert list(wav_source(path, speed=0)) == []


def test_mic_health_flips_after_10s_of_digital_silence_and_recovers():
    h = MicHealth()
    flips = [h.update(b) for b in blocks(110, value=0.0)]  # 11 s of zeros
    assert flips.count(True) == 1 and not h.ok
    h.update(AudioBlock(np.full(1600, 0.01, np.float32), 11.0))
    assert h.ok


class _FakeInputStream:
    """Delivers 6 fake blocks synchronously in start(), exactly as PortAudio's callback
    would -- but with no real concurrency, so the test controls exactly what's queued."""

    def __init__(self, *, samplerate, channels, dtype, blocksize, callback):
        self.blocksize = blocksize
        self._callback = callback

    def start(self):
        chunk = np.full((self.blocksize, 1), 0.05, np.float32)
        for _ in range(6):
            self._callback(chunk, self.blocksize, None, None)

    def close(self):
        pass


class _FakeSD:
    class PortAudioError(Exception):
        pass

    InputStream = _FakeInputStream

    @staticmethod
    def query_devices(kind=None):
        return {"default_samplerate": SAMPLE_RATE}


def test_mic_source_times_blocks_by_samples_not_by_consumption_delay(monkeypatch):
    # A slow consumer (e.g. the runner blocked on a synchronous uplink POST) must not
    # make mic_source's timestamps jump -- that would look like a dropped audio stream
    # to Windower's gap detection and needlessly split a live detection in two.
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD)
    gen = mic_source(block_s=0.1)
    first = next(gen)
    time.sleep(HOP_S + 0.1)  # well over one hop: simulate the runner being busy
    second = next(gen)
    assert second.t - first.t == pytest.approx(0.1, abs=1e-3)

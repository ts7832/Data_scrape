from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kuulo_node.audio import SAMPLE_RATE as SR
from kuulo_node.audio import AudioBlock
from kuulo_node.impulse import ImpulseConfig, ImpulseDetector, aic_pick, aic_sigma

T0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def noise(seconds, level=0.003, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def burst(amp=0.5, tau=0.03, seconds=0.4, seed=1):
    n = int(seconds * SR)
    env = np.exp(-np.arange(n) / (tau * SR))
    return (np.random.default_rng(seed).standard_normal(n) * amp * env).astype(np.float32)


def place(bg, event, at_s):
    x = bg.copy()
    i = int(at_s * SR)
    x[i:i + event.size] += event[: x.size - i]
    return np.clip(x, -1, 1)


def run(x, block_s=0.1, cfg=None, utc0=T0):
    det = ImpulseDetector(cfg or ImpulseConfig())
    n = int(block_s * SR)
    out = []
    for i in range(0, x.size, n):
        out += det.push(AudioBlock(x[i:i + n], i / SR), utc0 + timedelta(seconds=i / SR))
    return out


def test_aic_finds_the_change_point():
    x = np.concatenate([noise(0.05), burst(seconds=0.05)])
    assert abs(aic_pick(x.astype(np.float64)) - int(0.05 * SR)) <= 8


def test_aic_sigma_is_tight_for_a_sharp_change_and_loose_for_a_subtle_one():
    sharp = np.concatenate(
        [noise(0.05, level=0.001), burst(amp=0.5, seconds=0.05)]).astype(np.float64)
    subtle = np.concatenate(
        [noise(0.05, level=0.05), burst(amp=0.06, seconds=0.05)]).astype(np.float64)
    assert aic_sigma(sharp) < aic_sigma(subtle)
    assert aic_sigma(sharp) / SR < 0.001  # sub-millisecond for an unambiguous change point


def test_one_bang_one_report_with_sub_millisecond_onset():
    [c] = run(place(noise(6), burst(), 3.0))
    assert abs((c.onset_utc - (T0 + timedelta(seconds=3.0))).total_seconds()) < 0.001
    # A clean, high-SNR onset like this one should report a tight sigma: the AIC curve pins it
    # sharply, so sigma should track the true (near-zero) error, not sit at some fixed ceiling.
    assert 0 < c.onset_sigma_s <= 0.001
    assert c.features.peak_dbfs > -10 and c.features.rise_time_ms < 5 and not c.features.clipped


def test_stationary_noise_gives_nothing():
    assert run(noise(20)) == []


def test_faded_in_tone_is_narrowband_and_rejected():
    t = np.arange(int(4 * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t) * np.clip((t - 2.0) / 0.05, 0, 1)
    assert run(noise(4) + tone.astype(np.float32)) == []


def test_step_to_sustained_noise_is_not_an_impulse_and_does_not_retrigger():
    x = np.concatenate([noise(3), noise(12, level=0.1, seed=5)])
    assert run(x) == []


def test_echo_within_dead_time_is_not_reported_again():
    x = place(place(noise(6), burst(), 2.0), burst(amp=0.25, seed=2), 2.3)
    assert len(run(x)) == 1


def test_two_bangs_three_seconds_apart_are_two_reports():
    x = place(place(noise(8), burst(), 2.0), burst(seed=3), 5.0)
    assert len(run(x)) == 2


def test_clipping_is_flagged():
    [c] = run(place(noise(6), burst(amp=3.0), 3.0))
    assert c.features.clipped and c.features.peak_dbfs == pytest.approx(0, abs=0.1)


def test_block_size_does_not_change_the_onset():
    x = place(noise(6), burst(), 3.0)
    a, b = run(x, block_s=0.1), run(x, block_s=0.0137)
    assert abs((a[0].onset_utc - b[0].onset_utc).total_seconds()) < 1 / SR + 1e-9


def test_onset_uses_the_block_utc_anchor_not_the_audio_clock():
    x = place(noise(6), burst(), 3.0)
    [c] = run(x, utc0=T0 + timedelta(hours=1))
    assert abs((c.onset_utc - (T0 + timedelta(hours=1, seconds=3))).total_seconds()) < 0.001


def test_restart_resets_warm_up():
    det = ImpulseDetector()
    x = place(noise(3), burst(), 0.2)
    det.push(AudioBlock(noise(2), 0.0), T0)
    assert det.push(AudioBlock(x, 10.0, restart=True), T0 + timedelta(seconds=10)) == []


def test_quiet_click_below_min_peak_is_ignored():
    # burst peaks near 4 sigma = 0.004 (-48 dBFS), below the -40 dBFS default
    assert run(place(noise(6, level=0.0001), burst(amp=0.001), 3.0)) == []

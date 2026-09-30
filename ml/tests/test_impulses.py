import numpy as np
import pytest

from kuulo_ml.impulses import _snr_stats, evaluate, mix_at

SR = 16_000


def burst(seed, amp=0.5):
    n = int(0.4 * SR)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * amp * np.exp(-np.arange(n) / (0.03 * SR))).astype(np.float32)


def noise(seed, seconds=8.0, level=0.003):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def test_mix_places_the_event_at_the_offset_and_snr():
    bg = noise(0)
    ev = np.concatenate([np.zeros(100, np.float32), burst(1)])
    added = mix_at(bg, ev, offset=32_000, snr_db=20, ref=100) - bg
    assert np.all(added[:32_000] == 0)
    seg = added[32_100: 32_100 + int(0.05 * SR)].astype(np.float64)
    snr = 10 * np.log10(np.mean(seg ** 2) / np.mean(bg.astype(np.float64) ** 2))
    assert abs(snr - 20) < 0.2


def test_evaluate_on_synthetic_bursts_detects_them_with_small_error():
    positives = [np.concatenate([np.zeros(1600, np.float32), burst(s)]) for s in range(5)]
    backgrounds = [noise(10 + s) for s in range(3)]
    negatives = [noise(20 + s, seconds=30) for s in range(3)]
    r = evaluate(positives, backgrounds, negatives, np.random.default_rng(2), snrs=(30,))
    assert r["clean_detected"] == 1.0 and r["false_per_hour"] == 0.0
    assert r["snr"][30]["detected"] == 1.0 and r["snr"][30]["median_error_ms"] < 1.0
    # A reported sigma of the right order of magnitude for a clean, high-SNR onset (calibrated
    # against real audio -- see ml/IMPULSE_RESULTS.md), and no gross mis-picks at this SNR (a
    # real onset error, in a synthetic case with no other events in the clip).
    assert 0 < r["snr"][30]["median_sigma_ms"] < 5.0
    assert r["snr"][30]["gross_error_rate"] == 0.0


def test_gross_mispicks_are_excluded_from_calibration_but_counted_separately():
    # Two genuine, well-calibrated picks (error within 2x their own reported sigma) and one
    # gross mis-pick (a different event entirely -- a wrong pick, not picking-noise sigma).
    picks = [(0.3, 0.2), (0.5, 0.3), (50.0, 0.1)]
    r = _snr_stats(picks, detected=3, trials=3)
    assert r["gross_error_rate"] == pytest.approx(1 / 3)
    assert r["within_2sigma"] == 1.0

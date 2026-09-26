import numpy as np
import pytest

from kuulo_ml.augment import augment, mix_at_snr, rms

SR = 16_000


def tone(seconds=1.0, freq=180.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("snr", [-5.0, 0.0, 10.0, 20.0])
def test_mix_hits_the_requested_snr(snr):
    rng = np.random.default_rng(0)
    signal = tone()
    noise = (rng.standard_normal(SR) * 0.1).astype(np.float32)
    mixed = mix_at_snr(signal, noise, snr)
    residual = mixed - signal
    assert 20 * np.log10(rms(signal) / rms(residual)) == pytest.approx(snr, abs=0.5)


def test_augment_keeps_length_and_is_finite_and_bounded():
    rng = np.random.default_rng(1)
    backgrounds = [(rng.standard_normal(SR * 2) * 0.05).astype(np.float32)]
    x = tone(0.975)
    for _ in range(50):
        y = augment(x, rng, backgrounds)
        assert y.shape == x.shape and y.dtype == np.float32
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0


def test_augment_changes_the_signal_and_is_seeded():
    backgrounds = [(np.random.default_rng(2).standard_normal(SR) * 0.05).astype(np.float32)]
    x = tone(0.975)
    a = augment(x, np.random.default_rng(3), backgrounds)
    b = augment(x, np.random.default_rng(3), backgrounds)
    assert np.array_equal(a, b) and not np.allclose(a, x)


def test_augment_without_backgrounds_still_works():
    y = augment(tone(0.975), np.random.default_rng(4), [])
    assert np.all(np.isfinite(y))

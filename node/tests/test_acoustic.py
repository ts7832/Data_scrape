import numpy as np

from kuulo_node.acoustic import AcousticMeter
from kuulo_node.audio import SAMPLE_RATE, WINDOW_SAMPLES

t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE


def tone(freq, amp):
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_peak_frequency_of_a_tone():
    assert abs(AcousticMeter().measure(tone(1000, 0.1)).peak_freq_hz - 1000) < 5


def test_snr_rises_when_loud_sound_follows_quiet_floor():
    m = AcousticMeter()
    for _ in range(10):
        quiet = m.measure(tone(200, 0.001))
    loud = m.measure(tone(200, 0.1))
    assert abs(quiet.snr_db) < 3
    assert loud.snr_db > 30


def test_silence_is_valid_and_bounded():
    a = AcousticMeter().measure(np.zeros(WINDOW_SAMPLES, np.float32))
    assert -50 <= a.snr_db <= 150 and a.peak_freq_hz >= 0

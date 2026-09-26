import numpy as np
import soundfile as sf

from kuulo_ml.datasets import Clip
from kuulo_ml.embed import embed_clips, windows_of
from kuulo_node.classify import YamnetOutput

SR = 16_000


class FakeEmbedder:
    names = ["Propeller, airscrew", "Speech"]

    def __init__(self):
        self.calls = 0

    def run(self, window):
        self.calls += 1
        level = float(np.sqrt(np.mean(window**2)))
        return YamnetOutput(np.full(1024, level, np.float32), np.array([level, 0.1], np.float32))


def write(path, x, sr=SR):
    sf.write(path, x, sr)
    return path


def test_windows_are_yamnet_sized_with_half_overlap():
    w = windows_of(np.zeros(SR * 2, np.float32))
    assert w.shape == (3, 15_600)  # 0, 7800, 15600 fit in 32000 samples


def test_positive_windows_far_below_the_loudest_are_dropped(tmp_path):
    quiet = 0.5 * 10 ** (-30 / 20)
    x = np.concatenate([np.full(SR, 0.5), np.full(SR * 3, quiet)]).astype(np.float32)
    clip = Clip(write(tmp_path / "d.wav", x), 1, "g", "drone:M3", "test")
    out = embed_clips([clip], FakeEmbedder(), {"Propeller, airscrew": 1.0})
    # FakeEmbedder's Step A score is the window RMS: every kept window is within 20 dB of 0.5.
    assert 1 <= len(out["label"]) < len(windows_of(x))
    assert np.all(out["step_a"] >= 0.05)


def test_negative_digital_silence_is_dropped_but_quiet_sound_kept(tmp_path):
    x = np.concatenate([np.zeros(SR * 2), np.full(SR * 2, 1e-3)]).astype(np.float32)
    clip = Clip(write(tmp_path / "n.wav", x), 0, "g", "dog", "test")
    out = embed_clips([clip], FakeEmbedder(), {"Propeller, airscrew": 1.0})
    assert 1 <= len(out["label"]) < len(windows_of(x))


def test_resampled_and_step_a_scored_with_given_weights(tmp_path):
    x = (0.3 * np.ones(44100 * 2)).astype(np.float32)
    clip = Clip(write(tmp_path / "d.wav", x, 44100), 1, "g", "drone:M3", "val")
    out = embed_clips([clip], FakeEmbedder(), {"Propeller, airscrew": 0.5})
    assert np.allclose(out["step_a"], 0.15, atol=0.01)
    assert out["embedding"].shape[1] == 1024 and out["embedding"].dtype == np.float32
    assert set(out["kind"]) == {"drone:M3"} and set(out["group"]) == {"g"}


def test_augmented_copies_are_added_and_flagged(tmp_path):
    x = (0.3 * np.sin(np.arange(SR * 2) / 10)).astype(np.float32)
    clip = Clip(write(tmp_path / "d.wav", x), 1, "g", "drone:M3", "train")
    plain = embed_clips([clip], FakeEmbedder(), {"Propeller, airscrew": 1.0})
    aug = embed_clips([clip], FakeEmbedder(), {"Propeller, airscrew": 1.0},
                      copies={1: 2, 0: 1}, rng=np.random.default_rng(0), backgrounds=[])
    n = len(plain["label"])
    assert len(aug["label"]) == 3 * n and aug["augmented"].sum() == 2 * n

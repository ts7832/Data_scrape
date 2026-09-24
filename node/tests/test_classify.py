from pathlib import Path

import numpy as np
import pytest

from kuulo_node.audio import WINDOW_SAMPLES
from kuulo_node.classify import FakeClassifier, check_weights, drone_score

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"


def test_drone_score_is_weighted_max_and_ignores_unweighted_classes():
    weights = {"Propeller, airscrew": 1.0, "Helicopter": 0.8}
    scores = {"Propeller, airscrew": 0.3, "Helicopter": 0.9, "Speech": 0.99}
    assert drone_score(scores, weights) == pytest.approx(0.72)


def test_drone_score_clamps_to_unit_interval():
    assert drone_score({"A": 1.7}, {"A": 1.0}) == 1.0
    assert drone_score({}, {"A": 1.0}) == 0.0


def test_check_weights_names_unknown_classes():
    with pytest.raises(ValueError, match="Propellor"):
        check_weights(["Speech", "Propeller, airscrew"], {"Propellor": 1.0})


def test_fake_classifier_follows_script_by_window_index():
    fake = FakeClassifier(lambda i: {"A": float(i)})
    window = np.zeros(WINDOW_SAMPLES, np.float32)
    assert [fake.score(window)["A"] for _ in range(3)] == [0.0, 1.0, 2.0]
    assert fake.calls == 3


@pytest.mark.skipif(not (MODELS / "yamnet.tflite").exists(), reason="model not downloaded")
def test_yamnet_scores_silence_as_silence():
    from kuulo_node.classify import YamnetClassifier

    clf = YamnetClassifier(
        MODELS / "yamnet.tflite", MODELS / "yamnet_class_map.csv", {"Propeller, airscrew": 1.0}
    )
    scores = clf.score(np.zeros(WINDOW_SAMPLES, np.float32))
    assert len(scores) == 521
    assert max(scores, key=scores.get) == "Silence"

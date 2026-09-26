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


@pytest.mark.skipif(not (MODELS / "yamnet.tflite").exists(), reason="model not downloaded")
def test_embedder_returns_embedding_that_reproduces_yamnet_scores():
    from kuulo_node.classify import YamnetEmbedder

    emb = YamnetEmbedder(MODELS / "yamnet.tflite", MODELS / "yamnet_class_map.csv")
    rng = np.random.default_rng(0)
    out = emb.run((rng.standard_normal(WINDOW_SAMPLES) * 0.1).astype(np.float32))
    assert out.embedding.shape == (1024,) and out.embedding.dtype == np.float32
    assert out.scores.shape == (521,) and len(emb.names) == 521
    logits = emb.head_weights @ out.embedding + emb.head_bias
    assert np.max(np.abs(1 / (1 + np.exp(-logits)) - out.scores)) < 0.06


def test_head_probability_is_rescaled_so_its_threshold_maps_to_half():
    from kuulo_node.classify import rescale_to_half

    assert rescale_to_half(0.2, threshold=0.2) == pytest.approx(0.5)
    assert rescale_to_half(0.0, threshold=0.2) == 0.0 and rescale_to_half(1.0, threshold=0.2) == 1.0
    assert rescale_to_half(0.1, threshold=0.2) == pytest.approx(0.25)
    assert rescale_to_half(0.6, threshold=0.2) == pytest.approx(0.75)


def test_auto_falls_back_to_yamnet_with_a_warning_when_no_head(tmp_path, caplog):
    from dataclasses import replace

    from kuulo_node.classify import YamnetClassifier
    from kuulo_node.cli import build_classifier
    from kuulo_node.testing import make_test_config

    cfg = replace(make_test_config(tmp_path), model_path=MODELS / "yamnet.tflite",
                  class_map_path=MODELS / "yamnet_class_map.csv", classifier="auto",
                  head_path=tmp_path / "nope.onnx")
    if not cfg.model_path.exists():
        pytest.skip("model not downloaded")
    clf, used = build_classifier(cfg)
    assert isinstance(clf, YamnetClassifier) and used.weights == cfg.weights
    assert "make ml" in caplog.text


def test_head_mode_scores_the_drone_output(tmp_path, monkeypatch):
    from dataclasses import replace

    import kuulo_node.classify as classify
    from kuulo_node.cli import build_classifier
    from kuulo_node.testing import make_test_config

    head = tmp_path / "head.onnx"
    head.touch()
    monkeypatch.setattr(classify, "YamnetEmbedder", lambda *a: object())
    monkeypatch.setattr(classify, "HeadClassifier", lambda emb, path: ("head", path))
    cfg = replace(make_test_config(tmp_path), classifier="auto", head_path=head)
    clf, used = build_classifier(cfg)
    assert clf == ("head", head) and used.weights == {"drone": 1.0}


def test_build_classifier_explains_a_missing_head(tmp_path):
    from dataclasses import replace

    from kuulo_node.cli import build_classifier
    from kuulo_node.testing import make_test_config

    cfg = replace(make_test_config(tmp_path), model_path=MODELS / "yamnet.tflite",
                  class_map_path=MODELS / "yamnet_class_map.csv", classifier="head",
                  head_path=tmp_path / "nope.onnx", weights={"drone": 1.0})
    with pytest.raises(ValueError, match="make train"):
        build_classifier(cfg)


@pytest.mark.skipif(not (MODELS / "yamnet.tflite").exists(), reason="model not downloaded")
def test_build_classifier_yamnet_by_default(tmp_path):
    from dataclasses import replace

    from kuulo_node.classify import YamnetClassifier
    from kuulo_node.cli import build_classifier
    from kuulo_node.testing import make_test_config

    cfg = replace(make_test_config(tmp_path), model_path=MODELS / "yamnet.tflite",
                  class_map_path=MODELS / "yamnet_class_map.csv")
    assert isinstance(build_classifier(cfg)[0], YamnetClassifier)

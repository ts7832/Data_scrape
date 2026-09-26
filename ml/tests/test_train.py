import json

import numpy as np
import pytest

from kuulo_ml.evaluate import evaluate_method, results_markdown
from kuulo_ml.train import export_onnx, onnx_probabilities, train_head


def synthetic(n=600, seed=0, shift=1.5):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    x = rng.standard_normal((n, 1024)).astype(np.float32)
    x[:, :8] += shift * y[:, None]
    return {"embedding": x, "label": y}


def test_head_learns_a_separable_problem_and_picks_a_threshold():
    head = train_head(synthetic(seed=0), synthetic(seed=1))
    assert head.val_ap > 0.95
    assert 0.0 < head.threshold < 1.0
    assert head.name in {"logreg", "mlp"}


def test_onnx_export_matches_sklearn(tmp_path):
    head = train_head(synthetic(seed=0), synthetic(seed=1))
    path = tmp_path / "head.onnx"
    export_onnx(head, path)
    x = synthetic(seed=2)["embedding"][:50]
    expected = head.pipeline.predict_proba(x)[:, 1]
    assert np.allclose(onnx_probabilities(path, x), expected, atol=1e-5)
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["threshold"] == pytest.approx(head.threshold)
    assert "DroneNoise" in meta["training_data"]


def test_evaluate_counts_confusion_and_hard_negative_false_positives():
    scores = np.array([0.9, 0.2, 0.8, 0.7, 0.1, 0.6])
    label = np.array([1, 1, 0, 0, 0, 0])
    kind = np.array(["drone:M3", "drone:Yn", "chainsaw", "dog", "chainsaw", "engine"])
    r = evaluate_method(scores, label, kind, threshold=0.5)
    assert r["confusion"] == {"tp": 1, "fn": 1, "fp": 3, "tn": 1}
    assert r["precision"] == pytest.approx(0.25) and r["recall"] == pytest.approx(0.5)
    assert r["hard_negative_fpr"]["chainsaw"] == pytest.approx(0.5)
    assert r["hard_negative_fpr"]["engine"] == pytest.approx(1.0)
    assert "dog" not in r["hard_negative_fpr"]
    assert r["recall_by_drone"] == {"drone:M3": 1.0, "drone:Yn": 0.0}


def test_results_markdown_has_a_row_per_method():
    r = evaluate_method(np.array([0.9, 0.1]), np.array([1, 0]), np.array(["drone:M3", "dog"]), 0.5)
    text = results_markdown({"Step A": r, "Step B": r}, windows=2, drone_windows=1)
    assert "| Step A |" in text and "| Step B |" in text


def test_node_head_classifier_uses_the_exported_model(tmp_path):
    from kuulo_node.classify import HeadClassifier, YamnetOutput

    head = train_head(synthetic(seed=0), synthetic(seed=1))
    path = tmp_path / "head.onnx"
    export_onnx(head, path)
    positive = synthetic(seed=3, shift=6.0)
    x = positive["embedding"][positive["label"] == 1][0]

    class Fake:
        names = ["Propeller, airscrew", "Speech"]

        def run(self, window):
            return YamnetOutput(x, np.array([0.2, 0.7], np.float32))

    clf = HeadClassifier(Fake(), path)
    scores = clf.score(np.zeros(15_600, np.float32))
    assert scores["drone"] > 0.5 and scores["Speech"] == pytest.approx(0.7)


def test_level_match_sets_rms_and_stays_in_range():
    from kuulo_ml.evaluate import level_match

    w = (0.001 * np.sin(np.arange(15_600) / 5)).astype(np.float32)
    out = level_match(w, -20.0)
    level = 20 * np.log10(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert level == pytest.approx(-20, abs=0.1)
    assert np.max(np.abs(level_match(w, 20.0))) <= 1.0


def test_results_markdown_includes_notes():
    r = evaluate_method(np.array([0.9, 0.1]), np.array([1, 0]), np.array(["drone:M3", "dog"]), 0.5)
    assert "Note: hello" in results_markdown({"A": r}, 2, 1, notes=["Note: hello"])


def test_first_start_uses_the_node_smoother_timing():
    from kuulo_ml.evaluate import first_start_s

    # 3 of 5 windows >= 0.5 fires START on the 3rd high window (index 4 here).
    assert first_start_s([0, 0, 0.9, 0.9, 0.9, 0.9]) == pytest.approx(4 * 0.4875 + 0.975)
    assert first_start_s([0.9, 0, 0.9, 0, 0, 0, 0.9]) is None


def test_event_summary_counts_recordings():
    from kuulo_ml.evaluate import event_summary

    rows = [(1, "drone:M3", 2.0), (1, "drone:Yn", None), (0, "chainsaw", 1.0), (0, "dog", None)]
    s = event_summary(rows)
    assert s["drone_detected"] == 1 and s["drone_total"] == 2
    assert s["median_start_s"] == pytest.approx(2.0)
    assert s["hard_false_starts"] == 1 and s["hard_total"] == 1
    assert s["other_false_starts"] == 0 and s["other_total"] == 1

"""Step B: a small head on frozen YAMNet embeddings, chosen on validation, exported to ONNX."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

TRAINING_DATA = (
    "DroneNoise Database (Univ. of Salford, CC BY 4.0) positives; "
    "ESC-50 (CC BY-NC 3.0) negatives: research use only, retrain before commercial use"
)


@dataclass
class TrainedHead:
    name: str  # "logreg" | "mlp"
    detail: str  # e.g. "logreg C=0.01"
    pipeline: Pipeline
    threshold: float
    val_ap: float
    val_f1: float


def _balanced(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Oversample the minority class (MLPClassifier has no class_weight)."""
    pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    small, big = (pos, neg) if pos.size < neg.size else (neg, pos)
    extra = rng.choice(small, size=big.size - small.size, replace=True)
    idx = rng.permutation(np.concatenate([pos, neg, extra]))
    return x[idx], y[idx]


def _candidates(seed: int) -> dict[str, tuple[Pipeline, bool]]:
    """name -> (pipeline, needs oversampling)."""
    out: dict[str, tuple[Pipeline, bool]] = {}
    for c in (0.001, 0.01, 0.1):
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
        out[f"logreg C={c}"] = (make_pipeline(StandardScaler(), clf), False)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, early_stopping=True,
                        max_iter=200, random_state=seed)
    out["mlp 128"] = (make_pipeline(StandardScaler(), mlp), True)
    return out


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def train_head(train: dict, val: dict, *, seed: int = 0, log=None) -> TrainedHead:
    rng = np.random.default_rng(seed)
    x, y = train["embedding"], train["label"].astype(int)
    vx, vy = val["embedding"], val["label"].astype(int)
    best: TrainedHead | None = None
    for name, (pipeline, oversample) in _candidates(seed).items():
        fx, fy = _balanced(x, y, rng) if oversample else (x, y)
        pipeline.fit(fx, fy)
        p = pipeline.predict_proba(vx)[:, 1]
        ap = float(average_precision_score(vy, p))
        threshold, f1 = best_f1_threshold(vy, p)
        if log:
            log(f"  {name:14s} val AP {ap:.3f}  best F1 {f1:.3f} at {threshold:.3f}")
        if best is None or ap > best.val_ap:
            best = TrainedHead(name.split()[0], name, pipeline, threshold, ap, f1)
    assert best is not None
    return best


def export_onnx(head: TrainedHead, path: Path) -> None:
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    clf = head.pipeline.steps[-1][1]
    model = to_onnx(
        head.pipeline, initial_types=[("embedding", FloatTensorType([None, 1024]))],
        options={id(clf): {"zipmap": False}}, target_opset=17,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    path.with_suffix(".json").write_text(json.dumps({
        "model": head.detail,
        "threshold": head.threshold,
        "val_average_precision": head.val_ap,
        "val_f1": head.val_f1,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": "YAMNet layer28 embedding (1024, dequantised), MediaPipe yamnet.tflite",
        "training_data": TRAINING_DATA,
    }, indent=2))


def onnx_probabilities(path: Path, embeddings: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"embedding": np.asarray(embeddings, np.float32)})
    return np.asarray(outputs[1])[:, 1]

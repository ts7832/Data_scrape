"""Window -> per-class scores -> one drone score. YAMNet now; a trained head can drop in later."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import WINDOW_SAMPLES


class Classifier(Protocol):
    def score(self, window: np.ndarray) -> dict[str, float]: ...


def drone_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    best = max((w * scores.get(name, 0.0) for name, w in weights.items()), default=0.0)
    return min(1.0, max(0.0, best))


def check_weights(class_names: list[str], weights: dict[str, float]) -> None:
    unknown = sorted(set(weights) - set(class_names))
    if unknown:
        raise ValueError(
            f"[weights] names classes YAMNet does not have: {unknown}. "
            "Use exact display names from data/models/yamnet_class_map.csv."
        )


class FakeClassifier:
    """Scripted scores for tests: script(i) gives the scores for the i-th window."""

    def __init__(self, script: Callable[[int], dict[str, float]]) -> None:
        self._script = script
        self.calls = 0

    def score(self, window: np.ndarray) -> dict[str, float]:
        result = self._script(self.calls)
        self.calls += 1
        return result


def load_class_names(path: Path) -> list[str]:
    with open(path, newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


class YamnetClassifier:
    """YAMNet TFLite via ai-edge-litert (no TensorFlow). 521 AudioSet class scores per window."""

    def __init__(self, model_path: Path, class_map_path: Path, weights: dict[str, float]) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.names = load_class_names(class_map_path)
        check_weights(self.names, weights)
        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        inp = self._interp.get_input_details()[0]
        self._in_index = inp["index"]
        self._in_shape = tuple(int(d) for d in inp["shape"])
        if int(np.prod(self._in_shape)) != WINDOW_SAMPLES:
            raise ValueError(f"unexpected YAMNet input shape {self._in_shape}")
        self._out_index = self._interp.get_output_details()[0]["index"]

    def score(self, window: np.ndarray) -> dict[str, float]:
        x = np.asarray(window, dtype=np.float32).reshape(self._in_shape)
        self._interp.set_tensor(self._in_index, x)
        self._interp.invoke()
        out = self._interp.get_tensor(self._out_index).reshape(-1, len(self.names)).mean(axis=0)
        return dict(zip(self.names, (float(v) for v in out), strict=True))

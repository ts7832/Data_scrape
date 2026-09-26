"""Window -> per-class scores -> one drone score. YAMNet now; a trained head can drop in later."""

from __future__ import annotations

import csv
import warnings
from collections.abc import Callable
from dataclasses import dataclass
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


@dataclass(frozen=True)
class YamnetOutput:
    embedding: np.ndarray  # float32 [1024]: YAMNet's pooled penultimate layer
    scores: np.ndarray  # float32 [521]: AudioSet class probabilities


def _find(details: list[dict], suffix: str) -> dict:
    matches = [d for d in details if d["name"].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"YAMNet model has no unique tensor ending in {suffix!r}")
    return matches[0]


class YamnetEmbedder:
    """One YAMNet pass giving both the 1024-d embedding (Step B's input) and the class scores.

    The MediaPipe model is int8-quantised inside, so the embedding tensor is dequantised with
    its (scale, zero point). Reading an intermediate tensor needs the plain built-in kernels:
    the default XNNPACK delegate computes the graph internally and never writes it.
    """

    EMBEDDING = "layer28/reduce_mean"
    HEAD_WEIGHTS = "layer29/fc/MatMul"
    HEAD_BIAS = "layer29/fc/biases"

    def __init__(self, model_path: Path, class_map_path: Path) -> None:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType

        self.names = load_class_names(class_map_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "preserve_all_tensors is for debugging" notice
            self._interp = Interpreter(
                model_path=str(model_path), experimental_preserve_all_tensors=True,
                experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
        self._interp.allocate_tensors()
        details = self._interp.get_tensor_details()
        self._in_index = self._interp.get_input_details()[0]["index"]
        self._out_index = self._interp.get_output_details()[0]["index"]
        self._emb = _find(details, self.EMBEDDING)
        self.head_weights = self._dequantised(_find(details, self.HEAD_WEIGHTS))
        self.head_bias = self._dequantised(_find(details, self.HEAD_BIAS))

    def _dequantised(self, detail: dict) -> np.ndarray:
        raw = self._interp.get_tensor(detail["index"])
        scale, zero = detail["quantization"]
        if scale == 0:  # float tensor
            return raw.astype(np.float32)
        return ((raw.astype(np.float32) - zero) * scale).astype(np.float32)

    def run(self, window: np.ndarray) -> YamnetOutput:
        x = np.asarray(window, dtype=np.float32).reshape(-1)
        if x.size != WINDOW_SAMPLES:
            raise ValueError(f"expected {WINDOW_SAMPLES} samples, got {x.size}")
        self._interp.set_tensor(self._in_index, x)
        self._interp.invoke()
        scores = self._interp.get_tensor(self._out_index).reshape(-1).astype(np.float32)
        return YamnetOutput(self._dequantised(self._emb).reshape(-1), scores)

"""Clips -> YAMNet windows -> (1024-d embedding, Step A drone score) per window.

Window rules (documented in ml/RESULTS.md):
- drone recordings: keep windows within 20 dB of the recording's loudest window. The DroneNoise
  overflights start and end with the drone far away and possibly inaudible at that microphone;
  labelling those windows "drone" would teach the head that silence is a drone.
- negatives: drop only digital silence (ESC-50 pads short clips with exact zeros).
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from kuulo_node.audio import HOP_SAMPLES, WINDOW_SAMPLES, to_mono_16k
from kuulo_node.classify import drone_score

from .augment import augment
from .datasets import Clip

GATE_DB = 20.0
SILENCE = 1e-6


def load_16k(path) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    return to_mono_16k(data, rate)


def windows_of(x: np.ndarray) -> np.ndarray:
    if x.size < WINDOW_SAMPLES:
        x = np.pad(x, (0, WINDOW_SAMPLES - x.size))
    starts = range(0, x.size - WINDOW_SAMPLES + 1, HOP_SAMPLES)
    return np.stack([x[s : s + WINDOW_SAMPLES] for s in starts])


def _keep(windows: np.ndarray, label: int) -> np.ndarray:
    if label == 1:
        level = 20 * np.log10(np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1)) + 1e-12)
        return level >= level.max() - GATE_DB
    return np.max(np.abs(windows), axis=1) >= SILENCE


def embed_clips(
    clips: list[Clip],
    embedder,
    weights: dict[str, float],
    *,
    copies: dict[int, int] | None = None,
    rng: np.random.Generator | None = None,
    backgrounds: list[np.ndarray] | None = None,
    progress=None,
) -> dict[str, np.ndarray]:
    """copies: augmented copies per kept window by label, e.g. {1: 2, 0: 1} (training only)."""
    copies = copies or {}
    rng = rng or np.random.default_rng(0)
    rows: dict[str, list] = {k: [] for k in
                             ("embedding", "step_a", "label", "group", "kind", "clip", "augmented")}

    def add(window, clip: Clip, augmented: bool) -> None:
        out = embedder.run(window)
        scores = dict(zip(embedder.names, out.scores.tolist(), strict=True))
        rows["embedding"].append(out.embedding)
        rows["step_a"].append(drone_score(scores, weights))
        rows["label"].append(clip.label)
        rows["group"].append(clip.group)
        rows["kind"].append(clip.kind)
        rows["clip"].append(clip.path.name)
        rows["augmented"].append(augmented)

    for i, clip in enumerate(clips):
        windows = windows_of(load_16k(clip.path))
        for window in windows[_keep(windows, clip.label)]:
            add(window, clip, False)
            for _ in range(copies.get(clip.label, 0)):
                add(augment(window, rng, backgrounds or []), clip, True)
        if progress:
            progress(i + 1, len(clips))
    return {
        "embedding": np.asarray(rows["embedding"], np.float32).reshape(-1, 1024),
        "step_a": np.asarray(rows["step_a"], np.float32),
        "label": np.asarray(rows["label"], np.int8),
        "group": np.asarray(rows["group"], dtype=str),
        "kind": np.asarray(rows["kind"], dtype=str),
        "clip": np.asarray(rows["clip"], dtype=str),
        "augmented": np.asarray(rows["augmented"], bool),
    }

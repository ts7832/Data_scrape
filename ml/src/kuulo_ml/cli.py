"""kuulo-ml: datasets -> embed -> train -> evaluate. Everything lands in data/ (gitignored)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATASETS = REPO / "data" / "datasets"
FEATURES = REPO / "data" / "ml"
MODELS = REPO / "data" / "models"
NODE_CONFIG = REPO / "node" / "config.example.toml"
BACKGROUNDS = 200


def _progress(done: int, total: int) -> None:
    if done == total or done % 50 == 0:
        print(f"  {done}/{total} clips", flush=True)


def cmd_datasets(_args) -> int:
    from .datasets import download_dronenoise, download_esc50

    download_dronenoise(DATASETS)
    download_esc50(DATASETS)
    print(f"datasets ready in {DATASETS}")
    return 0


def cmd_embed(args) -> int:
    from kuulo_node.classify import YamnetEmbedder
    from kuulo_node.config import load_config

    from .datasets import HARD_NEGATIVES, manifest
    from .embed import embed_clips, load_16k

    weights = load_config(NODE_CONFIG).weights
    embedder = YamnetEmbedder(MODELS / "yamnet.tflite", MODELS / "yamnet_class_map.csv")
    clips = manifest(DATASETS)
    rng = np.random.default_rng(args.seed)
    pool = [c for c in clips
            if c.split == "train" and c.label == 0 and c.kind not in HARD_NEGATIVES]
    picks = rng.choice(len(pool), size=min(BACKGROUNDS, len(pool)), replace=False)
    backgrounds = [load_16k(pool[i].path) for i in picks]
    FEATURES.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        subset = [c for c in clips if c.split == split]
        print(f"{split}: {len(subset)} clips", flush=True)
        copies = {1: 2, 0: 1} if split == "train" else {}
        out = embed_clips(subset, embedder, weights, copies=copies, rng=rng,
                          backgrounds=backgrounds, progress=_progress)
        np.savez_compressed(FEATURES / f"features_{split}.npz", **out)
        print(f"  {len(out['label'])} windows ({int(out['label'].sum())} drone)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuulo-ml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="download DroneNoise and ESC-50 into data/datasets/")
    embed = sub.add_parser("embed", help="YAMNet embeddings + Step A scores per window")
    embed.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    return {"datasets": cmd_datasets, "embed": cmd_embed}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

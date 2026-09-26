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


def _load(split: str) -> dict:
    path = FEATURES / f"features_{split}.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `make embed` first")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _clean(features: dict) -> dict:
    keep = ~features["augmented"]
    return {k: v[keep] for k, v in features.items()}


def cmd_train(args) -> int:
    from .train import export_onnx, train_head

    train, val = _load("train"), _clean(_load("val"))
    print(f"train {len(train['label'])} windows ({int(train['label'].sum())} drone), "
          f"val {len(val['label'])} ({int(val['label'].sum())} drone)")
    head = train_head(train, val, seed=args.seed, log=print)
    export_onnx(head, MODELS / "drone_head.onnx")
    print(f"chose {head.detail}: val AP {head.val_ap:.3f}, threshold {head.threshold:.3f} -> "
          f"{MODELS / 'drone_head.onnx'}")
    return 0


def cmd_evaluate(_args) -> int:
    import json

    from .evaluate import evaluate_method, results_markdown
    from .train import best_f1_threshold, onnx_probabilities

    test, val = _clean(_load("test")), _clean(_load("val"))
    head_path = MODELS / "drone_head.onnx"
    meta = json.loads(head_path.with_suffix(".json").read_text())
    tuned_a, _ = best_f1_threshold(val["label"], val["step_a"])
    step_b = onnx_probabilities(head_path, test["embedding"])
    results = {
        "Step A (YAMNet, node default 0.5)": evaluate_method(
            test["step_a"], test["label"], test["kind"], 0.5),
        "Step A (threshold tuned on val)": evaluate_method(
            test["step_a"], test["label"], test["kind"], tuned_a),
        f"Step B ({meta['model']})": evaluate_method(
            step_b, test["label"], test["kind"], meta["threshold"]),
    }
    text = results_markdown(results, windows=len(test["label"]),
                            drone_windows=int(test["label"].sum()))
    (REPO / "ml" / "RESULTS.md").write_text(text)
    (FEATURES / "results.json").write_text(json.dumps(results, indent=2))
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuulo-ml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("datasets", help="download DroneNoise and ESC-50 into data/datasets/")
    embed = sub.add_parser("embed", help="YAMNet embeddings + Step A scores per window")
    embed.add_argument("--seed", type=int, default=0)
    train = sub.add_parser("train", help="fit the Step B head and export ONNX")
    train.add_argument("--seed", type=int, default=0)
    sub.add_parser("evaluate", help="Step A vs Step B on the held-out test set -> ml/RESULTS.md")
    args = parser.parse_args(argv)
    commands = {"datasets": cmd_datasets, "embed": cmd_embed, "train": cmd_train,
                "evaluate": cmd_evaluate}
    return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

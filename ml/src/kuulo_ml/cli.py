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


def _replay_recordings(clips, embedder, weights, head_path, head_threshold):
    """One YAMNet pass per window of every held-out recording -> START time per method."""
    from kuulo_node.classify import drone_score, rescale_to_half

    from .embed import load_16k, windows_of
    from .evaluate import first_start_s
    from .train import onnx_probabilities

    rows_a, rows_b = [], []
    for i, clip in enumerate(clips):
        outs = [embedder.run(w) for w in windows_of(load_16k(clip.path))]
        a = [drone_score(dict(zip(embedder.names, o.scores.tolist(), strict=True)), weights)
             for o in outs]
        p = onnx_probabilities(head_path, np.stack([o.embedding for o in outs]))
        b = [rescale_to_half(float(x), head_threshold) for x in p]
        rows_a.append((clip.label, clip.kind, first_start_s(a)))
        rows_b.append((clip.label, clip.kind, first_start_s(b)))
        _progress(i + 1, len(clips))
    return rows_a, rows_b


def _level_probe(clips, embedder, head_path, threshold, target_db) -> dict[str, float]:
    """Step B false-positive rate on negatives rescaled to the drone recordings' median level."""
    from .datasets import HARD_NEGATIVES
    from .embed import _keep, load_16k, windows_of
    from .evaluate import level_match
    from .train import onnx_probabilities

    out = {}
    for name, subset in (("hard", [c for c in clips if c.kind in HARD_NEGATIVES]),
                         ("other", [c for c in clips if c.label == 0
                                    and c.kind not in HARD_NEGATIVES])):
        emb = []
        for clip in subset:
            windows = windows_of(load_16k(clip.path))
            emb += [embedder.run(level_match(w, target_db)).embedding
                    for w in windows[_keep(windows, 0)]]
        out[name] = float(np.mean(onnx_probabilities(head_path, np.stack(emb)) >= threshold))
    return out


def cmd_evaluate(_args) -> int:
    import json

    from kuulo_node.classify import YamnetEmbedder
    from kuulo_node.config import load_config

    from .datasets import HARD_NEGATIVES, manifest
    from .evaluate import evaluate_method, event_summary, results_markdown
    from .train import best_f1_threshold, onnx_probabilities

    test, val = _clean(_load("test")), _clean(_load("val"))
    head_path = MODELS / "drone_head.onnx"
    meta = json.loads(head_path.with_suffix(".json").read_text())
    tuned_a, _ = best_f1_threshold(val["label"], val["step_a"])
    step_b = onnx_probabilities(head_path, test["embedding"])
    name_b = f"Step B ({meta['model']})"
    results = {
        "Step A (YAMNet, node default 0.5)": evaluate_method(
            test["step_a"], test["label"], test["kind"], 0.5),
        "Step A (threshold tuned on val)": evaluate_method(
            test["step_a"], test["label"], test["kind"], tuned_a),
        name_b: evaluate_method(step_b, test["label"], test["kind"], meta["threshold"]),
    }

    print("replaying held-out recordings through the node's smoother", flush=True)
    embedder = YamnetEmbedder(MODELS / "yamnet.tflite", MODELS / "yamnet_class_map.csv")
    weights = load_config(NODE_CONFIG).weights
    clips = [c for c in manifest(DATASETS) if c.split == "test"]
    rows_a, rows_b = _replay_recordings(clips, embedder, weights, head_path, meta["threshold"])
    events = {"Step A": event_summary(rows_a), "Step B": event_summary(rows_b)}
    drone_db = [float(x) for x in _drone_levels(clips)]
    target = float(np.median(drone_db))
    print("level-matching negatives to the drone recordings", flush=True)
    probe = _level_probe(clips, embedder, head_path, meta["threshold"], target)

    hard = np.isin(test["kind"], list(HARD_NEGATIVES)) & (test["label"] == 0)
    rest = ~np.isin(test["kind"], list(HARD_NEGATIVES)) & (test["label"] == 0)
    fired = step_b >= meta["threshold"]
    notes = [
        f"Robustness: with every negative test window rescaled to the drone recordings' median "
        f"level ({target:.1f} dBFS), Step B's false-positive rate is {probe['hard']:.3f} on hard "
        f"negatives and {probe['other']:.3f} on the rest (unscaled: "
        f"{float(np.mean(fired[hard])):.3f} and {float(np.mean(fired[rest])):.3f}). A large jump "
        "would mean the head keys on loudness rather than on the sound itself.",
        "The tuned Step A threshold is the best F1 on validation. When it is ~0, no threshold "
        "beats flagging every window: YAMNet's class scores do not separate these distant, "
        "quiet field recordings (YAMNet mostly labels them 'Silence').",
        "All drone positives come from one outdoor campaign (Edzell, Scotland, 2022) recorded "
        "with measurement microphones. Performance through a laptop microphone in a city is not "
        "measured here, and a recording-site bias cannot be ruled out with one campaign.",
    ]
    text = results_markdown(results, windows=len(test["label"]),
                            drone_windows=int(test["label"].sum()), notes=notes,
                            sections=[_event_table(events)])
    (REPO / "ml" / "RESULTS.md").write_text(text)
    (FEATURES / "results.json").write_text(json.dumps(
        {"windows": results, "recordings": events, "level_probe": probe}, indent=2))
    print(text)
    return 0


def _drone_levels(clips):
    from .embed import _keep, load_16k, windows_of

    for clip in (c for c in clips if c.label == 1):
        windows = windows_of(load_16k(clip.path))
        for w in windows[_keep(windows, 1)]:
            yield 20 * np.log10(np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-12)


def _event_table(events: dict[str, dict]) -> str:
    rows = ["## Per recording (what a node would report)", "",
            "Each held-out recording replayed through the node's 3-of-5 smoother. Time to START "
            "is audio time from the start of each recording (DroneNoise recordings are 20 s of "
            "an overflight; ESC-50 clips are 5 s). With a 0.49 s hop, 1.95 s is the earliest "
            "the 3-of-5 rule can fire.", "",
            "| Method | Drone recordings detected | Median / max time to START | "
            "False START, hard negatives | False START, other |",
            "|---|---|---|---|---|"]
    for name, e in events.items():
        timing = (f"{e['median_start_s']:.2f} s / {e['max_start_s']:.2f} s"
                  if e["median_start_s"] is not None else "—")
        rows.append(f"| {name} | {e['drone_detected']}/{e['drone_total']} | {timing} | "
                    f"{e['hard_false_starts']}/{e['hard_total']} | "
                    f"{e['other_false_starts']}/{e['other_total']} |")
    return "\n".join(rows)


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

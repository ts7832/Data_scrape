"""`kuulo-node run`: listen (or replay a wav) and report drone detections to a Kuulo server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx

from .audio import mic_source, wav_source
from .config import ConfigError, NodeConfig, load_config
from .keys import load_or_create_keys
from .outbox import Outbox
from .runner import NodeRunner
from .tracestore import TraceStore
from .traceupload import TraceUploader
from .uplink import RegistrationConflict, Uplink

log = logging.getLogger("kuulo.node")


def build_classifier(cfg: NodeConfig):
    """Step A (YAMNet class scores) or Step B (trained head on YAMNet embeddings)."""
    from .classify import HeadClassifier, YamnetClassifier, YamnetEmbedder

    if cfg.classifier == "head":
        if cfg.head_path is None or not cfg.head_path.exists():
            raise ValueError(f"trained head not found at {cfg.head_path}. Run `make ml` "
                             "(or `make train` after `make datasets embed`), or set "
                             'classifier = "yamnet".')
        return HeadClassifier(YamnetEmbedder(cfg.model_path, cfg.class_map_path), cfg.head_path)
    return YamnetClassifier(cfg.model_path, cfg.class_map_path, cfg.weights)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuulo-node")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run the node")
    run_p.add_argument("--config", type=Path, default=Path("node/config.local.toml"))
    run_p.add_argument("--input", type=Path, help="replay a wav file instead of the microphone")
    run_p.add_argument("--speed", type=float, default=1.0, help="wav replay speed; 0 = max")
    run_p.add_argument("--print-scores", action="store_true", help="print scores per window")
    run_p.add_argument("--debug-save-clips", type=Path, metavar="DIR",
                       help="save each detection's raw audio as a wav in DIR (local only; off "
                            "by default because it records whatever the microphone hears)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    if not cfg.model_path.exists() or not cfg.class_map_path.exists():
        log.error("YAMNet model not found at %s. Run `make model` first.", cfg.model_path)
        return 2
    try:
        classifier = build_classifier(cfg)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    keys = load_or_create_keys(cfg.key_file)
    with httpx.Client(base_url=cfg.server_url, timeout=5.0) as client:
        outbox = Outbox(cfg.state_dir / "outbox.db")
        if len(outbox):
            log.info("%d unsent messages from a previous run will be delivered", len(outbox))
        uplink = Uplink(client, outbox=outbox)
        store = uploader = None
        if cfg.traces.enabled:
            store = TraceStore(cfg.state_dir / "traces", cfg.node_id, keys.private_key,
                               cfg.time_quality, budget_bytes=int(cfg.traces.budget_mb * 2**20))
            uploader = TraceUploader(
                client, store, node_id=cfg.node_id, private_key=keys.private_key,
                upload_bytes_per_day=int(cfg.traces.upload_mb_per_day * 2**20),
                sample_rate=cfg.traces.sample_rate,
            )
        runner = NodeRunner(cfg, keys, classifier, uplink, print_scores=args.print_scores,
                            traces=store, trace_uploader=uploader,
                            debug_clip_dir=args.debug_save_clips)
        if args.input:
            blocks = wav_source(args.input, speed=args.speed)
        else:
            blocks = mic_source(on_error=runner.health.mark_failed)
        try:
            stats = runner.run(blocks)
        except RegistrationConflict as exc:
            log.error("%s", exc)
            return 1
    log.info("done: %d windows, %d observations", stats.windows, stats.observations)
    return 0


if __name__ == "__main__":
    sys.exit(main())

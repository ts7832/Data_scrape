"""`kuulo-node run`: listen (or replay a wav) and report drone detections to a Kuulo server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx

from .audio import mic_source, wav_source
from .config import ConfigError, load_config
from .keys import load_or_create_keys
from .runner import NodeRunner
from .uplink import RegistrationConflict, Uplink

log = logging.getLogger("kuulo.node")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuulo-node")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run the node")
    run_p.add_argument("--config", type=Path, default=Path("node/config.local.toml"))
    run_p.add_argument("--input", type=Path, help="replay a wav file instead of the microphone")
    run_p.add_argument("--speed", type=float, default=1.0, help="wav replay speed; 0 = max")
    run_p.add_argument("--print-scores", action="store_true", help="print scores per window")
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
    from .classify import YamnetClassifier

    try:
        classifier = YamnetClassifier(cfg.model_path, cfg.class_map_path, cfg.weights)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    keys = load_or_create_keys(cfg.key_file)
    with httpx.Client(base_url=cfg.server_url, timeout=5.0) as client:
        runner = NodeRunner(cfg, keys, classifier, Uplink(client), print_scores=args.print_scores)
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

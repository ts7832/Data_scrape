"""kuulo-sim: run bundled or custom scenarios against a local server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

from .engine import SimulationRun
from .runner import run_realtime, utc_now
from .scenario import bundled_scenarios, load_scenario, resolve_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuulo-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list bundled scenarios")
    run_p = sub.add_parser("run", help="play a scenario against a server")
    run_p.add_argument("scenario", help="bundled scenario name or path to a .yaml file")
    run_p.add_argument("--server", default="http://127.0.0.1:8000")
    run_p.add_argument("--speed", type=float, default=1.0, help="time compression factor")
    run_p.add_argument("--truth", type=Path, default=Path("data/truth.json"))
    args = parser.parse_args(argv)

    if args.command == "list":
        print("\n".join(bundled_scenarios()))
        return 0

    try:
        path = resolve_scenario(args.scenario)
        scenario = load_scenario(path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"Invalid scenario {args.scenario}:", file=sys.stderr)
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            print(f"  {loc}: {error['msg']}", file=sys.stderr)
        return 2

    run = SimulationRun(scenario, utc_now(), time_scale=args.speed)
    try:
        with httpx.Client(base_url=args.server, timeout=5.0) as client:
            stats = run_realtime(run, client, truth_path=args.truth)
    except httpx.HTTPStatusError as exc:
        print(f"Server rejected {exc.request.url}: {exc.response.status_code} {exc.response.text}",
              file=sys.stderr)
        return 1
    except httpx.ConnectError:
        print(f"Could not connect to {args.server}. Is the server running? (make server)",
              file=sys.stderr)
        return 1
    print(f"{scenario.name}: sent {stats.sent}, rejected {stats.errors}, "
          f"trace segments uploaded {stats.trace_segments}")
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Plays a simulation against a running server in real (or scaled) time."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .engine import SimulationRun

log = logging.getLogger("kuulo.sim")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunStats:
    sent: int = 0
    errors: int = 0


def write_truth(run: SimulationRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "at": p.at.isoformat(), "drone_id": p.drone_id,
            "lat": p.position.lat, "lon": p.position.lon,
        }
        for p in run.truth()
    ]
    path.write_text(json.dumps(rows, indent=1))


def run_realtime(
    run: SimulationRun,
    client: httpx.Client,
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = utc_now,
    truth_path: Path | None = None,
) -> RunStats:
    for reg in run.registrations():
        client.post("/v1/nodes/register", json=reg.model_dump(mode="json")).raise_for_status()
    if truth_path is not None:
        write_truth(run, truth_path)
    stats = RunStats()
    for msg in run.messages():
        wait = (msg.at - now()).total_seconds()
        if wait > 0:
            sleep(wait)
        path = "/v1/observations" if msg.kind == "observation" else "/v1/heartbeats"
        headers = {"content-type": "application/json"}
        response = client.post(path, content=msg.payload.model_dump_json(), headers=headers)
        if response.status_code >= 400:
            stats.errors += 1
            log.warning("%s rejected (%s): %s", path, response.status_code, response.text)
        else:
            stats.sent += 1
    return stats

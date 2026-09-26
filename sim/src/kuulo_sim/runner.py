"""Plays a simulation against a running server in real (or scaled) time."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from kuulo_protocol.signing import sign
from kuulo_protocol.traces import TraceRequest, TraceUnavailable

from .engine import SimulationRun

TRACE_POLL_S = 10.0

log = logging.getLogger("kuulo.sim")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunStats:
    sent: int = 0
    errors: int = 0
    trace_segments: int = 0


class TraceResponder:
    """Answers the server's trace requests the way real nodes would (spec §5.1 server pull)."""

    def __init__(self, run: SimulationRun, client: httpx.Client) -> None:
        self.run, self.client = run, client
        self.uploaded: set[tuple[str, str]] = set()  # (trace_id, node_id)

    def service(self, now: datetime) -> int:
        """Upload every requested segment a node would have closed by `now`. Returns the count."""
        sent = 0
        for node in self.run.scenario.nodes:
            try:
                r = self.client.get("/v1/traces/requests", params={"node_id": node.id})
            except httpx.TransportError:
                return sent
            if r.status_code != 200 or not isinstance(r.json(), list):
                continue
            for request in (TraceRequest.model_validate(item) for item in r.json()):
                sent += self._answer(node.id, request, now)
        return sent

    def _answer(self, node_id: str, request: TraceRequest, now: datetime) -> int:
        segments = self.run.trace_segments(node_id, request.detection_id)
        if not segments:
            msg = TraceUnavailable(node_id=node_id, detection_id=request.detection_id, sent_at=now)
            signed = sign(msg, self.run.node_keys[node_id][0])
            self.client.post("/v1/traces/unavailable", content=signed.model_dump_json(),
                             headers={"content-type": "application/json"})
            return 0
        sent = 0
        for seg in segments:
            key = (str(seg.header.trace_id), node_id)
            if seg.available_at > now or key in self.uploaded:
                continue
            r = self.client.post(
                "/v1/traces", data={"header": seg.header.model_dump_json()},
                files={"body": ("trace.npz", seg.body, "application/octet-stream")},
            )
            if r.status_code == 200:
                self.uploaded.add(key)
                sent += 1
            else:
                log.warning("trace rejected (%s): %s", r.status_code, r.text[:200])
        return sent


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
    responder = TraceResponder(run, client)
    next_poll = run.t0
    for msg in run.messages():
        wait = (msg.at - now()).total_seconds()
        if wait > 0:
            sleep(wait)
        if msg.at >= next_poll:
            stats.trace_segments += responder.service(msg.at)
            next_poll = msg.at + timedelta(seconds=TRACE_POLL_S)
        path = "/v1/observations" if msg.kind == "observation" else "/v1/heartbeats"
        headers = {"content-type": "application/json"}
        response = client.post(path, content=msg.payload.model_dump_json(), headers=headers)
        if response.status_code >= 400:
            stats.errors += 1
            log.warning("%s rejected (%s): %s", path, response.status_code, response.text)
        else:
            stats.sent += 1
    # A detection's last segment closes after its END; give the server one last answer.
    stats.trace_segments += responder.service(run.end_at + timedelta(seconds=60))
    return stats

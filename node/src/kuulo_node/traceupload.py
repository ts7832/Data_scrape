"""Which traces leave the node, and when (spec §5.1 upload policy).

- Server pull: every segment of a detection the server requests (it joined a confirmed track).
  Bypasses the daily budget. A detection the node no longer holds is answered "unavailable".
- Auto-push: a detection with confidence >= 0.8 sustained for >= 20 s.
- Hard-negative sampling: 1 % of the remaining (uncorroborated) detections, first 3 minutes only.
Pushes and samples share a daily upload budget (default 50 MB).
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

import httpx

from kuulo_protocol.signing import sign
from kuulo_protocol.traces import TraceRequest, TraceUnavailable

from .tracestore import SegmentInfo, TraceStore, utc_now

log = logging.getLogger("kuulo.node")
PUSH_AFTER_S = 20.0
PUSH_CONFIDENCE = 0.8
SAMPLE_SEGMENTS = 3  # 3 minutes of 60 s segments


class _Stop(Exception):
    """The server is unreachable or does not know us: try again later."""


class TraceUploader:
    def __init__(
        self,
        client: httpx.Client,
        store: TraceStore,
        *,
        node_id: str,
        private_key: str,
        now: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        upload_bytes_per_day: int = 50 * 2**20,
        sample_rate: float = 0.01,
        poll_every_s: float = 10.0,
    ) -> None:
        self.client, self.store = client, store
        self.node_id, self._key = node_id, private_key
        self._now, self._monotonic = now, monotonic
        self._rng = rng or random.Random()
        self.upload_bytes_per_day = upload_bytes_per_day
        self.sample_rate = sample_rate
        self.poll_every_s = poll_every_s
        self._last_poll: float | None = None

    # ---------------------------------------------------------------- policy

    def plan_for(self, sustained_high_s: float) -> str | None:
        if sustained_high_s >= PUSH_AFTER_S:
            return "push"
        if self._rng.random() < self.sample_rate:
            return "sample"
        return None

    def on_detection_end(self, detection_id: UUID, sustained_high_s: float) -> None:
        """sustained_high_s: longest run with confidence >= 0.8 during the detection."""
        plan = self.plan_for(sustained_high_s)
        if plan is not None and self.store.has(detection_id):
            self.store.set_upload_plan(detection_id, plan)
            log.info("traces for %s queued (%s)", str(detection_id)[:8], plan)

    # ---------------------------------------------------------------- transport

    def _upload(self, seg: SegmentInfo) -> None:
        try:
            r = self.client.post(
                "/v1/traces", data={"header": seg.header_path.read_text()},
                files={"body": ("trace.npz", seg.body_path.read_bytes(),
                                "application/octet-stream")},
            )
        except httpx.TransportError as exc:
            raise _Stop(str(exc)) from exc
        if r.status_code == 401 or r.status_code >= 500:
            raise _Stop(f"HTTP {r.status_code}")
        if r.status_code >= 400:  # this segment will never be accepted: do not retry it forever
            log.error("server rejected trace %s/%d: %s", seg.detection_id, seg.index, r.text[:200])
        self.store.mark_uploaded(seg.detection_id, seg.index)

    # ---------------------------------------------------------------- server pull

    def poll(self) -> None:
        now = self._monotonic()
        if self._last_poll is not None and now - self._last_poll < self.poll_every_s:
            return
        self._last_poll = now
        try:
            r = self.client.get("/v1/traces/requests", params={"node_id": self.node_id})
            if r.status_code != 200:
                return
            for request in (TraceRequest.model_validate(item) for item in r.json()):
                self._answer(request.detection_id)
        except (httpx.TransportError, _Stop) as exc:
            log.warning("trace poll stopped (%s); will retry", exc)

    def _answer(self, detection_id: UUID) -> None:
        if not self.store.has(detection_id) and self.store.recording != detection_id:
            msg = sign(TraceUnavailable(node_id=self.node_id, detection_id=detection_id,
                                        sent_at=self._now()), self._key)
            try:
                self.client.post("/v1/traces/unavailable", content=msg.model_dump_json(),
                                 headers={"content-type": "application/json"})
            except httpx.TransportError as exc:
                raise _Stop(str(exc)) from exc
            return
        self.store.mark_corroborated(detection_id)
        self.store.clear_plan(detection_id)  # a pull supersedes any push or sample
        for seg in self.store.segments(detection_id):
            if not seg.uploaded:
                self._upload(seg)

    # ---------------------------------------------------------------- push and sample

    def pump(self) -> None:
        day = self._now().date().isoformat()
        try:
            for detection_id, plan in self.store.planned():
                segments = self.store.segments(detection_id)
                if plan == "sample":
                    segments = [s for s in segments if s.index < SAMPLE_SEGMENTS]
                pending = [s for s in segments if not s.uploaded]
                for seg in pending:
                    if self.store.usage(day) + seg.size > self.upload_bytes_per_day:
                        return  # budget spent for today; the plan stays for tomorrow
                    self._upload(seg)
                    self.store.add_usage(day, seg.size)
                self.store.clear_plan(detection_id)
        except _Stop as exc:
            log.warning("trace upload stopped (%s); will retry", exc)

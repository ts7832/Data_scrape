"""HTTP uplink: register once, then deliver the durable outbox in order with backoff."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from kuulo_protocol.models import NodeRegistration

from .outbox import Outbox

log = logging.getLogger("kuulo.node")
OBSERVATIONS = "/v1/observations"


class RegistrationConflict(RuntimeError):
    pass


class Uplink:
    """Sends outbox messages in order: runs of observations as one batch, heartbeats singly."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        outbox: Outbox | None = None,
        max_pending: int = 10_000,
        max_batch: int = 100,
        monotonic: Callable[[], float] = time.monotonic,
        backoff_max_s: float = 30.0,
    ) -> None:
        self.client = client
        self.outbox = outbox if outbox is not None else Outbox(cap=max_pending)
        self.max_batch = max_batch
        self._monotonic = monotonic
        self._backoff_max = backoff_max_s
        self._backoff = 1.0
        self._next_try = 0.0
        self.registered = False
        self._rejected = 0

    @property
    def pending_count(self) -> int:
        return len(self.outbox)

    @property
    def dropped(self) -> int:
        return self.outbox.dropped + self._rejected

    def _due(self) -> bool:
        return self._monotonic() >= self._next_try

    def _fail(self, why: str) -> None:
        log.warning("server unreachable (%s); retrying in %.0f s", why, self._backoff)
        self._next_try = self._monotonic() + self._backoff
        self._backoff = min(self._backoff * 2, self._backoff_max)

    def _ok(self) -> None:
        self._backoff = 1.0
        self._next_try = 0.0

    def ensure_registered(self, reg: NodeRegistration) -> bool:
        if self.registered:
            return True
        if not self._due():
            return False
        try:
            r = self.client.post("/v1/nodes/register", json=reg.model_dump(mode="json"))
        except httpx.TransportError as exc:
            self._fail(str(exc))
            return False
        if r.status_code == 409:
            raise RegistrationConflict(
                f"The server already has node '{reg.node_id}' registered with a different key "
                "(was the key file deleted?). Choose a new node_id in your config, or delete "
                "the server's database (data/kuulo.db) if this is a local demo."
            )
        if r.status_code >= 500:
            self._fail(f"HTTP {r.status_code}")
            return False
        r.raise_for_status()
        self.registered = True
        self._ok()
        log.info("registered as %s", reg.node_id)
        return True

    def send(self, path: str, msg: BaseModel) -> None:
        self.outbox.put(path, msg.model_dump_json())

    def _reject(self, path: str, why: str) -> None:
        self._rejected += 1
        log.error("server rejected %s: %s", path, why[:200])

    def _next_run(self) -> list[tuple[int, str, str]]:
        rows = self.outbox.peek(self.max_batch)
        if not rows or rows[0][1] != OBSERVATIONS:
            return rows[:1]
        run = []
        for row in rows:
            if row[1] != OBSERVATIONS:
                break
            run.append(row)
        return run

    def _post(self, path: str, content: str) -> httpx.Response | None:
        try:
            r = self.client.post(path, content=content,
                                 headers={"content-type": "application/json"})
        except httpx.TransportError as exc:
            self._fail(str(exc))
            return None
        if r.status_code >= 500:
            self._fail(f"HTTP {r.status_code}")
            return None
        return r

    def flush(self, force: bool = False) -> None:
        if not self.registered or (not force and not self._due()):
            return
        while run := self._next_run():
            path = run[0][1]
            batch = path == OBSERVATIONS
            content = "[" + ",".join(body for _, _, body in run) + "]" if batch else run[0][2]
            r = self._post(path, content)
            if r is None:
                return
            if r.status_code == 401:  # server forgot us (fresh database): register again
                self.registered = False
                return
            if r.status_code >= 400:
                for _ in run:
                    self._reject(path, f"HTTP {r.status_code} {r.text}")
                self.outbox.delete([row_id for row_id, _, _ in run])
                continue
            if not batch:
                self.outbox.delete([run[0][0]])
                continue
            done = []
            for (row_id, _, _), result in zip(run, r.json(), strict=True):
                reason = str(result.get("reason", ""))
                if result.get("status") == "rejected" and reason.startswith("unknown node"):
                    self.outbox.delete(done)
                    self.registered = False
                    return
                if result.get("status") == "rejected":
                    self._reject(path, reason)
                done.append(row_id)
            self.outbox.delete(done)
        self._ok()

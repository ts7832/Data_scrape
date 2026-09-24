"""HTTP uplink: register once, then send messages in order with bounded retry and backoff."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from kuulo_protocol.models import NodeRegistration

log = logging.getLogger("kuulo.node")
HEARTBEATS = "/v1/heartbeats"


class RegistrationConflict(RuntimeError):
    pass


class Uplink:
    def __init__(
        self,
        client: httpx.Client,
        *,
        max_pending: int = 1000,
        monotonic: Callable[[], float] = time.monotonic,
        backoff_max_s: float = 30.0,
    ) -> None:
        self.client = client
        self.max_pending = max_pending
        self._monotonic = monotonic
        self._backoff_max = backoff_max_s
        self._backoff = 1.0
        self._next_try = 0.0
        self._pending: deque[tuple[str, BaseModel]] = deque()
        self.registered = False
        self.dropped = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

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
        if len(self._pending) >= self.max_pending:
            idx = next((i for i, (p, _) in enumerate(self._pending) if p == HEARTBEATS), 0)
            del self._pending[idx]
            self.dropped += 1
            log.warning("uplink queue full; dropped 1 message (%d dropped so far)", self.dropped)
        self._pending.append((path, msg))

    def flush(self, force: bool = False) -> None:
        if not self.registered or (not force and not self._due()):
            return
        while self._pending:
            path, msg = self._pending[0]
            try:
                r = self.client.post(
                    path, content=msg.model_dump_json(),
                    headers={"content-type": "application/json"},
                )
            except httpx.TransportError as exc:
                self._fail(str(exc))
                return
            if r.status_code >= 500:
                self._fail(f"HTTP {r.status_code}")
                return
            if r.status_code == 401:  # server forgot us (fresh database): register again
                self.registered = False
                return
            self._pending.popleft()
            if r.status_code >= 400:
                self.dropped += 1
                log.error("server rejected %s: HTTP %d %s", path, r.status_code, r.text[:200])
        self._ok()

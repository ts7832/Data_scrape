"""Fan-out of live events to WebSocket clients. A client that falls behind gets a resync."""

from __future__ import annotations

import asyncio

from kuulo_protocol.api import LiveEvent

RESYNC = LiveEvent(type="resync").model_dump_json()


class LiveHub:
    def __init__(self, max_queue: int = 1000):
        self.max_queue = max_queue
        self._clients: set[asyncio.Queue[str]] = set()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_queue)
        self._clients.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._clients.discard(queue)

    def publish(self, event: LiveEvent) -> None:
        payload = event.model_dump_json()
        for queue in list(self._clients):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(RESYNC)

"""Durable, ordered outbox: messages are written to SQLite first and deleted only once delivered.

A crash, reboot or long outage loses nothing. When the cap is reached the oldest heartbeat is
dropped first (a newer heartbeat says the same thing); observations are dropped only when no
heartbeat is left, and every drop is counted.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger("kuulo.node")
HEARTBEATS = "/v1/heartbeats"


class Outbox:
    def __init__(self, path: Path | str = ":memory:", cap: int = 10_000) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.cap = cap
        self.dropped = 0
        self._db = sqlite3.connect(str(path), isolation_level=None)  # autocommit
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS outbox ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, body TEXT NOT NULL)"
        )

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

    def put(self, path: str, body: str) -> None:
        with self._db:
            while len(self) >= self.cap:
                row = self._db.execute(
                    "SELECT id FROM outbox WHERE path = ? ORDER BY id LIMIT 1", (HEARTBEATS,)
                ).fetchone() or self._db.execute(
                    "SELECT id FROM outbox ORDER BY id LIMIT 1"
                ).fetchone()
                self._db.execute("DELETE FROM outbox WHERE id = ?", row)
                self.dropped += 1
                log.warning("outbox full; dropped 1 message (%d dropped so far)", self.dropped)
            self._db.execute("INSERT INTO outbox (path, body) VALUES (?, ?)", (path, body))

    def peek(self, limit: int) -> list[tuple[int, str, str]]:
        return self._db.execute(
            "SELECT id, path, body FROM outbox ORDER BY id LIMIT ?", (limit,)
        ).fetchall()

    def delete(self, ids: list[int]) -> None:
        with self._db:
            self._db.executemany("DELETE FROM outbox WHERE id = ?", [(i,) for i in ids])

    def close(self) -> None:
        self._db.close()

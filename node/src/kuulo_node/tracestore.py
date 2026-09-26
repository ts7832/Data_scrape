"""FeatureTrace recorder and local store (spec §5.1).

While a detection is active the node records 20 ms log-mel frames (plus a short pre-roll from
before the detection fired) into 60 s segments on disk. A disk budget is enforced by evicting
whole detections, least valuable first:

1. uncorroborated detections older than 7 days
2. any uncorroborated detection, oldest first
3. corroborated detections already uploaded, oldest first

Never evicted: the detection being recorded, and a corroborated (server-requested) detection
that has not been fully uploaded yet.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sqlite3
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import numpy as np

from kuulo_protocol.features import (
    BAND_EDGES_HZ,
    FRAME_PERIOD_MS,
    FRAME_SAMPLES,
    N_BANDS,
    SAMPLE_RATE,
    extract_frames,
)
from kuulo_protocol.models import TimeQuality
from kuulo_protocol.signing import sign
from kuulo_protocol.traces import SEGMENT_FRAMES, FeatureTraceHeader, encode_body

log = logging.getLogger("kuulo.node")
FRAME_S = FRAME_PERIOD_MS / 1000
OLD_AFTER = timedelta(days=7)


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SegmentInfo:
    detection_id: UUID
    index: int
    final: bool
    size: int
    uploaded: bool
    header_path: Path
    body_path: Path


class _Segment:
    """Frames of the segment being recorded, held in memory until it closes."""

    def __init__(self, detection_id: UUID, index: int, anchor: datetime) -> None:
        self.detection_id = detection_id
        self.index = index
        self.anchor = anchor  # wall-clock time of audio time 0 for this detection
        self.walls: list[float] = []  # audio times, seconds
        self.bands: list[np.ndarray] = []
        self.rms: list[float] = []
        self.peak: list[float] = []

    def add(self, wall: float, band: np.ndarray, rms: float, peak: float) -> None:
        self.walls.append(wall)
        self.bands.append(band)
        self.rms.append(rms)
        self.peak.append(peak)

    def __len__(self) -> int:
        return len(self.walls)


class TraceStore:
    def __init__(
        self,
        root: Path,
        node_id: str,
        private_key: str,
        time_quality: TimeQuality | str,
        *,
        budget_bytes: int = 500 * 2**20,
        now: Callable[[], datetime] = utc_now,
        pre_roll_s: float = 3.0,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.node_id = node_id
        self._key = private_key
        self._time_quality = TimeQuality(time_quality)
        self.budget_bytes = budget_bytes
        self._now = now
        self._pre_roll: deque[tuple[float, np.ndarray, float, float]] = deque(
            maxlen=round(pre_roll_s / FRAME_S)
        )
        self._leftover = np.zeros(0, np.float32)
        self._leftover_t = 0.0
        self._anchor: datetime | None = None
        self._open: _Segment | None = None
        self._db = sqlite3.connect(str(self.root / "index.db"), isolation_level=None)
        self._db.executescript(
            "PRAGMA journal_mode=WAL;"
            "CREATE TABLE IF NOT EXISTS detections (detection_id TEXT PRIMARY KEY,"
            " created_at TEXT NOT NULL, corroborated INTEGER NOT NULL DEFAULT 0,"
            " plan TEXT);"
            "CREATE TABLE IF NOT EXISTS segments (detection_id TEXT NOT NULL,"
            " idx INTEGER NOT NULL, final INTEGER NOT NULL, size INTEGER NOT NULL,"
            " uploaded INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (detection_id, idx));"
            "CREATE TABLE IF NOT EXISTS upload_usage (day TEXT PRIMARY KEY, bytes INTEGER);"
        )

    # ---------------------------------------------------------------- recording

    @property
    def recording(self) -> UUID | None:
        return self._open.detection_id if self._open is not None else None

    def feed(self, samples: np.ndarray, t_audio: float, detection_id: UUID | None) -> None:
        """Add 16 kHz mono audio (first sample at audio time t_audio) under a detection.

        Frames are timed by audio time, never by when a block happens to be processed: a mic
        queue draining after a slow upload hands over many blocks at one instant. The wall-clock
        anchor (wall time of audio time 0) follows the clock while idle and is frozen for the
        length of a detection, so a segment's frame offsets always increase.
        """
        samples = np.asarray(samples, np.float32)
        expected = self._leftover_t + self._leftover.size / SAMPLE_RATE
        if self._leftover.size == 0 or abs(t_audio - expected) > FRAME_S:
            self._leftover, self._leftover_t = np.zeros(0, np.float32), t_audio  # gap/restart
        x = np.concatenate([self._leftover, samples])
        start_t = self._leftover_t
        block = extract_frames(x)
        used = block.frames * FRAME_SAMPLES
        self._leftover, self._leftover_t = x[used:], start_t + used / SAMPLE_RATE
        if self._open is None or self._anchor is None:
            audio_end = t_audio + samples.size / SAMPLE_RATE
            self._anchor = self._now() - timedelta(seconds=audio_end)

        if self._open is not None and self._open.detection_id != detection_id:
            self._close(final=True)
        for i in range(block.frames):
            t = start_t + i * FRAME_S
            frame = (t, block.band_db[i], float(block.rms_db[i]), float(block.peak_freq_hz[i]))
            if detection_id is None:
                self._pre_roll.append(frame)
                continue
            if self._open is None:
                self._start(detection_id)
            self._open.add(*frame)
            if len(self._open) >= SEGMENT_FRAMES:
                anchor = self._open.anchor
                self._close(final=False)
                self._open = _Segment(detection_id, self._next_index(detection_id), anchor)
        if detection_id is not None and self._open is None:
            self._start(detection_id)  # the detection began on a block with no complete frame

    def _start(self, detection_id: UUID) -> None:
        self._open = _Segment(detection_id, self._next_index(detection_id), self._anchor)
        for frame in self._pre_roll:
            self._open.add(*frame)
        self._pre_roll.clear()
        self._db.execute(
            "INSERT OR IGNORE INTO detections (detection_id, created_at) VALUES (?, ?)",
            (str(detection_id), self._now().isoformat()),
        )

    def _next_index(self, detection_id: UUID) -> int:
        row = self._db.execute(
            "SELECT MAX(idx) FROM segments WHERE detection_id = ?", (str(detection_id),)
        ).fetchone()
        return 0 if row[0] is None else row[0] + 1

    def _close(self, final: bool) -> None:
        seg, self._open = self._open, None
        if seg is None:
            return
        if len(seg) == 0:
            if final:  # mark the previous segment final instead
                self._db.execute(
                    "UPDATE segments SET final = 1 WHERE detection_id = ? AND idx = ?",
                    (str(seg.detection_id), seg.index - 1),
                )
                self._rewrite_final(seg.detection_id, seg.index - 1)
            return
        start = seg.anchor + timedelta(seconds=seg.walls[0])
        offsets = np.round((np.array(seg.walls) - seg.walls[0]) * 1000).astype(np.int64)
        body = encode_body(
            t_offset_ms=offsets, band_db=np.stack(seg.bands).reshape(-1, N_BANDS),
            rms_db=np.array(seg.rms), peak_freq_hz=np.array(seg.peak),
        )
        self._write(seg.detection_id, seg.index, final, start, len(seg), body)
        self.evict_to_budget()

    def _write(self, detection_id, index, final, start, frames, body: bytes) -> None:
        header = sign(FeatureTraceHeader(
            node_id=self.node_id, detection_id=detection_id, segment_index=index, final=final,
            time_quality=self._time_quality, start_at=start, frame_count=frames,
            band_edges_hz=list(BAND_EDGES_HZ), body_sha256=hashlib.sha256(body).hexdigest(),
        ), self._key)
        folder = self.root / str(detection_id)
        folder.mkdir(exist_ok=True)
        (folder / f"{index:05d}.npz").write_bytes(body)
        (folder / f"{index:05d}.json").write_text(header.model_dump_json())
        self._db.execute(
            "INSERT OR REPLACE INTO segments (detection_id, idx, final, size) VALUES (?, ?, ?, ?)",
            (str(detection_id), index, int(final), len(body)),
        )

    def _rewrite_final(self, detection_id: UUID, index: int) -> None:
        """The detection ended exactly on a segment boundary: re-sign its last segment as final."""
        folder = self.root / str(detection_id)
        header_path = folder / f"{index:05d}.json"
        if index < 0 or not header_path.exists():
            return
        header = FeatureTraceHeader.model_validate_json(header_path.read_text())
        header_path.write_text(sign(header.model_copy(update={"final": True}), self._key)
                               .model_dump_json())

    # ---------------------------------------------------------------- queries

    def has(self, detection_id: UUID) -> bool:
        return self._db.execute(
            "SELECT 1 FROM segments WHERE detection_id = ? LIMIT 1", (str(detection_id),)
        ).fetchone() is not None

    def segments(self, detection_id: UUID) -> list[SegmentInfo]:
        folder = self.root / str(detection_id)
        rows = self._db.execute(
            "SELECT idx, final, size, uploaded FROM segments WHERE detection_id = ? ORDER BY idx",
            (str(detection_id),),
        )
        return [
            SegmentInfo(detection_id, idx, bool(final), size, bool(uploaded),
                        folder / f"{idx:05d}.json", folder / f"{idx:05d}.npz")
            for idx, final, size, uploaded in rows
        ]

    def total_bytes(self) -> int:
        return self._db.execute("SELECT COALESCE(SUM(size), 0) FROM segments").fetchone()[0]

    def mark_corroborated(self, detection_id: UUID) -> None:
        self._db.execute("UPDATE detections SET corroborated = 1 WHERE detection_id = ?",
                         (str(detection_id),))

    def mark_uploaded(self, detection_id: UUID, index: int) -> None:
        self._db.execute("UPDATE segments SET uploaded = 1 WHERE detection_id = ? AND idx = ?",
                         (str(detection_id), index))

    def set_upload_plan(self, detection_id: UUID, plan: str) -> None:
        self._db.execute("UPDATE detections SET plan = ? WHERE detection_id = ?",
                         (plan, str(detection_id)))

    def clear_plan(self, detection_id: UUID) -> None:
        self._db.execute("UPDATE detections SET plan = NULL WHERE detection_id = ?",
                         (str(detection_id),))

    def planned(self) -> list[tuple[UUID, str]]:
        rows = self._db.execute(
            "SELECT detection_id, plan FROM detections WHERE plan IS NOT NULL ORDER BY created_at"
        )
        return [(UUID(d), plan) for d, plan in rows]

    def usage(self, day: str) -> int:
        row = self._db.execute("SELECT bytes FROM upload_usage WHERE day = ?", (day,)).fetchone()
        return row[0] if row else 0

    def add_usage(self, day: str, size: int) -> None:
        self._db.execute(
            "INSERT INTO upload_usage (day, bytes) VALUES (?, ?)"
            " ON CONFLICT(day) DO UPDATE SET bytes = bytes + excluded.bytes", (day, size),
        )

    # ---------------------------------------------------------------- eviction

    def evict_to_budget(self) -> int:
        """Delete whole detections until under budget. Returns how many were evicted."""
        evicted = 0
        while self.total_bytes() > self.budget_bytes:
            victim = self._next_victim()
            if victim is None:
                log.warning("trace store over budget but nothing is evictable")
                break
            self._delete(victim)
            evicted += 1
        return evicted

    def _next_victim(self) -> str | None:
        recording = str(self.recording) if self.recording else ""
        cutoff = (self._now() - OLD_AFTER).isoformat()
        not_fully_uploaded = (
            "EXISTS (SELECT 1 FROM segments s WHERE s.detection_id = d.detection_id"
            " AND s.uploaded = 0)"
        )
        tiers: list[tuple[str, tuple]] = [
            ("d.corroborated = 0 AND d.created_at < ?", (cutoff,)),
            ("d.corroborated = 0", ()),
            (f"d.corroborated = 1 AND NOT {not_fully_uploaded}", ()),
        ]
        for where, params in tiers:
            row = self._db.execute(
                "SELECT d.detection_id FROM detections d"
                f" WHERE d.detection_id != ? AND {where}"
                " AND EXISTS (SELECT 1 FROM segments s WHERE s.detection_id = d.detection_id)"
                " ORDER BY d.created_at LIMIT 1",
                (recording, *params),
            ).fetchone()
            if row:
                return row[0]
        return None

    def _delete(self, detection_id: str) -> None:
        shutil.rmtree(self.root / detection_id, ignore_errors=True)
        with self._db:
            self._db.execute("DELETE FROM segments WHERE detection_id = ?", (detection_id,))
            self._db.execute("DELETE FROM detections WHERE detection_id = ?", (detection_id,))

    def close(self) -> None:
        self._close(final=True)
        self._db.close()

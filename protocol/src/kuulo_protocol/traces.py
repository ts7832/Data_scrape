"""FeatureTrace wire format: a signed JSON header plus a compressed .npz body.

The header carries the body's SHA-256, so the header's one Ed25519 signature covers both.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
from pydantic import Field, field_validator

from .features import FRAME_PERIOD_MS, N_BANDS
from .models import SCHEMA_VERSION, NodeId, TimeQuality, WireModel, to_utc_ms

MAX_FRAMES = 3000  # 60 s at 20 ms
SEGMENT_FRAMES = MAX_FRAMES
# Uncompressed size of the largest legitimate member (band_db, float64 worst case) plus headroom.
_MAX_MEMBER_BYTES = MAX_FRAMES * N_BANDS * 8 + 4096
BODY_KEYS = ("t_offset_ms", "band_db", "rms_db", "peak_freq_hz")


class FeatureTraceHeader(WireModel):
    """One ≤60 s segment of one detection's features. `final` marks the detection's last segment."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    trace_id: UUID = Field(default_factory=uuid4)
    node_id: NodeId
    detection_id: UUID
    segment_index: int = Field(ge=0, le=100_000)
    final: bool
    time_quality: TimeQuality
    frame_period_ms: Literal[20] = FRAME_PERIOD_MS
    start_at: datetime
    frame_count: int = Field(ge=1, le=MAX_FRAMES)
    band_edges_hz: list[float] = Field(min_length=N_BANDS + 1, max_length=N_BANDS + 1)
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature: str = ""

    @field_validator("start_at")
    @classmethod
    def _start_at_utc(cls, value: datetime) -> datetime:
        return to_utc_ms(value)


class TraceRequest(WireModel):
    """The server wants every segment of this detection (it contributed to a confirmed track)."""

    request_id: UUID
    node_id: str
    detection_id: UUID
    created_at: datetime


class TraceUnavailable(WireModel):
    """A node's signed answer that it no longer holds a requested detection's segments."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    node_id: NodeId
    detection_id: UUID
    sent_at: datetime
    signature: str = ""

    @field_validator("sent_at")
    @classmethod
    def _sent_at_utc(cls, value: datetime) -> datetime:
        return to_utc_ms(value)


def encode_body(
    t_offset_ms: np.ndarray, band_db: np.ndarray, rms_db: np.ndarray, peak_freq_hz: np.ndarray
) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        t_offset_ms=np.asarray(t_offset_ms, dtype=np.int32),
        band_db=np.asarray(band_db, dtype=np.float32),
        rms_db=np.asarray(rms_db, dtype=np.float32),
        peak_freq_hz=np.asarray(peak_freq_hz, dtype=np.float32),
    )
    return buf.getvalue()


def decode_body(data: bytes, frame_count: int) -> dict[str, np.ndarray]:
    """Parse and validate an untrusted body. Never unpickles; raises ValueError on anything odd."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = {i.filename: i for i in zf.infolist()}
            if set(infos) != {f"{k}.npy" for k in BODY_KEYS}:
                raise ValueError(f"body must contain exactly {BODY_KEYS}")
            if any(i.file_size > _MAX_MEMBER_BYTES for i in infos.values()):
                raise ValueError("body member too large")
            arrays = {
                k: np.load(io.BytesIO(zf.read(f"{k}.npy")), allow_pickle=False) for k in BODY_KEYS
            }
    except (zipfile.BadZipFile, OSError, EOFError, KeyError) as exc:
        raise ValueError(f"unreadable body: {exc}") from exc
    for key, value in arrays.items():
        if value.dtype.kind not in "iuf":
            raise ValueError(f"{key} must be numeric")
        want = (frame_count, N_BANDS) if key == "band_db" else (frame_count,)
        if value.shape != want:
            raise ValueError(f"{key} has shape {value.shape}, expected {want}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains non-finite values")
    if np.any(np.diff(arrays["t_offset_ms"]) < 0):
        raise ValueError("t_offset_ms must not decrease")
    return arrays

import hashlib
import io
import zipfile
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from kuulo_protocol.features import (
    BAND_EDGES_HZ,
    FRAME_SAMPLES,
    N_BANDS,
    SAMPLE_RATE,
    extract_frames,
)
from kuulo_protocol.models import TimeQuality
from kuulo_protocol.signing import generate_keypair, sign, verify
from kuulo_protocol.traces import FeatureTraceHeader, decode_body, encode_body


def tone(freq: float, seconds: float = 1.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_band_edges_are_mel_spaced_50_to_8000():
    assert len(BAND_EDGES_HZ) == N_BANDS + 1 == 33
    assert BAND_EDGES_HZ[0] == pytest.approx(50.0)
    assert BAND_EDGES_HZ[-1] == pytest.approx(8000.0)
    widths = np.diff(BAND_EDGES_HZ)
    assert np.all(widths > 0)
    assert widths[-1] > widths[0] * 10  # mel: much wider at the top


def test_one_second_is_fifty_frames():
    block = extract_frames(tone(1000))
    assert block.band_db.shape == (50, N_BANDS)
    assert block.rms_db.shape == block.peak_freq_hz.shape == (50,)
    assert FRAME_SAMPLES == 320


def test_partial_frame_is_not_emitted():
    assert extract_frames(np.zeros(FRAME_SAMPLES * 2 + 100, np.float32)).band_db.shape[0] == 2


def test_tone_lands_in_its_band_with_correct_peak():
    block = extract_frames(tone(1000))
    band = int(np.searchsorted(BAND_EDGES_HZ, 1000.0) - 1)
    assert np.all(np.argmax(block.band_db, axis=1) == band)
    assert np.all(np.abs(block.peak_freq_hz - 1000) <= 16)


def test_rms_db_of_full_scale_sine():
    block = extract_frames(tone(1000, amp=1.0))
    assert np.allclose(block.rms_db, 20 * np.log10(1 / np.sqrt(2)), atol=0.1)


def test_silence_is_finite():
    block = extract_frames(np.zeros(SAMPLE_RATE, np.float32))
    assert np.all(np.isfinite(block.band_db)) and np.all(np.isfinite(block.rms_db))


def _body(frames: int = 10) -> tuple[bytes, dict]:
    arrays = {
        "t_offset_ms": np.arange(frames, dtype=np.int32) * 20,
        "band_db": np.full((frames, N_BANDS), -40.0, np.float32),
        "rms_db": np.full(frames, -30.0, np.float32),
        "peak_freq_hz": np.full(frames, 180.0, np.float32),
    }
    return encode_body(**arrays), arrays


def test_body_round_trip():
    data, arrays = _body()
    decoded = decode_body(data, frame_count=10)
    for key, value in arrays.items():
        np.testing.assert_array_equal(decoded[key], value)


def test_decode_rejects_truncated_body():
    data, _ = _body()
    with pytest.raises(ValueError):
        decode_body(data[: len(data) // 2], frame_count=10)


def test_decode_rejects_frame_count_mismatch():
    data, _ = _body()
    with pytest.raises(ValueError):
        decode_body(data, frame_count=11)


def test_decode_rejects_wrong_band_count():
    buf = io.BytesIO()
    np.savez_compressed(buf, t_offset_ms=np.zeros(3, np.int32),
                        band_db=np.zeros((3, 31), np.float32),
                        rms_db=np.zeros(3, np.float32), peak_freq_hz=np.zeros(3, np.float32))
    with pytest.raises(ValueError):
        decode_body(buf.getvalue(), frame_count=3)


def test_decode_never_unpickles_object_arrays():
    buf = io.BytesIO()
    evil = np.empty(3, dtype=object)
    np.savez(buf, t_offset_ms=evil, band_db=np.zeros((3, 32)), rms_db=np.zeros(3),
             peak_freq_hz=np.zeros(3))
    with pytest.raises(ValueError):
        decode_body(buf.getvalue(), frame_count=3)


def test_decode_rejects_oversized_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("band_db.npy", b"\0" * (10 * 1024 * 1024))  # 10 MB of zeros compresses tiny
    with pytest.raises(ValueError):
        decode_body(buf.getvalue(), frame_count=3)


def test_decode_rejects_non_finite_values():
    data, arrays = _body()
    arrays["rms_db"][0] = np.nan
    with pytest.raises(ValueError):
        decode_body(encode_body(**arrays), frame_count=10)


def test_header_signature_covers_body_hash():
    data, _ = _body()
    priv, pub = generate_keypair()
    header = FeatureTraceHeader(
        trace_id=uuid4(), node_id="n1", detection_id=uuid4(), segment_index=0, final=True,
        time_quality=TimeQuality.NTP, start_at=datetime(2026, 9, 26, tzinfo=UTC), frame_count=10,
        band_edges_hz=list(BAND_EDGES_HZ), body_sha256=hashlib.sha256(data).hexdigest(),
    )
    signed = sign(header, priv)
    assert verify(signed, pub)
    tampered = signed.model_copy(update={"body_sha256": "0" * 64})
    assert not verify(tampered, pub)


def test_header_rejects_bad_hash_and_frame_count():
    with pytest.raises(ValueError):
        FeatureTraceHeader(
            trace_id=uuid4(), node_id="n1", detection_id=uuid4(), segment_index=0, final=False,
            time_quality=TimeQuality.NTP, start_at=datetime(2026, 9, 26, tzinfo=UTC),
            frame_count=0, band_edges_hz=list(BAND_EDGES_HZ), body_sha256="xyz",
        )

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np

from kuulo_node.tracestore import TraceStore
from kuulo_protocol.features import SAMPLE_RATE
from kuulo_protocol.signing import generate_keypair, verify
from kuulo_protocol.traces import FeatureTraceHeader, decode_body

T0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
BLOCK_S = 0.5


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


def make_store(tmp_path, budget=500 * 2**20, clock=None):
    priv, pub = generate_keypair()
    store = TraceStore(tmp_path / "traces", "n1", priv, "ntp", budget_bytes=budget,
                       now=clock or Clock())
    return store, pub


def play(store, seconds, detection_id, *, start_t=0.0, clock=None, noise=True):
    """Feed `seconds` of audio in 0.5 s blocks; returns the audio time reached."""
    rng = np.random.default_rng(0)
    n = int(BLOCK_S * SAMPLE_RATE)
    t = start_t
    for _ in range(int(round(seconds / BLOCK_S))):
        x = (rng.standard_normal(n) * 0.05).astype(np.float32) if noise else np.zeros(n, np.float32)
        if clock is not None:
            clock.now += timedelta(seconds=BLOCK_S)
        store.feed(x, t, detection_id)
        t += BLOCK_S
    return t


def test_nothing_recorded_without_a_detection(tmp_path):
    store, _ = make_store(tmp_path)
    play(store, 10, None)
    assert store.total_bytes() == 0


def test_long_detection_splits_into_60_s_segments_last_is_final(tmp_path):
    clock = Clock()
    store, pub = make_store(tmp_path, clock=clock)
    d = uuid4()
    t = play(store, 130, d, clock=clock)
    play(store, 1, None, start_t=t, clock=clock)
    segs = store.segments(d)
    assert [s.index for s in segs] == [0, 1, 2]
    assert [s.final for s in segs] == [False, False, True]
    for seg in segs:
        header = FeatureTraceHeader.model_validate_json(seg.header_path.read_text())
        assert verify(header, pub)
        body = decode_body(seg.body_path.read_bytes(), header.frame_count)
        assert body["band_db"].shape == (header.frame_count, 32)
    first = FeatureTraceHeader.model_validate_json(segs[0].header_path.read_text())
    assert first.frame_count == 3000


def test_pre_roll_is_included_in_first_segment(tmp_path):
    clock = Clock()
    store, _ = make_store(tmp_path, clock=clock)
    t = play(store, 5, None, clock=clock)
    d = uuid4()
    t = play(store, 2, d, start_t=t, clock=clock)
    play(store, 1, None, start_t=t, clock=clock)
    [seg] = store.segments(d)
    header = json.loads(seg.header_path.read_text())
    assert header["frame_count"] == 250  # 3 s pre-roll + 2 s detection, 50 frames/s
    start = datetime.fromisoformat(header["start_at"])
    assert abs((start - (T0 + timedelta(seconds=2))).total_seconds()) < 0.1


def test_switching_detection_closes_the_old_one_as_final(tmp_path):
    store, _ = make_store(tmp_path)
    a, b = uuid4(), uuid4()
    t = play(store, 2, a)
    play(store, 2, b, start_t=t)
    assert [s.final for s in store.segments(a)] == [True]
    assert store.recording == b


def _detection(store, clock, seconds=2):
    d = uuid4()
    play(store, seconds, d, clock=clock)
    play(store, 0.5, None, clock=clock)
    return d


def test_eviction_order_old_uncorroborated_then_uncorroborated_then_uploaded(tmp_path):
    clock = Clock()
    store, _ = make_store(tmp_path, clock=clock)
    old = _detection(store, clock)
    clock.now += timedelta(days=8)
    corroborated_uploaded = _detection(store, clock)
    store.mark_corroborated(corroborated_uploaded)
    for seg in store.segments(corroborated_uploaded):
        store.mark_uploaded(corroborated_uploaded, seg.index)
    fresh = _detection(store, clock)
    size = store.total_bytes() // 3
    store.budget_bytes = 2 * size + size // 2
    assert store.evict_to_budget() == 1
    assert not store.has(old) and store.has(fresh) and store.has(corroborated_uploaded)
    store.budget_bytes = size + size // 2
    store.evict_to_budget()
    assert not store.has(fresh) and store.has(corroborated_uploaded)
    store.budget_bytes = size // 2
    store.evict_to_budget()
    assert not store.has(corroborated_uploaded)
    assert not (tmp_path / "traces" / str(corroborated_uploaded)).exists()


def test_eviction_never_touches_requested_not_uploaded_or_recording(tmp_path):
    clock = Clock()
    store, _ = make_store(tmp_path, clock=clock)
    requested = _detection(store, clock)
    store.mark_corroborated(requested)
    live = uuid4()
    play(store, 3, live, clock=clock)  # closed segment? no: still recording, nothing on disk yet
    long_live = uuid4()
    play(store, 61, long_live, clock=clock)  # one closed 60 s segment of a live detection
    store.budget_bytes = 1
    store.evict_to_budget()
    assert store.has(requested) and store.has(long_live)


def test_budget_is_enforced_as_segments_are_written(tmp_path):
    clock = Clock()
    store, _ = make_store(tmp_path, clock=clock)
    first = _detection(store, clock)
    one = store.total_bytes()
    store.budget_bytes = int(one * 1.5)
    _detection(store, clock)
    assert not store.has(first) and store.total_bytes() <= store.budget_bytes


def test_upload_plan_persists_across_reopen(tmp_path):
    clock = Clock()
    priv, _ = generate_keypair()
    store = TraceStore(tmp_path / "t", "n1", priv, "ntp", now=clock)
    d = _detection(store, clock)
    store.set_upload_plan(d, "push")
    again = TraceStore(tmp_path / "t", "n1", priv, "ntp", now=clock)
    assert again.planned() == [(d, "push")]
    again.clear_plan(d)
    assert again.planned() == []


def test_detection_ending_exactly_on_a_segment_boundary_marks_it_final(tmp_path):
    clock = Clock()
    store, pub = make_store(tmp_path, clock=clock)
    d = uuid4()
    t = play(store, 60, d, clock=clock)
    play(store, 1, None, start_t=t, clock=clock)
    [seg] = store.segments(d)
    header = FeatureTraceHeader.model_validate_json(seg.header_path.read_text())
    assert seg.final and header.final and verify(header, pub)

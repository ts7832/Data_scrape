"""Trace upload policy against the real server, in-process (spec §5.1 upload policy)."""

import random
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from kuulo_node.tracestore import TraceStore
from kuulo_node.traceupload import TraceUploader
from kuulo_protocol.features import SAMPLE_RATE
from kuulo_protocol.testing import make_observation
from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.db import TraceRow
from kuulo_server.testing import T0, FakeClock, post_signed, register


class Mono:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    clock = FakeClock(T0)
    app = create_app(Settings(db_path=tmp_path / "k.db", clock=clock, tick_interval_s=None))
    with TestClient(app) as client:
        me, other = register(client, "me"), register(client, "other")
        mono = Mono()
        store = TraceStore(tmp_path / "traces", "me", me.private_key, "ntp", now=clock)

        def uploader(**kw):
            kw.setdefault("rng", random.Random(1))
            return TraceUploader(client, store, node_id="me", private_key=me.private_key,
                                 now=clock, monotonic=mono, **kw)

        yield {"client": client, "app": app, "me": me, "other": other, "store": store,
               "clock": clock, "mono": mono, "uploader": uploader, "audio_t": 0.0}


def record(env, seconds, detection_id=None, *, finish=True):
    d = detection_id or uuid4()
    n = SAMPLE_RATE // 2
    rng = np.random.default_rng(0)
    for _ in range(int(seconds * 2)):
        env["clock"].advance(0.5)
        env["store"].feed((rng.standard_normal(n) * 0.05).astype(np.float32), env["audio_t"], d)
        env["audio_t"] += 0.5
    if finish:
        env["clock"].advance(0.5)
        env["store"].feed(np.zeros(n, np.float32), env["audio_t"], None)
        env["audio_t"] += 0.5
    return d


def confirm(env, detection_id):
    """Our detection plus another node's make a confirmed track: the server requests traces."""
    at = env["clock"]()
    for keys, det in ((env["me"], detection_id), (env["other"], uuid4())):
        obs = make_observation(keys.node_id, observed_at=at, detection_id=det)
        response = post_signed(env["client"], "/v1/observations", obs, keys.private_key)
        assert response.status_code == 200


def open_requests(env):
    return env["client"].get("/v1/traces/requests", params={"node_id": "me"}).json()


def stored_segments(env):
    with env["app"].state.sessions() as session:
        return sorted(r.segment_index for r in session.scalars(select(TraceRow)))


def test_server_pull_uploads_every_segment_and_closes_the_request(env):
    d = record(env, 70)
    confirm(env, d)
    env["uploader"]().poll()
    assert stored_segments(env) == [0, 1]
    assert open_requests(env) == []
    assert all(s.uploaded for s in env["store"].segments(d))


def test_requested_but_evicted_detection_is_reported_unavailable(env):
    d = uuid4()  # never recorded (or already evicted)
    confirm(env, d)
    env["uploader"]().poll()
    assert open_requests(env) == []
    assert stored_segments(env) == []


def test_request_for_a_detection_still_recording_stays_open_until_final(env):
    d = record(env, 65, finish=False)
    confirm(env, d)
    up = env["uploader"]()
    up.poll()
    assert stored_segments(env) == [0] and len(open_requests(env)) == 1
    record(env, 5, d)
    env["mono"].t += 11
    up.poll()
    assert stored_segments(env) == [0, 1] and open_requests(env) == []


def test_polls_at_most_every_poll_interval(env):
    d = record(env, 5)
    up = env["uploader"](poll_every_s=10)
    up.poll()
    confirm(env, d)
    env["mono"].t += 5
    up.poll()
    assert stored_segments(env) == []
    env["mono"].t += 5
    up.poll()
    assert stored_segments(env) == [0]


def test_sustained_confident_detection_is_auto_pushed(env):
    d = record(env, 30)
    up = env["uploader"](sample_rate=0.0)
    up.on_detection_end(d, sustained_high_s=25.0)
    up.pump()
    assert stored_segments(env) == [0]


def test_short_or_unconfident_detection_is_not_pushed(env):
    d = record(env, 30)
    up = env["uploader"](sample_rate=0.0)
    up.on_detection_end(d, sustained_high_s=19.0)
    up.pump()
    assert stored_segments(env) == []


def test_hard_negative_sample_is_capped_at_three_minutes(env):
    d = record(env, 290)  # five segments
    up = env["uploader"](sample_rate=1.0)
    up.on_detection_end(d, sustained_high_s=0.0)
    up.pump()
    assert stored_segments(env) == [0, 1, 2]


def test_sampling_rate_is_about_one_percent(env):
    up = env["uploader"](sample_rate=0.01, rng=random.Random(7))
    picks = sum(up.plan_for(sustained_high_s=0.0) == "sample" for _ in range(20_000))
    assert 150 <= picks <= 250


def test_daily_budget_limits_pushes_but_not_pulls(env):
    d = record(env, 130)  # three segments
    size = env["store"].segments(d)[0].size
    up = env["uploader"](upload_bytes_per_day=size + 10, sample_rate=0.0)
    up.on_detection_end(d, sustained_high_s=100.0)
    up.pump()
    assert stored_segments(env) == [0]
    pulled = record(env, 70)
    confirm(env, pulled)
    env["mono"].t += 11
    up.poll()
    assert stored_segments(env) == [0, 0, 1]  # the pull ignores the spent budget
    env["clock"].advance(24 * 3600)
    up.pump()
    assert stored_segments(env) == [0, 0, 1, 1]  # one segment's worth per day
    env["clock"].advance(24 * 3600)
    up.pump()
    assert stored_segments(env) == [0, 0, 1, 1, 2]


def test_budget_survives_restart(env):
    d = record(env, 130)
    size = env["store"].segments(d)[0].size
    up = env["uploader"](upload_bytes_per_day=size + 10, sample_rate=0.0)
    up.on_detection_end(d, sustained_high_s=100.0)
    up.pump()
    again = env["uploader"](upload_bytes_per_day=size + 10, sample_rate=0.0)
    again.pump()
    assert stored_segments(env) == [0]


def test_server_down_keeps_plans_for_later(env, monkeypatch):
    import httpx

    d = record(env, 30)
    up = env["uploader"](sample_rate=0.0)
    up.on_detection_end(d, sustained_high_s=30.0)
    real_post = env["client"].post
    monkeypatch.setattr(env["client"], "post",
                        lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("down")))
    up.pump()
    monkeypatch.setattr(env["client"], "post", real_post)
    up.pump()
    assert stored_segments(env) == [0]

import logging

import numpy as np

from kuulo_node.audio import SAMPLE_RATE, AudioBlock
from kuulo_node.classify import FakeClassifier
from kuulo_node.keys import load_or_create_keys
from kuulo_node.runner import NodeRunner, format_scores
from kuulo_node.testing import PROPELLER, make_test_config


class RecordingUplink:
    def __init__(self):
        self.sent = []
        self.registered_with = None

    def ensure_registered(self, reg):
        self.registered_with = reg
        return True

    def send(self, path, msg):
        self.sent.append((path, msg))

    def flush(self, force=False):
        pass

    pending_count = 0


def blocks(seconds, value=0.05):
    n = SAMPLE_RATE // 10
    for i in range(int(seconds * 10)):
        yield AudioBlock(np.full(n, value, np.float32), i / 10)


def runner(tmp_path, script, **kw):
    cfg = make_test_config(tmp_path)
    up = RecordingUplink()
    clock = iter(range(0, 10_000))
    r = NodeRunner(cfg, load_or_create_keys(cfg.key_file), FakeClassifier(script), up,
                   monotonic=lambda: float(next(clock)), **kw)
    return r, up


def test_detection_produces_start_and_end_and_heartbeats(tmp_path):
    r, up = runner(tmp_path, lambda i: {PROPELLER: 0.9 if 3 <= i < 10 else 0.0})
    stats = r.run(blocks(12))
    obs = [m for p, m in up.sent if p == "/v1/observations"]
    assert [o.event.phase.value for o in obs][0] == "start"
    assert obs[-1].event.phase.value == "end"
    assert stats.observations == len(obs) and stats.windows > 20
    assert stats.heartbeats >= 1 and up.registered_with.node_id == "test-node"


def test_short_wav_yields_no_windows_and_no_crash(tmp_path):
    r, up = runner(tmp_path, lambda i: {PROPELLER: 0.9})
    stats = r.run(blocks(0.5))
    assert stats.windows == 0 and stats.observations == 0


def test_blocked_mic_warns_with_settings_path_and_reports_mic_not_ok(tmp_path, caplog):
    r, up = runner(tmp_path, lambda i: {PROPELLER: 0.0}, heartbeat_every_s=1.0)
    with caplog.at_level(logging.WARNING, logger="kuulo.node"):
        r.run(blocks(12, value=0.0))
    assert "Privacy & Security > Microphone" in caplog.text
    heartbeats = [m for p, m in up.sent if p == "/v1/heartbeats"]
    assert heartbeats[-1].mic_ok is False


def test_format_scores_shows_drone_score_and_top_classes():
    line = format_scores(1.5, 0.72, {"Speech": 0.9, PROPELLER: 0.3, "Music": 0.1, "Bird": 0.05})
    assert "drone=0.72" in line and "Speech 0.90" in line and "Bird" not in line


class RecordingTraceUploader:
    def __init__(self):
        self.ended = []
        self.polls = self.pumps = 0

    def on_detection_end(self, detection_id, sustained_high_s):
        self.ended.append((detection_id, sustained_high_s))

    def poll(self):
        self.polls += 1

    def pump(self):
        self.pumps += 1


def test_detection_is_recorded_as_traces_and_reported_to_the_uploader(tmp_path):
    from kuulo_node.tracestore import TraceStore

    cfg = make_test_config(tmp_path)
    keys = load_or_create_keys(cfg.key_file)
    store = TraceStore(tmp_path / "traces", cfg.node_id, keys.private_key, cfg.time_quality)
    tu = RecordingTraceUploader()
    # 0.9 for windows 3..59 (57 windows at a 0.4875 s hop, ~27.8 s sustained), then quiet.
    r, up = runner(tmp_path, lambda i: {PROPELLER: 0.9 if 3 <= i < 60 else 0.0},
                   traces=store, trace_uploader=tu)
    r.run(blocks(45))
    starts = [m for p, m in up.sent if p == "/v1/observations" and m.event.phase.value == "start"]
    [(detection_id, sustained)] = tu.ended
    assert detection_id == starts[0].event.detection_id
    assert 26 <= sustained <= 29
    segments = store.segments(detection_id)
    assert segments and segments[-1].final
    assert tu.polls > 0 and tu.pumps > 0


def test_debug_clips_are_saved_locally_only_when_enabled(tmp_path):
    import soundfile as sf

    clips = tmp_path / "clips"
    r, up = runner(tmp_path, lambda i: {PROPELLER: 0.9 if 3 <= i < 10 else 0.0},
                   debug_clip_dir=clips)
    r.run(blocks(12))
    [start] = [m for p, m in up.sent if p == "/v1/observations" and m.event.phase.value == "start"]
    [wav] = list(clips.glob("*.wav"))
    assert wav.stem == str(start.event.detection_id)
    data, rate = sf.read(wav)
    assert rate == SAMPLE_RATE and 3 < data.size / rate < 12


def test_no_clips_are_written_by_default(tmp_path):
    r, _ = runner(tmp_path, lambda i: {PROPELLER: 0.9 if 3 <= i < 10 else 0.0})
    r.run(blocks(12))
    assert not list(tmp_path.rglob("*.wav"))

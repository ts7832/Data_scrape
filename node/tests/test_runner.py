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

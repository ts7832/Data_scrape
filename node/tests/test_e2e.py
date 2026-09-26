import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from kuulo_node.audio import wav_source
from kuulo_node.classify import FakeClassifier
from kuulo_node.keys import load_or_create_keys
from kuulo_node.runner import NodeRunner
from kuulo_node.testing import PROPELLER, make_test_config
from kuulo_node.uplink import Uplink
from kuulo_server.app import create_app
from kuulo_server.config import Settings


def test_wav_bang_through_real_server_creates_a_coarse_impulse_event(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 * 6) * 0.003).astype(np.float32)
    n, i = int(0.4 * 16000), 3 * 16000
    x[i:i + n] += (rng.standard_normal(n) * 0.5 * np.exp(-np.arange(n) / 480)).astype(np.float32)
    wav = tmp_path / "bang.wav"
    sf.write(wav, x, 16000)
    cfg = make_test_config(tmp_path)
    app = create_app(Settings(db_path=tmp_path / "kuulo.db", tick_interval_s=None))
    with TestClient(app) as client:
        clf = FakeClassifier(lambda i: {PROPELLER: 0.0})
        NodeRunner(cfg, load_or_create_keys(cfg.key_file), clf, Uplink(client)).run(
            wav_source(wav, speed=0))
        [event] = client.get("/v1/impulse-events").json()
    assert event["quality"] == "coarse" and event["node_ids"] == ["test-node"]


def test_wav_replay_through_real_server_creates_tentative_track(tmp_path):
    wav = tmp_path / "clip.wav"
    t = np.arange(44100 * 12) / 44100
    sf.write(wav, np.stack([0.1 * np.sin(2 * np.pi * 180 * t)] * 2, axis=1), 44100)
    cfg = make_test_config(tmp_path)
    clf = FakeClassifier(lambda i: {PROPELLER: 0.9 if 4 <= i < 12 else 0.01})
    app = create_app(Settings(db_path=tmp_path / "kuulo.db", tick_interval_s=None))
    with TestClient(app) as client:
        node = NodeRunner(cfg, load_or_create_keys(cfg.key_file), clf, Uplink(client))
        stats = node.run(wav_source(wav, speed=0))
        tracks = client.get("/v1/tracks").json()
        nodes = client.get("/v1/nodes").json()
    assert stats.observations >= 2
    assert len(tracks) == 1 and tracks[0]["status"] == "tentative"
    assert nodes[0]["node_id"] == "test-node" and nodes[0]["status"] == "online"

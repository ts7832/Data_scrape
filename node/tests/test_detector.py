from datetime import UTC, datetime
from uuid import UUID

from kuulo_node.detector import Detector
from kuulo_node.keys import load_or_create_keys
from kuulo_node.testing import make_test_config
from kuulo_protocol.models import Acoustic, Label, Phase, SourceType
from kuulo_protocol.signing import verify

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
AC = Acoustic(snr_db=12.0, peak_freq_hz=180.0)
HOP = 0.4875


def make(tmp_path):
    cfg = make_test_config(tmp_path)
    keys = load_or_create_keys(cfg.key_file)
    ids = iter(UUID(int=i) for i in range(1, 100))
    return cfg, keys, Detector(cfg, keys, now=lambda: NOW, id_factory=lambda: next(ids))


def feed(det, scores, start=0.0):
    out = []
    for i, s in enumerate(scores):
        obs = det.process(start + i * HOP, s, AC)
        if obs is not None:
            out.append(obs)
    return out


def test_three_high_windows_start_a_signed_detection(tmp_path):
    cfg, keys, det = make(tmp_path)
    [obs] = feed(det, [0.9, 0.9, 0.9])
    assert obs.event.phase is Phase.START and obs.event.detection_id == UUID(int=1)
    assert obs.source.type is SourceType.ACOUSTIC_NODE and obs.source.id == cfg.node_id
    assert obs.detection.label is Label.DRONE_MULTIROTOR and obs.detection.confidence == 0.9
    assert obs.sensor_location == cfg.location and obs.observed_at == NOW
    assert obs.acoustic == AC
    assert verify(obs, keys.public_key)


def test_quiet_after_detection_ends_it(tmp_path):
    _, _, det = make(tmp_path)
    phases = [o.event.phase for o in feed(det, [0.9] * 3 + [0.0] * 12)]
    assert phases == [Phase.START, Phase.END]


def test_reset_ends_open_detection_and_forgets_old_windows(tmp_path):
    _, _, det = make(tmp_path)
    feed(det, [0.9] * 3)
    end = det.reset()
    assert end is not None and end.event.phase is Phase.END
    assert end.event.detection_id == UUID(int=1)
    # two high windows before the gap + two after must NOT start a detection
    assert feed(det, [0.9, 0.9], start=10.0) == []


def test_reset_without_open_detection_is_silent(tmp_path):
    _, _, det = make(tmp_path)
    feed(det, [0.9, 0.9])
    assert det.reset() is None
    assert feed(det, [0.9], start=5.0) == []  # pre-reset windows were discarded

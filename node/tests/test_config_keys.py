import json
import stat
from pathlib import Path

import pytest

from kuulo_node.config import ConfigError, load_config
from kuulo_node.keys import load_or_create_keys
from kuulo_protocol.models import TimeQuality

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"


def test_example_config_loads_with_demo_location():
    cfg = load_config(EXAMPLE)
    assert cfg.node_id == "demo-laptop"
    assert (cfg.location.lat, cfg.location.lon, cfg.location.accuracy_m) == (60.1694, 24.9490, 50)
    assert cfg.time_quality is TimeQuality.NTP
    assert cfg.smoother.threshold == 0.5 and cfg.smoother.k == 3 and cfg.smoother.n == 5
    assert cfg.weights["Propeller, airscrew"] == 1.0
    # relative paths resolve against the config file's directory
    assert cfg.model_path == (EXAMPLE.parent / "../data/models/yamnet.tflite").resolve()
    assert cfg.key_file == EXAMPLE.parent / "demo-laptop.key"


def test_missing_key_is_a_clear_error(tmp_path):
    bad = tmp_path / "c.toml"
    bad.write_text('node_id = "x"\n')
    with pytest.raises(ConfigError, match="server_url"):
        load_config(bad)


def test_weights_must_be_in_unit_interval(tmp_path):
    text = EXAMPLE.read_text().replace('"Helicopter" = 0.8', '"Helicopter" = 1.5')
    path = tmp_path / "c.toml"
    path.write_text(text)
    with pytest.raises(ConfigError, match="Helicopter"):
        load_config(path)


def test_keys_are_created_once_and_reused(tmp_path):
    path = tmp_path / "n.key"
    first = load_or_create_keys(path)
    second = load_or_create_keys(path)
    assert first == second
    assert json.loads(path.read_text())["public_key"] == first.public_key
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_example_state_dir_is_inside_gitignored_data():
    cfg = load_config(EXAMPLE)
    assert cfg.state_dir == (EXAMPLE.parent / "../data/node-state/demo-laptop").resolve()


def test_state_dir_defaults_next_to_config(tmp_path):
    lines = [ln for ln in EXAMPLE.read_text().splitlines() if not ln.startswith("state_dir")]
    path = tmp_path / "c.toml"
    path.write_text("\n".join(lines))
    assert load_config(path).state_dir == tmp_path / "demo-laptop-state"


def test_trace_settings_default_to_the_spec_values(tmp_path):
    cfg = load_config(EXAMPLE)
    assert cfg.traces.enabled is True
    assert cfg.traces.budget_mb == 500 and cfg.traces.upload_mb_per_day == 50
    assert cfg.traces.sample_rate == 0.01


def test_trace_sample_rate_must_be_a_probability(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(EXAMPLE.read_text() + "\n[traces]\nsample_rate = 2.0\n")
    with pytest.raises(ConfigError, match="sample_rate"):
        load_config(path)

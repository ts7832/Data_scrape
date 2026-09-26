"""Test helpers shared by the node's tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from kuulo_protocol.models import SensorLocation, TimeQuality
from kuulo_protocol.smoothing import SmootherConfig

from .config import NodeConfig

PROPELLER = "Propeller, airscrew"


def make_test_config(tmp_path: Path, **overrides) -> NodeConfig:
    cfg = NodeConfig(
        node_id="test-node",
        server_url="http://testserver",
        location=SensorLocation(lat=60.1694, lon=24.9490, accuracy_m=50),
        time_quality=TimeQuality.NTP,
        key_file=tmp_path / "test-node.key",
        model_path=tmp_path / "missing.tflite",
        class_map_path=tmp_path / "missing.csv",
        smoother=SmootherConfig(),
        weights={PROPELLER: 1.0, "Helicopter": 0.8},
        state_dir=tmp_path / "state",
    )
    return replace(cfg, **overrides)

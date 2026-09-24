"""Node configuration: one TOML file, relative paths resolved against its directory."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from kuulo_protocol.models import SensorLocation, TimeQuality
from kuulo_protocol.smoothing import SmootherConfig


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    server_url: str
    location: SensorLocation
    time_quality: TimeQuality
    key_file: Path
    model_path: Path
    class_map_path: Path
    smoother: SmootherConfig
    weights: dict[str, float]


def _require(table: dict, key: str, where: str):
    if key not in table:
        raise ConfigError(f"config is missing '{key}'{where}")
    return table[key]


def load_config(path: Path) -> NodeConfig:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    base = path.parent

    def rel(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (base / p).resolve()

    node_id = str(_require(raw, "node_id", ""))
    if not 1 <= len(node_id) <= 64:
        raise ConfigError("node_id must be 1-64 characters")
    server_url = str(_require(raw, "server_url", ""))
    loc = _require(raw, "location", "")
    location = SensorLocation(
        lat=_require(loc, "lat", " in [location]"),
        lon=_require(loc, "lon", " in [location]"),
        accuracy_m=_require(loc, "accuracy_m", " in [location]"),
    )
    det = raw.get("detection", {})
    defaults = SmootherConfig()
    smoother = SmootherConfig(
        threshold=float(det.get("threshold", defaults.threshold)),
        k=int(det.get("k", defaults.k)),
        n=int(det.get("n", defaults.n)),
        end_after_s=float(det.get("end_after_s", defaults.end_after_s)),
        update_every_s=float(det.get("update_every_s", defaults.update_every_s)),
    )
    weights = {str(k): float(v) for k, v in _require(raw, "weights", "").items()}
    if not weights:
        raise ConfigError("[weights] must name at least one class")
    for name, w in weights.items():
        if not 0 < w <= 1:
            raise ConfigError(f"weight for '{name}' must be in (0, 1], got {w}")
    key_file = Path(_require(raw, "key_file", ""))
    return NodeConfig(
        node_id=node_id,
        server_url=server_url,
        location=location,
        time_quality=TimeQuality(raw.get("time_quality", "ntp")),
        key_file=key_file if key_file.is_absolute() else base / key_file,
        model_path=rel(_require(raw, "model_path", "")),
        class_map_path=rel(_require(raw, "class_map_path", "")),
        smoother=smoother,
        weights=weights,
    )

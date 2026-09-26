"""Node configuration: one TOML file, relative paths resolved against its directory."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from kuulo_protocol.models import NODE_ID_PATTERN, SensorLocation, TimeQuality
from kuulo_protocol.smoothing import SmootherConfig


class ConfigError(ValueError):
    pass


CLASSIFIERS = ("yamnet", "head", "auto")


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = True
    budget_mb: float = 500.0
    upload_mb_per_day: float = 50.0
    sample_rate: float = 0.01


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
    state_dir: Path  # outbox and feature traces; never inside the repository's tracked files
    traces: TraceConfig = TraceConfig()
    classifier: str = "yamnet"  # "yamnet" (Step A), "head" (Step B), "auto" (B if trained)
    head_path: Path | None = None


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
    if not re.fullmatch(NODE_ID_PATTERN, node_id):
        raise ConfigError("node_id must be 1-64 letters, digits, '.', '_' or '-', "
                          "starting with a letter or digit (no spaces or slashes)")
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
    classifier = str(raw.get("classifier", "yamnet"))
    if classifier not in CLASSIFIERS:
        raise ConfigError(f"classifier must be one of {CLASSIFIERS}, got {classifier!r}")
    head_path = rel(raw["head_path"]) if "head_path" in raw else None
    if classifier == "head" and "weights" not in raw:
        raw["weights"] = {"drone": 1.0}  # the head outputs a single drone probability
    weights = {str(k): float(v) for k, v in _require(raw, "weights", "").items()}
    if not weights:
        raise ConfigError("[weights] must name at least one class")
    for name, w in weights.items():
        if not 0 < w <= 1:
            raise ConfigError(f"weight for '{name}' must be in (0, 1], got {w}")
    key_file = Path(_require(raw, "key_file", ""))
    state_dir = rel(raw["state_dir"]) if "state_dir" in raw else base / f"{node_id}-state"
    tr = raw.get("traces", {})
    tdef = TraceConfig()
    traces = TraceConfig(
        enabled=bool(tr.get("enabled", tdef.enabled)),
        budget_mb=float(tr.get("budget_mb", tdef.budget_mb)),
        upload_mb_per_day=float(tr.get("upload_mb_per_day", tdef.upload_mb_per_day)),
        sample_rate=float(tr.get("sample_rate", tdef.sample_rate)),
    )
    if not 0 <= traces.sample_rate <= 1:
        raise ConfigError(f"[traces] sample_rate must be in [0, 1], got {traces.sample_rate}")
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
        state_dir=state_dir,
        traces=traces,
        classifier=classifier,
        head_path=head_path,
    )

# Kuulo Part 2: Demo Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `kuulo-node` that listens to the laptop microphone (or replays a wav file), detects hobby drones with pretrained YAMNet, and sends signed observations to the existing Kuulo server, plus the README, LICENSE and CI needed to publish the repo.

**Architecture:** A new uv workspace member `node/` (package `kuulo_node`). The pipeline is a set of small, pure units (windower, drone score, acoustic meter, detector, uplink), wired by a `NodeRunner` that takes any iterable of audio blocks. Hardware (`sounddevice`) and the model (`ai-edge-litert`) sit behind thin adapters, so every unit and the end-to-end test run in CI with a scripted `FakeClassifier` and synthetic audio.

**Tech Stack:** Python 3.12, numpy, scipy (`resample_poly`), soundfile, sounddevice, ai-edge-litert (YAMNet `.tflite`), httpx, pydantic models and Ed25519 signing from `kuulo_protocol`, FastAPI `TestClient` for the end-to-end test.

**Spec:** `docs/superpowers/specs/2026-09-24-kuulo-part2-demo-node-design.md` (parent: `2026-09-24-kuulo-milestone1-design.md` §5, §10)

## Global Constraints

- Python `>=3.12,<3.13`; run everything with `uv run --no-sync …` (see Makefile comment on why `--no-sync`).
- No full TensorFlow dependency. The model runtime is `ai-edge-litert`; ONNX only if the Task 1 spike fails, and then **stop and ask the user** (the ONNX route needs an Opus-level decision on model provenance).
- Audio: 16 000 Hz mono float32; window 15 600 samples (0.975 s); hop 7 800 samples (0.4875 s).
- Smoothing defaults: threshold 0.5, k=3 of n=5, end after 5.0 s below threshold, update every 5.0 s (reuse `kuulo_protocol.smoothing`).
- Demo location: lat 60.1694, lon 24.9490, accuracy 50 m (University of Helsinki main building). No other real coordinates are ever committed.
- Never commit: audio files, model weights (`*.tflite`, `*.onnx`), key files (`node/*.key`), local configs (`node/*.local.toml`), `data/`.
- Raw audio never leaves memory; the node writes no audio to disk.
- The model is downloaded only by `node/scripts/download_model.py` into `data/models/`; downloads need the user's approval (the user runs `make model` in their own terminal).
- Heartbeat every 60 s; uplink retry list capped at 1 000 messages, heartbeats dropped first, every drop logged and counted.
- License: AGPL-3.0.
- Ruff: line length 100, rules E, F, I, UP, B. Every new module starts with `from __future__ import annotations` and a one-line docstring, matching Part 1.

## Review Focus

1. **A stereo 44.1/48 kHz wav** (what most downloaded clips are): it must be downmixed and resampled, not rejected or played at the wrong speed. Test in Task 3.
2. **A wav shorter than one window, or empty:** the node exits cleanly with zero windows and no observations. Test in Task 7.
3. **The server not running when the node starts:** the node keeps running, queues messages, and registers once the server appears. Test in Task 6.
4. **The server already knows this node_id with a different key** (a deleted key file, or a copied config): a clear error naming the node_id and what to do, not a stack trace. Test in Task 6.
5. **The microphone blocked by macOS privacy settings** (all-zero audio): after 10 s the node logs the System Settings path and heartbeats report `mic_ok=false`. Test in Task 7.

---

## File Structure

```text
node/
  pyproject.toml                    package kuulo-node, script `kuulo-node`
  config.example.toml               committed demo config (UH campus)
  scripts/download_model.py         fetch YAMNet .tflite + class map into data/models/
  src/kuulo_node/
    __init__.py                     __version__
    config.py                       NodeConfig, load_config, ConfigError
    keys.py                         NodeKeys, load_or_create_keys
    audio.py                        constants, AudioBlock, Window, Windower, to_mono_16k,
                                    wav_source, mic_source, MicHealth
    classify.py                     Classifier protocol, drone_score, check_weights,
                                    FakeClassifier, YamnetClassifier, load_class_names
    acoustic.py                     AcousticMeter (SNR vs noise floor, peak frequency)
    detector.py                     Detector: smoothing + reset rule -> signed Observation
    uplink.py                       Uplink (register, queue, backoff), RegistrationConflict
    runner.py                       NodeRunner, RunStats, format_scores
    cli.py                          `kuulo-node run`
    testing.py                      make_test_config (shared by tests)
  tests/
    test_config_keys.py  test_audio.py  test_classify.py  test_acoustic.py
    test_detector.py     test_uplink.py test_runner.py    test_e2e.py
ml/DATASETS.md                      model/dataset sources and licences
.github/workflows/ci.yml
LICENSE
README.md                           rewritten
Makefile, pyproject.toml, .gitignore modified
```

---

### Task 1: Node package scaffold and YAMNet runtime spike

This task is a gate. If the model does not run, stop and report to the user.

**Files:**
- Create: `node/pyproject.toml`, `node/src/kuulo_node/__init__.py`, `node/scripts/download_model.py`, `ml/DATASETS.md`
- Modify: `pyproject.toml` (root), `Makefile`, `.gitignore`

**Interfaces:**
- Produces: installable package `kuulo_node`; files `data/models/yamnet.tflite` and `data/models/yamnet_class_map.csv` (user-downloaded); `make model`.

- [ ] **Step 1: Create `node/pyproject.toml`**

```toml
[project]
name = "kuulo-node"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "kuulo-protocol",
    "httpx>=0.27",
    "numpy>=1.26",
    "scipy>=1.13",
    "soundfile>=0.12",
    "sounddevice>=0.5",
    "ai-edge-litert>=1.2",
]

[project.scripts]
kuulo-node = "kuulo_node.cli:main"

[tool.uv.sources]
kuulo-protocol = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kuulo_node"]
```

- [ ] **Step 2: Create `node/src/kuulo_node/__init__.py`**

```python
"""Kuulo acoustic node: microphone or wav in, signed drone observations out."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Register the member in the root `pyproject.toml`**

Apply these edits:
- `dependencies = ["kuulo-protocol", "kuulo-server", "kuulo-sim", "kuulo-node"]`
- under `[tool.uv.sources]` add `kuulo-node = { workspace = true }`
- `members = ["protocol", "server", "sim", "node"]`
- `testpaths = ["protocol/tests", "server/tests", "sim/tests", "node/tests"]`
- `pythonpath = ["protocol/src", "server/src", "sim/src", "node/src"]`
- `known-first-party = ["kuulo_protocol", "kuulo_server", "kuulo_sim", "kuulo_node"]`

- [ ] **Step 4: Install**

Run: `uv sync` (a one-off; it needs network access to PyPI)
Expected: finishes without error and installs ai-edge-litert, numpy, scipy, soundfile, sounddevice.
If `ai-edge-litert` has no wheel for this platform: **stop and report to the user** (ONNX fallback decision).

- [ ] **Step 5: Create `node/scripts/download_model.py`**

```python
"""Download the YAMNet TFLite model and its class map into data/models/ (gitignored)."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

FILES = {
    # MediaPipe's float32 YAMNet audio classifier (Apache-2.0).
    "yamnet.tflite": "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite",  # noqa: E501
    # AudioSet class names in model output order (Apache-2.0, tensorflow/models).
    "yamnet_class_map.csv": "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",  # noqa: E501
}


def main() -> int:
    out_dir = Path(__file__).resolve().parents[2] / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = out_dir / name
        if target.exists():
            print(f"exists  {target}")
            continue
        print(f"fetch   {url}")
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        target.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()[:16]
        print(f"wrote   {target}  {len(data):,} bytes  sha256:{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Makefile targets and gitignore**

In `Makefile`, change the first line to `.PHONY: dev server dashboard sim test types node model prepublish`, and append:

```make
NODE_CONFIG ?= node/config.local.toml

node/config.local.toml:
	cp node/config.example.toml $@

model:
	uv run --no-sync python node/scripts/download_model.py

node: $(NODE_CONFIG)
	uv run --no-sync kuulo-node run --config $(NODE_CONFIG) $(if $(INPUT),--input $(INPUT)) $(if $(SPEED),--speed $(SPEED)) $(if $(SCORES),--print-scores)
```

Append to `.gitignore`:

```gitignore
# Node: local configs and keys are personal (real locations, private keys)
node/*.local.toml
node/*.key
```

- [ ] **Step 7: Ask the user to download the model (download gate)**

Tell the user: "Task 1 needs the YAMNet model, about 4 MB: `yamnet.tflite` from storage.googleapis.com (MediaPipe, Apache-2.0) plus a 14 KB class-name CSV from raw.githubusercontent.com. The sandbox doesn't allow storage.googleapis.com, so please run this in your own terminal: `cd ~/Developer/Data_scrape && make model`". Wait for them to confirm.

- [ ] **Step 8: Spike: load and run the model**

Run:

```bash
uv run --no-sync python - <<'EOF'
import csv, numpy as np
from ai_edge_litert.interpreter import Interpreter
it = Interpreter(model_path="data/models/yamnet.tflite"); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
print("input", i["shape"], i["dtype"], "output", o["shape"], o["dtype"])
names = [r["display_name"] for r in csv.DictReader(open("data/models/yamnet_class_map.csv"))]
x = np.zeros(int(np.prod(i["shape"])), np.float32).reshape(i["shape"])
it.set_tensor(i["index"], x); it.invoke()
s = it.get_tensor(o["index"]).reshape(-1, len(names)).mean(axis=0)
print(len(names), "classes; top for silence:", names[int(s.argmax())], round(float(s.max()), 3))
for n in ["Propeller, airscrew", "Helicopter", "Aircraft", "Aircraft engine"]:
    print(n, n in names)
EOF
```

Expected: input with 15 600 elements (e.g. `[15600]`), output with 521 columns, `521 classes; top for silence: Silence`, and all four class names `True`.
If the input size is not 15 600, or the output is not a multiple of 521: **stop and report to the user**.

- [ ] **Step 9: Create `ml/DATASETS.md`**

```markdown
# Models and datasets

Nothing listed here is committed to the repository. Scripts download into `data/` (gitignored).

| Item | Source | Licence | Commercial use | Used for |
|---|---|---|---|---|
| YAMNet float32 TFLite | `storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite` (Google MediaPipe) | Apache-2.0 | Yes | Step A classifier (zero training) |
| YAMNet class map | `tensorflow/models` `research/audioset/yamnet/yamnet_class_map.csv` | Apache-2.0 | Yes | Class names in output order |

Training datasets (Step B) are not used yet. Each will be added here with its licence verified before use.
```

- [ ] **Step 10: Commit**

```bash
git add node/pyproject.toml node/src/kuulo_node/__init__.py node/scripts/download_model.py ml/DATASETS.md pyproject.toml uv.lock Makefile .gitignore
git commit -m "feat(node): scaffold kuulo-node package, YAMNet download script, runtime spike passes"
```

---

### Task 2: Config and node keys

**Files:**
- Create: `node/src/kuulo_node/config.py`, `node/src/kuulo_node/keys.py`, `node/src/kuulo_node/testing.py`, `node/config.example.toml`
- Test: `node/tests/test_config_keys.py`

**Interfaces:**
- Consumes: `kuulo_protocol.models.SensorLocation`, `TimeQuality`; `kuulo_protocol.smoothing.SmootherConfig` (frozen dataclass: `threshold, k, n, end_after_s, update_every_s`); `kuulo_protocol.signing.generate_keypair() -> (private_b64, public_b64)`.
- Produces:
  - `NodeConfig` (frozen dataclass) with fields `node_id: str, server_url: str, location: SensorLocation, time_quality: TimeQuality, key_file: Path, model_path: Path, class_map_path: Path, smoother: SmootherConfig, weights: dict[str, float]`
  - `load_config(path: Path) -> NodeConfig`; `ConfigError(ValueError)`
  - `NodeKeys` (frozen dataclass: `private_key: str, public_key: str`); `load_or_create_keys(path: Path) -> NodeKeys`
  - `kuulo_node.testing.make_test_config(tmp_path: Path, **overrides) -> NodeConfig`

- [ ] **Step 1: Write `node/config.example.toml`**

```toml
# Kuulo node demo config. `make node` copies this to node/config.local.toml (gitignored);
# edit the local copy, not this file. Relative paths are relative to this file.

node_id = "demo-laptop"
server_url = "http://127.0.0.1:8000"
time_quality = "ntp"
key_file = "demo-laptop.key"
model_path = "../data/models/yamnet.tflite"
class_map_path = "../data/models/yamnet_class_map.csv"

# Reported sensor position. Demo value: University of Helsinki main building,
# city centre campus. Never commit a real node location.
[location]
lat = 60.1694
lon = 24.9490
accuracy_m = 50

[detection]
threshold = 0.5
k = 3
n = 5
end_after_s = 5.0
update_every_s = 5.0

# drone score = max over classes of (weight x YAMNet score). Keys are AudioSet class names.
[weights]
"Propeller, airscrew" = 1.0
"Helicopter" = 0.8
"Aircraft" = 0.6
"Aircraft engine" = 0.6
```

- [ ] **Step 2: Write the failing tests `node/tests/test_config_keys.py`**

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run --no-sync pytest node/tests/test_config_keys.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kuulo_node.config'`

- [ ] **Step 4: Implement `node/src/kuulo_node/config.py`**

```python
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
```

- [ ] **Step 5: Implement `node/src/kuulo_node/keys.py`**

```python
"""The node's Ed25519 identity: created on first run, then reused (the server pins it)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from kuulo_protocol.signing import generate_keypair


@dataclass(frozen=True)
class NodeKeys:
    private_key: str
    public_key: str


def load_or_create_keys(path: Path) -> NodeKeys:
    path = Path(path)
    if path.exists():
        data = json.loads(path.read_text())
        return NodeKeys(private_key=data["private_key"], public_key=data["public_key"])
    private_key, public_key = generate_keypair()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"private_key": private_key, "public_key": public_key}))
    os.chmod(path, 0o600)
    return NodeKeys(private_key=private_key, public_key=public_key)
```

- [ ] **Step 6: Implement `node/src/kuulo_node/testing.py`**

```python
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
    )
    return replace(cfg, **overrides)
```

- [ ] **Step 7: Run tests**

Run: `uv run --no-sync pytest node/tests/test_config_keys.py -q`
Expected: 4 passed

- [ ] **Step 8: Commit**

```bash
git add node/config.example.toml node/src/kuulo_node/config.py node/src/kuulo_node/keys.py node/src/kuulo_node/testing.py node/tests/test_config_keys.py
git commit -m "feat(node): TOML config with demo location, persistent Ed25519 node keys"
```

---

### Task 3: Audio: blocks, windowing, resampling, wav replay, mic health

**Files:**
- Create: `node/src/kuulo_node/audio.py`
- Test: `node/tests/test_audio.py`

**Interfaces:**
- Produces:
  - constants `SAMPLE_RATE = 16_000`, `WINDOW_SAMPLES = 15_600`, `HOP_SAMPLES = 7_800`, `HOP_S = HOP_SAMPLES / SAMPLE_RATE`
  - `AudioBlock(samples: np.ndarray, t: float, restart: bool = False)`, frozen dataclass; `t` = audio time (s) of the first sample
  - `Window(samples: np.ndarray, t: float, gap_before: bool)`, frozen dataclass
  - `Windower().push(block: AudioBlock) -> list[Window]`
  - `to_mono_16k(data: np.ndarray, rate: int) -> np.ndarray`
  - `wav_source(path: Path, *, speed: float = 1.0, block_s: float = 0.1, sleep=time.sleep) -> Iterator[AudioBlock]` (`speed <= 0` means no sleeping)
  - `mic_source(*, block_s=0.1, retry_s=10.0, on_error: Callable[[], None] | None = None) -> Iterator[AudioBlock]`
  - `MicHealth(silent_after_s: float = 10.0)`, with `.ok: bool`, `.update(block) -> bool` (True exactly when it flips to not-ok because of silence), `.mark_failed()`
  - `MIC_HELP: str` (the actionable message)

- [ ] **Step 1: Write the failing tests `node/tests/test_audio.py`**

```python
import numpy as np
import soundfile as sf

from kuulo_node.audio import (
    HOP_S,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    AudioBlock,
    MicHealth,
    Windower,
    to_mono_16k,
    wav_source,
)


def blocks(n_blocks, block=1600, start=0.0, value=0.1):
    for i in range(n_blocks):
        yield AudioBlock(np.full(block, value, np.float32), start + i * block / SAMPLE_RATE)


def test_windower_emits_overlapping_windows_with_times():
    w = Windower()
    windows = [win for b in blocks(20) for win in w.push(b)]  # 2.0 s of audio
    assert [round(x.t, 4) for x in windows] == [0.0, round(HOP_S, 4), round(2 * HOP_S, 4)]
    assert all(x.samples.shape == (WINDOW_SAMPLES,) for x in windows)
    assert not any(x.gap_before for x in windows)


def test_windower_gap_clears_buffer_and_flags_next_window():
    w = Windower()
    before = [x for b in blocks(20) for x in w.push(b)]
    after = [x for b in blocks(20, start=5.0) for x in w.push(b)]  # 3 s jump
    assert before and after
    assert after[0].gap_before and after[0].t == 5.0
    assert not any(x.gap_before for x in after[1:])


def test_windower_restart_flag_is_a_gap_even_without_time_jump():
    w = Windower()
    list(w.push(b) for b in blocks(20))
    restarted = AudioBlock(np.zeros(16000, np.float32), 2.0, restart=True)
    out = w.push(restarted)
    assert out and out[0].gap_before


def test_to_mono_16k_downmixes_and_resamples_stereo_44k():
    stereo = np.zeros((44100, 2), np.float32)
    stereo[:, 0] = 0.5
    x = to_mono_16k(stereo, 44100)
    assert x.dtype == np.float32 and x.ndim == 1
    assert abs(x.size - 16000) <= 1
    assert abs(float(np.median(x)) - 0.25) < 1e-3


def test_wav_source_replays_48k_stereo_file_at_16k(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros((48000 * 2, 2), np.float32), 48000)
    slept = []
    got = list(wav_source(path, speed=2.0, sleep=slept.append))
    assert abs(sum(b.samples.size for b in got) - 32000) <= 2
    assert got[0].t == 0.0 and got[1].t == 0.1
    assert abs(sum(slept) - 1.0) < 0.01  # 2 s of audio at 2x speed


def test_wav_source_empty_file_yields_nothing(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(0, np.float32), 16000)
    assert list(wav_source(path, speed=0)) == []


def test_mic_health_flips_after_10s_of_digital_silence_and_recovers():
    h = MicHealth()
    flips = [h.update(b) for b in blocks(110, value=0.0)]  # 11 s of zeros
    assert flips.count(True) == 1 and not h.ok
    h.update(AudioBlock(np.full(1600, 0.01, np.float32), 11.0))
    assert h.ok
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest node/tests/test_audio.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kuulo_node.audio'`

- [ ] **Step 3: Implement `node/src/kuulo_node/audio.py`**

```python
"""Audio in: 16 kHz mono blocks from a wav file or the microphone, cut into YAMNet windows."""

from __future__ import annotations

import logging
import math
import queue
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

log = logging.getLogger("kuulo.node")

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_600  # YAMNet's fixed input: 0.975 s
HOP_SAMPLES = 7_800  # 50 % overlap
HOP_S = HOP_SAMPLES / SAMPLE_RATE

MIC_HELP = (
    "No audio from the microphone. On macOS, allow your terminal app in System Settings > "
    "Privacy & Security > Microphone, then restart it. On Linux, install libportaudio2 and "
    "check the input device (`python -m sounddevice`)."
)


@dataclass(frozen=True)
class AudioBlock:
    samples: np.ndarray  # float32, mono, 16 kHz
    t: float  # audio time of the first sample, seconds
    restart: bool = False  # the source reopened; earlier audio is not contiguous


@dataclass(frozen=True)
class Window:
    samples: np.ndarray
    t: float  # time of the window's first sample
    gap_before: bool  # audio before this window was lost: detection state must reset


class Windower:
    """Ring buffer that emits 15 600-sample windows every 7 800 samples."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, np.float32)
        self._buf_t = 0.0
        self._expected: float | None = None
        self._gap = False

    def push(self, block: AudioBlock) -> list[Window]:
        jumped = self._expected is not None and abs(block.t - self._expected) > HOP_S
        if block.restart or jumped:
            self._buf = np.zeros(0, np.float32)
            self._gap = True
        if self._buf.size == 0:
            self._buf_t = block.t
        self._buf = np.concatenate([self._buf, block.samples.astype(np.float32, copy=False)])
        self._expected = block.t + block.samples.size / SAMPLE_RATE
        out: list[Window] = []
        while self._buf.size >= WINDOW_SAMPLES:
            out.append(Window(self._buf[:WINDOW_SAMPLES].copy(), self._buf_t, self._gap))
            self._gap = False
            self._buf = self._buf[HOP_SAMPLES:]
            self._buf_t += HOP_S
        return out


def to_mono_16k(data: np.ndarray, rate: int) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = math.gcd(int(rate), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
    return x


def wav_source(
    path: Path,
    *,
    speed: float = 1.0,
    block_s: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[AudioBlock]:
    """Replay a file as if it were live. speed=1 is real time; speed<=0 is as fast as possible."""
    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    x = to_mono_16k(data, rate)
    n = int(round(block_s * SAMPLE_RATE))
    for i in range(0, x.size, n):
        chunk = x[i : i + n]
        yield AudioBlock(chunk, round(i / SAMPLE_RATE, 6))
        if speed > 0:
            sleep(chunk.size / SAMPLE_RATE / speed)


def _open_stream(sd, block_s: float, q: queue.Queue):
    """Open at 16 kHz mono; if the device refuses, open at its native rate. Returns (stream, rate)."""

    def callback(indata, _frames, _time, status):
        if status:
            log.debug("audio status: %s", status)
        q.put(indata[:, 0].copy())

    for rate in (SAMPLE_RATE, None):
        try:
            if rate is None:
                rate = int(sd.query_devices(kind="input")["default_samplerate"])
            stream = sd.InputStream(
                samplerate=rate, channels=1, dtype="float32",
                blocksize=int(block_s * rate), callback=callback,
            )
            stream.start()
            return stream, rate
        except sd.PortAudioError:
            if rate != SAMPLE_RATE:
                raise
    raise RuntimeError("unreachable")


def mic_source(
    *,
    block_s: float = 0.1,
    retry_s: float = 10.0,
    on_error: Callable[[], None] | None = None,
) -> Iterator[AudioBlock]:
    """Live microphone blocks, timed on the monotonic clock; reopens every retry_s on failure."""
    import sounddevice as sd  # imported here so tests and CI never need PortAudio

    t0 = time.monotonic()
    restart = False
    while True:
        q: queue.Queue = queue.Queue()
        stream = None
        try:
            stream, rate = _open_stream(sd, block_s, q)
            log.info("microphone open at %d Hz", rate)
            while True:
                chunk = q.get(timeout=2.0)
                samples = chunk if rate == SAMPLE_RATE else to_mono_16k(chunk, rate)
                yield AudioBlock(samples, time.monotonic() - t0, restart)
                restart = False
        except (sd.PortAudioError, queue.Empty, OSError) as exc:
            log.error("%s (%s). Retrying in %.0f s.", MIC_HELP, exc, retry_s)
            if on_error is not None:
                on_error()
            restart = True
            time.sleep(retry_s)
        finally:
            if stream is not None:
                stream.close()


class MicHealth:
    """Tracks digital silence: a blocked mic delivers exact zeros, a quiet room does not."""

    SILENCE = 1e-6

    def __init__(self, silent_after_s: float = 10.0) -> None:
        self.silent_after_s = silent_after_s
        self.ok = True
        self._silent_since: float | None = None

    def update(self, block: AudioBlock) -> bool:
        silent = block.samples.size == 0 or float(np.max(np.abs(block.samples))) < self.SILENCE
        if not silent:
            self._silent_since = None
            self.ok = True
            return False
        if self._silent_since is None:
            self._silent_since = block.t
        end = block.t + block.samples.size / SAMPLE_RATE
        if self.ok and end - self._silent_since >= self.silent_after_s:
            self.ok = False
            return True
        return False

    def mark_failed(self) -> None:
        self.ok = False
```

- [ ] **Step 4: Run tests**

Run: `uv run --no-sync pytest node/tests/test_audio.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add node/src/kuulo_node/audio.py node/tests/test_audio.py
git commit -m "feat(node): audio windowing with gap detection, wav replay, mic source, mic health"
```

---

### Task 4: Classifier: drone score, fake and YAMNet

**Files:**
- Create: `node/src/kuulo_node/classify.py`
- Test: `node/tests/test_classify.py`

**Interfaces:**
- Consumes: `WINDOW_SAMPLES` from `kuulo_node.audio`.
- Produces:
  - `class Classifier(Protocol): def score(self, window: np.ndarray) -> dict[str, float]`
  - `drone_score(scores: dict[str, float], weights: dict[str, float]) -> float` (in [0, 1])
  - `check_weights(class_names: list[str], weights: dict[str, float]) -> None` (raises `ValueError` naming unknown classes)
  - `FakeClassifier(script: Callable[[int], dict[str, float]])`, called with the 0-based window index; exposes `.calls`
  - `load_class_names(path: Path) -> list[str]`
  - `YamnetClassifier(model_path: Path, class_map_path: Path, weights: dict[str, float])`

- [ ] **Step 1: Write the failing tests `node/tests/test_classify.py`**

```python
from pathlib import Path

import numpy as np
import pytest

from kuulo_node.audio import WINDOW_SAMPLES
from kuulo_node.classify import FakeClassifier, check_weights, drone_score

MODELS = Path(__file__).resolve().parents[2] / "data" / "models"


def test_drone_score_is_weighted_max_and_ignores_unweighted_classes():
    weights = {"Propeller, airscrew": 1.0, "Helicopter": 0.8}
    scores = {"Propeller, airscrew": 0.3, "Helicopter": 0.9, "Speech": 0.99}
    assert drone_score(scores, weights) == pytest.approx(0.72)


def test_drone_score_clamps_to_unit_interval():
    assert drone_score({"A": 1.7}, {"A": 1.0}) == 1.0
    assert drone_score({}, {"A": 1.0}) == 0.0


def test_check_weights_names_unknown_classes():
    with pytest.raises(ValueError, match="Propellor"):
        check_weights(["Speech", "Propeller, airscrew"], {"Propellor": 1.0})


def test_fake_classifier_follows_script_by_window_index():
    fake = FakeClassifier(lambda i: {"A": float(i)})
    window = np.zeros(WINDOW_SAMPLES, np.float32)
    assert [fake.score(window)["A"] for _ in range(3)] == [0.0, 1.0, 2.0]
    assert fake.calls == 3


@pytest.mark.skipif(not (MODELS / "yamnet.tflite").exists(), reason="model not downloaded")
def test_yamnet_scores_silence_as_silence():
    from kuulo_node.classify import YamnetClassifier

    clf = YamnetClassifier(
        MODELS / "yamnet.tflite", MODELS / "yamnet_class_map.csv", {"Propeller, airscrew": 1.0}
    )
    scores = clf.score(np.zeros(WINDOW_SAMPLES, np.float32))
    assert len(scores) == 521
    assert max(scores, key=scores.get) == "Silence"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest node/tests/test_classify.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kuulo_node.classify'`

- [ ] **Step 3: Implement `node/src/kuulo_node/classify.py`**

```python
"""Window -> per-class scores -> one drone score. YAMNet now; a trained head can drop in later."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import WINDOW_SAMPLES


class Classifier(Protocol):
    def score(self, window: np.ndarray) -> dict[str, float]: ...


def drone_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    best = max((w * scores.get(name, 0.0) for name, w in weights.items()), default=0.0)
    return min(1.0, max(0.0, best))


def check_weights(class_names: list[str], weights: dict[str, float]) -> None:
    unknown = sorted(set(weights) - set(class_names))
    if unknown:
        raise ValueError(
            f"[weights] names classes YAMNet does not have: {unknown}. "
            "Use exact display names from data/models/yamnet_class_map.csv."
        )


class FakeClassifier:
    """Scripted scores for tests: script(i) gives the scores for the i-th window."""

    def __init__(self, script: Callable[[int], dict[str, float]]) -> None:
        self._script = script
        self.calls = 0

    def score(self, window: np.ndarray) -> dict[str, float]:
        result = self._script(self.calls)
        self.calls += 1
        return result


def load_class_names(path: Path) -> list[str]:
    with open(path, newline="") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


class YamnetClassifier:
    """YAMNet TFLite via ai-edge-litert (no TensorFlow). 521 AudioSet class scores per window."""

    def __init__(self, model_path: Path, class_map_path: Path, weights: dict[str, float]) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.names = load_class_names(class_map_path)
        check_weights(self.names, weights)
        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        inp = self._interp.get_input_details()[0]
        self._in_index = inp["index"]
        self._in_shape = tuple(int(d) for d in inp["shape"])
        if int(np.prod(self._in_shape)) != WINDOW_SAMPLES:
            raise ValueError(f"unexpected YAMNet input shape {self._in_shape}")
        self._out_index = self._interp.get_output_details()[0]["index"]

    def score(self, window: np.ndarray) -> dict[str, float]:
        x = np.asarray(window, dtype=np.float32).reshape(self._in_shape)
        self._interp.set_tensor(self._in_index, x)
        self._interp.invoke()
        out = self._interp.get_tensor(self._out_index).reshape(-1, len(self.names)).mean(axis=0)
        return dict(zip(self.names, (float(v) for v in out), strict=True))
```

- [ ] **Step 4: Run tests**

Run: `uv run --no-sync pytest node/tests/test_classify.py -q`
Expected: 5 passed (4 if the model has not been downloaded; the YAMNet test is then skipped)

- [ ] **Step 5: Commit**

```bash
git add node/src/kuulo_node/classify.py node/tests/test_classify.py
git commit -m "feat(node): weighted drone score, scripted fake classifier, YAMNet via LiteRT"
```

---

### Task 5: Acoustic meter and detector

**Files:**
- Create: `node/src/kuulo_node/acoustic.py`, `node/src/kuulo_node/detector.py`
- Test: `node/tests/test_acoustic.py`, `node/tests/test_detector.py`

**Interfaces:**
- Consumes: `NodeConfig`, `NodeKeys`, `SAMPLE_RATE`, `HOP_S`; `kuulo_protocol.models` (`Observation, Source, SourceType, Detection, Label, EventRef, Phase, Acoustic`); `kuulo_protocol.signing.sign/verify`; `DetectionSmoother(config, id_factory)` with `.push(t, score) -> Phase | None`, `.active`, `.detection_id`.
- Produces:
  - `AcousticMeter(floor_window_s: float = 10.0, alpha: float = 0.2)` with `.measure(samples: np.ndarray) -> Acoustic`
  - `Detector(cfg: NodeConfig, keys: NodeKeys, *, now: Callable[[], datetime] = utc_now, id_factory: Callable[[], UUID] = uuid4)`
    - `.process(t: float, score: float, acoustic: Acoustic) -> Observation | None`
    - `.reset() -> Observation | None` (ends an open detection and replaces the smoother)
  - `utc_now() -> datetime` in `kuulo_node.detector`

- [ ] **Step 1: Write the failing tests `node/tests/test_acoustic.py`**

```python
import numpy as np

from kuulo_node.acoustic import AcousticMeter
from kuulo_node.audio import SAMPLE_RATE, WINDOW_SAMPLES

t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE


def tone(freq, amp):
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_peak_frequency_of_a_tone():
    assert abs(AcousticMeter().measure(tone(1000, 0.1)).peak_freq_hz - 1000) < 5


def test_snr_rises_when_loud_sound_follows_quiet_floor():
    m = AcousticMeter()
    for _ in range(10):
        quiet = m.measure(tone(200, 0.001))
    loud = m.measure(tone(200, 0.1))
    assert abs(quiet.snr_db) < 3
    assert loud.snr_db > 30


def test_silence_is_valid_and_bounded():
    a = AcousticMeter().measure(np.zeros(WINDOW_SAMPLES, np.float32))
    assert -50 <= a.snr_db <= 150 and a.peak_freq_hz >= 0
```

- [ ] **Step 2: Write the failing tests `node/tests/test_detector.py`**

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run --no-sync pytest node/tests/test_acoustic.py node/tests/test_detector.py -q`
Expected: FAIL with `ModuleNotFoundError` for `kuulo_node.acoustic` / `kuulo_node.detector`

- [ ] **Step 4: Implement `node/src/kuulo_node/acoustic.py`**

```python
"""Rough acoustics for an observation: SNR against a running noise floor, and peak frequency."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from kuulo_protocol.models import Acoustic

from .audio import HOP_S, SAMPLE_RATE


class AcousticMeter:
    """Noise floor = EMA of the quietest window RMS seen in the last floor_window_s."""

    def __init__(self, floor_window_s: float = 10.0, alpha: float = 0.2) -> None:
        self._recent: deque[float] = deque(maxlen=max(1, round(floor_window_s / HOP_S)))
        self._alpha = alpha
        self._floor: float | None = None

    def measure(self, samples: np.ndarray) -> Acoustic:
        x = np.asarray(samples, dtype=np.float64)
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        self._recent.append(rms)
        quietest = min(self._recent)
        if self._floor is None:
            self._floor = quietest
        else:
            self._floor = (1 - self._alpha) * self._floor + self._alpha * quietest
        snr = 20 * math.log10(max(rms, 1e-9) / max(self._floor, 1e-9))
        peak_hz = 0.0
        if x.size:
            spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size)))
            peak_hz = float(np.argmax(spectrum)) * SAMPLE_RATE / x.size
        return Acoustic(snr_db=round(min(150.0, max(-50.0, snr)), 1), peak_freq_hz=round(peak_hz, 1))
```

- [ ] **Step 5: Implement `node/src/kuulo_node/detector.py`**

```python
"""Per-window drone scores -> smoothed detections -> signed Observations."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from kuulo_protocol.models import (
    Acoustic,
    Detection,
    EventRef,
    Label,
    Observation,
    Phase,
    Source,
    SourceType,
)
from kuulo_protocol.signing import sign
from kuulo_protocol.smoothing import DetectionSmoother

from .config import NodeConfig
from .keys import NodeKeys


def utc_now() -> datetime:
    return datetime.now(UTC)


class Detector:
    def __init__(
        self,
        cfg: NodeConfig,
        keys: NodeKeys,
        *,
        now: Callable[[], datetime] = utc_now,
        id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self.cfg = cfg
        self.keys = keys
        self._now = now
        self._id_factory = id_factory
        self._smoother = self._new_smoother()
        self._last_score = 0.0
        self._last_acoustic: Acoustic | None = None

    def _new_smoother(self) -> DetectionSmoother:
        return DetectionSmoother(self.cfg.smoother, id_factory=self._id_factory)

    def process(self, t: float, score: float, acoustic: Acoustic) -> Observation | None:
        self._last_score, self._last_acoustic = score, acoustic
        phase = self._smoother.push(t, score)
        if phase is None:
            return None
        return self._observation(phase, self._smoother.detection_id, score, acoustic)

    def reset(self) -> Observation | None:
        """Audio was interrupted: close any open detection and start from a clean slate."""
        obs = None
        if self._smoother.active:
            obs = self._observation(
                Phase.END, self._smoother.detection_id, self._last_score, self._last_acoustic
            )
        self._smoother = self._new_smoother()
        return obs

    def _observation(
        self, phase: Phase, detection_id: UUID | None, score: float, acoustic: Acoustic | None
    ) -> Observation:
        assert detection_id is not None
        obs = Observation(
            source=Source(type=SourceType.ACOUSTIC_NODE, id=self.cfg.node_id),
            observed_at=self._now(),
            time_quality=self.cfg.time_quality,
            sensor_location=self.cfg.location,
            detection=Detection(
                label=Label.DRONE_MULTIROTOR, confidence=round(min(1.0, max(0.0, score)), 4)
            ),
            event=EventRef(detection_id=detection_id, phase=phase),
            acoustic=acoustic,
        )
        return sign(obs, self.keys.private_key)
```

- [ ] **Step 6: Run tests**

Run: `uv run --no-sync pytest node/tests/test_acoustic.py node/tests/test_detector.py -q`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add node/src/kuulo_node/acoustic.py node/src/kuulo_node/detector.py node/tests/test_acoustic.py node/tests/test_detector.py
git commit -m "feat(node): SNR/peak-frequency meter and detector with reset-on-gap rule"
```

---

### Task 6: Uplink: registration, queue, backoff

**Files:**
- Create: `node/src/kuulo_node/uplink.py`
- Test: `node/tests/test_uplink.py`

**Interfaces:**
- Consumes: `httpx.Client` (a FastAPI `TestClient` is one); `NodeRegistration`, `Observation`, `Heartbeat` from `kuulo_protocol.models`.
- Produces:
  - `RegistrationConflict(RuntimeError)`
  - `Uplink(client: httpx.Client, *, max_pending: int = 1000, monotonic: Callable[[], float] = time.monotonic, backoff_max_s: float = 30.0)`
    - `.ensure_registered(reg: NodeRegistration) -> bool`
    - `.send(path: str, msg: BaseModel) -> None` (paths: `"/v1/observations"`, `"/v1/heartbeats"`)
    - `.flush(force: bool = False) -> None`
    - attributes `.registered: bool`, `.dropped: int`, `.pending_count: int` (property)

- [ ] **Step 1: Write the failing tests `node/tests/test_uplink.py`**

```python
import httpx
import pytest

from kuulo_node.uplink import RegistrationConflict, Uplink
from kuulo_protocol.models import Heartbeat, NodeRegistration, SensorLocation, TimeQuality

REG = NodeRegistration(
    node_id="demo-laptop", public_key="cHVi",
    location=SensorLocation(lat=60.1694, lon=24.949, accuracy_m=50), time_quality=TimeQuality.NTP,
)


def hb(i=0):
    from datetime import UTC, datetime
    return Heartbeat(node_id="demo-laptop", sent_at=datetime(2026, 9, 24, tzinfo=UTC),
                     software_version=str(i), mic_ok=True, queue_depth=0)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def client(handler):
    return httpx.Client(base_url="http://node.test", transport=httpx.MockTransport(handler))


def test_server_down_at_start_is_not_fatal_and_backs_off():
    calls = []

    def down(request):
        calls.append(request.url.path)
        raise httpx.ConnectError("refused")

    clock = Clock()
    up = Uplink(client(down), monotonic=clock)
    assert up.ensure_registered(REG) is False
    assert up.ensure_registered(REG) is False  # within backoff: no second attempt
    assert calls == ["/v1/nodes/register"]
    up.send("/v1/heartbeats", hb())
    assert up.pending_count == 1
    clock.t = 1.5
    up.ensure_registered(REG)
    assert len(calls) == 2


def test_registers_then_flushes_in_order():
    seen = []

    def ok(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"status": "ok"})

    up = Uplink(client(ok))
    up.send("/v1/heartbeats", hb(1))
    up.flush()
    assert seen == []  # not registered yet: nothing sent
    assert up.ensure_registered(REG)
    up.flush()
    assert seen == ["/v1/nodes/register", "/v1/heartbeats"] and up.pending_count == 0


def test_key_conflict_names_the_node_and_the_fix():
    up = Uplink(client(lambda r: httpx.Response(409, json={"detail": "different key"})))
    with pytest.raises(RegistrationConflict, match="demo-laptop") as err:
        up.ensure_registered(REG)
    assert "node_id" in str(err.value)


def test_unknown_node_after_server_reset_triggers_reregistration():
    responses = iter([httpx.Response(200), httpx.Response(401, json={"detail": "unknown node"})])
    up = Uplink(client(lambda r: next(responses)))
    up.ensure_registered(REG)
    up.send("/v1/heartbeats", hb())
    up.flush()
    assert up.registered is False and up.pending_count == 1  # kept, not dropped


def test_rejected_message_is_dropped_and_counted():
    responses = iter([httpx.Response(200), httpx.Response(400, json={"detail": "bad"})])
    up = Uplink(client(lambda r: next(responses)))
    up.ensure_registered(REG)
    up.send("/v1/heartbeats", hb())
    up.flush()
    assert up.pending_count == 0 and up.dropped == 1


def test_full_queue_drops_heartbeats_before_observations():
    up = Uplink(client(lambda r: httpx.Response(500)), max_pending=3)
    up.send("/v1/observations", hb(1))
    up.send("/v1/heartbeats", hb(2))
    up.send("/v1/observations", hb(3))
    up.send("/v1/observations", hb(4))
    kept = [m.software_version for _, m in up._pending]
    assert kept == ["1", "3", "4"] and up.dropped == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --no-sync pytest node/tests/test_uplink.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kuulo_node.uplink'`

- [ ] **Step 3: Implement `node/src/kuulo_node/uplink.py`**

```python
"""HTTP uplink: register once, then send messages in order with bounded retry and backoff."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable

import httpx
from pydantic import BaseModel

from kuulo_protocol.models import NodeRegistration

log = logging.getLogger("kuulo.node")
HEARTBEATS = "/v1/heartbeats"


class RegistrationConflict(RuntimeError):
    pass


class Uplink:
    def __init__(
        self,
        client: httpx.Client,
        *,
        max_pending: int = 1000,
        monotonic: Callable[[], float] = time.monotonic,
        backoff_max_s: float = 30.0,
    ) -> None:
        self.client = client
        self.max_pending = max_pending
        self._monotonic = monotonic
        self._backoff_max = backoff_max_s
        self._backoff = 1.0
        self._next_try = 0.0
        self._pending: deque[tuple[str, BaseModel]] = deque()
        self.registered = False
        self.dropped = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _due(self) -> bool:
        return self._monotonic() >= self._next_try

    def _fail(self, why: str) -> None:
        log.warning("server unreachable (%s); retrying in %.0f s", why, self._backoff)
        self._next_try = self._monotonic() + self._backoff
        self._backoff = min(self._backoff * 2, self._backoff_max)

    def _ok(self) -> None:
        self._backoff = 1.0
        self._next_try = 0.0

    def ensure_registered(self, reg: NodeRegistration) -> bool:
        if self.registered:
            return True
        if not self._due():
            return False
        try:
            r = self.client.post("/v1/nodes/register", json=reg.model_dump(mode="json"))
        except httpx.TransportError as exc:
            self._fail(str(exc))
            return False
        if r.status_code == 409:
            raise RegistrationConflict(
                f"The server already has node '{reg.node_id}' registered with a different key "
                "(was the key file deleted?). Choose a new node_id in your config, or delete "
                "the server's database (data/kuulo.db) if this is a local demo."
            )
        if r.status_code >= 500:
            self._fail(f"HTTP {r.status_code}")
            return False
        r.raise_for_status()
        self.registered = True
        self._ok()
        log.info("registered as %s", reg.node_id)
        return True

    def send(self, path: str, msg: BaseModel) -> None:
        if len(self._pending) >= self.max_pending:
            idx = next((i for i, (p, _) in enumerate(self._pending) if p == HEARTBEATS), 0)
            del self._pending[idx]
            self.dropped += 1
            log.warning("uplink queue full; dropped 1 message (%d dropped so far)", self.dropped)
        self._pending.append((path, msg))

    def flush(self, force: bool = False) -> None:
        if not self.registered or (not force and not self._due()):
            return
        while self._pending:
            path, msg = self._pending[0]
            try:
                r = self.client.post(
                    path, content=msg.model_dump_json(),
                    headers={"content-type": "application/json"},
                )
            except httpx.TransportError as exc:
                self._fail(str(exc))
                return
            if r.status_code >= 500:
                self._fail(f"HTTP {r.status_code}")
                return
            if r.status_code == 401:  # server forgot us (fresh database): register again
                self.registered = False
                return
            self._pending.popleft()
            if r.status_code >= 400:
                self.dropped += 1
                log.error("server rejected %s: HTTP %d %s", path, r.status_code, r.text[:200])
        self._ok()
```

- [ ] **Step 4: Run tests**

Run: `uv run --no-sync pytest node/tests/test_uplink.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add node/src/kuulo_node/uplink.py node/tests/test_uplink.py
git commit -m "feat(node): uplink with registration, ordered retry queue, backoff, clear key-conflict error"
```

---

### Task 7: Runner, CLI and end-to-end test

**Files:**
- Create: `node/src/kuulo_node/runner.py`, `node/src/kuulo_node/cli.py`
- Test: `node/tests/test_runner.py`, `node/tests/test_e2e.py`

**Interfaces:**
- Consumes: everything from Tasks 2 to 6; `kuulo_server.app.create_app`, `kuulo_server.config.Settings(db_path=..., tick_interval_s=None)` (tests only).
- Produces:
  - `RunStats` dataclass: `windows: int = 0, observations: int = 0, heartbeats: int = 0`
  - `NodeRunner(cfg, keys, classifier, uplink, *, now=utc_now, monotonic=time.monotonic, heartbeat_every_s=60.0, print_scores=False, out=print)` with `.run(blocks: Iterable[AudioBlock]) -> RunStats`, `.health: MicHealth`
  - `format_scores(t: float, drone: float, scores: dict[str, float]) -> str`
  - CLI `kuulo-node run [--config PATH] [--input WAV] [--speed X] [--print-scores]`; exit code 0 ok, 1 registration conflict, 2 missing model or config error

- [ ] **Step 1: Write the failing tests `node/tests/test_runner.py`**

```python
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
```

- [ ] **Step 2: Write the failing end-to-end test `node/tests/test_e2e.py`**

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run --no-sync pytest node/tests/test_runner.py node/tests/test_e2e.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kuulo_node.runner'`

- [ ] **Step 4: Implement `node/src/kuulo_node/runner.py`**

```python
"""Wires audio -> windows -> classifier -> detector -> uplink, plus heartbeats and mic health."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation
from kuulo_protocol.signing import sign

from . import __version__
from .acoustic import AcousticMeter
from .audio import MIC_HELP, AudioBlock, MicHealth, Windower
from .classify import Classifier, drone_score
from .config import NodeConfig
from .detector import Detector, utc_now
from .keys import NodeKeys

log = logging.getLogger("kuulo.node")


@dataclass
class RunStats:
    windows: int = 0
    observations: int = 0
    heartbeats: int = 0


def format_scores(t: float, drone: float, scores: dict[str, float]) -> str:
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return f"{t:8.2f}s  drone={drone:.2f}  | " + ", ".join(f"{n} {p:.2f}" for n, p in top)


class NodeRunner:
    def __init__(
        self,
        cfg: NodeConfig,
        keys: NodeKeys,
        classifier: Classifier,
        uplink,
        *,
        now: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        heartbeat_every_s: float = 60.0,
        print_scores: bool = False,
        out: Callable[[str], None] = print,
    ) -> None:
        self.cfg, self.keys, self.classifier, self.uplink = cfg, keys, classifier, uplink
        self._now, self._monotonic = now, monotonic
        self._heartbeat_every_s = heartbeat_every_s
        self._print_scores, self._out = print_scores, out
        self.health = MicHealth()
        self._windower = Windower()
        self._meter = AcousticMeter()
        self._detector = Detector(cfg, keys, now=now)
        self._last_heartbeat: float | None = None
        self.stats = RunStats()

    def _registration(self) -> NodeRegistration:
        return NodeRegistration(
            node_id=self.cfg.node_id, public_key=self.keys.public_key,
            location=self.cfg.location, time_quality=self.cfg.time_quality,
        )

    def _emit(self, obs: Observation | None) -> None:
        if obs is not None:
            self.uplink.send("/v1/observations", obs)
            self.stats.observations += 1
            log.info("detection %s %s conf=%.2f", obs.event.phase.value.upper(),
                     str(obs.event.detection_id)[:8], obs.detection.confidence)

    def _heartbeat(self) -> None:
        hb = Heartbeat(
            node_id=self.cfg.node_id, sent_at=self._now(), software_version=__version__,
            mic_ok=self.health.ok, queue_depth=self.uplink.pending_count,
        )
        self.uplink.send("/v1/heartbeats", sign(hb, self.keys.private_key))
        self.stats.heartbeats += 1
        self._last_heartbeat = self._monotonic()

    def _maybe_heartbeat(self) -> None:
        if self._last_heartbeat is None or (
            self._monotonic() - self._last_heartbeat >= self._heartbeat_every_s
        ):
            self._heartbeat()

    def _process(self, block: AudioBlock) -> None:
        if self.health.update(block):
            log.warning(MIC_HELP)
        for window in self._windower.push(block):
            if window.gap_before:
                self._emit(self._detector.reset())
            scores = self.classifier.score(window.samples)
            score = drone_score(scores, self.cfg.weights)
            if self._print_scores:
                self._out(format_scores(window.t, score, scores))
            self._emit(self._detector.process(window.t, score, self._meter.measure(window.samples)))
            self.stats.windows += 1

    def run(self, blocks: Iterable[AudioBlock]) -> RunStats:
        reg = self._registration()
        self.uplink.ensure_registered(reg)
        try:
            for block in blocks:
                self._process(block)
                self._maybe_heartbeat()
                self.uplink.ensure_registered(reg)
                self.uplink.flush()
        except KeyboardInterrupt:
            log.info("stopping")
        self._emit(self._detector.reset())
        self._heartbeat()
        self.uplink.ensure_registered(reg)
        self.uplink.flush(force=True)
        if self.uplink.pending_count:
            log.warning("%d messages were not delivered", self.uplink.pending_count)
        return self.stats
```

- [ ] **Step 5: Implement `node/src/kuulo_node/cli.py`**

```python
"""`kuulo-node run`: listen (or replay a wav) and report drone detections to a Kuulo server."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import httpx

from .audio import mic_source, wav_source
from .config import ConfigError, load_config
from .keys import load_or_create_keys
from .runner import NodeRunner
from .uplink import RegistrationConflict, Uplink

log = logging.getLogger("kuulo.node")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuulo-node")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="run the node")
    run_p.add_argument("--config", type=Path, default=Path("node/config.local.toml"))
    run_p.add_argument("--input", type=Path, help="replay a wav file instead of the microphone")
    run_p.add_argument("--speed", type=float, default=1.0, help="wav replay speed; 0 = max")
    run_p.add_argument("--print-scores", action="store_true", help="print scores per window")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    if not cfg.model_path.exists() or not cfg.class_map_path.exists():
        log.error("YAMNet model not found at %s. Run `make model` first.", cfg.model_path)
        return 2
    from .classify import YamnetClassifier

    try:
        classifier = YamnetClassifier(cfg.model_path, cfg.class_map_path, cfg.weights)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    keys = load_or_create_keys(cfg.key_file)
    with httpx.Client(base_url=cfg.server_url, timeout=5.0) as client:
        runner = NodeRunner(cfg, keys, classifier, Uplink(client), print_scores=args.print_scores)
        if args.input:
            blocks = wav_source(args.input, speed=args.speed)
        else:
            blocks = mic_source(on_error=runner.health.mark_failed)
        try:
            stats = runner.run(blocks)
        except RegistrationConflict as exc:
            log.error("%s", exc)
            return 1
    log.info("done: %d windows, %d observations", stats.windows, stats.observations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the node tests**

Run: `uv run --no-sync pytest node/tests -q`
Expected: all pass (the YAMNet test is skipped if the model is absent)

- [ ] **Step 7: Run the full suite and lint**

Run: `make test`
Expected: ruff clean, all pytest and vitest pass

- [ ] **Step 8: Commit**

```bash
git add node/src/kuulo_node/runner.py node/src/kuulo_node/cli.py node/tests/test_runner.py node/tests/test_e2e.py
git commit -m "feat(node): runner with heartbeats and mic health, kuulo-node CLI, end-to-end test"
```

---

### Task 8: Real-model tuning with the user

A human is required: this needs the laptop microphone and real sounds.

**Files:**
- Modify: `node/config.example.toml` (weights, threshold, only if tuning changes them)

- [ ] **Step 1: Start the stack.** Ask the user to run `make dev` in one terminal, and `make node SCORES=1` in a second. On first run, macOS asks for microphone permission for the terminal app; the user must click Allow.
- [ ] **Step 2: Collect scores.** Ask the user to produce, about 20 s each: silence, speech, typing, a fan or traffic, and a hobby-drone recording played through the speakers, or a real drone. The user pastes the `--print-scores` output.
- [ ] **Step 3: Tune.** Choose `threshold` and `[weights]` so that the drone clip reaches the threshold on most windows and the other sounds stay below it. Keep "Propeller, airscrew" as the highest weight. If a weight or threshold change is needed, edit `node/config.example.toml`, and tell the user to delete `node/config.local.toml` so that `make node` re-copies it.
- [ ] **Step 4: Verify.** With `make node` running, play the drone clip. Expected: a TENTATIVE track appears at the University of Helsinki main building within about 3 s; after 5 s of quiet, the log shows END.
- [ ] **Step 5: Record the results** (sound → typical drone score → triggered yes/no) for the README's "How well it works" table in Task 9.
- [ ] **Step 6: Commit** (only if the config changed)

```bash
git add node/config.example.toml
git commit -m "tune(node): YAMNet class weights and threshold from laptop-mic trials"
```

---

### Task 9: README, LICENSE, CI and pre-publish check

**Files:**
- Create: `LICENSE`, `.github/workflows/ci.yml`
- Modify: `README.md` (rewrite), `Makefile` (add `prepublish`)

- [ ] **Step 1: LICENSE.** Fetch the AGPL-3.0 text into `LICENSE`:

Run: `curl -fsSL https://raw.githubusercontent.com/spdx/license-list-data/main/text/AGPL-3.0-only.txt -o LICENSE && head -3 LICENSE`
Expected: starts with "GNU AFFERO GENERAL PUBLIC LICENSE" and "Version 3, 19 November 2007"

- [ ] **Step 2: CI workflow `.github/workflows/ci.yml`**

```yaml
name: ci
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: sudo apt-get update && sudo apt-get install -y libportaudio2
      - run: uv sync
      - run: uv run --no-sync ruff check .
      - run: uv run --no-sync pytest
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: dashboard/package-lock.json
      - run: npm ci
        working-directory: dashboard
      - run: npx vitest run && npm run build
        working-directory: dashboard
```

- [ ] **Step 3: Pre-publish check in the `Makefile`**

```make
prepublish:
	@! git ls-files | grep -E '\.(wav|flac|mp3|tflite|onnx|db|key)$$|(^|/)data/|\.local\.toml$$' \
		|| (echo "FAIL: forbidden files are tracked (see above)"; exit 1)
	@echo "Tracked coordinates (must all be simulated or the demo location 60.1694, 24.9490):"
	@git grep -nE 'lat[" =:]+[0-9]{2}\.[0-9]{3}' -- ':!docs' | cut -c1-120
	@echo "OK: no audio, models, keys, databases or local configs are tracked."
```

Run: `make prepublish`
Expected: ends with `OK: …`; the coordinates listed are only the sim scenarios, test fixtures and `node/config.example.toml`.

- [ ] **Step 4: Rewrite `README.md`**

Write it with exactly these sections, in this order. Where content is quoted it is required verbatim; the rest is written from the spec.

1. `# Kuulo`, then the one-line pitch: "Civic acoustic drone detection for Finnish cities: cheap microphones that hear hobby drones, and a fusion layer that turns many noisy detections into tracks you can trust." Then a `![demo](docs/demo.gif)` line, with an HTML comment above it: `<!-- record with the node + dashboard running; see "Demo" below -->`.
2. `## Why`: 3 or 4 sentences taken from spec §1 of `2026-09-24-kuulo-milestone1-design.md` (radar gaps for small low drones, city interiors uncovered, false alarms are costly).
3. `## How it works`: a `text` code-block diagram:

   ```text
   laptop mic / wav ─► kuulo-node ─► POST /v1/observations (Ed25519-signed) ─► server ─► fusion ─► dashboard
                      YAMNet (AudioSet) ─► drone score ─► 3-of-5 smoothing        SQLite   tracks   MapLibre + Blueprint
   ```

   Follow it with one short paragraph per stage: node, server, fusion (single node = TENTATIVE, agreement between nodes = CONFIRMED, silent neighbours lower confidence), dashboard.
4. `## Quick start`: requirements (uv, Node 20+); `make setup`, `(cd dashboard && npm install)`, `make model` (with a note: "downloads the ~4 MB YAMNet model into `data/models/`; model files are never committed"), `make dev`, then `make node` in a second terminal. Platform notes: "macOS: the first run asks for microphone access for your terminal app; if you denied it, enable it in System Settings → Privacy & Security → Microphone." and "Linux: `sudo apt-get install libportaudio2`." Also `make node INPUT=path/to/clip.wav` and `make sim` for the multi-node simulator.
5. `## How well it works`: the table from Task 8, Step 5, then this line: "Step A is zero-training YAMNet. It also responds to propeller aircraft and some engines; a trained head (Step B) is the planned fix."
6. `## Known limits`: single-laptop demo gives TENTATIVE tracks only; location is the node's configured position (hundreds of metres coarse); Step A false positives as above; localhost only, no authentication, do not expose to the internet.
7. `## Privacy`: raw audio never leaves the node's memory; the node reports only its configured location (the demo uses the University of Helsinki main building, not a real deployment); no model weights, audio or keys are in the repo.
8. `## Development`: `make test`, the repo layout (protocol/, server/, sim/, node/, dashboard/, ml/), and a link to the specs and plans in `docs/superpowers/`.
9. `## Licence`: "AGPL-3.0. See LICENSE. Models and datasets: see ml/DATASETS.md."

- [ ] **Step 5: Verify**

Run: `make test && make prepublish`
Expected: all green; `OK: …`

- [ ] **Step 6: Commit**

```bash
git add LICENSE .github/workflows/ci.yml README.md Makefile
git commit -m "docs: full README, AGPL-3.0 licence, CI workflow, pre-publish check"
```

- [ ] **Step 7: Hand off to the user.** Tell them: record the GIF (dashboard plus a drone clip playing) to `docs/demo.gif`; run `make prepublish`; push; then make the repo public on GitHub (Settings → General → Danger Zone → Change visibility), remembering that clones made while it is public cannot be recalled.

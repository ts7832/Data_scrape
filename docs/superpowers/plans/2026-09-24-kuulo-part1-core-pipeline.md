# Kuulo Part 1 — Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A locally runnable pipeline — signed protocol, ingest server, scenario simulator, basic fusion and a live Helsinki map — where a simulated drone pass becomes a confirmed track on the dashboard.

**Architecture:** A uv workspace with three Python packages (`protocol/`, `server/`, `sim/`) plus a Vite/React dashboard. Every component speaks the Pydantic models in `kuulo_protocol`; every message is Ed25519-signed. The server (FastAPI + SQLite) validates, stores, runs a pluggable fusion engine and pushes live events over a WebSocket. The simulator produces the same signed messages real nodes will, through the real HTTP API, and also drives in-process scenario tests with a fake clock.

**Tech Stack:** Python 3.12, uv, Pydantic v2, PyNaCl, FastAPI, Uvicorn, SQLAlchemy 2.0 (SQLite, WAL), httpx, PyYAML, pytest, ruff; Node 26 + npm, Vite, React 18+, TypeScript, MapLibre GL, vitest, json-schema-to-typescript.

**Spec:** `docs/superpowers/specs/2026-09-24-kuulo-milestone1-design.md` (read it before starting).

**Scope of this plan (Part 1):** spec §§ 2, 3, 4.1–4.3, 4.5, 6 (minus traces), 7, 8, 9 (minus FeatureTraces and the `long_event` scenario), 11 and 12 for those components. Build-order steps 1–5.
**Deferred to Part 2 plan:** node software (§5), FeatureTraces (§4.4, §5.1, trace endpoints/tables, trace requests created on confirmation in §7 step 5, the `long_event` scenario), ML (§10), CI, full README/demo GIF, LICENSE file.

**Deliberate spec clarifications (decided here, not in the spec):**
- "Accept older messages but mark them late": the late threshold is **60 s** (`late_after_s`).
- Simulator "accelerated mode" = `time_scale` factor that compresses all timestamps (a `--speed 10` run of a 200 s scenario takes 20 s).
- `NodeView.uncorroborated_rate_24h` is `null` for a node with no drone detections in 24 h.

## Global Constraints

- Python `>=3.12,<3.13` via `uv` (run `uv python pin 3.12`); all Python commands run from the repo root as `uv run ...`.
- Node.js 26 / npm 11 are installed via Homebrew; dashboard commands run inside `dashboard/`.
- The server binds to `127.0.0.1` only. No auth in M1; never expose it to the internet.
- Never commit audio, model weights, datasets or databases (`data/` is gitignored). Never commit `.claude/settings.local.json`. Never run `git push`.
- Every Observation and Heartbeat is Ed25519-signed over canonical JSON: `json.dumps(model_dump(mode="json", exclude={"signature"}), sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, UTF-8.
- All timestamps are timezone-aware; they are normalised to UTC with millisecond precision. A naive timestamp is a validation error.
- Validation errors return HTTP **400** (not FastAPI's default 422); bad signature or unknown node **401**; key conflict **409**.
- Server defaults: `future_tolerance_s=30`, `late_after_s=60`; node status `online` < 120 s since last heartbeat, `stale` < 300 s, else `offline` (never heard = `offline`).
- Fusion defaults: `group_radius_m=2000`, `group_window_s=10`, `base_range_m=300`, `gate_m=1000`, `gate_s=10`, `close_after_s=30`, `expected_hearing_m=500`, `silent_penalty=0.7`, `smoothing_alpha=0.5`.
- Smoother defaults: `threshold=0.5`, start when ≥3 of last 5 windows are above threshold, `end_after_s=5`, `update_every_s=5`.
- Every commit message ends with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the executing model may substitute its own name).
- Line length 100; `uv run ruff check .` must pass before each commit.

## Review Focus

1. **A signed message that travels through JSON** (sub-millisecond or non-UTC timestamps, non-ASCII node ids) must still verify on the server → test in Task 2.
2. **Re-registering an existing `node_id` with a different public key** must be refused with 409, otherwise anyone could take over a node's identity → test in Task 3.
3. **A server restart with open tracks in the database** must close them at startup, not leave "confirmed" drones on the map forever → test in Task 8.
4. **A stalled dashboard WebSocket client** must not block or break publishing to other clients; it gets a `resync` event instead → test in Task 4.
5. **Live events that arrive while the dashboard is loading its snapshot** must not be lost or overwritten → test in Task 7.

---

## File Structure

```
pyproject.toml                      uv workspace root (virtual), dev deps, pytest/ruff config
.python-version                     3.12
Makefile                            dev / server / dashboard / sim / test targets
README.md                           short development quick start (full README in Part 2)

protocol/pyproject.toml
protocol/src/kuulo_protocol/__init__.py
protocol/src/kuulo_protocol/models.py     wire messages: Observation, Heartbeat, Track, NodeRegistration
protocol/src/kuulo_protocol/api.py        API views: NodeStatus, NodeView, TrackDetail, LiveEvent, IngestResult
protocol/src/kuulo_protocol/signing.py    canonical JSON, keypairs, sign, verify
protocol/src/kuulo_protocol/schema.py     JSON Schema export for the dashboard
protocol/src/kuulo_protocol/geo.py        local metric projection + distances
protocol/src/kuulo_protocol/smoothing.py  DetectionSmoother (shared by sim now, node in Part 2)
protocol/src/kuulo_protocol/testing.py    make_observation / make_heartbeat factories for tests
protocol/tests/test_models.py
protocol/tests/test_signing.py
protocol/tests/test_geo.py
protocol/tests/test_smoothing.py

server/pyproject.toml
server/src/kuulo_server/__init__.py
server/src/kuulo_server/config.py         Settings dataclass, utc_now
server/src/kuulo_server/db.py             SQLAlchemy tables, session factory, as_utc
server/src/kuulo_server/ingest.py         register / observation / heartbeat ingest rules
server/src/kuulo_server/status.py         node status + NodeView building
server/src/kuulo_server/live.py           LiveHub (WebSocket fan-out)
server/src/kuulo_server/tracks.py         persist tracks, evidence queries, uncorroborated rate
server/src/kuulo_server/fusion/__init__.py
server/src/kuulo_server/fusion/base.py    FusionEngine / FusionContext protocols, NodeInfo, TrackUpdate
server/src/kuulo_server/fusion/basic.py   BasicFusion
server/src/kuulo_server/fusion/context.py DbFusionContext
server/src/kuulo_server/fusion/loader.py  load_engine("module:Class")
server/src/kuulo_server/app.py            create_app(): routes, lifespan, tick
server/src/kuulo_server/main.py           uvicorn entry: app = create_app()
server/src/kuulo_server/testing.py        T0, FakeClock, register, post_signed (test helpers)
server/tests/conftest.py                  app/client/node fixtures
server/tests/test_ingest.py
server/tests/test_status_live.py
server/tests/test_fusion_basic.py
server/tests/test_fusion_server.py
server/tests/scenario_harness.py
server/tests/test_scenarios.py

sim/pyproject.toml
sim/src/kuulo_sim/__init__.py
sim/src/kuulo_sim/scenario.py             scenario YAML models, load/resolve, drone_position
sim/src/kuulo_sim/physics.py              propagation, detection probability, clock error
sim/src/kuulo_sim/engine.py               SimulationRun: registrations, messages, truth
sim/src/kuulo_sim/runner.py               realtime HTTP runner
sim/src/kuulo_sim/cli.py                  kuulo-sim run / list
sim/src/kuulo_sim/scenarios/*.yaml        single_node, helsinki_pass, false_alarm, two_drones, node_failure
sim/tests/test_scenario_physics.py
sim/tests/test_engine.py
sim/tests/test_runner_cli.py

dashboard/package.json, tsconfig.json, vite.config.ts, index.html
dashboard/src/main.tsx, App.tsx, styles.css
dashboard/src/api/schema.json, types.ts   generated from kuulo_protocol
dashboard/src/api/client.ts               fetchSnapshot, fetchTrackDetail
dashboard/src/state/reducer.ts            State, Action, reducer
dashboard/src/state/live.ts               backoffDelay, startLive
dashboard/src/map/geo.ts                  circlePolygon
dashboard/src/map/layers.ts               state → GeoJSON
dashboard/src/components/MapView.tsx
dashboard/src/components/SidePanel.tsx
dashboard/src/state/reducer.test.ts, live.test.ts, dashboard/src/map/map.test.ts
```

---

### Task 1: Workspace and protocol models

**Files:**
- Create: `pyproject.toml`, `.python-version` (via `uv python pin`), `protocol/pyproject.toml`, `protocol/src/kuulo_protocol/__init__.py`, `protocol/src/kuulo_protocol/models.py`, `protocol/src/kuulo_protocol/testing.py`
- Test: `protocol/tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces (module `kuulo_protocol.models`): `SCHEMA_VERSION`, enums `SourceType`, `TimeQuality`, `Label`, `Phase`, `TrackStatus`, constant `DRONE_LABELS`, function `to_utc_ms(datetime) -> datetime`, base `WireModel`, models `Source`, `GeoPoint`, `SensorLocation`, `Detection`, `EventRef`, `Acoustic`, `Observation`, `Heartbeat`, `Velocity`, `Track`, `NodeRegistration`. Module `kuulo_protocol.testing`: `make_observation(...)`, `make_heartbeat(...)` (signatures below).

- [ ] **Step 1: Create the workspace root**

`pyproject.toml`:
```toml
[project]
name = "kuulo"
version = "0.1.0"
description = "Civic acoustic drone-detection network"
requires-python = ">=3.12,<3.13"
dependencies = ["kuulo-protocol"]

[tool.uv]
package = false

[tool.uv.sources]
kuulo-protocol = { workspace = true }

[tool.uv.workspace]
members = ["protocol"]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.6", "httpx>=0.27"]

[tool.pytest.ini_options]
addopts = "--import-mode=importlib -q"
testpaths = ["protocol/tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

`protocol/pyproject.toml`:
```toml
[project]
name = "kuulo-protocol"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["pydantic>=2.8", "pynacl>=1.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kuulo_protocol"]
```

`protocol/src/kuulo_protocol/__init__.py`:
```python
"""Kuulo wire protocol: messages, signing and shared helpers."""
```

Run:
```bash
uv python pin 3.12
uv sync
```
Expected: `.python-version` contains `3.12`; `uv sync` creates `.venv/` and installs pydantic, pynacl, pytest, ruff, httpx.

- [ ] **Step 2: Write the failing tests**

`protocol/tests/test_models.py`:
```python
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kuulo_protocol.models import (
    DRONE_LABELS,
    Label,
    Observation,
    Phase,
    to_utc_ms,
)
from kuulo_protocol.testing import make_heartbeat, make_observation

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_observation_json_round_trip():
    obs = make_observation(observed_at=T0)
    again = Observation.model_validate_json(obs.model_dump_json())
    assert again == obs


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError, match="timezone"):
        make_observation(observed_at=datetime(2026, 9, 24, 12, 0))


def test_timestamp_normalised_to_utc_milliseconds():
    helsinki = timezone(timedelta(hours=3))
    ts = datetime(2026, 9, 24, 14, 0, 0, 123456, tzinfo=helsinki)
    assert to_utc_ms(ts) == datetime(2026, 9, 24, 11, 0, 0, 123000, tzinfo=UTC)
    assert make_observation(observed_at=ts).observed_at.microsecond == 123000


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        make_observation(observed_at=T0, confidence=1.5)


def test_unknown_field_rejected():
    data = make_observation(observed_at=T0).model_dump(mode="json")
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        Observation.model_validate(data)


def test_factory_overrides():
    det = uuid4()
    obs = make_observation(
        "n7", observed_at=T0, label=Label.BIRD, phase=Phase.END, detection_id=det, snr_db=None
    )
    assert obs.source.id == "n7"
    assert obs.detection.label is Label.BIRD
    assert obs.event.phase is Phase.END
    assert obs.event.detection_id == det
    assert obs.acoustic is None


def test_drone_labels():
    assert Label.DRONE_MULTIROTOR in DRONE_LABELS
    assert Label.BIRD not in DRONE_LABELS


def test_heartbeat_factory():
    hb = make_heartbeat("n1", sent_at=T0)
    assert hb.node_id == "n1" and hb.mic_ok and hb.queue_depth == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest protocol/tests/test_models.py`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'kuulo_protocol.models'`.

- [ ] **Step 4: Implement the models**

`protocol/src/kuulo_protocol/models.py`:
```python
"""Wire-format messages shared by every Kuulo component."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"


class SourceType(StrEnum):
    ACOUSTIC_NODE = "acoustic_node"
    SIMULATED_NODE = "simulated_node"
    CITIZEN_REPORT = "citizen_report"
    REMOTE_ID = "remote_id"
    ADSB = "adsb"
    EXTERNAL_NETWORK = "external_network"


class TimeQuality(StrEnum):
    GPS = "gps"
    NTP = "ntp"
    MANUAL = "manual"


class Label(StrEnum):
    DRONE_MULTIROTOR = "drone_multirotor"
    DRONE_FIXEDWING_ENGINE = "drone_fixedwing_engine"
    AIRCRAFT = "aircraft"
    BIRD = "bird"
    UNKNOWN = "unknown"


DRONE_LABELS = frozenset({Label.DRONE_MULTIROTOR, Label.DRONE_FIXEDWING_ENGINE})


class Phase(StrEnum):
    START = "start"
    UPDATE = "update"
    END = "end"


class TrackStatus(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DOWNGRADED = "downgraded"
    CLOSED = "closed"


def to_utc_ms(value: datetime) -> datetime:
    """Require a timezone-aware timestamp; return it in UTC with millisecond precision."""
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    value = value.astimezone(UTC)
    return value.replace(microsecond=value.microsecond // 1000 * 1000)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(WireModel):
    type: SourceType
    id: str = Field(min_length=1, max_length=64)


class GeoPoint(WireModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class SensorLocation(GeoPoint):
    accuracy_m: float = Field(ge=0)


class Detection(WireModel):
    label: Label
    confidence: float = Field(ge=0, le=1)
    bearing_deg: float | None = Field(default=None, ge=0, lt=360)


class EventRef(WireModel):
    detection_id: UUID
    phase: Phase


class Acoustic(WireModel):
    snr_db: float
    peak_freq_hz: float = Field(ge=0)


class Observation(WireModel):
    """A source noticed something. Used by every source type."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    observation_id: UUID = Field(default_factory=uuid4)
    source: Source
    observed_at: datetime
    time_quality: TimeQuality
    sensor_location: SensorLocation
    detection: Detection
    event: EventRef
    acoustic: Acoustic | None = None
    signature: str = ""

    @field_validator("observed_at")
    @classmethod
    def _observed_at_utc(cls, value: datetime) -> datetime:
        return to_utc_ms(value)


class Heartbeat(WireModel):
    """A node saying it is alive, so silence can be told apart from failure."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION
    node_id: str = Field(min_length=1, max_length=64)
    sent_at: datetime
    software_version: str
    mic_ok: bool
    cpu_temp_c: float | None = None
    queue_depth: int = Field(ge=0)
    signature: str = ""

    @field_validator("sent_at")
    @classmethod
    def _sent_at_utc(cls, value: datetime) -> datetime:
        return to_utc_ms(value)


class Velocity(WireModel):
    speed_mps: float = Field(ge=0)
    heading_deg: float = Field(ge=0, lt=360)


class Track(WireModel):
    """The server's belief that a drone is at a place, with its evidence."""

    track_id: UUID
    status: TrackStatus
    label: Label
    confidence: float = Field(ge=0, le=1)
    position: GeoPoint
    uncertainty_m: float = Field(ge=0)
    velocity: Velocity | None = None
    first_seen: datetime
    last_seen: datetime
    observation_ids: list[UUID]
    silent_neighbour_ids: list[str]


class NodeRegistration(WireModel):
    node_id: str = Field(min_length=1, max_length=64)
    public_key: str = Field(min_length=1)
    location: SensorLocation
    time_quality: TimeQuality
```

`protocol/src/kuulo_protocol/testing.py`:
```python
"""Factories for building valid messages in tests (unsigned unless a key is given)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from .models import (
    Acoustic,
    Detection,
    EventRef,
    Heartbeat,
    Label,
    Observation,
    Phase,
    SensorLocation,
    Source,
    SourceType,
    TimeQuality,
)


def make_observation(
    node_id: str = "n1",
    *,
    observed_at: datetime,
    label: Label = Label.DRONE_MULTIROTOR,
    confidence: float = 0.9,
    lat: float = 60.1699,
    lon: float = 24.9384,
    snr_db: float | None = 12.0,
    phase: Phase = Phase.START,
    detection_id: UUID | None = None,
    time_quality: TimeQuality = TimeQuality.NTP,
) -> Observation:
    return Observation(
        source=Source(type=SourceType.SIMULATED_NODE, id=node_id),
        observed_at=observed_at,
        time_quality=time_quality,
        sensor_location=SensorLocation(lat=lat, lon=lon, accuracy_m=10.0),
        detection=Detection(label=label, confidence=confidence),
        event=EventRef(detection_id=detection_id or uuid4(), phase=phase),
        acoustic=None if snr_db is None else Acoustic(snr_db=snr_db, peak_freq_hz=180.0),
    )


def make_heartbeat(node_id: str = "n1", *, sent_at: datetime, mic_ok: bool = True) -> Heartbeat:
    return Heartbeat(
        node_id=node_id,
        sent_at=sent_at,
        software_version="0.1.0",
        mic_ok=mic_ok,
        queue_depth=0,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest protocol/tests/test_models.py && uv run ruff check .`
Expected: `8 passed`; ruff prints `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version uv.lock protocol
git commit -m "feat(protocol): workspace and wire message models

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Signing, API views and JSON Schema export

**Files:**
- Create: `protocol/src/kuulo_protocol/signing.py`, `protocol/src/kuulo_protocol/api.py`, `protocol/src/kuulo_protocol/schema.py`
- Test: `protocol/tests/test_signing.py`

**Interfaces:**
- Consumes: `kuulo_protocol.models` (Task 1).
- Produces:
  - `kuulo_protocol.signing`: `canonical_bytes(msg: BaseModel) -> bytes`, `generate_keypair() -> tuple[str, str]` (private_b64, public_b64), `keypair_from_seed(seed: bytes) -> tuple[str, str]`, `sign(msg: M, private_key_b64: str) -> M` (returns a copy with `signature` set), `verify(msg: BaseModel, public_key_b64: str) -> bool` (never raises).
  - `kuulo_protocol.api`: `NodeStatus` (StrEnum online/stale/offline), `NodeView`, `TrackDetail`, `LiveEvent` (`type` ∈ observation/track/node_status/resync; `data` optional), `IngestResult` (`status` ∈ accepted/duplicate, `late: bool`).
  - `kuulo_protocol.schema`: `export_schema() -> dict`; `python -m kuulo_protocol.schema` prints it.

- [ ] **Step 1: Write the failing tests**

`protocol/tests/test_signing.py`:
```python
from datetime import UTC, datetime, timedelta, timezone

from kuulo_protocol.api import LiveEvent, NodeStatus
from kuulo_protocol.models import Observation
from kuulo_protocol.schema import export_schema
from kuulo_protocol.signing import (
    canonical_bytes,
    generate_keypair,
    keypair_from_seed,
    sign,
    verify,
)
from kuulo_protocol.testing import make_heartbeat, make_observation

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_sign_and_verify():
    priv, pub = generate_keypair()
    obs = sign(make_observation(observed_at=T0), priv)
    assert obs.signature
    assert verify(obs, pub)


def test_tampered_message_fails():
    priv, pub = generate_keypair()
    obs = sign(make_observation(observed_at=T0, confidence=0.4), priv)
    forged = obs.model_copy(update={"detection": obs.detection.model_copy(update={"confidence": 0.99})})
    assert not verify(forged, pub)


def test_wrong_key_fails():
    priv, _ = generate_keypair()
    _, other_pub = generate_keypair()
    assert not verify(sign(make_observation(observed_at=T0), priv), other_pub)


def test_garbage_signature_returns_false_not_exception():
    _, pub = generate_keypair()
    obs = make_observation(observed_at=T0).model_copy(update={"signature": "not base64!!"})
    assert verify(obs, pub) is False
    assert verify(make_observation(observed_at=T0), pub) is False  # empty signature


def test_signature_survives_json_round_trip_with_awkward_inputs():
    # Review Focus 1: sub-millisecond, non-UTC timestamp and a non-ASCII node id.
    priv, pub = generate_keypair()
    ts = datetime(2026, 9, 24, 15, 0, 0, 987654, tzinfo=timezone(timedelta(hours=3)))
    obs = sign(make_observation("solmu-ÄÖ-1", observed_at=ts), priv)
    received = Observation.model_validate_json(obs.model_dump_json().encode("utf-8"))
    assert verify(received, pub)


def test_canonical_bytes_exclude_signature_and_sort_keys():
    obs = make_observation(observed_at=T0)
    a = canonical_bytes(obs)
    b = canonical_bytes(obs.model_copy(update={"signature": "xyz"}))
    assert a == b
    assert b'"signature"' not in a
    assert a.index(b'"acoustic"') < a.index(b'"detection"')


def test_heartbeat_signing():
    priv, pub = generate_keypair()
    assert verify(sign(make_heartbeat(sent_at=T0), priv), pub)


def test_keypair_from_seed_is_deterministic():
    assert keypair_from_seed(b"a" * 32) == keypair_from_seed(b"a" * 32)
    assert keypair_from_seed(b"a" * 32) != keypair_from_seed(b"b" * 32)


def test_live_event_resync_has_no_data():
    ev = LiveEvent(type="resync")
    assert ev.data is None
    assert NodeStatus.ONLINE == "online"


def test_schema_export_contains_all_models():
    schema = export_schema()
    for name in [
        "Observation", "Heartbeat", "Track", "NodeRegistration",
        "NodeView", "TrackDetail", "LiveEvent", "IngestResult",
    ]:
        assert name in schema["$defs"]
        assert schema["properties"][name] == {"$ref": f"#/$defs/{name}"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest protocol/tests/test_signing.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuulo_protocol.api'`.

- [ ] **Step 3: Implement signing**

`protocol/src/kuulo_protocol/signing.py`:
```python
"""Ed25519 signing over canonical JSON. Every node message is signed with the node's key."""

from __future__ import annotations

import base64
import binascii
import json
from typing import TypeVar

from nacl.exceptions import BadSignatureError, CryptoError
from nacl.signing import SigningKey, VerifyKey
from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def canonical_bytes(msg: BaseModel) -> bytes:
    """Deterministic bytes of a message, excluding its signature field."""
    data = msg.model_dump(mode="json", exclude={"signature"})
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text, validate=True)


def _pair(key: SigningKey) -> tuple[str, str]:
    return _b64(bytes(key)), _b64(bytes(key.verify_key))


def generate_keypair() -> tuple[str, str]:
    """Return (private_key_b64, public_key_b64)."""
    return _pair(SigningKey.generate())


def keypair_from_seed(seed: bytes) -> tuple[str, str]:
    """Deterministic keypair from a 32-byte seed (used by the simulator)."""
    return _pair(SigningKey(seed))


def sign(msg: M, private_key_b64: str) -> M:
    signature = SigningKey(_unb64(private_key_b64)).sign(canonical_bytes(msg)).signature
    return msg.model_copy(update={"signature": _b64(signature)})


def verify(msg: BaseModel, public_key_b64: str) -> bool:
    """True only for a valid signature by this key. Never raises on bad input."""
    signature = getattr(msg, "signature", "")
    if not signature:
        return False
    try:
        VerifyKey(_unb64(public_key_b64)).verify(canonical_bytes(msg), _unb64(signature))
    except (BadSignatureError, CryptoError, binascii.Error, ValueError, TypeError):
        return False
    return True
```

- [ ] **Step 4: Implement API views and schema export**

`protocol/src/kuulo_protocol/api.py`:
```python
"""Server API views: what the dashboard reads."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from .models import Observation, SensorLocation, TimeQuality, Track, WireModel


class NodeStatus(StrEnum):
    ONLINE = "online"
    STALE = "stale"
    OFFLINE = "offline"


class NodeView(WireModel):
    node_id: str
    location: SensorLocation
    time_quality: TimeQuality
    status: NodeStatus
    last_heartbeat_at: datetime | None = None
    mic_ok: bool | None = None
    software_version: str | None = None
    uncorroborated_rate_24h: float | None = None


class TrackDetail(WireModel):
    track: Track
    observations: list[Observation]
    silent_neighbours: list[NodeView]


class LiveEvent(WireModel):
    """One message on the /v1/live WebSocket. `resync` tells the client to reload its snapshot."""

    type: Literal["observation", "track", "node_status", "resync"]
    data: Observation | Track | NodeView | None = None


class IngestResult(WireModel):
    status: Literal["accepted", "duplicate"]
    late: bool = False
```

`protocol/src/kuulo_protocol/schema.py`:
```python
"""Export one JSON Schema containing every wire and API model (input for dashboard types)."""

from __future__ import annotations

import json
import sys

from pydantic.json_schema import models_json_schema

from .api import IngestResult, LiveEvent, NodeView, TrackDetail
from .models import Heartbeat, NodeRegistration, Observation, Track

EXPORTED = (
    Observation, Heartbeat, Track, NodeRegistration, NodeView, TrackDetail, LiveEvent, IngestResult,
)


def export_schema() -> dict:
    _, schema = models_json_schema(
        [(model, "serialization") for model in EXPORTED], title="KuuloSchema"
    )
    # json-schema-to-typescript only emits types reachable from the root,
    # so the root object references every exported model.
    schema["type"] = "object"
    schema["properties"] = {m.__name__: {"$ref": f"#/$defs/{m.__name__}"} for m in EXPORTED}
    schema["additionalProperties"] = False
    return schema


def main() -> None:
    json.dump(export_schema(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest protocol/tests && uv run ruff check . && uv run python -m kuulo_protocol.schema | head -5`
Expected: all protocol tests pass; ruff clean; the schema prints starting with `{` and `"$defs"`.

- [ ] **Step 6: Commit**

```bash
git add protocol
git commit -m "feat(protocol): Ed25519 signing, API views and JSON Schema export

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 3: Server — storage and signed ingest

**Files:**
- Modify: `pyproject.toml` (add `server` member)
- Create: `server/pyproject.toml`, `server/src/kuulo_server/__init__.py`, `server/src/kuulo_server/config.py`, `server/src/kuulo_server/db.py`, `server/src/kuulo_server/ingest.py`, `server/src/kuulo_server/app.py`, `server/src/kuulo_server/testing.py`
- Test: `server/tests/conftest.py`, `server/tests/test_ingest.py`

**Interfaces:**
- Consumes: `kuulo_protocol.models`, `kuulo_protocol.signing.verify/sign/generate_keypair`, `kuulo_protocol.api.IngestResult`, `kuulo_protocol.testing.make_observation`.
- Produces:
  - `kuulo_server.config`: `utc_now() -> datetime`, `@dataclass Settings(db_path: Path = Path("data/kuulo.db"), clock: Callable[[], datetime] = utc_now, tick_interval_s: float | None = 1.0, future_tolerance_s: float = 30.0, late_after_s: float = 60.0, fusion_engine: str = "kuulo_server.fusion.basic:BasicFusion", on_track_update: Callable[[Track], None] | None = None)`.
  - `kuulo_server.db`: `Base`, `NodeRow`, `ObservationRow`, `TrackRow`, `TrackObservationRow`, `make_session_factory(db_path: Path) -> sessionmaker`, `as_utc(dt: datetime | None) -> datetime | None`.
  - `kuulo_server.ingest`: `IngestError(status_code: int, reason: str)`, `register_node(session, reg, now) -> None`, `ingest_observation(session, obs, now, settings) -> IngestResult`, `ingest_heartbeat(session, hb, now, settings) -> NodeRow`.
  - `kuulo_server.app`: `create_app(settings: Settings | None = None) -> FastAPI`; `app.state.settings`, `app.state.sessions`.
  - `kuulo_server.testing`: `T0`, `FakeClock` (callable; `.set(dt)`, `.advance(seconds)`), `NodeKeys(node_id, private_key, public_key)`, `register(client, node_id, lat=..., lon=...) -> NodeKeys`, `post_signed(client, path, msg, private_key)`. Fixtures in `server/tests/conftest.py`: `clock`, `app`, `client`, `node`.

- [ ] **Step 1: Add the server package**

Edit root `pyproject.toml`: `dependencies = ["kuulo-protocol", "kuulo-server"]`, add `kuulo-server = { workspace = true }` under `[tool.uv.sources]`, set `members = ["protocol", "server"]`, set `testpaths = ["protocol/tests", "server/tests"]`.

`server/pyproject.toml`:
```toml
[project]
name = "kuulo-server"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "kuulo-protocol",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "sqlalchemy>=2.0",
]

[tool.uv.sources]
kuulo-protocol = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kuulo_server"]
```

`server/src/kuulo_server/__init__.py`:
```python
"""Kuulo ingest server: stores observations, runs fusion, pushes live updates."""
```

Run: `uv sync`
Expected: installs fastapi, uvicorn, sqlalchemy.

- [ ] **Step 2: Write test fixtures and failing tests**

`server/src/kuulo_server/testing.py` (test helpers live in the package so test modules can import them absolutely — test directories are not packages):
```python
"""Helpers for server tests and scenario harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from pydantic import BaseModel

from kuulo_protocol.models import NodeRegistration, SensorLocation, TimeQuality
from kuulo_protocol.signing import generate_keypair, sign

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


class FakeClock:
    def __init__(self, now: datetime):
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def set(self, now: datetime) -> None:
        self.now = now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@dataclass(frozen=True)
class NodeKeys:
    node_id: str
    private_key: str
    public_key: str


def register(client: TestClient, node_id: str, lat: float = 60.1699, lon: float = 24.9384) -> NodeKeys:
    priv, pub = generate_keypair()
    reg = NodeRegistration(
        node_id=node_id,
        public_key=pub,
        location=SensorLocation(lat=lat, lon=lon, accuracy_m=10),
        time_quality=TimeQuality.NTP,
    )
    response = client.post("/v1/nodes/register", json=reg.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    return NodeKeys(node_id, priv, pub)


def post_signed(client: TestClient, path: str, msg: BaseModel, private_key: str):
    return client.post(
        path,
        content=sign(msg, private_key).model_dump_json(),
        headers={"content-type": "application/json"},
    )
```

`server/tests/conftest.py`:
```python
import pytest
from fastapi.testclient import TestClient

from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.testing import T0, FakeClock, NodeKeys, register


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(T0)


@pytest.fixture
def app(tmp_path, clock):
    return create_app(Settings(db_path=tmp_path / "kuulo.db", clock=clock, tick_interval_s=None))


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def node(client) -> NodeKeys:
    return register(client, "n1")
```

`server/tests/test_ingest.py`:
```python
from datetime import timedelta

from kuulo_protocol.models import NodeRegistration, SensorLocation, TimeQuality
from kuulo_protocol.signing import generate_keypair
from kuulo_protocol.testing import make_observation

from kuulo_server.testing import T0, post_signed


def test_signed_observation_accepted(client, node):
    obs = make_observation(node.node_id, observed_at=T0)
    response = post_signed(client, "/v1/observations", obs, node.private_key)
    assert response.status_code == 200
    assert response.json() == {"status": "accepted", "late": False}


def test_retry_is_duplicate_not_double(client, node):
    obs = make_observation(node.node_id, observed_at=T0)
    post_signed(client, "/v1/observations", obs, node.private_key)
    again = post_signed(client, "/v1/observations", obs, node.private_key)
    assert again.status_code == 200
    assert again.json()["status"] == "duplicate"


def test_bad_signature_rejected(client, node):
    other_priv, _ = generate_keypair()
    obs = make_observation(node.node_id, observed_at=T0)
    response = post_signed(client, "/v1/observations", obs, other_priv)
    assert response.status_code == 401
    assert "signature" in response.json()["detail"]


def test_unsigned_rejected(client, node):
    obs = make_observation(node.node_id, observed_at=T0)
    response = client.post("/v1/observations", content=obs.model_dump_json(),
                           headers={"content-type": "application/json"})
    assert response.status_code == 401


def test_unknown_node_rejected(client):
    priv, _ = generate_keypair()
    obs = make_observation("ghost", observed_at=T0)
    response = post_signed(client, "/v1/observations", obs, priv)
    assert response.status_code == 401
    assert "unknown node" in response.json()["detail"]


def test_malformed_body_is_400(client, node):
    response = client.post("/v1/observations", json={"hello": "world"})
    assert response.status_code == 400


def test_future_timestamp_rejected(client, node, clock):
    obs = make_observation(node.node_id, observed_at=T0)
    clock.set(T0 - timedelta(seconds=31))
    response = post_signed(client, "/v1/observations", obs, node.private_key)
    assert response.status_code == 400
    assert "future" in response.json()["detail"]


def test_old_observation_accepted_but_late(client, node, clock):
    obs = make_observation(node.node_id, observed_at=T0)
    clock.advance(120)
    response = post_signed(client, "/v1/observations", obs, node.private_key)
    assert response.json() == {"status": "accepted", "late": True}


def test_reregister_with_different_key_is_conflict(client, node):
    # Review Focus 2: a node identity cannot be taken over by re-registering.
    _, attacker_pub = generate_keypair()
    reg = NodeRegistration(
        node_id=node.node_id, public_key=attacker_pub,
        location=SensorLocation(lat=60.0, lon=25.0, accuracy_m=5), time_quality=TimeQuality.NTP,
    )
    response = client.post("/v1/nodes/register", json=reg.model_dump(mode="json"))
    assert response.status_code == 409


def test_reregister_same_key_updates_location(client, node):
    reg = NodeRegistration(
        node_id=node.node_id, public_key=node.public_key,
        location=SensorLocation(lat=60.2, lon=25.0, accuracy_m=5), time_quality=TimeQuality.GPS,
    )
    assert client.post("/v1/nodes/register", json=reg.model_dump(mode="json")).status_code == 200
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_ingest.py`
Expected: collection error — `ModuleNotFoundError: No module named 'kuulo_server.app'`.

- [ ] **Step 4: Implement config and db**

`server/src/kuulo_server/config.py`:
```python
"""Server settings. Tests inject a fake clock and disable the background tick."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kuulo_protocol.models import Track


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Settings:
    db_path: Path = Path("data/kuulo.db")
    clock: Callable[[], datetime] = utc_now
    tick_interval_s: float | None = 1.0
    future_tolerance_s: float = 30.0
    late_after_s: float = 60.0
    fusion_engine: str = "kuulo_server.fusion.basic:BasicFusion"
    on_track_update: Callable[[Track], None] | None = None
```

`server/src/kuulo_server/db.py`:
```python
"""SQLite storage (WAL mode) via SQLAlchemy 2.0."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def as_utc(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; everything we store is UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class Base(DeclarativeBase):
    pass


class NodeRow(Base):
    __tablename__ = "nodes"
    node_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    public_key: Mapped[str] = mapped_column(String(64))
    lat: Mapped[float]
    lon: Mapped[float]
    accuracy_m: Mapped[float]
    time_quality: Mapped[str] = mapped_column(String(16))
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    software_version: Mapped[str | None] = mapped_column(String(32))
    mic_ok: Mapped[bool | None] = mapped_column(Boolean)
    queue_depth: Mapped[int | None] = mapped_column(Integer)


class ObservationRow(Base):
    __tablename__ = "observations"
    observation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(64), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    late: Mapped[bool] = mapped_column(Boolean)
    label: Mapped[str] = mapped_column(String(32))
    detection_id: Mapped[str] = mapped_column(String(36), index=True)
    raw_json: Mapped[str] = mapped_column(Text)


class TrackRow(Base):
    __tablename__ = "tracks"
    track_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    ever_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    raw_json: Mapped[str] = mapped_column(Text)


class TrackObservationRow(Base):
    __tablename__ = "track_observations"
    track_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    observation_id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)


def make_session_factory(db_path: Path) -> sessionmaker:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
```

- [ ] **Step 5: Implement ingest rules**

`server/src/kuulo_server/ingest.py`:
```python
"""Ingest rules: validate identity, verify signatures, deduplicate, check time, store."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from kuulo_protocol.api import IngestResult
from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation
from kuulo_protocol.signing import verify

from .config import Settings
from .db import NodeRow, ObservationRow


class IngestError(Exception):
    def __init__(self, status_code: int, reason: str):
        super().__init__(reason)
        self.status_code = status_code
        self.reason = reason


def register_node(session: Session, reg: NodeRegistration, now: datetime) -> None:
    row = session.get(NodeRow, reg.node_id)
    if row is not None and row.public_key != reg.public_key:
        raise IngestError(409, "node_id already registered with a different key")
    if row is None:
        row = NodeRow(node_id=reg.node_id, public_key=reg.public_key, registered_at=now)
        session.add(row)
    row.lat = reg.location.lat
    row.lon = reg.location.lon
    row.accuracy_m = reg.location.accuracy_m
    row.time_quality = reg.time_quality.value
    session.commit()


def _node_for(session: Session, node_id: str) -> NodeRow:
    row = session.get(NodeRow, node_id)
    if row is None:
        raise IngestError(401, f"unknown node: {node_id}")
    return row


def _check_not_future(ts: datetime, now: datetime, settings: Settings, field: str) -> None:
    if ts > now + timedelta(seconds=settings.future_tolerance_s):
        raise IngestError(400, f"{field} is in the future")


def ingest_observation(
    session: Session, obs: Observation, now: datetime, settings: Settings
) -> IngestResult:
    node = _node_for(session, obs.source.id)
    if not verify(obs, node.public_key):
        raise IngestError(401, "bad signature")
    _check_not_future(obs.observed_at, now, settings, "observed_at")
    if session.get(ObservationRow, str(obs.observation_id)) is not None:
        return IngestResult(status="duplicate")
    late = obs.observed_at < now - timedelta(seconds=settings.late_after_s)
    session.add(
        ObservationRow(
            observation_id=str(obs.observation_id),
            node_id=obs.source.id,
            observed_at=obs.observed_at,
            received_at=now,
            late=late,
            label=obs.detection.label.value,
            detection_id=str(obs.event.detection_id),
            raw_json=obs.model_dump_json(),
        )
    )
    session.commit()
    return IngestResult(status="accepted", late=late)


def ingest_heartbeat(session: Session, hb: Heartbeat, now: datetime, settings: Settings) -> NodeRow:
    node = _node_for(session, hb.node_id)
    if not verify(hb, node.public_key):
        raise IngestError(401, "bad signature")
    _check_not_future(hb.sent_at, now, settings, "sent_at")
    node.last_heartbeat_at = now
    node.software_version = hb.software_version
    node.mic_ok = hb.mic_ok
    node.queue_depth = hb.queue_depth
    session.commit()
    return node
```

- [ ] **Step 6: Implement the app (routes for this task)**

`server/src/kuulo_server/app.py`:
```python
"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from kuulo_protocol.api import IngestResult
from kuulo_protocol.models import NodeRegistration, Observation

from .config import Settings
from .db import make_session_factory
from .ingest import IngestError, ingest_observation, register_node


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    sessions = make_session_factory(settings.db_path)

    app = FastAPI(title="Kuulo")
    app.state.settings = settings
    app.state.sessions = sessions

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(IngestError)
    async def _ingest_error(_request: Request, exc: IngestError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.reason})

    @app.post("/v1/nodes/register")
    async def post_register(reg: NodeRegistration) -> dict:
        with sessions() as session:
            register_node(session, reg, settings.clock())
        return {"status": "registered"}

    @app.post("/v1/observations")
    async def post_observation(obs: Observation) -> IngestResult:
        with sessions() as session:
            return ingest_observation(session, obs, settings.clock(), settings)

    return app
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest server/tests/test_ingest.py && uv run ruff check .`
Expected: `10 passed`; ruff clean.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock server
git commit -m "feat(server): SQLite storage and signed, idempotent observation ingest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Server — heartbeats, node status, read API and live WebSocket

**Files:**
- Create: `server/src/kuulo_server/status.py`, `server/src/kuulo_server/live.py`, `server/src/kuulo_server/tracks.py`
- Modify: `server/src/kuulo_server/app.py`
- Test: `server/tests/test_status_live.py`

**Interfaces:**
- Consumes: Task 3 (`Settings`, `db`, `ingest`), `kuulo_protocol.api` (`NodeStatus`, `NodeView`, `LiveEvent`, `TrackDetail`).
- Produces:
  - `kuulo_server.status`: `ONLINE_S = 120`, `STALE_S = 300`, `node_status(last_heartbeat_at: datetime | None, now: datetime) -> NodeStatus`, `node_view(row: NodeRow, now: datetime, uncorroborated_rate: float | None = None) -> NodeView`.
  - `kuulo_server.live`: `class LiveHub(max_queue: int = 1000)` with `subscribe() -> asyncio.Queue[str]`, `unsubscribe(q)`, `publish(event: LiveEvent) -> None`, property `client_count`. On a full client queue the queue is emptied and a `resync` event is enqueued.
  - `kuulo_server.tracks`: `open_tracks(session, now) -> list[Track]` (status != closed, or last_seen within 10 min), `track_detail(session, track_id: str, now) -> TrackDetail | None`. (Task 8 adds more to this module.)
  - `app.state.hub: LiveHub`, `app.state.run_tick() -> None` (synchronous; publishes `node_status` events for nodes whose status changed since the previous tick; Task 8 adds fusion to it).
  - Routes: `POST /v1/heartbeats`, `GET /v1/nodes -> list[NodeView]`, `GET /v1/tracks -> list[Track]`, `GET /v1/tracks/{track_id} -> TrackDetail` (404 if unknown), `WS /v1/live`.
  - Accepted, non-late observations are published as `LiveEvent(type="observation")`.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_status_live.py`:
```python
import asyncio
from datetime import timedelta

from kuulo_protocol.api import LiveEvent, NodeStatus
from kuulo_protocol.testing import make_heartbeat, make_observation
from kuulo_server.live import LiveHub
from kuulo_server.status import node_status

from kuulo_server.testing import T0, post_signed, register


def test_node_status_thresholds():
    assert node_status(None, T0) is NodeStatus.OFFLINE
    assert node_status(T0, T0 + timedelta(seconds=119)) is NodeStatus.ONLINE
    assert node_status(T0, T0 + timedelta(seconds=120)) is NodeStatus.STALE
    assert node_status(T0, T0 + timedelta(seconds=299)) is NodeStatus.STALE
    assert node_status(T0, T0 + timedelta(seconds=300)) is NodeStatus.OFFLINE


def test_registered_node_without_heartbeat_is_offline(client, node):
    nodes = client.get("/v1/nodes").json()
    assert nodes[0]["node_id"] == "n1"
    assert nodes[0]["status"] == "offline"


def test_heartbeat_makes_node_online_then_it_ages(client, node, clock):
    response = post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0),
                           node.private_key)
    assert response.status_code == 200
    assert client.get("/v1/nodes").json()[0]["status"] == "online"
    clock.advance(180)
    assert client.get("/v1/nodes").json()[0]["status"] == "stale"
    clock.advance(180)
    assert client.get("/v1/nodes").json()[0]["status"] == "offline"


def test_heartbeat_bad_signature(client, node):
    other = register(client, "n2")
    response = post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0),
                           other.private_key)
    assert response.status_code == 401


def test_websocket_receives_observation(client, node):
    with client.websocket_connect("/v1/live") as ws:
        obs = make_observation("n1", observed_at=T0)
        post_signed(client, "/v1/observations", obs, node.private_key)
        event = ws.receive_json()
        assert event["type"] == "observation"
        assert event["data"]["observation_id"] == str(obs.observation_id)


def test_late_observation_not_published_live(client, node, clock):
    with client.websocket_connect("/v1/live") as ws:
        clock.advance(120)
        post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0),
                    node.private_key)
        post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=clock()),
                    node.private_key)
        assert ws.receive_json()["type"] == "node_status"  # the late observation was skipped


def test_tick_publishes_status_transition(app, client, node, clock):
    post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0), node.private_key)
    app.state.run_tick()  # records "online"
    with client.websocket_connect("/v1/live") as ws:
        clock.advance(150)
        app.state.run_tick()
        event = ws.receive_json()
        assert event == {"type": "node_status", "data": event["data"]}
        assert event["data"]["status"] == "stale"


def test_unknown_track_is_404(client):
    assert client.get("/v1/tracks/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get("/v1/tracks").json() == []


def test_stalled_client_gets_resync_and_others_unaffected():
    # Review Focus 4.
    async def scenario():
        hub = LiveHub(max_queue=2)
        slow, fast = hub.subscribe(), hub.subscribe()
        for _ in range(2):
            hub.publish(LiveEvent(type="resync"))
            await fast.get()
        hub.publish(LiveEvent(type="resync"))  # slow queue is full now
        assert slow.qsize() == 1
        assert '"resync"' in slow.get_nowait()
        assert fast.qsize() == 1
        assert hub.client_count == 2

    asyncio.run(scenario())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest server/tests/test_status_live.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'kuulo_server.live'`.

- [ ] **Step 3: Implement status and live hub**

`server/src/kuulo_server/status.py`:
```python
"""Node health from heartbeat age."""

from __future__ import annotations

from datetime import datetime

from kuulo_protocol.api import NodeStatus, NodeView
from kuulo_protocol.models import SensorLocation, TimeQuality

from .db import NodeRow, as_utc

ONLINE_S = 120
STALE_S = 300


def node_status(last_heartbeat_at: datetime | None, now: datetime) -> NodeStatus:
    if last_heartbeat_at is None:
        return NodeStatus.OFFLINE
    age = (now - last_heartbeat_at).total_seconds()
    if age < ONLINE_S:
        return NodeStatus.ONLINE
    if age < STALE_S:
        return NodeStatus.STALE
    return NodeStatus.OFFLINE


def node_view(row: NodeRow, now: datetime, uncorroborated_rate: float | None = None) -> NodeView:
    last = as_utc(row.last_heartbeat_at)
    return NodeView(
        node_id=row.node_id,
        location=SensorLocation(lat=row.lat, lon=row.lon, accuracy_m=row.accuracy_m),
        time_quality=TimeQuality(row.time_quality),
        status=node_status(last, now),
        last_heartbeat_at=last,
        mic_ok=row.mic_ok,
        software_version=row.software_version,
        uncorroborated_rate_24h=uncorroborated_rate,
    )
```

`server/src/kuulo_server/live.py`:
```python
"""Fan-out of live events to WebSocket clients. A client that falls behind gets a resync."""

from __future__ import annotations

import asyncio

from kuulo_protocol.api import LiveEvent

RESYNC = LiveEvent(type="resync").model_dump_json()


class LiveHub:
    def __init__(self, max_queue: int = 1000):
        self.max_queue = max_queue
        self._clients: set[asyncio.Queue[str]] = set()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.max_queue)
        self._clients.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        self._clients.discard(queue)

    def publish(self, event: LiveEvent) -> None:
        payload = event.model_dump_json()
        for queue in list(self._clients):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(RESYNC)
```

- [ ] **Step 4: Implement track read queries**

`server/src/kuulo_server/tracks.py`:
```python
"""Track persistence and evidence queries."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from kuulo_protocol.api import TrackDetail
from kuulo_protocol.models import Observation, Track, TrackStatus

from .db import NodeRow, ObservationRow, TrackObservationRow, TrackRow
from .status import node_view

RECENT_CLOSED = timedelta(minutes=10)


def open_tracks(session: Session, now: datetime) -> list[Track]:
    rows = session.scalars(
        select(TrackRow).where(
            or_(TrackRow.status != TrackStatus.CLOSED.value, TrackRow.last_seen >= now - RECENT_CLOSED)
        )
    )
    return [Track.model_validate_json(row.raw_json) for row in rows]


def track_detail(session: Session, track_id: str, now: datetime) -> TrackDetail | None:
    row = session.get(TrackRow, track_id)
    if row is None:
        return None
    track = Track.model_validate_json(row.raw_json)
    obs_rows = session.scalars(
        select(ObservationRow)
        .join(TrackObservationRow, TrackObservationRow.observation_id == ObservationRow.observation_id)
        .where(TrackObservationRow.track_id == track_id)
        .order_by(ObservationRow.observed_at)
    )
    observations = [Observation.model_validate_json(r.raw_json) for r in obs_rows]
    silent = [
        node_view(n, now)
        for n in session.scalars(select(NodeRow).where(NodeRow.node_id.in_(track.silent_neighbour_ids)))
    ]
    return TrackDetail(track=track, observations=observations, silent_neighbours=silent)
```

- [ ] **Step 5: Extend the app**

Replace `server/src/kuulo_server/app.py` with:
```python
"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select

from kuulo_protocol.api import IngestResult, LiveEvent, NodeStatus, NodeView, TrackDetail
from kuulo_protocol.models import Heartbeat, NodeRegistration, Observation, Track

from .config import Settings
from .db import NodeRow, make_session_factory
from .ingest import IngestError, ingest_heartbeat, ingest_observation, register_node
from .live import LiveHub
from .status import node_view
from .tracks import open_tracks, track_detail

log = logging.getLogger("kuulo.server")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    sessions = make_session_factory(settings.db_path)
    hub = LiveHub()
    last_status: dict[str, NodeStatus] = {}

    def run_tick() -> None:
        now = settings.clock()
        with sessions() as session:
            for row in session.scalars(select(NodeRow)):
                view = node_view(row, now)
                if last_status.get(row.node_id) not in (None, view.status):
                    hub.publish(LiveEvent(type="node_status", data=view))
                last_status[row.node_id] = view.status

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = None
        if settings.tick_interval_s:
            async def loop():
                while True:
                    await asyncio.sleep(settings.tick_interval_s)
                    try:
                        run_tick()
                    except Exception:
                        log.exception("tick failed")

            task = asyncio.create_task(loop())
        yield
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Kuulo", lifespan=lifespan)
    app.state.settings = settings
    app.state.sessions = sessions
    app.state.hub = hub
    app.state.run_tick = run_tick

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})

    @app.exception_handler(IngestError)
    async def _ingest_error(_request: Request, exc: IngestError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.reason})

    @app.post("/v1/nodes/register")
    async def post_register(reg: NodeRegistration) -> dict:
        with sessions() as session:
            register_node(session, reg, settings.clock())
        return {"status": "registered"}

    @app.post("/v1/observations")
    async def post_observation(obs: Observation) -> IngestResult:
        with sessions() as session:
            result = ingest_observation(session, obs, settings.clock(), settings)
        if result.status == "accepted" and not result.late:
            hub.publish(LiveEvent(type="observation", data=obs))
        return result

    @app.post("/v1/heartbeats")
    async def post_heartbeat(hb: Heartbeat) -> dict:
        now = settings.clock()
        with sessions() as session:
            row = ingest_heartbeat(session, hb, now, settings)
            view = node_view(row, now)
        hub.publish(LiveEvent(type="node_status", data=view))
        return {"status": "ok"}

    @app.get("/v1/nodes")
    async def get_nodes() -> list[NodeView]:
        now = settings.clock()
        with sessions() as session:
            return [node_view(row, now) for row in session.scalars(select(NodeRow))]

    @app.get("/v1/tracks")
    async def get_tracks() -> list[Track]:
        with sessions() as session:
            return open_tracks(session, settings.clock())

    @app.get("/v1/tracks/{track_id}")
    async def get_track(track_id: str) -> TrackDetail:
        with sessions() as session:
            detail = track_detail(session, track_id, settings.clock())
        if detail is None:
            raise HTTPException(status_code=404, detail="unknown track")
        return detail

    @app.websocket("/v1/live")
    async def live(ws: WebSocket) -> None:
        await ws.accept()
        queue = hub.subscribe()
        try:
            while True:
                await ws.send_text(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.unsubscribe(queue)

    return app
```


- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest server/tests && uv run ruff check .`
Expected: all server tests pass (Task 3 + Task 4); ruff clean.

- [ ] **Step 7: Commit**

```bash
git add server
git commit -m "feat(server): heartbeats, node status, read API and live WebSocket

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 5: Shared geo + smoothing, simulator scenario model and physics

**Files:**
- Create: `protocol/src/kuulo_protocol/geo.py`, `protocol/src/kuulo_protocol/smoothing.py`
- Modify: `pyproject.toml` (add `sim` member)
- Create: `sim/pyproject.toml`, `sim/src/kuulo_sim/__init__.py`, `sim/src/kuulo_sim/scenario.py`, `sim/src/kuulo_sim/physics.py`
- Test: `protocol/tests/test_geo.py`, `protocol/tests/test_smoothing.py`, `sim/tests/test_scenario_physics.py`

**Interfaces:**
- Consumes: `kuulo_protocol.models` (`GeoPoint`, `Label`, `Phase`, `TimeQuality`).
- Produces:
  - `kuulo_protocol.geo`: `to_local(origin, point) -> tuple[float, float]` (metres east, north), `from_local(origin, x, y) -> GeoPoint`, `distance_m(a, b) -> float`. `origin`/`point` are any objects with `.lat`/`.lon`.
  - `kuulo_protocol.smoothing`: `SmootherConfig(threshold=0.5, k=3, n=5, end_after_s=5.0, update_every_s=5.0)`, `DetectionSmoother(config=SmootherConfig(), id_factory=uuid4)` with `push(t: float, score: float) -> Phase | None`, attributes `active: bool`, `detection_id: UUID | None` (kept after END so the caller can build the end message).
  - `kuulo_sim.scenario`: models `NodeSpec`, `DroneSpec`, `FalseAlarmSpec`, `NodeFailureSpec`, `Scenario`; `load_scenario(path) -> Scenario`; `resolve_scenario(name_or_path: str) -> Path`; `bundled_scenarios() -> list[str]`; `drone_position(drone: DroneSpec, t: float) -> GeoPoint | None`; `node_offline(scenario, node_id, t) -> bool`.
  - `kuulo_sim.physics`: `SPEED_OF_SOUND_MPS = 343.0`, `ABSORPTION_DB_PER_M = 0.005`, `CLOCK_SIGMA_S`, `received_level_db(source_db, distance_m)`, `detection_probability(snr_db, snr50=6.0, width=2.0)`, `confidence_from_snr(snr_db, rng)`, `propagation_delay_s(distance_m)`, `clock_offset_s(time_quality, rng)`.

- [ ] **Step 1: Add the sim package**

Edit root `pyproject.toml`: `dependencies = ["kuulo-protocol", "kuulo-server", "kuulo-sim"]`, add `kuulo-sim = { workspace = true }` to `[tool.uv.sources]`, `members = ["protocol", "server", "sim"]`, `testpaths = ["protocol/tests", "server/tests", "sim/tests"]`.

`sim/pyproject.toml`:
```toml
[project]
name = "kuulo-sim"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["kuulo-protocol", "httpx>=0.27", "pyyaml>=6"]

[project.scripts]
kuulo-sim = "kuulo_sim.cli:main"

[tool.uv.sources]
kuulo-protocol = { workspace = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/kuulo_sim"]
```

`sim/src/kuulo_sim/__init__.py`:
```python
"""Kuulo scenario simulator: fake nodes that speak the real protocol."""
```

Create an empty directory marker `sim/src/kuulo_sim/scenarios/.gitkeep` (scenario files arrive in Task 6).

Run: `uv sync` — Expected: installs pyyaml.

- [ ] **Step 2: Write the failing tests**

`protocol/tests/test_geo.py`:
```python
import pytest

from kuulo_protocol.geo import distance_m, from_local, to_local
from kuulo_protocol.models import GeoPoint

HKI = GeoPoint(lat=60.1699, lon=24.9384)


def test_zero_distance():
    assert distance_m(HKI, HKI) == 0


def test_600m_north_and_east():
    north = GeoPoint(lat=HKI.lat + 600 / 110_540, lon=HKI.lon)
    assert distance_m(HKI, north) == pytest.approx(600, rel=1e-3)
    x, y = to_local(HKI, from_local(HKI, 600, 0))
    assert x == pytest.approx(600, rel=1e-6) and y == pytest.approx(0, abs=1e-6)


def test_round_trip():
    p = from_local(HKI, -1234.5, 987.0)
    assert to_local(HKI, p) == pytest.approx((-1234.5, 987.0), rel=1e-9)
```

`protocol/tests/test_smoothing.py`:
```python
from uuid import UUID

from kuulo_protocol.models import Phase
from kuulo_protocol.smoothing import DetectionSmoother, SmootherConfig


def feed(smoother, scores, start=0.0, dt=1.0):
    return [smoother.push(start + i * dt, s) for i, s in enumerate(scores)]


def test_isolated_spikes_never_start():
    s = DetectionSmoother()
    assert feed(s, [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1]) == [None] * 8
    assert not s.active


def test_three_of_five_starts_then_updates_then_ends():
    ids = iter([UUID(int=1), UUID(int=2)])
    s = DetectionSmoother(SmootherConfig(), id_factory=lambda: next(ids))
    phases = feed(s, [0.9, 0.9, 0.9])
    assert phases == [None, None, Phase.START]
    assert s.detection_id == UUID(int=1)
    phases = feed(s, [0.9] * 5, start=3.0)          # t = 3..7; START was at t=2
    assert phases == [None, None, None, None, Phase.UPDATE]


def test_update_timing_and_end():
    s = DetectionSmoother()
    out = feed(s, [0.9] * 8)                         # t = 0..7, start at t=2
    assert out[2] is Phase.START and out[7] is Phase.UPDATE
    assert [p for p in out if p is Phase.UPDATE] == [Phase.UPDATE]
    tail = feed(s, [0.0] * 6, start=8.0)             # last above at t=7 → end at t=12
    assert tail == [None, None, None, None, Phase.END, None]
    assert not s.active
    assert s.detection_id is not None                # kept for the END message


def test_new_detection_gets_new_id():
    s = DetectionSmoother()
    feed(s, [0.9] * 3)
    first = s.detection_id
    feed(s, [0.0] * 6, start=3.0)
    feed(s, [0.9] * 3, start=9.0)
    assert s.active and s.detection_id != first
```

`sim/tests/test_scenario_physics.py`:
```python
import random

import pytest
from pydantic import ValidationError

from kuulo_protocol.geo import distance_m
from kuulo_protocol.models import GeoPoint, TimeQuality
from kuulo_sim.physics import (
    clock_offset_s,
    confidence_from_snr,
    detection_probability,
    propagation_delay_s,
    received_level_db,
)
from kuulo_sim.scenario import DroneSpec, Scenario, drone_position, node_offline


def test_farther_is_quieter():
    levels = [received_level_db(100, r) for r in (1, 10, 100, 500, 1000, 3000)]
    assert levels == sorted(levels, reverse=True)
    assert received_level_db(100, 0) == received_level_db(100, 1)


def test_detection_probability_shape():
    assert detection_probability(6.0) == pytest.approx(0.5)
    assert detection_probability(20) > 0.99
    assert detection_probability(-10) < 0.01


def test_confidence_bounds():
    rng = random.Random(0)
    values = [confidence_from_snr(snr, rng) for snr in range(-20, 60, 5) for _ in range(20)]
    assert all(0 <= v <= 1 for v in values)


def test_clock_offsets_scale_with_quality():
    rng = random.Random(1)
    gps = [abs(clock_offset_s(TimeQuality.GPS, rng)) for _ in range(200)]
    manual = [abs(clock_offset_s(TimeQuality.MANUAL, rng)) for _ in range(200)]
    assert max(gps) < 1e-5 < sum(manual) / len(manual)


def test_propagation_delay():
    assert propagation_delay_s(343) == pytest.approx(1.0)


def test_drone_position_along_route():
    d = DroneSpec(id="d", speed_mps=10, start_s=5, waypoints=[(60.17, 24.90), (60.17, 24.95)])
    start = GeoPoint(lat=60.17, lon=24.90)
    assert drone_position(d, 4.9) is None
    assert distance_m(drone_position(d, 5), start) == pytest.approx(0, abs=1e-6)
    assert distance_m(drone_position(d, 105), start) == pytest.approx(1000, rel=1e-3)
    assert drone_position(d, 10_000) is None  # route finished


def _scenario(**extra):
    base = {"name": "t", "duration_s": 10, "nodes": [{"id": "n1", "lat": 60.17, "lon": 24.94}]}
    base.update(extra)
    return Scenario.model_validate(base)


def test_duplicate_node_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate node id"):
        _scenario(nodes=[{"id": "n1", "lat": 60, "lon": 24}, {"id": "n1", "lat": 60, "lon": 25}])


def test_failure_must_reference_known_node():
    with pytest.raises(ValidationError, match="unknown node"):
        _scenario(node_failures=[{"node_id": "nope"}])


def test_node_offline_window():
    sc = _scenario(node_failures=[{"node_id": "n1", "offline_from_s": 5, "offline_to_s": 8}])
    assert not node_offline(sc, "n1", 4.9)
    assert node_offline(sc, "n1", 5) and node_offline(sc, "n1", 7.9)
    assert not node_offline(sc, "n1", 8)
    forever = _scenario(node_failures=[{"node_id": "n1"}])
    assert node_offline(forever, "n1", 9999)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest protocol/tests/test_geo.py protocol/tests/test_smoothing.py sim/tests`
Expected: collection errors — `No module named 'kuulo_protocol.geo'` / `'kuulo_sim.physics'`.

- [ ] **Step 4: Implement geo and smoothing**

`protocol/src/kuulo_protocol/geo.py`:
```python
"""Local flat-earth projection. Accurate to well under 0.1 % over the few-km scale Kuulo fuses at."""

from __future__ import annotations

from math import cos, hypot, radians
from typing import Protocol

from .models import GeoPoint

M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQUATOR = 111_320.0


class HasLatLon(Protocol):
    lat: float
    lon: float


def to_local(origin: HasLatLon, point: HasLatLon) -> tuple[float, float]:
    """Metres (east, north) of `point` relative to `origin`."""
    x = (point.lon - origin.lon) * M_PER_DEG_LON_EQUATOR * cos(radians(origin.lat))
    y = (point.lat - origin.lat) * M_PER_DEG_LAT
    return x, y


def from_local(origin: HasLatLon, x: float, y: float) -> GeoPoint:
    return GeoPoint(
        lat=origin.lat + y / M_PER_DEG_LAT,
        lon=origin.lon + x / (M_PER_DEG_LON_EQUATOR * cos(radians(origin.lat))),
    )


def distance_m(a: HasLatLon, b: HasLatLon) -> float:
    return hypot(*to_local(a, b))
```

`protocol/src/kuulo_protocol/smoothing.py`:
```python
"""Turns a stream of per-window drone scores into start / update / end events.

Shared by the simulator now and the real node later, so both behave identically.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

from .models import Phase


@dataclass(frozen=True)
class SmootherConfig:
    threshold: float = 0.5
    k: int = 3
    n: int = 5
    end_after_s: float = 5.0
    update_every_s: float = 5.0


class DetectionSmoother:
    def __init__(
        self, config: SmootherConfig = SmootherConfig(), id_factory: Callable[[], UUID] = uuid4
    ):
        self.config = config
        self._id_factory = id_factory
        self._recent: deque[bool] = deque(maxlen=config.n)
        self._last_above: float | None = None
        self._last_emit: float | None = None
        self.active = False
        self.detection_id: UUID | None = None

    def push(self, t: float, score: float) -> Phase | None:
        above = score >= self.config.threshold
        self._recent.append(above)
        if above:
            self._last_above = t
        if not self.active:
            if sum(self._recent) >= self.config.k:
                self.active = True
                self.detection_id = self._id_factory()
                self._last_emit = t
                return Phase.START
            return None
        if self._last_above is not None and t - self._last_above >= self.config.end_after_s:
            self.active = False
            self._recent.clear()
            self._last_emit = None
            return Phase.END
        if self._last_emit is not None and t - self._last_emit >= self.config.update_every_s:
            self._last_emit = t
            return Phase.UPDATE
        return None
```

- [ ] **Step 5: Implement scenario model and physics**

`sim/src/kuulo_sim/physics.py`:
```python
"""Deliberately simple acoustic physics: spherical spreading + air absorption vs a noise floor."""

from __future__ import annotations

import random
from math import exp, log10

from kuulo_protocol.models import TimeQuality

SPEED_OF_SOUND_MPS = 343.0
ABSORPTION_DB_PER_M = 0.005
CLOCK_SIGMA_S = {TimeQuality.GPS: 1e-6, TimeQuality.NTP: 0.02, TimeQuality.MANUAL: 0.5}


def received_level_db(source_db: float, distance_m: float) -> float:
    r = max(distance_m, 1.0)
    return source_db - 20 * log10(r) - ABSORPTION_DB_PER_M * r


def detection_probability(snr_db: float, snr50: float = 6.0, width: float = 2.0) -> float:
    return 1.0 / (1.0 + exp(-(snr_db - snr50) / width))


def confidence_from_snr(snr_db: float, rng: random.Random) -> float:
    return min(1.0, max(0.0, 0.5 + 0.03 * snr_db + rng.gauss(0, 0.05)))


def propagation_delay_s(distance_m: float) -> float:
    return distance_m / SPEED_OF_SOUND_MPS


def clock_offset_s(time_quality: TimeQuality, rng: random.Random) -> float:
    return rng.gauss(0, CLOCK_SIGMA_S[time_quality])
```

`sim/src/kuulo_sim/scenario.py`:
```python
"""Scenario files: nodes, drones, false alarms and failures, validated with Pydantic."""

from __future__ import annotations

from importlib import resources
from itertools import pairwise
from math import hypot
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kuulo_protocol.geo import from_local, to_local
from kuulo_protocol.models import GeoPoint, Label, TimeQuality


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeSpec(_Spec):
    id: str = Field(min_length=1, max_length=64)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    time_quality: TimeQuality = TimeQuality.NTP
    noise_floor_db: float = 35.0


class DroneSpec(_Spec):
    id: str
    label: Label = Label.DRONE_MULTIROTOR
    source_db: float = 100.0
    speed_mps: float = Field(gt=0)
    start_s: float = Field(default=0.0, ge=0)
    waypoints: list[tuple[float, float]] = Field(min_length=2)  # (lat, lon)


class FalseAlarmSpec(_Spec):
    """A sound that fools the classifier (leaf blower, two-stroke scooter) near some nodes."""

    lat: float
    lon: float
    radius_m: float = Field(default=100.0, gt=0)
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    confidence: float = Field(default=0.9, ge=0, le=1)
    label: Label = Label.DRONE_MULTIROTOR


class NodeFailureSpec(_Spec):
    node_id: str
    offline_from_s: float = Field(default=0.0, ge=0)
    offline_to_s: float | None = None  # None = until the end


class Scenario(_Spec):
    name: str
    seed: int = 0
    duration_s: float = Field(gt=0)
    tick_s: float = Field(default=1.0, gt=0)
    heartbeat_every_s: float = Field(default=30.0, gt=0)
    nodes: list[NodeSpec] = Field(min_length=1)
    drones: list[DroneSpec] = []
    false_alarms: list[FalseAlarmSpec] = []
    node_failures: list[NodeFailureSpec] = []

    @model_validator(mode="after")
    def _check_references(self) -> Scenario:
        ids = [n.id for n in self.nodes]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"duplicate node id(s): {sorted(duplicates)}")
        for failure in self.node_failures:
            if failure.node_id not in ids:
                raise ValueError(f"node_failures refers to unknown node {failure.node_id!r}")
        return self


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(Path(path).read_text()))


def _scenario_dir() -> Path:
    return Path(str(resources.files("kuulo_sim") / "scenarios"))


def bundled_scenarios() -> list[str]:
    return sorted(p.stem for p in _scenario_dir().glob("*.yaml"))


def resolve_scenario(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    bundled = _scenario_dir() / f"{name_or_path}.yaml"
    if bundled.exists():
        return bundled
    raise FileNotFoundError(f"no scenario file or bundled scenario named {name_or_path!r}")


def drone_position(drone: DroneSpec, t: float) -> GeoPoint | None:
    """Where the drone is at scenario time t, or None before start / after its route ends."""
    if t < drone.start_s:
        return None
    origin = GeoPoint(lat=drone.waypoints[0][0], lon=drone.waypoints[0][1])
    points = [to_local(origin, GeoPoint(lat=lat, lon=lon)) for lat, lon in drone.waypoints]
    remaining = (t - drone.start_s) * drone.speed_mps
    for (x0, y0), (x1, y1) in pairwise(points):
        length = hypot(x1 - x0, y1 - y0)
        if remaining <= length:
            f = remaining / length if length else 0.0
            return from_local(origin, x0 + f * (x1 - x0), y0 + f * (y1 - y0))
        remaining -= length
    return None


def node_offline(scenario: Scenario, node_id: str, t: float) -> bool:
    for failure in scenario.node_failures:
        if failure.node_id != node_id:
            continue
        end = failure.offline_to_s if failure.offline_to_s is not None else float("inf")
        if failure.offline_from_s <= t < end:
            return True
    return False
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest && uv run ruff check .`
Expected: all tests pass (protocol, server, sim); ruff clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock protocol sim
git commit -m "feat(sim): shared geo + detection smoother, scenario model and acoustic physics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Simulator engine, bundled scenarios, realtime runner and CLI

**Files:**
- Create: `sim/src/kuulo_sim/engine.py`, `sim/src/kuulo_sim/runner.py`, `sim/src/kuulo_sim/cli.py`
- Create: `sim/src/kuulo_sim/scenarios/single_node.yaml`, `helsinki_pass.yaml`, `false_alarm.yaml`, `two_drones.yaml`, `node_failure.yaml` (delete `.gitkeep`)
- Test: `sim/tests/test_engine.py`, `sim/tests/test_runner_cli.py`

**Interfaces:**
- Consumes: Task 5 (`Scenario`, `drone_position`, `node_offline`, physics, `DetectionSmoother`), `kuulo_protocol.signing.keypair_from_seed/sign`.
- Produces:
  - `kuulo_sim.engine`: `@dataclass(frozen=True) SimMessage(at: datetime, kind: Literal["observation", "heartbeat"], payload: Observation | Heartbeat)`; `@dataclass(frozen=True) TruthPoint(at: datetime, drone_id: str, position: GeoPoint)`; `class SimulationRun(scenario: Scenario, t0: datetime, time_scale: float = 1.0)` with `.registrations() -> list[NodeRegistration]`, `.messages() -> list[SimMessage]` (sorted by `at`, deterministic for a seed, all signed), `.truth() -> list[TruthPoint]`, `.node_keys: dict[str, tuple[str, str]]`, `.end_at: datetime`.
  - `kuulo_sim.runner`: `@dataclass RunStats(sent: int = 0, errors: int = 0)`; `run_realtime(run, client: httpx.Client, *, sleep=time.sleep, now=utc_now, truth_path: Path | None = None) -> RunStats`.
  - `kuulo_sim.cli`: `main(argv: list[str] | None = None) -> int`; commands `kuulo-sim list`, `kuulo-sim run SCENARIO [--server URL] [--speed X] [--truth PATH]`. Invalid scenario → prints `Invalid scenario <path>:` then one `  <loc>: <msg>` line per error, exit code 2.

- [ ] **Step 1: Write the bundled scenarios**

Node grid for `helsinki_pass`: two rows 600 m apart (lat 60.1672 / 60.1726), five columns 600 m apart (lon 24.9168 … 24.9600). The drone flies between the rows at 20 m/s.

`sim/src/kuulo_sim/scenarios/helsinki_pass.yaml`:
```yaml
name: helsinki_pass
seed: 42
duration_s: 200
nodes:
  - {id: n01, lat: 60.1672, lon: 24.9168}
  - {id: n02, lat: 60.1672, lon: 24.9276}
  - {id: n03, lat: 60.1672, lon: 24.9384}
  - {id: n04, lat: 60.1672, lon: 24.9492}
  - {id: n05, lat: 60.1672, lon: 24.9600}
  - {id: n06, lat: 60.1726, lon: 24.9168}
  - {id: n07, lat: 60.1726, lon: 24.9276}
  - {id: n08, lat: 60.1726, lon: 24.9384}
  - {id: n09, lat: 60.1726, lon: 24.9492}
  - {id: n10, lat: 60.1726, lon: 24.9600}
drones:
  - id: d1
    speed_mps: 20
    start_s: 5
    waypoints: [[60.1699, 24.9060], [60.1699, 24.9708]]
```

`sim/src/kuulo_sim/scenarios/single_node.yaml`:
```yaml
name: single_node
seed: 1
duration_s: 150
nodes:
  - {id: n01, lat: 60.1699, lon: 24.9384}
drones:
  - id: d1
    speed_mps: 15
    start_s: 5
    waypoints: [[60.1699, 24.9200], [60.1699, 24.9570]]
```

`sim/src/kuulo_sim/scenarios/false_alarm.yaml` (a leaf blower next to n01 fools its classifier; n02 and n03 are 250 m away and hear nothing):
```yaml
name: false_alarm
seed: 7
duration_s: 100
nodes:
  - {id: n01, lat: 60.1699, lon: 24.9384}
  - {id: n02, lat: 60.1699, lon: 24.94291}
  - {id: n03, lat: 60.17216, lon: 24.9384}
false_alarms:
  - {lat: 60.1699, lon: 24.9384, radius_m: 50, start_s: 10, duration_s: 60, confidence: 0.9}
```

`sim/src/kuulo_sim/scenarios/two_drones.yaml` (two clusters ~5 km apart, one drone over each):
```yaml
name: two_drones
seed: 11
duration_s: 130
nodes:
  - {id: a1, lat: 60.1699, lon: 24.9384}
  - {id: a2, lat: 60.1699, lon: 24.9438}
  - {id: a3, lat: 60.1726, lon: 24.9411}
  - {id: b1, lat: 60.1699, lon: 25.0286}
  - {id: b2, lat: 60.1699, lon: 25.0340}
  - {id: b3, lat: 60.1726, lon: 25.0313}
drones:
  - {id: d1, speed_mps: 15, start_s: 5, waypoints: [[60.1712, 24.9250], [60.1712, 24.9550]]}
  - {id: d2, speed_mps: 15, start_s: 5, waypoints: [[60.1712, 25.0152], [60.1712, 25.0452]]}
```

`sim/src/kuulo_sim/scenarios/node_failure.yaml` (same leaf blower, but the only neighbour is offline — its silence must not count):
```yaml
name: node_failure
seed: 3
duration_s: 100
nodes:
  - {id: n01, lat: 60.1699, lon: 24.9384}
  - {id: n02, lat: 60.1699, lon: 24.94291}
false_alarms:
  - {lat: 60.1699, lon: 24.9384, radius_m: 50, start_s: 10, duration_s: 60, confidence: 0.9}
node_failures:
  - {node_id: n02}
```

Delete `sim/src/kuulo_sim/scenarios/.gitkeep`.

- [ ] **Step 2: Write the failing tests**

`sim/tests/test_engine.py`:
```python
from datetime import UTC, datetime

from kuulo_protocol.models import Observation
from kuulo_protocol.signing import verify
from kuulo_sim.engine import SimulationRun
from kuulo_sim.scenario import bundled_scenarios, load_scenario, resolve_scenario

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def run_of(name: str, **kw) -> SimulationRun:
    return SimulationRun(load_scenario(resolve_scenario(name)), T0, **kw)


def observations(run):
    return [m.payload for m in run.messages() if m.kind == "observation"]


def test_all_bundled_scenarios_load_and_run():
    assert set(bundled_scenarios()) >= {
        "single_node", "helsinki_pass", "false_alarm", "two_drones", "node_failure"
    }
    for name in bundled_scenarios():
        assert run_of(name).messages()


def test_deterministic_for_a_seed():
    a = [m.payload.model_dump_json() for m in run_of("helsinki_pass").messages()]
    b = [m.payload.model_dump_json() for m in run_of("helsinki_pass").messages()]
    assert a == b


def test_messages_sorted_and_signed():
    run = run_of("helsinki_pass")
    msgs = run.messages()
    assert [m.at for m in msgs] == sorted(m.at for m in msgs)
    keys = {r.node_id: r.public_key for r in run.registrations()}
    for m in msgs:
        node_id = m.payload.source.id if isinstance(m.payload, Observation) else m.payload.node_id
        assert verify(m.payload, keys[node_id])


def test_helsinki_pass_heard_by_several_nodes():
    starts = [o for o in observations(run_of("helsinki_pass")) if o.event.phase == "start"]
    assert len({o.source.id for o in starts}) >= 4


def test_false_alarm_only_at_n01():
    obs = observations(run_of("false_alarm"))
    assert obs and {o.source.id for o in obs} == {"n01"}


def test_failed_node_is_silent():
    run = run_of("node_failure")
    senders = {
        m.payload.source.id if m.kind == "observation" else m.payload.node_id
        for m in run.messages()
    }
    assert "n02" not in senders


def test_observations_arrive_after_emission_and_end_phase_exists():
    obs = observations(run_of("single_node"))
    phases = [o.event.phase for o in obs]
    assert phases[0] == "start" and "end" in phases


def test_time_scale_compresses_timestamps():
    fast = run_of("helsinki_pass", time_scale=10)
    assert (fast.end_at - T0).total_seconds() == 20


def test_truth_follows_drone():
    truth = run_of("helsinki_pass").truth()
    assert truth[0].drone_id == "d1"
    assert truth[0].at >= T0
```

`sim/tests/test_runner_cli.py`:
```python
import json
from datetime import UTC, datetime, timedelta

import httpx

from kuulo_sim.cli import main
from kuulo_sim.engine import SimulationRun
from kuulo_sim.runner import run_realtime
from kuulo_sim.scenario import load_scenario, resolve_scenario

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_runner_registers_then_sends_everything_in_time(tmp_path):
    run = SimulationRun(load_scenario(resolve_scenario("false_alarm")), T0)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"status": "accepted", "late": False})

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    fake_now = [T0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        fake_now[0] = fake_now[0] + timedelta(seconds=seconds)

    truth = tmp_path / "truth.json"
    stats = run_realtime(run, client, sleep=sleep, now=lambda: fake_now[0], truth_path=truth)
    registrations = seen.count("/v1/nodes/register")
    assert registrations == 3 and seen[:3] == ["/v1/nodes/register"] * 3
    assert stats.sent == len(run.messages()) and stats.errors == 0
    assert sum(slept) > 60
    assert json.loads(truth.read_text()) == []  # no drones in this scenario


def test_runner_counts_errors():
    run = SimulationRun(load_scenario(resolve_scenario("single_node")), T0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/nodes/register":
            return httpx.Response(200, json={})
        return httpx.Response(401, json={"detail": "bad signature"})

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    stats = run_realtime(run, client, sleep=lambda s: None, now=lambda: T0)
    assert stats.errors == len(run.messages()) and stats.sent == 0


def test_cli_list(capsys):
    assert main(["list"]) == 0
    assert "helsinki_pass" in capsys.readouterr().out


def test_cli_invalid_scenario_names_the_field(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nduration_s: 10\nnodes:\n  - {id: n1, lat: 95, lon: 24}\n")
    assert main(["run", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "Invalid scenario" in err and "nodes.0.lat" in err
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest sim/tests/test_engine.py sim/tests/test_runner_cli.py`
Expected: collection errors — `No module named 'kuulo_sim.engine'`.

- [ ] **Step 4: Implement the engine**

`sim/src/kuulo_sim/engine.py`:
```python
"""Turns a scenario into the exact signed messages real nodes would send."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import UUID

from kuulo_protocol.geo import distance_m
from kuulo_protocol.models import (
    Acoustic,
    Detection,
    EventRef,
    GeoPoint,
    Heartbeat,
    Label,
    NodeRegistration,
    Observation,
    SensorLocation,
    Source,
    SourceType,
)
from kuulo_protocol.signing import keypair_from_seed, sign
from kuulo_protocol.smoothing import DetectionSmoother

from .physics import (
    clock_offset_s,
    confidence_from_snr,
    detection_probability,
    propagation_delay_s,
    received_level_db,
)
from .scenario import Scenario, drone_position, node_offline

SOFTWARE_VERSION = "sim-0.1.0"
FALSE_ALARM_SNR_DB = 20.0


@dataclass(frozen=True)
class SimMessage:
    at: datetime
    kind: Literal["observation", "heartbeat"]
    payload: Observation | Heartbeat


@dataclass(frozen=True)
class TruthPoint:
    at: datetime
    drone_id: str
    position: GeoPoint


@dataclass(frozen=True)
class _Heard:
    confidence: float
    snr_db: float
    label: Label
    delay_s: float


class SimulationRun:
    def __init__(self, scenario: Scenario, t0: datetime, time_scale: float = 1.0):
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self.scenario = scenario
        self.t0 = t0
        self.time_scale = time_scale
        self.node_keys: dict[str, tuple[str, str]] = {
            n.id: keypair_from_seed(hashlib.sha256(f"{scenario.seed}:{n.id}".encode()).digest())
            for n in scenario.nodes
        }
        self._messages: list[SimMessage] | None = None

    @property
    def end_at(self) -> datetime:
        return self._at(self.scenario.duration_s)

    def _at(self, sim_seconds: float) -> datetime:
        return self.t0 + timedelta(seconds=sim_seconds / self.time_scale)

    def registrations(self) -> list[NodeRegistration]:
        return [
            NodeRegistration(
                node_id=n.id,
                public_key=self.node_keys[n.id][1],
                location=SensorLocation(lat=n.lat, lon=n.lon, accuracy_m=10.0),
                time_quality=n.time_quality,
            )
            for n in self.scenario.nodes
        ]

    def truth(self) -> list[TruthPoint]:
        points = []
        steps = int(self.scenario.duration_s / self.scenario.tick_s)
        for step in range(steps + 1):
            t = step * self.scenario.tick_s
            for drone in self.scenario.drones:
                pos = drone_position(drone, t)
                if pos is not None:
                    points.append(TruthPoint(self._at(t), drone.id, pos))
        return points

    def messages(self) -> list[SimMessage]:
        if self._messages is None:
            self._messages = self._build()
        return self._messages

    def _build(self) -> list[SimMessage]:
        sc = self.scenario
        rng = random.Random(sc.seed)

        def new_id() -> UUID:
            return UUID(int=rng.getrandbits(128), version=4)

        nodes = sorted(sc.nodes, key=lambda n: n.id)
        offsets = {n.id: clock_offset_s(n.time_quality, rng) for n in nodes}
        smoothers = {n.id: DetectionSmoother(id_factory=new_id) for n in nodes}
        last_heard: dict[str, _Heard] = {}
        heartbeat_every = max(1, round(sc.heartbeat_every_s / sc.tick_s))
        out: list[SimMessage] = []

        steps = int(sc.duration_s / sc.tick_s)
        for step in range(steps + 1):
            t = step * sc.tick_s
            for node in nodes:
                if node_offline(sc, node.id, t):
                    continue
                if step % heartbeat_every == 0:
                    hb = Heartbeat(
                        node_id=node.id, sent_at=self._at(t), software_version=SOFTWARE_VERSION,
                        mic_ok=True, queue_depth=0,
                    )
                    out.append(SimMessage(self._at(t), "heartbeat", sign(hb, self.node_keys[node.id][0])))

                heard: _Heard | None = None
                for drone in sc.drones:
                    pos = drone_position(drone, t)
                    if pos is None:
                        continue
                    r = distance_m(node, pos)
                    snr = received_level_db(drone.source_db, r) - node.noise_floor_db
                    if rng.random() < detection_probability(snr):
                        candidate = _Heard(
                            confidence_from_snr(snr, rng), snr, drone.label, propagation_delay_s(r)
                        )
                        if heard is None or candidate.confidence > heard.confidence:
                            heard = candidate
                for alarm in sc.false_alarms:
                    active = alarm.start_s <= t < alarm.start_s + alarm.duration_s
                    if active and distance_m(node, alarm) <= alarm.radius_m:
                        candidate = _Heard(alarm.confidence, FALSE_ALARM_SNR_DB, alarm.label, 0.0)
                        if heard is None or candidate.confidence > heard.confidence:
                            heard = candidate

                smoother = smoothers[node.id]
                phase = smoother.push(t, heard.confidence if heard else 0.0)
                if heard:
                    last_heard[node.id] = heard
                if phase is None:
                    continue
                basis = heard or last_heard[node.id]
                sim_observed = t + basis.delay_s + offsets[node.id]
                obs = Observation(
                    observation_id=new_id(),
                    source=Source(type=SourceType.SIMULATED_NODE, id=node.id),
                    observed_at=self._at(sim_observed),
                    time_quality=node.time_quality,
                    sensor_location=SensorLocation(lat=node.lat, lon=node.lon, accuracy_m=10.0),
                    detection=Detection(label=basis.label, confidence=round(basis.confidence, 3)),
                    event=EventRef(detection_id=smoother.detection_id, phase=phase),
                    acoustic=Acoustic(snr_db=round(basis.snr_db, 2), peak_freq_hz=180.0),
                )
                send_at = self._at(t + basis.delay_s)
                out.append(SimMessage(send_at, "observation", sign(obs, self.node_keys[node.id][0])))

        out.sort(key=lambda m: m.at)
        return out
```

- [ ] **Step 5: Implement the runner and CLI**

`sim/src/kuulo_sim/runner.py`:
```python
"""Plays a simulation against a running server in real (or scaled) time."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .engine import SimulationRun

log = logging.getLogger("kuulo.sim")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunStats:
    sent: int = 0
    errors: int = 0


def write_truth(run: SimulationRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"at": p.at.isoformat(), "drone_id": p.drone_id, "lat": p.position.lat, "lon": p.position.lon}
        for p in run.truth()
    ]
    path.write_text(json.dumps(rows, indent=1))


def run_realtime(
    run: SimulationRun,
    client: httpx.Client,
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = utc_now,
    truth_path: Path | None = None,
) -> RunStats:
    for reg in run.registrations():
        client.post("/v1/nodes/register", json=reg.model_dump(mode="json")).raise_for_status()
    if truth_path is not None:
        write_truth(run, truth_path)
    stats = RunStats()
    for msg in run.messages():
        wait = (msg.at - now()).total_seconds()
        if wait > 0:
            sleep(wait)
        path = "/v1/observations" if msg.kind == "observation" else "/v1/heartbeats"
        response = client.post(
            path, content=msg.payload.model_dump_json(), headers={"content-type": "application/json"}
        )
        if response.status_code >= 400:
            stats.errors += 1
            log.warning("%s rejected (%s): %s", path, response.status_code, response.text)
        else:
            stats.sent += 1
    return stats
```

`sim/src/kuulo_sim/cli.py`:
```python
"""kuulo-sim: run bundled or custom scenarios against a local server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

from .engine import SimulationRun
from .runner import run_realtime, utc_now
from .scenario import bundled_scenarios, load_scenario, resolve_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuulo-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list bundled scenarios")
    run_p = sub.add_parser("run", help="play a scenario against a server")
    run_p.add_argument("scenario", help="bundled scenario name or path to a .yaml file")
    run_p.add_argument("--server", default="http://127.0.0.1:8000")
    run_p.add_argument("--speed", type=float, default=1.0, help="time compression factor")
    run_p.add_argument("--truth", type=Path, default=Path("data/truth.json"))
    args = parser.parse_args(argv)

    if args.command == "list":
        print("\n".join(bundled_scenarios()))
        return 0

    try:
        path = resolve_scenario(args.scenario)
        scenario = load_scenario(path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"Invalid scenario {args.scenario}:", file=sys.stderr)
        for error in exc.errors():
            loc = ".".join(str(part) for part in error["loc"])
            print(f"  {loc}: {error['msg']}", file=sys.stderr)
        return 2

    run = SimulationRun(scenario, utc_now(), time_scale=args.speed)
    with httpx.Client(base_url=args.server, timeout=5.0) as client:
        stats = run_realtime(run, client, truth_path=args.truth)
    print(f"{scenario.name}: sent {stats.sent}, rejected {stats.errors}")
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `resolve_scenario` treats an existing `.yaml` path as a file. In `test_cli_invalid_scenario_names_the_field`, the tmp file exists, so it is loaded and fails validation as intended.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest && uv run ruff check .`
Expected: all tests pass; ruff clean. If `test_helsinki_pass_heard_by_several_nodes` fails because fewer than 4 nodes start, print the per-node SNR at closest approach and fix the scenario geometry (not the assertion) — the design intent is that rows 300 m from the track hear it clearly (SNR ≈ 14 dB).

- [ ] **Step 7: Commit**

```bash
git add sim
git commit -m "feat(sim): deterministic scenario engine, bundled scenarios, realtime runner and CLI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 7: Dashboard — live map

**Files:**
- Create: `dashboard/package.json` (via npm), `dashboard/tsconfig.json`, `dashboard/vite.config.ts`, `dashboard/index.html`, `dashboard/src/main.tsx`, `dashboard/src/App.tsx`, `dashboard/src/styles.css`
- Create: `dashboard/src/api/schema.json`, `dashboard/src/api/types.ts` (generated), `dashboard/src/api/client.ts`
- Create: `dashboard/src/state/reducer.ts`, `dashboard/src/state/live.ts`, `dashboard/src/map/geo.ts`, `dashboard/src/map/layers.ts`
- Create: `dashboard/src/components/MapView.tsx`, `dashboard/src/components/SidePanel.tsx`
- Test: `dashboard/src/state/reducer.test.ts`, `dashboard/src/state/live.test.ts`, `dashboard/src/map/map.test.ts`

**Interfaces:**
- Consumes: HTTP API from Task 4 (`GET /v1/nodes`, `GET /v1/tracks`, `GET /v1/tracks/{id}`, `WS /v1/live`), JSON Schema from Task 2.
- Produces: generated TS types `Observation`, `Track`, `NodeView`, `TrackDetail`, `LiveEvent`; `reducer(state, action)`, `initialState`, `backoffDelay(attempt)`, `startLive(deps) -> stop()`, `circlePolygon(lat, lon, radiusM, steps?)`, GeoJSON builders `nodesToGeoJSON`, `tracksToGeoJSON`, `uncertaintyToGeoJSON`, `trailsToGeoJSON`, `pulsesToGeoJSON`.

- [ ] **Step 1: Scaffold the project (no interactive generators)**

```bash
mkdir -p dashboard/src/api dashboard/src/state dashboard/src/map dashboard/src/components
cd dashboard
npm init -y
npm install react react-dom maplibre-gl
npm install -D vite @vitejs/plugin-react typescript vitest @types/react @types/react-dom json-schema-to-typescript
cd ..
```

Then set `dashboard/package.json` fields (keep the dependency versions npm wrote):
```json
{
  "name": "kuulo-dashboard",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "test": "vitest run",
    "gen:types": "cd .. && uv run python -m kuulo_protocol.schema > dashboard/src/api/schema.json && cd dashboard && npx json2ts -i src/api/schema.json -o src/api/types.ts"
  }
}
```
Remove `main`, `keywords`, `author`, `license`, `description` keys that `npm init` added.

`dashboard/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noEmit": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "resolveJsonModule": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
```

`dashboard/vite.config.ts`:
```ts
/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/v1": { target: "http://127.0.0.1:8000", ws: true } },
  },
  test: { environment: "node" },
});
```

`dashboard/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Kuulo</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

Generate types: `cd dashboard && npm run gen:types && cd ..`
Expected: `dashboard/src/api/types.ts` exists and contains `export interface Track`, `export interface NodeView`, `export interface LiveEvent`.

- [ ] **Step 2: Write the failing tests**

`dashboard/src/state/reducer.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import type { NodeView, Track } from "../api/types";
import { initialState, reducer, TRAIL_MAX, PULSE_MS } from "./reducer";

const node = (id: string, status: NodeView["status"] = "online"): NodeView => ({
  node_id: id,
  location: { lat: 60.17, lon: 24.94, accuracy_m: 10 },
  time_quality: "ntp",
  status,
});

const track = (id: string, lat = 60.17, status: Track["status"] = "tentative"): Track => ({
  track_id: id,
  status,
  label: "drone_multirotor",
  confidence: 0.8,
  position: { lat, lon: 24.94 },
  uncertainty_m: 300,
  first_seen: "2026-09-24T12:00:00Z",
  last_seen: "2026-09-24T12:00:05Z",
  observation_ids: [],
  silent_neighbour_ids: [],
});

describe("reducer", () => {
  it("snapshot replaces everything", () => {
    let s = reducer(initialState, { type: "live", event: { type: "track", data: track("old") }, receivedAt: 0 });
    s = reducer(s, { type: "snapshot", nodes: [node("n1")], tracks: [track("t1")] });
    expect(Object.keys(s.tracks)).toEqual(["t1"]);
    expect(Object.keys(s.nodes)).toEqual(["n1"]);
    expect(s.trails["old"]).toBeUndefined();
  });

  it("live track upserts and builds a capped trail", () => {
    let s = initialState;
    for (let i = 0; i < TRAIL_MAX + 10; i++) {
      s = reducer(s, { type: "live", event: { type: "track", data: track("t1", 60 + i / 1000) }, receivedAt: 0 });
    }
    expect(s.tracks["t1"].position.lat).toBeCloseTo(60 + (TRAIL_MAX + 9) / 1000);
    expect(s.trails["t1"].length).toBe(TRAIL_MAX);
  });

  it("closed track is removed and deselected", () => {
    let s = reducer(initialState, { type: "live", event: { type: "track", data: track("t1") }, receivedAt: 0 });
    s = reducer(s, { type: "select", trackId: "t1" });
    s = reducer(s, { type: "live", event: { type: "track", data: track("t1", 60.17, "closed") }, receivedAt: 0 });
    expect(s.tracks["t1"]).toBeUndefined();
    expect(s.selectedTrackId).toBeNull();
  });

  it("node_status upserts; observation adds a pulse that expires", () => {
    let s = reducer(initialState, { type: "live", event: { type: "node_status", data: node("n1", "stale") }, receivedAt: 0 });
    expect(s.nodes["n1"].status).toBe("stale");
    const obs = { source: { type: "simulated_node", id: "n1" } } as never;
    s = reducer(s, { type: "live", event: { type: "observation", data: obs }, receivedAt: 1000 });
    expect(s.pulses).toEqual([{ nodeId: "n1", at: 1000 }]);
    s = reducer(s, { type: "expirePulses", now: 1000 + PULSE_MS + 1 });
    expect(s.pulses).toEqual([]);
  });

  it("resync event does not change state", () => {
    const s = reducer(initialState, { type: "live", event: { type: "resync" }, receivedAt: 0 });
    expect(s).toBe(initialState);
  });
});
```

`dashboard/src/state/live.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import type { Action } from "./reducer";
import { backoffDelay, startLive, type SocketLike } from "./live";

class FakeSocket implements SocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  close() { this.closed = true; }
}

function harness() {
  const sockets: FakeSocket[] = [];
  const actions: Action[] = [];
  const timers: { fn: () => void; ms: number }[] = [];
  let resolveSnapshot: (v: { nodes: never[]; tracks: never[] }) => void = () => {};
  let snapshots = 0;
  const stop = startLive({
    url: "ws://x/v1/live",
    openSocket: () => { const s = new FakeSocket(); sockets.push(s); return s; },
    fetchSnapshot: () => { snapshots++; return new Promise((r) => { resolveSnapshot = r; }); },
    dispatch: (a) => actions.push(a),
    schedule: (fn, ms) => { timers.push({ fn, ms }); },
    now: () => 42,
  });
  return { sockets, actions, timers, stop, resolve: () => resolveSnapshot({ nodes: [], tracks: [] }), snapshots: () => snapshots };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("live connection", () => {
  it("backoff doubles and caps", () => {
    expect([0, 1, 2, 3, 4, 10].map(backoffDelay)).toEqual([1000, 2000, 4000, 8000, 15000, 15000]);
  });

  it("buffers events that arrive while the snapshot loads, then replays them", async () => {
    // Review Focus 5.
    const h = harness();
    h.sockets[0].onopen!();
    h.sockets[0].onmessage!({ data: JSON.stringify({ type: "node_status", data: { node_id: "n1" } }) });
    expect(h.actions.some((a) => a.type === "live")).toBe(false);
    h.resolve();
    await flush();
    const types = h.actions.map((a) => a.type);
    expect(types.indexOf("snapshot")).toBeLessThan(types.indexOf("live"));
    expect(h.actions.at(-1)).toEqual({ type: "connection", connection: "live" });
  });

  it("reconnects with backoff after close", async () => {
    const h = harness();
    h.sockets[0].onclose!();
    expect(h.actions.at(-1)).toEqual({ type: "connection", connection: "reconnecting" });
    expect(h.timers[0].ms).toBe(1000);
    h.timers[0].fn();
    expect(h.sockets.length).toBe(2);
    h.sockets[1].onclose!();
    expect(h.timers[1].ms).toBe(2000);
  });

  it("resync event reloads the snapshot", async () => {
    const h = harness();
    h.sockets[0].onopen!();
    h.resolve();
    await flush();
    h.sockets[0].onmessage!({ data: JSON.stringify({ type: "resync" }) });
    expect(h.snapshots()).toBe(2);
  });

  it("stop closes the socket and prevents reconnect", () => {
    const h = harness();
    h.stop();
    expect(h.sockets[0].closed).toBe(true);
    h.sockets[0].onclose?.();
    expect(h.timers.length).toBe(0);
  });
});
```

`dashboard/src/map/map.test.ts`:
```ts
import { describe, expect, it } from "vitest";
import { circlePolygon, haversineM } from "./geo";
import { nodesToGeoJSON, tracksToGeoJSON } from "./layers";

describe("geo", () => {
  it("circle is closed and has the right radius", () => {
    const ring = circlePolygon(60.17, 24.94, 500, 32);
    expect(ring.length).toBe(33);
    expect(ring[0]).toEqual(ring[32]);
    for (const [lon, lat] of ring) expect(haversineM(60.17, 24.94, lat, lon)).toBeCloseTo(500, -1);
  });
});

describe("layers", () => {
  it("nodes and tracks become GeoJSON points with status", () => {
    const nodes = nodesToGeoJSON({
      n1: { node_id: "n1", location: { lat: 60.1, lon: 24.9, accuracy_m: 1 }, time_quality: "ntp", status: "stale" },
    });
    expect(nodes.features[0].geometry.coordinates).toEqual([24.9, 60.1]);
    expect(nodes.features[0].properties.status).toBe("stale");
    const tracks = tracksToGeoJSON({});
    expect(tracks.features).toEqual([]);
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd dashboard && npx vitest run; cd ..`
Expected: FAIL — cannot resolve `./reducer`, `./live`, `./geo`.

- [ ] **Step 4: Implement state, live connection and map helpers**

`dashboard/src/state/reducer.ts`:
```ts
import type { LiveEvent, NodeView, Track } from "../api/types";

export type Connection = "connecting" | "live" | "reconnecting";
export interface Pulse { nodeId: string; at: number }

export interface State {
  nodes: Record<string, NodeView>;
  tracks: Record<string, Track>;
  trails: Record<string, [number, number][]>; // [lon, lat]
  pulses: Pulse[];
  connection: Connection;
  selectedTrackId: string | null;
}

export type Action =
  | { type: "snapshot"; nodes: NodeView[]; tracks: Track[] }
  | { type: "live"; event: LiveEvent; receivedAt: number }
  | { type: "connection"; connection: Connection }
  | { type: "select"; trackId: string | null }
  | { type: "expirePulses"; now: number };

export const TRAIL_MAX = 60;
export const PULSE_MS = 3000;

export const initialState: State = {
  nodes: {}, tracks: {}, trails: {}, pulses: [], connection: "connecting", selectedTrackId: null,
};

function upsertTrack(state: State, track: Track): State {
  const tracks = { ...state.tracks };
  const trails = { ...state.trails };
  if (track.status === "closed") {
    delete tracks[track.track_id];
    delete trails[track.track_id];
    const selectedTrackId = state.selectedTrackId === track.track_id ? null : state.selectedTrackId;
    return { ...state, tracks, trails, selectedTrackId };
  }
  tracks[track.track_id] = track;
  const point: [number, number] = [track.position.lon, track.position.lat];
  const trail = trails[track.track_id] ?? [];
  const last = trail[trail.length - 1];
  const next = last && last[0] === point[0] && last[1] === point[1] ? trail : [...trail, point];
  trails[track.track_id] = next.slice(-TRAIL_MAX);
  return { ...state, tracks, trails };
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "snapshot": {
      const nodes = Object.fromEntries(action.nodes.map((n) => [n.node_id, n]));
      const tracks = Object.fromEntries(action.tracks.map((t) => [t.track_id, t]));
      const trails = Object.fromEntries(
        Object.entries(state.trails).filter(([id]) => id in tracks),
      );
      for (const t of action.tracks) {
        if (!trails[t.track_id]) trails[t.track_id] = [[t.position.lon, t.position.lat]];
      }
      const selectedTrackId = state.selectedTrackId && state.selectedTrackId in tracks ? state.selectedTrackId : null;
      return { ...state, nodes, tracks, trails, selectedTrackId };
    }
    case "live": {
      const { event } = action;
      switch (event.type) {
        case "track":
          return upsertTrack(state, event.data as Track);
        case "node_status": {
          const n = event.data as NodeView;
          return { ...state, nodes: { ...state.nodes, [n.node_id]: n } };
        }
        case "observation": {
          const nodeId = (event.data as { source: { id: string } }).source.id;
          return { ...state, pulses: [...state.pulses, { nodeId, at: action.receivedAt }] };
        }
        default:
          return state;
      }
    }
    case "connection":
      return { ...state, connection: action.connection };
    case "select":
      return { ...state, selectedTrackId: action.trackId };
    case "expirePulses": {
      const pulses = state.pulses.filter((p) => action.now - p.at <= PULSE_MS);
      return pulses.length === state.pulses.length ? state : { ...state, pulses };
    }
  }
}
```

`dashboard/src/state/live.ts`:
```ts
import type { LiveEvent, NodeView, Track } from "../api/types";
import type { Action } from "./reducer";

export interface SocketLike {
  onopen: (() => void) | null;
  onmessage: ((e: { data: string }) => void) | null;
  onclose: (() => void) | null;
  close(): void;
}

export interface LiveDeps {
  url: string;
  openSocket(url: string): SocketLike;
  fetchSnapshot(): Promise<{ nodes: NodeView[]; tracks: Track[] }>;
  dispatch(action: Action): void;
  schedule(fn: () => void, ms: number): void;
  now(): number;
}

export function backoffDelay(attempt: number): number {
  return Math.min(1000 * 2 ** attempt, 15000);
}

export function defaultLiveUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/v1/live`;
}

/** Connects, loads a snapshot, applies live events; reconnects with backoff. Returns stop(). */
export function startLive(deps: LiveDeps): () => void {
  let attempt = 0;
  let stopped = false;
  let socket: SocketLike | null = null;

  const connect = () => {
    if (stopped) return;
    let loading = true;
    let buffer: LiveEvent[] = [];
    const ws = deps.openSocket(deps.url);
    socket = ws;

    const loadSnapshot = async () => {
      loading = true;
      try {
        const snap = await deps.fetchSnapshot();
        if (stopped || socket !== ws) return;
        deps.dispatch({ type: "snapshot", nodes: snap.nodes, tracks: snap.tracks });
        for (const event of buffer) deps.dispatch({ type: "live", event, receivedAt: deps.now() });
        buffer = [];
        loading = false;
        deps.dispatch({ type: "connection", connection: "live" });
      } catch {
        ws.close();
      }
    };

    ws.onopen = () => {
      attempt = 0;
      void loadSnapshot();
    };
    ws.onmessage = (e) => {
      const event = JSON.parse(e.data) as LiveEvent;
      if (event.type === "resync") {
        buffer = [];
        void loadSnapshot();
      } else if (loading) {
        buffer.push(event);
      } else {
        deps.dispatch({ type: "live", event, receivedAt: deps.now() });
      }
    };
    ws.onclose = () => {
      if (stopped) return;
      deps.dispatch({ type: "connection", connection: "reconnecting" });
      deps.schedule(connect, backoffDelay(attempt++));
    };
  };

  connect();
  return () => {
    stopped = true;
    socket?.close();
  };
}
```

`dashboard/src/map/geo.ts`:
```ts
const R = 6_371_000;
const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

export function haversineM(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const a = Math.sin(rad(lat2 - lat1) / 2) ** 2 +
    Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

/** Closed ring of [lon, lat] points approximating a circle on the ground. */
export function circlePolygon(lat: number, lon: number, radiusM: number, steps = 48): [number, number][] {
  const ring: [number, number][] = [];
  const d = radiusM / R;
  for (let i = 0; i < steps; i++) {
    const b = (2 * Math.PI * i) / steps;
    const lat2 = Math.asin(Math.sin(rad(lat)) * Math.cos(d) + Math.cos(rad(lat)) * Math.sin(d) * Math.cos(b));
    const lon2 = rad(lon) + Math.atan2(
      Math.sin(b) * Math.sin(d) * Math.cos(rad(lat)),
      Math.cos(d) - Math.sin(rad(lat)) * Math.sin(lat2),
    );
    ring.push([deg(lon2), deg(lat2)]);
  }
  ring.push(ring[0]);
  return ring;
}
```

`dashboard/src/map/layers.ts`:
```ts
import type { NodeView, Track } from "../api/types";
import type { Pulse } from "../state/reducer";
import { circlePolygon } from "./geo";

type Props = Record<string, string | number>;
export interface PointFeature { type: "Feature"; geometry: { type: "Point"; coordinates: [number, number] }; properties: Props }
export interface Collection<F> { type: "FeatureCollection"; features: F[] }

const point = (lon: number, lat: number, properties: Props): PointFeature => ({
  type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties,
});

export function nodesToGeoJSON(nodes: Record<string, NodeView>): Collection<PointFeature> {
  return {
    type: "FeatureCollection",
    features: Object.values(nodes).map((n) =>
      point(n.location.lon, n.location.lat, { id: n.node_id, status: n.status })),
  };
}

export function tracksToGeoJSON(tracks: Record<string, Track>): Collection<PointFeature> {
  return {
    type: "FeatureCollection",
    features: Object.values(tracks).map((t) =>
      point(t.position.lon, t.position.lat, { id: t.track_id, status: t.status, confidence: t.confidence })),
  };
}

export function uncertaintyToGeoJSON(tracks: Record<string, Track>) {
  return {
    type: "FeatureCollection" as const,
    features: Object.values(tracks).map((t) => ({
      type: "Feature" as const,
      geometry: { type: "Polygon" as const, coordinates: [circlePolygon(t.position.lat, t.position.lon, t.uncertainty_m)] },
      properties: { id: t.track_id, status: t.status },
    })),
  };
}

export function trailsToGeoJSON(trails: Record<string, [number, number][]>, tracks: Record<string, Track>) {
  return {
    type: "FeatureCollection" as const,
    features: Object.entries(trails)
      .filter(([id, coords]) => id in tracks && coords.length > 1)
      .map(([id, coords]) => ({
        type: "Feature" as const,
        geometry: { type: "LineString" as const, coordinates: coords },
        properties: { id, status: tracks[id].status },
      })),
  };
}

export function pulsesToGeoJSON(pulses: Pulse[], nodes: Record<string, NodeView>): Collection<PointFeature> {
  return {
    type: "FeatureCollection",
    features: pulses
      .filter((p) => p.nodeId in nodes)
      .map((p) => point(nodes[p.nodeId].location.lon, nodes[p.nodeId].location.lat, { id: p.nodeId })),
  };
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd dashboard && npx vitest run; cd ..`
Expected: all dashboard tests pass. If the generated `types.ts` marks a field as optional that the code reads as required (e.g. `status`), adjust the code with `?? ` defaults, never by hand-editing `types.ts`.

- [ ] **Step 6: Implement the UI**

`dashboard/src/api/client.ts`:
```ts
import type { NodeView, Track, TrackDetail } from "./types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return (await response.json()) as T;
}

export async function fetchSnapshot(): Promise<{ nodes: NodeView[]; tracks: Track[] }> {
  const [nodes, tracks] = await Promise.all([getJson<NodeView[]>("/v1/nodes"), getJson<Track[]>("/v1/tracks")]);
  return { nodes, tracks };
}

export const fetchTrackDetail = (id: string) => getJson<TrackDetail>(`/v1/tracks/${id}`);
```

`dashboard/src/components/MapView.tsx`:
```tsx
import { useEffect, useRef } from "react";
import maplibregl, { type GeoJSONSource } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Action, State } from "../state/reducer";
import { nodesToGeoJSON, pulsesToGeoJSON, trailsToGeoJSON, tracksToGeoJSON, uncertaintyToGeoJSON } from "../map/layers";

const STYLE_URL = "https://tiles.openfreemap.org/styles/liberty";
const HELSINKI: [number, number] = [24.9384, 60.1699];
const NODE_COLORS = ["match", ["get", "status"], "online", "#2e7d32", "stale", "#f9a825", "#9e9e9e"];
const TRACK_COLORS = ["match", ["get", "status"], "confirmed", "#d32f2f", "downgraded", "#757575", "#ffa000"];

export function MapView({ state, dispatch }: { state: State; dispatch: (a: Action) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const loaded = useRef(false);

  useEffect(() => {
    if (!container.current) return;
    const map = new maplibregl.Map({ container: container.current, style: STYLE_URL, center: HELSINKI, zoom: 13 });
    mapRef.current = map;
    map.on("load", () => {
      for (const id of ["nodes", "tracks", "uncertainty", "trails", "pulses"]) {
        map.addSource(id, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      }
      map.addLayer({ id: "uncertainty", type: "fill", source: "uncertainty",
        paint: { "fill-color": TRACK_COLORS as never, "fill-opacity": 0.15 } });
      map.addLayer({ id: "trails", type: "line", source: "trails",
        paint: { "line-color": TRACK_COLORS as never, "line-width": 2 } });
      map.addLayer({ id: "pulses", type: "circle", source: "pulses",
        paint: { "circle-radius": 18, "circle-color": "#1e88e5", "circle-opacity": 0.35 } });
      map.addLayer({ id: "nodes", type: "circle", source: "nodes",
        paint: { "circle-radius": 6, "circle-color": NODE_COLORS as never, "circle-stroke-width": 1, "circle-stroke-color": "#fff" } });
      map.addLayer({ id: "tracks", type: "circle", source: "tracks",
        paint: { "circle-radius": 9, "circle-color": TRACK_COLORS as never, "circle-stroke-width": 2, "circle-stroke-color": "#fff" } });
      map.on("click", "tracks", (e) => {
        const id = e.features?.[0]?.properties?.id;
        if (typeof id === "string") dispatch({ type: "select", trackId: id });
      });
      loaded.current = true;
      map.fire("kuulo:ready");
    });
    return () => map.remove();
  }, [dispatch]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const update = () => {
      const set = (id: string, data: unknown) => (map.getSource(id) as GeoJSONSource | undefined)?.setData(data as never);
      set("nodes", nodesToGeoJSON(state.nodes));
      set("tracks", tracksToGeoJSON(state.tracks));
      set("uncertainty", uncertaintyToGeoJSON(state.tracks));
      set("trails", trailsToGeoJSON(state.trails, state.tracks));
      set("pulses", pulsesToGeoJSON(state.pulses, state.nodes));
    };
    if (loaded.current) update();
    else map.once("kuulo:ready", update);
  }, [state]);

  return <div ref={container} className="map" />;
}
```

`dashboard/src/components/SidePanel.tsx`:
```tsx
import { useEffect, useState } from "react";
import { fetchTrackDetail } from "../api/client";
import type { TrackDetail } from "../api/types";
import type { Action, State } from "../state/reducer";

const ORDER: Record<string, number> = { confirmed: 0, tentative: 1, downgraded: 2 };
const time = (iso?: string | null) => (iso ? new Date(iso).toLocaleTimeString() : "never");

export function SidePanel({ state, dispatch }: { state: State; dispatch: (a: Action) => void }) {
  const [detail, setDetail] = useState<TrackDetail | null>(null);
  const selected = state.selectedTrackId;
  const selectedTrack = selected ? state.tracks[selected] : undefined;

  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    let cancelled = false;
    fetchTrackDetail(selected).then((d) => { if (!cancelled) setDetail(d); }).catch(() => setDetail(null));
    return () => { cancelled = true; };
  }, [selected, selectedTrack?.last_seen]);

  const tracks = Object.values(state.tracks).sort(
    (a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9) || b.confidence - a.confidence);

  return (
    <aside className="panel">
      <header>
        <h1>Kuulo</h1>
        <span className={`conn conn-${state.connection}`}>{state.connection}</span>
      </header>

      <section>
        <h2>Tracks ({tracks.length})</h2>
        {tracks.length === 0 && <p className="muted">No active tracks</p>}
        <ul>
          {tracks.map((t) => (
            <li key={t.track_id} className={t.track_id === selected ? "selected" : ""}
                onClick={() => dispatch({ type: "select", trackId: t.track_id })}>
              <span className={`badge badge-${t.status}`}>{t.status}</span>
              {t.label} · {(t.confidence * 100).toFixed(0)}% · ±{Math.round(t.uncertainty_m)} m
            </li>
          ))}
        </ul>
      </section>

      {detail && (
        <section>
          <h2>Evidence</h2>
          <p>{detail.observations.length} observations from{" "}
            {new Set(detail.observations.map((o) => o.source.id)).size} node(s)</p>
          <ul className="evidence">
            {detail.observations.slice(-12).map((o) => (
              <li key={o.observation_id}>
                {time(o.observed_at)} · {o.source.id} · {o.event.phase} · {(o.detection.confidence * 100).toFixed(0)}%
                {o.acoustic ? ` · SNR ${o.acoustic.snr_db.toFixed(1)} dB` : ""}
              </li>
            ))}
          </ul>
          {detail.silent_neighbours.length > 0 && (
            <p className="muted">Silent online neighbours lowered confidence:{" "}
              {detail.silent_neighbours.map((n) => n.node_id).join(", ")}</p>
          )}
        </section>
      )}

      <section>
        <h2>Nodes ({Object.keys(state.nodes).length})</h2>
        <ul>
          {Object.values(state.nodes).map((n) => (
            <li key={n.node_id}>
              <span className={`dot dot-${n.status}`} /> {n.node_id} · {n.status} · last {time(n.last_heartbeat_at)}
              {n.uncorroborated_rate_24h != null &&
                ` · ${(n.uncorroborated_rate_24h * 100).toFixed(0)}% uncorroborated`}
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
```

`dashboard/src/App.tsx`:
```tsx
import { useEffect, useReducer } from "react";
import { fetchSnapshot } from "./api/client";
import { MapView } from "./components/MapView";
import { SidePanel } from "./components/SidePanel";
import { defaultLiveUrl, startLive } from "./state/live";
import { initialState, reducer } from "./state/reducer";

export function App() {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => startLive({
    url: defaultLiveUrl(),
    openSocket: (url) => new WebSocket(url),
    fetchSnapshot,
    dispatch,
    schedule: (fn, ms) => { window.setTimeout(fn, ms); },
    now: () => Date.now(),
  }), []);

  useEffect(() => {
    const id = window.setInterval(() => dispatch({ type: "expirePulses", now: Date.now() }), 1000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="layout">
      <MapView state={state} dispatch={dispatch} />
      <SidePanel state={state} dispatch={dispatch} />
    </div>
  );
}
```

`dashboard/src/main.tsx`:
```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
```

`dashboard/src/styles.css`:
```css
:root { font-family: system-ui, sans-serif; color-scheme: light dark; }
html, body, #root { margin: 0; height: 100%; }
.layout { display: grid; grid-template-columns: 1fr 360px; height: 100%; }
.map { width: 100%; height: 100%; }
.panel { overflow-y: auto; padding: 12px 16px; border-left: 1px solid #8884; }
.panel header { display: flex; align-items: center; justify-content: space-between; }
.panel h1 { font-size: 20px; margin: 0; }
.panel h2 { font-size: 14px; margin: 16px 0 6px; text-transform: uppercase; letter-spacing: .05em; }
.panel ul { list-style: none; margin: 0; padding: 0; font-size: 13px; }
.panel li { padding: 4px 2px; cursor: default; }
.panel li.selected { background: #1e88e522; }
.muted { opacity: .65; font-size: 13px; }
.conn { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #9e9e9e44; }
.conn-live { background: #2e7d3244; }
.conn-reconnecting { background: #f9a82566; }
.badge { font-size: 11px; padding: 1px 6px; border-radius: 8px; margin-right: 6px; color: #fff; }
.badge-confirmed { background: #d32f2f; } .badge-tentative { background: #ffa000; } .badge-downgraded { background: #757575; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
.dot-online { background: #2e7d32; } .dot-stale { background: #f9a825; } .dot-offline { background: #9e9e9e; }
@media (max-width: 800px) { .layout { grid-template-columns: 1fr; grid-template-rows: 60vh auto; } }
```

- [ ] **Step 7: Build and test**

Run: `cd dashboard && npx vitest run && npm run build; cd ..`
Expected: tests pass; `tsc --noEmit` reports no errors; `vite build` writes `dashboard/dist/` (gitignored). Fix type errors in app code, not in generated `types.ts` (regenerate instead).

- [ ] **Step 8: Commit**

```bash
git add dashboard ':!dashboard/node_modules' ':!dashboard/dist'
git commit -m "feat(dashboard): live MapLibre map with tracks, evidence panel and resilient WebSocket

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---
### Task 8: Basic fusion, server integration and scenario tests

**Files:**
- Create: `server/src/kuulo_server/fusion/__init__.py`, `fusion/base.py`, `fusion/basic.py`, `fusion/context.py`, `fusion/loader.py`
- Modify: `server/src/kuulo_server/tracks.py` (persist, close-at-startup, uncorroborated rate), `server/src/kuulo_server/app.py` (run fusion on ingest and tick, node views with rate)
- Test: `server/tests/test_fusion_basic.py`, `server/tests/test_fusion_server.py`, `server/tests/test_scenarios.py`

**Interfaces:**
- Consumes: `kuulo_protocol.geo.distance_m/to_local/from_local`, models, `NodeStatus`; server `Settings`, `db`, `status.node_status`, `LiveHub`; `kuulo_sim.engine.SimulationRun`, `kuulo_sim.scenario.load_scenario/resolve_scenario` (tests only; importable because the workspace root depends on `kuulo-sim`).
- Produces:
  - `kuulo_server.fusion.base`: `@dataclass(frozen=True) NodeInfo(node_id: str, location: GeoPoint, status: NodeStatus)`, `@dataclass(frozen=True) TrackUpdate(track: Track)`, `class FusionContext(Protocol)` with `nodes() -> list[NodeInfo]` and `recent_observations(since: datetime) -> list[Observation]` (non-late, drone-labelled), `class FusionEngine(Protocol)` with `on_observation(obs, ctx) -> list[TrackUpdate]` and `on_tick(now, ctx) -> list[TrackUpdate]`.
  - `kuulo_server.fusion.basic`: `@dataclass(frozen=True) FusionConfig(...)` (defaults from Global Constraints), `class BasicFusion(config: FusionConfig = FusionConfig())`.
  - `kuulo_server.fusion.context`: `class DbFusionContext(sessions: sessionmaker, now: datetime)`.
  - `kuulo_server.fusion.loader`: `load_engine(spec: str) -> FusionEngine`.
  - `kuulo_server.tracks` additions: `persist_track(session, track: Track) -> None`, `close_open_tracks(session) -> int`, `uncorroborated_rate(session, node_id: str, now: datetime) -> float | None`.

- [ ] **Step 1: Write the failing fusion unit tests**

`server/tests/test_fusion_basic.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from kuulo_protocol.api import NodeStatus
from kuulo_protocol.geo import distance_m, from_local
from kuulo_protocol.models import GeoPoint, Label, Observation, Phase, TrackStatus
from kuulo_protocol.testing import make_observation
from kuulo_server.fusion.base import NodeInfo
from kuulo_server.fusion.basic import BasicFusion
from kuulo_server.testing import T0

ORIGIN = GeoPoint(lat=60.1699, lon=24.9384)


def at(x: float, y: float) -> GeoPoint:
    return from_local(ORIGIN, x, y)


@dataclass
class Ctx:
    node_infos: dict[str, NodeInfo] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)

    def add_node(self, node_id, point, status=NodeStatus.ONLINE):
        self.node_infos[node_id] = NodeInfo(node_id, point, status)

    def nodes(self):
        return list(self.node_infos.values())

    def recent_observations(self, since: datetime):
        return [o for o in self.observations if o.observed_at >= since]


def observe(ctx, fusion, node_id, t: datetime, confidence=0.9, snr=12.0, label=Label.DRONE_MULTIROTOR,
            phase=Phase.START):
    p = ctx.node_infos[node_id].location
    obs = make_observation(node_id, observed_at=t, lat=p.lat, lon=p.lon, confidence=confidence,
                           snr_db=snr, label=label, phase=phase)
    ctx.observations.append(obs)
    return fusion.on_observation(obs, ctx)


def test_single_node_is_tentative_at_node():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    [update] = observe(ctx, fusion, "n1", T0)
    track = update.track
    assert track.status is TrackStatus.TENTATIVE
    assert distance_m(track.position, ORIGIN) < 1
    assert track.uncertainty_m == 300
    assert track.confidence == pytest.approx(0.9)


def test_two_nodes_confirm_and_centroid_leans_to_louder():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(600, 0))
    observe(ctx, fusion, "n1", T0, snr=20)
    [update] = observe(ctx, fusion, "n2", T0 + timedelta(seconds=2), snr=6)
    x_east = distance_m(ORIGIN, GeoPoint(lat=ORIGIN.lat, lon=update.track.position.lon))
    assert update.track.status is TrackStatus.CONFIRMED
    assert x_east < 300  # pulled toward the louder n1
    assert len(fusion.tracks) == 1


def test_silent_online_neighbour_downgrades_offline_does_not():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(250, 0))                          # online, within 500 m, silent
    ctx.add_node("n3", at(0, 250), NodeStatus.OFFLINE)      # offline: ignored
    [update] = observe(ctx, fusion, "n1", T0)
    assert update.track.status is TrackStatus.DOWNGRADED
    assert update.track.silent_neighbour_ids == ["n2"]
    assert update.track.confidence == pytest.approx(0.9 * 0.7)


def test_downgraded_is_promoted_when_second_node_hears():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(250, 0))
    observe(ctx, fusion, "n1", T0)
    [update] = observe(ctx, fusion, "n2", T0 + timedelta(seconds=3))
    assert update.track.status is TrackStatus.CONFIRMED
    assert update.track.silent_neighbour_ids == []


def test_non_drone_and_end_phase_ignored():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    assert observe(ctx, fusion, "n1", T0, label=Label.BIRD) == []
    assert observe(ctx, fusion, "n1", T0, phase=Phase.END) == []


def test_same_drone_updates_same_track_with_velocity():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(400, 0))
    [first] = observe(ctx, fusion, "n1", T0)
    [second] = observe(ctx, fusion, "n2", T0 + timedelta(seconds=5))
    assert second.track.track_id == first.track.track_id
    assert second.track.velocity is not None and second.track.velocity.speed_mps > 0
    assert len(second.track.observation_ids) == 2


def test_far_apart_groups_make_separate_tracks():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("a", at(0, 0))
    ctx.add_node("b", at(5000, 0))
    [ta] = observe(ctx, fusion, "a", T0)
    [tb] = observe(ctx, fusion, "b", T0)
    assert ta.track.track_id != tb.track.track_id


def test_tick_closes_quiet_tracks():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    observe(ctx, fusion, "n1", T0)
    assert fusion.on_tick(T0 + timedelta(seconds=29), ctx) == []
    [closed] = fusion.on_tick(T0 + timedelta(seconds=31), ctx)
    assert closed.track.status is TrackStatus.CLOSED
    assert fusion.tracks == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest server/tests/test_fusion_basic.py`
Expected: `ModuleNotFoundError: No module named 'kuulo_server.fusion'`.

- [ ] **Step 3: Implement the fusion package**

`server/src/kuulo_server/fusion/__init__.py`:
```python
"""Fusion engines turn observations into tracks. BasicFusion is the public baseline."""
```

`server/src/kuulo_server/fusion/base.py`:
```python
"""The plugin interface every fusion engine implements (the private engine included)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from kuulo_protocol.api import NodeStatus
from kuulo_protocol.models import GeoPoint, Observation, Track


@dataclass(frozen=True)
class NodeInfo:
    node_id: str
    location: GeoPoint
    status: NodeStatus


@dataclass(frozen=True)
class TrackUpdate:
    track: Track


class FusionContext(Protocol):
    def nodes(self) -> list[NodeInfo]: ...

    def recent_observations(self, since: datetime) -> list[Observation]:
        """Non-late, drone-labelled observations with observed_at >= since."""
        ...


class FusionEngine(Protocol):
    def on_observation(self, obs: Observation, ctx: FusionContext) -> list[TrackUpdate]: ...

    def on_tick(self, now: datetime, ctx: FusionContext) -> list[TrackUpdate]: ...
```

`server/src/kuulo_server/fusion/basic.py`:
```python
"""Baseline fusion: group nearby detections, weighted-centroid location, silent-neighbour logic.

Coarse by design (hundreds of metres). Precise localisation (TDOA, bearings, tracking filters)
belongs to the private engine behind the same interface.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import atan2, degrees, hypot
from uuid import uuid4

from kuulo_protocol.api import NodeStatus
from kuulo_protocol.geo import distance_m, from_local, to_local
from kuulo_protocol.models import (
    DRONE_LABELS,
    GeoPoint,
    Observation,
    Phase,
    Track,
    TrackStatus,
    Velocity,
)

from .base import FusionContext, TrackUpdate


@dataclass(frozen=True)
class FusionConfig:
    group_radius_m: float = 2000.0
    group_window_s: float = 10.0
    base_range_m: float = 300.0
    gate_m: float = 1000.0
    gate_s: float = 10.0
    close_after_s: float = 30.0
    expected_hearing_m: float = 500.0
    silent_penalty: float = 0.7
    smoothing_alpha: float = 0.5


def _weight(obs: Observation) -> float:
    snr = obs.acoustic.snr_db if obs.acoustic else 0.0
    return obs.detection.confidence * 10 ** (snr / 20)


class BasicFusion:
    def __init__(self, config: FusionConfig = FusionConfig()):
        self.config = config
        self.tracks: dict[str, Track] = {}

    def on_observation(self, obs: Observation, ctx: FusionContext) -> list[TrackUpdate]:
        cfg = self.config
        if obs.detection.label not in DRONE_LABELS or obs.event.phase is Phase.END:
            return []
        since = obs.observed_at - timedelta(seconds=cfg.group_window_s)
        candidates = [
            o for o in ctx.recent_observations(since)
            if o.detection.label in DRONE_LABELS
            and o.event.phase is not Phase.END
            and o.observed_at <= obs.observed_at
            and distance_m(o.sensor_location, obs.sensor_location) <= cfg.group_radius_m
        ]
        if all(o.observation_id != obs.observation_id for o in candidates):
            candidates.append(obs)
        latest: dict[str, Observation] = {}
        for o in sorted(candidates, key=lambda o: o.observed_at):
            latest[o.source.id] = o
        group = list(latest.values())

        centroid = self._centroid(group)
        spread = max(distance_m(centroid, o.sensor_location) for o in group)
        uncertainty = max(spread, cfg.base_range_m)
        detecting = set(latest)
        silent = sorted(
            n.node_id for n in ctx.nodes()
            if n.status is NodeStatus.ONLINE
            and n.node_id not in detecting
            and any(distance_m(n.location, o.sensor_location) <= cfg.expected_hearing_m for o in group)
        )
        if len(group) >= 2:
            status = TrackStatus.CONFIRMED
            confidence = 1.0
            for o in group:
                confidence *= 1 - o.detection.confidence
            confidence = 1 - confidence
            silent = []
        else:
            status = TrackStatus.DOWNGRADED if silent else TrackStatus.TENTATIVE
            confidence = group[0].detection.confidence * cfg.silent_penalty ** len(silent)
        label = Counter(o.detection.label for o in group).most_common(1)[0][0]

        existing = self._associate(centroid, obs.observed_at)
        if existing is None:
            track = Track(
                track_id=uuid4(), status=status, label=label, confidence=confidence,
                position=centroid, uncertainty_m=uncertainty, velocity=None,
                first_seen=obs.observed_at, last_seen=obs.observed_at,
                observation_ids=[o.observation_id for o in group],
                silent_neighbour_ids=silent,
            )
        else:
            track = self._update(existing, centroid, status, label, confidence, uncertainty, group,
                                 silent, obs.observed_at)
        self.tracks[str(track.track_id)] = track
        return [TrackUpdate(track)]

    def on_tick(self, now: datetime, ctx: FusionContext) -> list[TrackUpdate]:
        updates = []
        for key, track in list(self.tracks.items()):
            if (now - track.last_seen).total_seconds() > self.config.close_after_s:
                closed = track.model_copy(update={"status": TrackStatus.CLOSED})
                del self.tracks[key]
                updates.append(TrackUpdate(closed))
        return updates

    def _centroid(self, group: list[Observation]) -> GeoPoint:
        origin = group[0].sensor_location
        total = sum(_weight(o) for o in group) or 1.0
        x = y = 0.0
        for o in group:
            ox, oy = to_local(origin, o.sensor_location)
            x += ox * _weight(o) / total
            y += oy * _weight(o) / total
        return from_local(origin, x, y)

    def _associate(self, position: GeoPoint, at: datetime) -> Track | None:
        best, best_d = None, None
        for track in self.tracks.values():
            if abs((at - track.last_seen).total_seconds()) > self.config.gate_s:
                continue
            d = distance_m(track.position, position)
            if d <= self.config.gate_m and (best_d is None or d < best_d):
                best, best_d = track, d
        return best

    def _update(self, old: Track, centroid: GeoPoint, status: TrackStatus, label, confidence: float,
                uncertainty: float, group: list[Observation], silent: list[str],
                at: datetime) -> Track:
        a = self.config.smoothing_alpha
        dx, dy = to_local(old.position, centroid)
        position = from_local(old.position, a * dx, a * dy)
        dt = (at - old.last_seen).total_seconds()
        velocity = old.velocity
        if dt > 0:
            mx, my = to_local(old.position, position)
            velocity = Velocity(
                speed_mps=hypot(mx, my) / dt, heading_deg=degrees(atan2(mx, my)) % 360
            )
        if old.status is TrackStatus.CONFIRMED and status is not TrackStatus.CONFIRMED:
            status, silent = TrackStatus.CONFIRMED, []  # a confirmed track stays confirmed
            confidence = max(confidence, old.confidence)
        ids = list(dict.fromkeys([*old.observation_ids, *(o.observation_id for o in group)]))
        return old.model_copy(update={
            "status": status, "label": label, "confidence": confidence, "position": position,
            "uncertainty_m": uncertainty, "velocity": velocity, "last_seen": max(at, old.last_seen),
            "observation_ids": ids, "silent_neighbour_ids": silent,
        })
```

`server/src/kuulo_server/fusion/context.py`:
```python
"""FusionContext backed by the server database."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from kuulo_protocol.models import DRONE_LABELS, GeoPoint, Observation

from ..db import NodeRow, ObservationRow, as_utc
from ..status import node_status
from .base import NodeInfo


class DbFusionContext:
    def __init__(self, sessions: sessionmaker, now: datetime):
        self._sessions = sessions
        self._now = now

    def nodes(self) -> list[NodeInfo]:
        with self._sessions() as session:
            return [
                NodeInfo(row.node_id, GeoPoint(lat=row.lat, lon=row.lon),
                         node_status(as_utc(row.last_heartbeat_at), self._now))
                for row in session.scalars(select(NodeRow))
            ]

    def recent_observations(self, since: datetime) -> list[Observation]:
        labels = [label.value for label in DRONE_LABELS]
        with self._sessions() as session:
            rows = session.scalars(
                select(ObservationRow)
                .where(ObservationRow.observed_at >= since)
                .where(ObservationRow.late.is_(False))
                .where(ObservationRow.label.in_(labels))
                .order_by(ObservationRow.observed_at)
            )
            return [Observation.model_validate_json(r.raw_json) for r in rows]
```

SQLite stores datetimes without timezone; comparing `ObservationRow.observed_at >= since` with an aware `since` works because SQLAlchemy renders both as ISO strings in UTC. Keep every stored datetime in UTC (they are, via `to_utc_ms` and `Settings.clock`).

`server/src/kuulo_server/fusion/loader.py`:
```python
"""Load a fusion engine from a "module:Class" string, so a private engine can be dropped in."""

from __future__ import annotations

from importlib import import_module

from .base import FusionEngine


def load_engine(spec: str) -> FusionEngine:
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"fusion engine must look like 'module:Class', got {spec!r}")
    return getattr(import_module(module_name), class_name)()
```

- [ ] **Step 4: Run the unit tests**

Run: `uv run pytest server/tests/test_fusion_basic.py`
Expected: `8 passed`.

- [ ] **Step 5: Write failing server-integration tests**

`server/tests/test_fusion_server.py`:
```python
from fastapi.testclient import TestClient

from kuulo_protocol.models import Track, TrackStatus
from kuulo_protocol.testing import make_heartbeat, make_observation
from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.testing import T0, FakeClock, post_signed, register


def test_observations_create_confirmed_track_with_evidence(client, node, clock):
    n2 = register(client, "n2", lon=24.9438)
    post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0), node.private_key)
    post_signed(client, "/v1/observations",
                make_observation("n2", observed_at=T0, lon=24.9438), n2.private_key)
    [track] = client.get("/v1/tracks").json()
    assert track["status"] == "confirmed"
    detail = client.get(f"/v1/tracks/{track['track_id']}").json()
    assert {o["source"]["id"] for o in detail["observations"]} == {"n1", "n2"}


def test_track_published_live_and_closed_by_tick(app, client, node, clock):
    with client.websocket_connect("/v1/live") as ws:
        post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0),
                    node.private_key)
        assert ws.receive_json()["type"] == "observation"
        track_event = ws.receive_json()
        assert track_event["type"] == "track" and track_event["data"]["status"] == "tentative"
        clock.advance(40)
        app.state.run_tick()
        closed = ws.receive_json()
        assert closed["type"] == "track" and closed["data"]["status"] == "closed"


def test_fusion_crash_does_not_break_ingest(tmp_path):
    clock = FakeClock(T0)
    settings = Settings(db_path=tmp_path / "k.db", clock=clock, tick_interval_s=None,
                        fusion_engine="tests_support_broken:Broken")
    import sys
    import types

    module = types.ModuleType("tests_support_broken")

    class Broken:
        def on_observation(self, obs, ctx):
            raise RuntimeError("boom")

        def on_tick(self, now, ctx):
            raise RuntimeError("boom")

    module.Broken = Broken
    sys.modules["tests_support_broken"] = module
    app = create_app(settings)
    with TestClient(app) as c:
        n1 = register(c, "n1")
        r = post_signed(c, "/v1/observations", make_observation("n1", observed_at=T0), n1.private_key)
        assert r.status_code == 200 and r.json()["status"] == "accepted"
        app.state.run_tick()  # must not raise


def test_restart_closes_open_tracks(tmp_path):
    # Review Focus 3.
    clock = FakeClock(T0)
    settings = Settings(db_path=tmp_path / "k.db", clock=clock, tick_interval_s=None)
    with TestClient(create_app(settings)) as c:
        n1 = register(c, "n1")
        post_signed(c, "/v1/observations", make_observation("n1", observed_at=T0), n1.private_key)
        assert c.get("/v1/tracks").json()[0]["status"] == "tentative"
    with TestClient(create_app(settings)) as c:
        [track] = c.get("/v1/tracks").json()
        assert Track.model_validate(track).status is TrackStatus.CLOSED


def test_uncorroborated_rate(client, node, clock):
    n2 = register(client, "n2", lon=24.9438)
    post_signed(client, "/v1/heartbeats", make_heartbeat("n1", sent_at=T0), node.private_key)
    post_signed(client, "/v1/observations", make_observation("n1", observed_at=T0), node.private_key)
    views = {n["node_id"]: n for n in client.get("/v1/nodes").json()}
    assert views["n1"]["uncorroborated_rate_24h"] == 1.0
    assert views["n2"]["uncorroborated_rate_24h"] is None
    clock.advance(2)
    post_signed(client, "/v1/observations",
                make_observation("n2", observed_at=clock(), lon=24.9438), n2.private_key)
    views = {n["node_id"]: n for n in client.get("/v1/nodes").json()}
    assert views["n1"]["uncorroborated_rate_24h"] == 0.0
```

- [ ] **Step 6: Implement persistence helpers**

Append to `server/src/kuulo_server/tracks.py` (its existing imports already cover `datetime`, `timedelta`, `select`, `Session` and the row classes):
```python
def persist_track(session: Session, track: Track) -> None:
    track_id = str(track.track_id)
    row = session.get(TrackRow, track_id)
    if row is None:
        row = TrackRow(track_id=track_id, ever_confirmed=False)
        session.add(row)
    row.status = track.status.value
    row.ever_confirmed = bool(row.ever_confirmed) or track.status is TrackStatus.CONFIRMED
    row.last_seen = track.last_seen
    row.raw_json = track.model_dump_json()
    known = set(session.scalars(
        select(TrackObservationRow.observation_id).where(TrackObservationRow.track_id == track_id)
    ))
    for obs_id in map(str, track.observation_ids):
        if obs_id not in known:
            session.add(TrackObservationRow(track_id=track_id, observation_id=obs_id))
    session.commit()


def close_open_tracks(session: Session) -> int:
    """Tracks cannot survive a restart (fusion state is in memory), so close them explicitly."""
    rows = list(session.scalars(select(TrackRow).where(TrackRow.status != TrackStatus.CLOSED.value)))
    for row in rows:
        track = Track.model_validate_json(row.raw_json).model_copy(
            update={"status": TrackStatus.CLOSED}
        )
        row.status = TrackStatus.CLOSED.value
        row.raw_json = track.model_dump_json()
    session.commit()
    return len(rows)


def uncorroborated_rate(session: Session, node_id: str, now: datetime) -> float | None:
    """Share of this node's drone detections in 24 h that never joined a confirmed track."""
    since = now - timedelta(hours=24)
    drone = ["drone_multirotor", "drone_fixedwing_engine"]
    detections = set(session.scalars(
        select(ObservationRow.detection_id)
        .where(ObservationRow.node_id == node_id)
        .where(ObservationRow.observed_at >= since)
        .where(ObservationRow.label.in_(drone))
    ))
    if not detections:
        return None
    corroborated = set(session.scalars(
        select(ObservationRow.detection_id)
        .join(TrackObservationRow, TrackObservationRow.observation_id == ObservationRow.observation_id)
        .join(TrackRow, TrackRow.track_id == TrackObservationRow.track_id)
        .where(ObservationRow.node_id == node_id)
        .where(ObservationRow.detection_id.in_(detections))
        .where(TrackRow.ever_confirmed.is_(True))
    ))
    return 1 - len(corroborated) / len(detections)
```

- [ ] **Step 7: Wire fusion into the app**

In `server/src/kuulo_server/app.py`:

1. Imports: add
```python
from .fusion.context import DbFusionContext
from .fusion.loader import load_engine
from .tracks import close_open_tracks, open_tracks, persist_track, track_detail, uncorroborated_rate
```
(replace the existing `from .tracks import ...` line).

2. After `hub = LiveHub()` add:
```python
    engine = load_engine(settings.fusion_engine)
    with sessions() as session:
        closed = close_open_tracks(session)
    if closed:
        log.info("closed %d tracks left open by a previous run", closed)

    def apply_updates(updates) -> None:
        with sessions() as session:
            for update in updates:
                persist_track(session, update.track)
        for update in updates:
            hub.publish(LiveEvent(type="track", data=update.track))
            if settings.on_track_update:
                settings.on_track_update(update.track)

    def run_fusion(call) -> None:
        try:
            apply_updates(call(DbFusionContext(sessions, settings.clock())))
        except Exception:
            log.exception("fusion engine failed; ingest continues")

    def view_of(session, row, now) -> NodeView:
        return node_view(row, now, uncorroborated_rate(session, row.node_id, now))
```

3. In `run_tick`, replace `view = node_view(row, now)` with `view = view_of(session, row, now)`, and after the `with` block add:
```python
        run_fusion(lambda ctx: engine.on_tick(now, ctx))
```

4. In `post_observation`, after publishing the observation event (inside the same `if` block) add:
```python
            run_fusion(lambda ctx: engine.on_observation(obs, ctx))
```

5. In `post_heartbeat` use `view = view_of(session, row, now)`; in `get_nodes` return `[view_of(session, row, now) for row in session.scalars(select(NodeRow))]`.

6. `app.state.engine = engine`.

- [ ] **Step 8: Run server tests**

Run: `uv run pytest server/tests && uv run ruff check .`
Expected: all pass. `test_fusion_crash_does_not_break_ingest` logs a traceback (expected).

- [ ] **Step 9: Write the scenario harness and scenario tests**

`server/tests/conftest.py` — add these imports at the top of the file:
```python
from dataclasses import dataclass
from datetime import timedelta

from kuulo_protocol.models import Track
from kuulo_sim.engine import SimulationRun, TruthPoint
from kuulo_sim.scenario import load_scenario, resolve_scenario
```
and append:
```python
@dataclass
class ScenarioResult:
    run: SimulationRun
    updates: list[Track]
    truth: list[TruthPoint]


@pytest.fixture
def run_scenario(tmp_path):
    """Play a bundled scenario through the real server in-process with a fake clock."""

    def _run(name: str) -> ScenarioResult:
        run = SimulationRun(load_scenario(resolve_scenario(name)), T0)
        clock = FakeClock(T0)
        updates: list[Track] = []
        app = create_app(Settings(db_path=tmp_path / f"{name}.db", clock=clock,
                                  tick_interval_s=None, on_track_update=updates.append))
        with TestClient(app) as c:
            for reg in run.registrations():
                assert c.post("/v1/nodes/register", json=reg.model_dump(mode="json")).status_code == 200
            next_tick = T0
            for msg in run.messages():
                while next_tick <= msg.at:
                    clock.set(next_tick)
                    app.state.run_tick()
                    next_tick += timedelta(seconds=1)
                clock.set(msg.at)
                path = "/v1/observations" if msg.kind == "observation" else "/v1/heartbeats"
                r = c.post(path, content=msg.payload.model_dump_json(),
                           headers={"content-type": "application/json"})
                assert r.status_code == 200, r.text
            while next_tick <= run.end_at + timedelta(seconds=60):
                clock.set(next_tick)
                app.state.run_tick()
                next_tick += timedelta(seconds=1)
        return ScenarioResult(run, updates, run.truth())

    return _run
```

`server/tests/test_scenarios.py`:
```python
from kuulo_protocol.geo import distance_m
from kuulo_protocol.models import TrackStatus

LOCATION_ERROR_BOUND_M = 750.0


def statuses(result):
    return {u.status for u in result.updates}


def mean_location_error(result) -> float:
    errors = []
    for track in result.updates:
        if track.status is not TrackStatus.CONFIRMED:
            continue
        near = [p for p in result.truth if abs((p.at - track.last_seen).total_seconds()) <= 0.5]
        if near:
            errors.append(min(distance_m(track.position, p.position) for p in near))
    assert errors, "no confirmed track could be matched to ground truth"
    return sum(errors) / len(errors)


def test_single_node_only_tentative(run_scenario):
    result = run_scenario("single_node")
    assert result.updates
    assert TrackStatus.CONFIRMED not in statuses(result)
    assert TrackStatus.TENTATIVE in statuses(result)


def test_helsinki_pass_confirmed_and_located(run_scenario):
    result = run_scenario("helsinki_pass")
    assert TrackStatus.CONFIRMED in statuses(result)
    error = mean_location_error(result)
    print(f"helsinki_pass mean location error: {error:.0f} m")
    assert error < LOCATION_ERROR_BOUND_M


def test_false_alarm_never_confirmed(run_scenario):
    result = run_scenario("false_alarm")
    assert TrackStatus.CONFIRMED not in statuses(result)
    assert TrackStatus.DOWNGRADED in statuses(result)


def test_two_drones_two_tracks(run_scenario):
    result = run_scenario("two_drones")
    confirmed_ids = {u.track_id for u in result.updates if u.status is TrackStatus.CONFIRMED}
    assert len(confirmed_ids) >= 2
    node_of = {
        m.payload.observation_id: m.payload.source.id
        for m in result.run.messages() if m.kind == "observation"
    }
    for track in result.updates:
        clusters = {node_of[i][0] for i in track.observation_ids}
        assert len(clusters) == 1, f"track {track.track_id} mixes clusters {clusters}"


def test_offline_neighbour_silence_not_counted(run_scenario):
    result = run_scenario("node_failure")
    assert TrackStatus.DOWNGRADED not in statuses(result)
    assert TrackStatus.TENTATIVE in statuses(result)
```

- [ ] **Step 10: Run all tests**

Run: `uv run pytest -s server/tests/test_scenarios.py && uv run pytest && uv run ruff check .`
Expected: all pass; the helsinki_pass line prints the measured mean location error. Record that number — it goes into the README in Part 2. If a scenario test fails, investigate with `superpowers:systematic-debugging`; change fusion or scenario geometry only with a stated reason, never loosen an assertion silently.

- [ ] **Step 11: Commit**

```bash
git add server
git commit -m "feat(fusion): BasicFusion with silent-neighbour logic, server integration, scenario tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: One-command dev environment and end-to-end smoke run

**Files:**
- Create: `server/src/kuulo_server/main.py`, `Makefile`, `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `make dev`, `make server`, `make dashboard`, `make sim SCENARIO=<name> SPEED=<x>`, `make test`, `make types`.

- [ ] **Step 1: Server entry point**

`server/src/kuulo_server/main.py`:
```python
"""Uvicorn entry point: `uvicorn kuulo_server.main:app`."""

import logging

from .app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
```

- [ ] **Step 2: Makefile**

`Makefile` (recipes indented with a TAB):
```make
.PHONY: dev server dashboard sim test types

SCENARIO ?= helsinki_pass
SPEED ?= 1

server:
	uv run uvicorn kuulo_server.main:app --host 127.0.0.1 --port 8000

dashboard:
	cd dashboard && npm run dev

dev:
	$(MAKE) -j2 server dashboard

sim:
	uv run kuulo-sim run $(SCENARIO) --speed $(SPEED)

types:
	cd dashboard && npm run gen:types

test:
	uv run ruff check .
	uv run pytest
	cd dashboard && npx vitest run
```

- [ ] **Step 3: Development README**

`README.md`:
````markdown
# Kuulo

Civic acoustic drone-detection network, starting in Helsinki. This repository is the open core:
protocol, server, basic fusion, simulator and dashboard. See
`docs/superpowers/specs/2026-09-24-kuulo-milestone1-design.md` for the design.

> Status: Milestone 1, part 1 (core pipeline). The node software, ML detector and full README
> arrive in part 2. Localhost only — do not expose the server to the internet.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Node.js 20+.

```bash
uv sync
(cd dashboard && npm install)
make dev            # server on 127.0.0.1:8000, dashboard on http://127.0.0.1:5173
make sim            # in a second terminal: plays the helsinki_pass scenario
make sim SCENARIO=false_alarm SPEED=5
```

## Tests

```bash
make test
```
````

- [ ] **Step 4: End-to-end smoke run (real processes, real HTTP)**

Run each in the background with the Bash tool's background option (or `&`), then check:
```bash
uv run uvicorn kuulo_server.main:app --host 127.0.0.1 --port 8000 > "$TMPDIR/kuulo-server.log" 2>&1 &
sleep 3   # if foreground sleep is blocked, poll with: until curl -sf http://127.0.0.1:8000/v1/nodes; do :; done
uv run kuulo-sim run helsinki_pass --speed 10
curl -s http://127.0.0.1:8000/v1/tracks | python3 -c "import json,sys; t=json.load(sys.stdin); print(len(t), sorted({x['status'] for x in t}))"
```
Expected: the sim prints `helsinki_pass: sent N, rejected 0`; the curl line prints at least one track and its statuses (`closed` or `confirmed` depending on timing). Also confirm `cd dashboard && npm run build` still succeeds. Stop the server afterwards (`kill %1` or `pkill -f "uvicorn kuulo_server"`). Delete `data/kuulo.db` if you want a clean state (it is gitignored).

A person must still do the visual check in a browser (`make dev` then `make sim`, open http://127.0.0.1:5173): nodes appear green, a pulse flashes at each detecting node, an amber then red track moves west→east with a trail and an uncertainty circle, and clicking it shows the evidence list. Note in the final report that this visual check is pending for the user.

- [ ] **Step 5: Full test run and commit**

Run: `make test`
Expected: ruff clean, all pytest and vitest tests pass.

```bash
git add Makefile README.md server/src/kuulo_server/main.py
git commit -m "chore: make dev/test/sim targets, development README, smoke-tested end to end

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

# Kuulo Impulse Events Implementation Plan (public repository)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nodes detect sharp acoustic impulses (blasts, crashes) and report them with precise UTC
onset times. The server groups the reports into impulse events, places them on the map (coarsely
with the public locator, precisely with the private one) and exports them to the dashboard, API
and CAP 1.2 for emergency services and counter-UAS operators.

**Architecture:** A node-side `ImpulseDetector` (STA/LTA trigger + AIC onset picker) sends signed
`ImpulseReport`s, timestamped from the audio driver's capture clock. The server stores them,
groups physically consistent reports, and hands each group to a pluggable locator. The public
`CoarseLocator` draws a circle around the hearing nodes. The private fusion engine (not in this
repository) plugs in through `KUULO_IMPULSE_LOCATOR` for precise positions and drone attribution. The simulator gains blasts
and detonating drones; the dashboard gains an impulse layer. No audio leaves the node.

**Tech Stack:** Python 3.12 (uv workspace), NumPy/SciPy (`scipy.signal`), FastAPI, SQLAlchemy,
sounddevice/PortAudio, React + TypeScript + MapLibre + Blueprint, Vitest, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-kuulo-impulse-events-design.md` (read first).
Parent: `2026-09-24-kuulo-milestone1-design.md`.

**Interfaces other code depends on:** Task 3's `impact/base.py` (locator protocol, `LocatorConfig`,
grouping helpers) and `testing.py` helpers (`register_ring`, `post_bang`), and Task 7's scenario
files. Keep their names and signatures exactly as written.

## Decisions (resolved with the user, 2026-09-26)

| # | Decision | Resolution |
|---|---|---|
| D1 | Public or private | **Private** engine in a separate private repo; this repo gets the interface, a coarse public locator, and everything nodes, dashboard and alerts need |
| D2 | 3-node ambiguity | Show both points, marked AMBIGUOUS, and let a drone track break the tie (private locator). This repo only renders `alternatives` |
| D3 | CAP 1.2 export | Yes; status `Test` by default |
| D4 | Fix BasicFusion speed | Yes (Task 2) |
| D5 | Real-audio detector evaluation | Yes (Task 9) |
| D6 | Labels | IMPACT (red, attributed to a drone track) / IMPULSE (orange) |

## Under the hood (read before building)

- **STA/LTA trigger** (seismology). The node keeps a short-term average of energy (2 ms) and a
  long-term average (5 s). A bang makes the ratio jump by 20 dB or more; slow changes do not.
- **AIC onset picker** (Maeda, 1985, seismology). For every candidate split point *k* in a window,
  AIC(k) = k·log var(x[:k]) + (n−k−1)·log var(x[k:]). The minimum is where "noise" becomes
  "signal", to within a fraction of a millisecond at high SNR. It is more precise than the
  trigger instant.
- **Why the decay and bandwidth checks.** A bang rises in milliseconds, is broadband and dies
  away. A lorry pulling up is sustained; a whistle is narrowband. Those two checks, plus the rise
  time, reject most things that are not bangs, without any machine learning.
- **UTC from the audio driver.** PortAudio stamps each input buffer with the capture time of its
  first sample (`inputBufferAdcTime`). The node converts that to UTC. Using "time when Python
  processed the block" would add queue delay, and a sound card's sample clock drifts tens of
  milliseconds per hour. Re-anchoring every block removes both.
- **Physical grouping.** Two reports can belong to one bang only if their onset difference is at
  most the travel time between the two nodes (distance / speed of sound) plus timing slack. This
  single rule groups reports and separates simultaneous bangs in different places.
- **Plug-in seam.** The server loads its locator from a `"module:Class"` string, set by an
  environment variable. That is how a private package supplies the precise locator without the
  public code ever naming it, the same pattern the fusion engine already uses.
- **CAP 1.2** (OASIS Common Alerting Protocol) is the XML format emergency alerting systems
  exchange. Mapping an event onto it makes Kuulo's output consumable without custom integration.
- **C++?** No genuine fit: the detector handles 500 energy frames per second in NumPy, whose cores
  are already C.

## Global Constraints

- Python `>=3.12,<3.13`; all commands `uv run --no-sync …`; ruff line length 100.
- **No audio, waveform or envelope leaves the node.** `ImpulseReport` carries only onset time,
  σ, summary features and the 32-band spectrum at the peak (spec §2).
- **The public repository never names or imports the private package**, and never contains
  multilateration code (D1).
- Wire models forbid `inf`/`nan`: every number a locator produces must be finite.
- Existing scenario outputs must not change: the simulator's impulse randomness uses its own RNG.
- Late impulse reports are accepted and used; reports more than 30 s in the future are rejected.
- CAP `status` defaults to `Test`.
- CI never downloads data (Task 9's evaluation runs locally via `make impulse-eval`).
- Commits end with the attribution line from the executing session's system reminder.

## Review Focus

1. **`kuulo-sim --speed 4` compressing arrival differences.** Emission time follows the scenario's
   time scale; propagation, clock error and picking noise must not (Task 7 test
   `test_speed_factor_does_not_compress_arrival_differences`).
2. **Implausible track speeds** (the demo showed 2647 m/s) must be fixed at the source (Task 2
   test `test_helsinki_pass_reports_a_plausible_speed`); the private locator also clamps them.
3. **Reports arriving late or out of order** after an outage must update the same event, not
   duplicate it (Task 3 test `test_late_report_refines_the_same_event`).
4. **A sustained noise that starts abruptly** must not produce a report every dead-time interval
   (Task 5 test `test_step_to_sustained_noise_is_not_an_impulse_and_does_not_retrigger`).
5. **A misconfigured or crashing locator plug-in** must fail loudly at startup, or be isolated at
   runtime, never taking ingest down (Task 3 tests `test_unknown_locator_fails_at_startup_naming_the_setting`,
   `test_a_crashing_locator_does_not_break_ingest`).

---

## File structure

```
protocol/src/kuulo_protocol/impulses.py        ImpulseReport, ImpulseEvent, Ellipse, speed_of_sound
protocol/src/kuulo_protocol/{models,geo,api,schema,testing}.py
server/src/kuulo_server/impact/base.py         locator interface + shared grouping helpers
server/src/kuulo_server/impact/coarse.py       public CoarseLocator
server/src/kuulo_server/impact/context.py      DbImpulseContext
server/src/kuulo_server/impact/cap.py          CAP 1.2 XML
server/src/kuulo_server/impulses.py            persistence and queries
server/src/kuulo_server/fusion/basic.py        velocity fix
server/src/kuulo_server/{db,ingest,app,config,main,testing}.py, fusion/loader.py
node/src/kuulo_node/impulse.py                 ImpulseDetector, aic_pick
node/src/kuulo_node/{audio,runner,config,outbox,cli}.py
sim/src/kuulo_sim/{scenario,engine,runner}.py, scenarios/impact_strike.yaml, scenarios/firework.yaml
dashboard/src/{api,state,map,components}/…     impulse layer, list, detail
ml/src/kuulo_ml/impulses.py                    real-audio detector evaluation
```

---

### Task 1: Protocol — impulse wire models

**Files:** Create `protocol/src/kuulo_protocol/impulses.py`, `protocol/tests/test_impulses.py`;
modify `models.py`, `geo.py`, `api.py`, `schema.py`, `testing.py`.

**Interfaces (Produces):**
```python
# models.py
def to_utc_us(value: datetime) -> datetime          # tz required, UTC, microseconds kept
# impulses.py
CLOCK_SIGMA_S: dict[TimeQuality, float]             # gps 1e-6, ntp 0.02, manual 0.5
def speed_of_sound(temp_c: float) -> float
class ImpulseFeatures(WireModel)                    # peak_dbfs, snr_db, rise_time_ms, duration_ms, clipped, band_db[32]
class ImpulseReport(WireModel)                      # report_id, source, onset_at(µs), onset_sigma_s, time_quality, sensor_location, features, signature
class Ellipse(WireModel)                            # semi_major_m, semi_minor_m, bearing_deg [0,180), confidence
class ImpulseKind(StrEnum)                          # DRONE_IMPACT "drone_impact", UNASSOCIATED "unassociated"
class LocationQuality(StrEnum)                      # MULTILATERATED, AMBIGUOUS, COARSE
class ImpulseEvent(WireModel)                       # see code
# geo.py
def ellipse_ring(center, semi_major_m, semi_minor_m, bearing_deg, steps=64) -> list[GeoPoint]  # closed
# api.py
class ImpulseEventDetail(WireModel): event: ImpulseEvent; reports: list[ImpulseReport]
LiveEvent.type gains "impulse_event"; LiveEvent.data gains ImpulseEvent
# testing.py
def make_impulse_report(node_id, *, onset_at, lat, lon, time_quality=TimeQuality.GPS,
                        sigma=0.0005, peak_dbfs=-10.0) -> ImpulseReport   # unsigned
```

- [ ] **Step 1: Write the failing tests** — `protocol/tests/test_impulses.py`:

```python
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest

from kuulo_protocol.api import LiveEvent
from kuulo_protocol.geo import distance_m, ellipse_ring
from kuulo_protocol.impulses import (
    Ellipse,
    ImpulseEvent,
    ImpulseKind,
    LocationQuality,
    speed_of_sound,
)
from kuulo_protocol.models import GeoPoint
from kuulo_protocol.signing import generate_keypair, sign, verify
from kuulo_protocol.testing import make_impulse_report

T = datetime(2026, 9, 26, 12, 0, 0, 123456, tzinfo=UTC)


def test_onset_keeps_microseconds_and_converts_to_utc():
    helsinki = timezone(timedelta(hours=3))
    r = make_impulse_report("n1", onset_at=T.astimezone(helsinki), lat=60.17, lon=24.94)
    assert r.onset_at == T and r.onset_at.tzinfo == UTC


def test_naive_onset_is_rejected():
    with pytest.raises(ValueError):
        make_impulse_report("n1", onset_at=T.replace(tzinfo=None), lat=60.17, lon=24.94)


def test_report_signature_round_trip():
    priv, pub = generate_keypair()
    signed = sign(make_impulse_report("n1", onset_at=T, lat=60.17, lon=24.94), priv)
    assert verify(signed, pub)
    assert not verify(signed.model_copy(update={"onset_sigma_s": 0.5}), pub)


def test_features_need_32_bands():
    r = make_impulse_report("n1", onset_at=T, lat=60.17, lon=24.94)
    with pytest.raises(ValueError):
        r.features.model_validate({**r.features.model_dump(), "band_db": [0.0] * 31})


@pytest.mark.parametrize(("temp", "expected"), [(-20, 319.1), (0, 331.3), (20, 343.2)])
def test_speed_of_sound(temp, expected):
    assert speed_of_sound(temp) == pytest.approx(expected, abs=0.2)


def test_ellipse_ring_axes_and_orientation():
    c = GeoPoint(lat=60.17, lon=24.94)
    ring = ellipse_ring(c, 400.0, 100.0, 90.0, steps=64)  # major axis east-west
    assert len(ring) == 65 and ring[0] == ring[-1]
    assert distance_m(c, ring[0]) == pytest.approx(400, rel=1e-3)
    assert ring[0].lon > c.lon and ring[0].lat == pytest.approx(c.lat, abs=1e-6)
    assert distance_m(c, ring[16]) == pytest.approx(100, rel=1e-3)  # quarter turn: minor axis


def test_ellipse_bearing_must_be_half_turn():
    with pytest.raises(ValueError):
        Ellipse(semi_major_m=1, semi_minor_m=1, bearing_deg=180, confidence=0.95)


def test_live_event_carries_an_impulse_event():
    event = ImpulseEvent(
        event_id=uuid4(), kind=ImpulseKind.UNASSOCIATED, quality=LocationQuality.COARSE,
        position=GeoPoint(lat=60.17, lon=24.94),
        ellipse=Ellipse(semi_major_m=1500, semi_minor_m=1500, bearing_deg=0, confidence=0.95),
        occurred_at=T, node_ids=["n1"], report_ids=[uuid4()], updated_at=T,
    )
    parsed = LiveEvent.model_validate_json(
        LiveEvent(type="impulse_event", data=event).model_dump_json())
    assert parsed.data == event
```

- [ ] **Step 2: Run** `uv run --no-sync pytest protocol/tests/test_impulses.py` — Expected: collection error (module missing).

- [ ] **Step 3: Implement.** `models.py` (after `to_utc_ms`):

```python
def to_utc_us(value: datetime) -> datetime:
    """Require a timezone-aware timestamp; return it in UTC, keeping microseconds.

    Arrival-time differences between nodes need sub-millisecond precision.
    """
    if value.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)
```

`impulses.py`:

```python
"""Impulse wire format: what a node reports when it hears a bang, and what the server makes of it.

A node only says "I heard a sharp broadband event at this instant". The server decides where it
happened (an ImpulseEvent) and whether a drone track explains it (kind = drone_impact).
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, field_validator

from .features import N_BANDS
from .models import (
    SCHEMA_VERSION,
    GeoPoint,
    SensorLocation,
    Source,
    TimeQuality,
    WireModel,
    to_utc_us,
)

# 1-sigma clock error per time_quality (same values as the parent spec §9).
CLOCK_SIGMA_S = {TimeQuality.GPS: 1e-6, TimeQuality.NTP: 0.02, TimeQuality.MANUAL: 0.5}


def speed_of_sound(temp_c: float) -> float:
    """Speed of sound in dry air, m/s."""
    return 331.3 * math.sqrt(1.0 + temp_c / 273.15)


class ImpulseFeatures(WireModel):
    peak_dbfs: float = Field(ge=-200, le=0)
    snr_db: float = Field(ge=-50, le=150)
    rise_time_ms: float = Field(ge=0, le=1000)
    duration_ms: float = Field(ge=0, le=60_000)
    clipped: bool
    band_db: list[float] = Field(min_length=N_BANDS, max_length=N_BANDS)  # 20 ms at the peak


class ImpulseReport(WireModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    report_id: UUID = Field(default_factory=uuid4)
    source: Source
    onset_at: datetime  # UTC, microsecond precision
    onset_sigma_s: float = Field(gt=0, le=5)  # the node's own onset-picking uncertainty (1 sigma)
    time_quality: TimeQuality
    sensor_location: SensorLocation
    features: ImpulseFeatures
    signature: str = ""

    @field_validator("onset_at")
    @classmethod
    def _onset_utc(cls, value: datetime) -> datetime:
        return to_utc_us(value)


class Ellipse(WireModel):
    semi_major_m: float = Field(ge=0, le=100_000)
    semi_minor_m: float = Field(ge=0, le=100_000)
    bearing_deg: float = Field(ge=0, lt=180)  # major-axis direction, clockwise from north
    confidence: float = Field(gt=0, lt=1)


class ImpulseKind(StrEnum):
    DRONE_IMPACT = "drone_impact"
    UNASSOCIATED = "unassociated"


class LocationQuality(StrEnum):
    MULTILATERATED = "multilaterated"  # >= 3 nodes, one clear solution
    AMBIGUOUS = "ambiguous"  # an equally good second solution exists (see alternatives)
    COARSE = "coarse"  # 1-2 nodes: "heard near these sensors"


class ImpulseEvent(WireModel):
    event_id: UUID
    kind: ImpulseKind
    quality: LocationQuality
    position: GeoPoint
    ellipse: Ellipse
    alternatives: list[GeoPoint] = []
    occurred_at: datetime  # estimated emission time
    node_ids: list[str]
    excluded_node_ids: list[str] = []  # dropped as inconsistent (e.g. sound path blocked)
    report_ids: list[UUID]
    residuals_ms: dict[str, float] = {}
    rms_residual_ms: float | None = None
    associated_track_id: UUID | None = None
    updated_at: datetime

    @field_validator("occurred_at", "updated_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return to_utc_us(value)
```

`geo.py` (add `pi, sin, cos` to the `math` import):

```python
def ellipse_ring(
    center: HasLatLon, semi_major_m: float, semi_minor_m: float, bearing_deg: float,
    steps: int = 64,
) -> list[GeoPoint]:
    """Closed ring of points on an ellipse; bearing is the major axis, clockwise from north."""
    th = radians(bearing_deg)
    ux, uy = sin(th), cos(th)  # major axis (east, north)
    vx, vy = cos(th), -sin(th)  # minor axis, perpendicular
    ring = []
    for k in range(steps):
        a = 2 * pi * k / steps
        x = semi_major_m * cos(a) * ux + semi_minor_m * sin(a) * vx
        y = semi_major_m * cos(a) * uy + semi_minor_m * sin(a) * vy
        ring.append(from_local(center, x, y))
    ring.append(ring[0])
    return ring
```

`api.py`: add `ImpulseEventDetail(WireModel)` with `event: ImpulseEvent`, `reports: list[ImpulseReport]`;
`LiveEvent.type: Literal["observation", "track", "node_status", "impulse_event", "resync"]`,
`data: Observation | Track | NodeView | ImpulseEvent | None = None`.
`schema.py`: add `ImpulseReport, ImpulseEvent, ImpulseEventDetail` to `EXPORTED`.
`testing.py`:

```python
def make_impulse_report(
    node_id: str, *, onset_at: datetime, lat: float, lon: float,
    time_quality: TimeQuality = TimeQuality.GPS, sigma: float = 0.0005, peak_dbfs: float = -10.0,
) -> ImpulseReport:
    return ImpulseReport(
        source=Source(type=SourceType.SIMULATED_NODE, id=node_id), onset_at=onset_at,
        onset_sigma_s=sigma, time_quality=time_quality,
        sensor_location=SensorLocation(lat=lat, lon=lon, accuracy_m=5.0),
        features=ImpulseFeatures(peak_dbfs=peak_dbfs, snr_db=40.0, rise_time_ms=1.0,
                                 duration_ms=150.0, clipped=False, band_db=[-40.0] * 32),
    )
```

- [ ] **Step 4: Run** `uv run --no-sync pytest protocol` and `make types`; `cd dashboard && npx tsc --noEmit` — Expected: all pass (regenerated types compile).
- [ ] **Step 5: Commit** `feat(protocol): impulse reports, impulse events, uncertainty ellipse`.

---

### Task 2: Fix BasicFusion's track velocity

**Files:** Modify `server/src/kuulo_server/fusion/basic.py`; test `server/tests/test_fusion_basic.py`, `server/tests/test_scenarios.py`.

Cause: `_update` divides the displacement of the smoothed centroid by the time since the previous
observation. Observations from different nodes arrive milliseconds apart and their centroids
differ by hundreds of metres, giving speeds in the thousands of m/s (seen in the demo: 1295 and
2647 m/s for a 20 m/s drone).

- [ ] **Step 1: Failing tests.** In `test_fusion_basic.py`:

```python
def test_speed_is_not_computed_from_observations_milliseconds_apart():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(900, 0))
    observe(ctx, fusion, "n1", T0, snr=20)
    [update] = observe(ctx, fusion, "n2", T0 + timedelta(milliseconds=40), snr=5)
    v = update.track.velocity
    assert v is None or v.speed_mps < 100  # today: thousands of m/s
```

(`Ctx`, `at` and `observe` are the helpers already at the top of `test_fusion_basic.py`.)
In `test_scenarios.py`:

```python
def test_helsinki_pass_reports_a_plausible_speed(run_scenario):
    result = run_scenario("helsinki_pass")  # drone flies at 20 m/s
    speeds = [u.velocity.speed_mps for u in result.updates
              if u.status is TrackStatus.CONFIRMED and u.velocity is not None]
    assert speeds and 5 < sorted(speeds)[len(speeds) // 2] < 60
```

- [ ] **Step 2: Run** — Expected: both fail (speeds in the hundreds or thousands).
- [ ] **Step 3: Implement.** Keep a per-track velocity anchor `(position, time)` in `BasicFusion`
  (`self._anchor: dict[str, tuple[GeoPoint, datetime]]`). In `_update`, recompute velocity only
  when at least `FusionConfig.velocity_min_dt_s = 3.0` seconds have passed since the anchor:
  velocity = displacement(anchor → new smoothed position) / dt, then move the anchor. Otherwise
  keep the old velocity. Set the anchor when a track is created, and drop it when the track closes.
- [ ] **Step 4: Run** `uv run --no-sync pytest server` — Expected: all pass, including the existing
  scenario error bound.
- [ ] **Step 5: Commit** `fix(fusion): compute track velocity over >= 3 s, not between near-simultaneous observations`.

---

### Task 3: Server — impulse ingest, grouping, pluggable locator, public CoarseLocator

**Files:** Create `server/src/kuulo_server/impact/{__init__,base,coarse,context}.py`,
`server/src/kuulo_server/impulses.py`, `server/tests/test_impulses.py`; modify `db.py`, `ingest.py`,
`config.py`, `main.py`, `app.py`, `testing.py`, `fusion/loader.py`, `server/pyproject.toml`
(no new dependencies: grouping is plain Python).

**Interfaces (Produces)** — the private locator (private plan F3) is written against these, so
keep the names exactly:
```python
# impact/base.py
@dataclass(frozen=True)
class LocatorConfig:
    air_temperature_c: float = 10.0
    model_sigma_s: float = 0.005      # propagation modelling floor (wind, temperature, 2-D)
    max_array_m: float = 5000.0
    coarse_radius_m: float = 1500.0
    confidence: float = 0.95
    lookback_s: float = 60.0
    max_extrapolation_s: float = 30.0
    margin_m: float = 300.0
    max_track_speed_mps: float = 80.0
class ImpulseContext(Protocol):
    now: datetime
    def reports_between(self, start: datetime, end: datetime) -> list[ImpulseReport]: ...
    def event_ids_for(self, report_ids: list[UUID]) -> dict[UUID, UUID]: ...
    def tracks_seen_between(self, start: datetime, end: datetime) -> list[Track]: ...
class ImpulseLocator(Protocol):
    def on_report(self, report: ImpulseReport, ctx: ImpulseContext) -> ImpulseEvent | None: ...
def report_sigma(r: ImpulseReport, cfg: LocatorConfig) -> float
def consistent(a: ImpulseReport, b: ImpulseReport, c: float, cfg: LocatorConfig) -> bool
def nearby_reports(report, ctx, c, cfg) -> list[ImpulseReport]
def group_reports(report, nearby, c, cfg) -> list[ImpulseReport]   # sorted by onset
def event_id_for(group, ctx) -> UUID
def coarse_event(event_id, group, cfg, now) -> ImpulseEvent
# impact/coarse.py
class CoarseLocator: __init__(self, config: LocatorConfig | None = None); on_report(...)
# fusion/loader.py
def load_plugin(spec: str, *args)   # "module:Class" -> Class(*args); load_engine(spec) wraps it
# config.py
Settings.impulse_locator = "kuulo_server.impact.coarse:CoarseLocator"; Settings.air_temperature_c = 10.0
Settings.from_env(environ=os.environ) -> Settings  # KUULO_FUSION_ENGINE, KUULO_IMPULSE_LOCATOR,
                                                   # KUULO_AIR_TEMPERATURE_C
# testing.py (shared by Tasks 4, 7 and the private repo's tests)
IMPULSE_ORIGIN, RING_OFFSETS, RingNode, register_ring(client, n=6, prefix="s"),
post_bang(client, ring, source, at, *, only=None, temp_c=10.0) -> list[UUID]
# HTTP
POST /v1/impulses -> IngestResult
GET /v1/impulse-events?hours=24 -> list[ImpulseEvent]      (newest first)
GET /v1/impulse-events/{id} -> ImpulseEventDetail          (reports ordered by onset)
```

- [ ] **Step 1: Failing tests** — `server/tests/test_impulses.py`:

```python
from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from kuulo_protocol.geo import distance_m, from_local
from kuulo_protocol.models import GeoPoint
from kuulo_protocol.signing import generate_keypair
from kuulo_protocol.testing import make_impulse_report
from kuulo_server.app import create_app
from kuulo_server.config import Settings
from kuulo_server.testing import IMPULSE_ORIGIN as ORIGIN
from kuulo_server.testing import T0, FakeClock, post_bang, post_signed, register, register_ring


def events(client):
    return client.get("/v1/impulse-events").json()


def test_six_nodes_hearing_one_bang_make_one_coarse_event(client):
    ring = register_ring(client)
    source = from_local(ORIGIN, 150, -220)
    post_bang(client, ring, source, T0)
    [e] = events(client)
    assert e["quality"] == "coarse" and e["kind"] == "unassociated"
    assert set(e["node_ids"]) == {n.node_id for n in ring}
    assert distance_m(GeoPoint(**e["position"]), source) <= e["ellipse"]["semi_major_m"]


def test_the_same_event_grows_as_reports_arrive(client):
    ring = register_ring(client)
    post_bang(client, ring, ORIGIN, T0, only=[0])
    [first] = events(client)
    post_bang(client, ring, ORIGIN, T0, only=[1, 2, 3])
    [later] = events(client)
    assert later["event_id"] == first["event_id"] and len(later["report_ids"]) == 4


def test_late_report_refines_the_same_event(client, clock):
    ring = register_ring(client)
    post_bang(client, ring, from_local(ORIGIN, 100, 100), T0, only=[0, 1, 2, 3])
    clock.advance(600)  # the fifth node's report arrives ten minutes late
    post_bang(client, ring, from_local(ORIGIN, 100, 100), T0, only=[4])
    [e] = events(client)
    assert len(e["report_ids"]) == 5


def test_two_bangs_seconds_apart_in_different_places_are_two_events(client):
    ring = register_ring(client)
    post_bang(client, ring, from_local(ORIGIN, -400, 0), T0)
    post_bang(client, ring, from_local(ORIGIN, 3000, 3000), T0 + timedelta(seconds=3))
    assert len(events(client)) == 2


def test_duplicate_report_is_idempotent(client):
    ring = register_ring(client, 3)
    r = make_impulse_report("s0", onset_at=T0, lat=ring[0].location.lat, lon=ring[0].location.lon)
    post_signed(client, "/v1/impulses", r, ring[0].private_key)
    again = post_signed(client, "/v1/impulses", r, ring[0].private_key)
    assert again.json()["status"] == "duplicate" and len(events(client)) == 1


def test_bad_signature_unknown_node_and_future_are_rejected(client):
    ring = register_ring(client, 1)
    other, _ = generate_keypair()
    r = make_impulse_report("s0", onset_at=T0, lat=60.17, lon=24.94)
    assert post_signed(client, "/v1/impulses", r, other).status_code == 401
    ghost = make_impulse_report("ghost", onset_at=T0, lat=60.17, lon=24.94)
    assert post_signed(client, "/v1/impulses", ghost, other).status_code == 401
    future = make_impulse_report("s0", onset_at=T0 + timedelta(minutes=5), lat=60.17, lon=24.94)
    assert post_signed(client, "/v1/impulses", future, ring[0].private_key).status_code == 400


def _plugin(name: str, cls) -> str:
    import sys
    import types

    mod = types.ModuleType(name)
    mod.Locator = cls
    sys.modules[name] = mod
    return f"{name}:Locator"


def test_a_crashing_locator_does_not_break_ingest(tmp_path):
    class Broken:
        def __init__(self, *_):
            pass

        def on_report(self, *_):
            raise RuntimeError("boom")

    app = create_app(Settings(db_path=tmp_path / "k.db", clock=FakeClock(T0), tick_interval_s=None,
                              impulse_locator=_plugin("broken_locator", Broken)))
    with TestClient(app) as c:
        keys = register(c, "s0")
        r = make_impulse_report("s0", onset_at=T0, lat=60.17, lon=24.94)
        assert post_signed(c, "/v1/impulses", r, keys.private_key).status_code == 200
        assert c.get("/v1/impulse-events").json() == []


def test_the_plugin_receives_the_locator_config(tmp_path):
    seen = []

    class Spy:
        def __init__(self, config):
            seen.append(config)

        def on_report(self, *_):
            return None

    create_app(Settings(db_path=tmp_path / "k.db", tick_interval_s=None, air_temperature_c=-15.0,
                        impulse_locator=_plugin("spy_locator", Spy)))
    assert seen[0].air_temperature_c == -15.0


def test_unknown_locator_fails_at_startup_naming_the_setting(tmp_path):
    with pytest.raises(ImportError, match="impulse_locator"):
        create_app(Settings(db_path=tmp_path / "k.db", tick_interval_s=None,
                            impulse_locator="no_such_module:Locator"))


def test_settings_from_environment():
    s = Settings.from_env({"KUULO_IMPULSE_LOCATOR": "a.b:C", "KUULO_FUSION_ENGINE": "d.e:F",
                           "KUULO_AIR_TEMPERATURE_C": "-5"})
    assert (s.impulse_locator, s.fusion_engine, s.air_temperature_c) == ("a.b:C", "d.e:F", -5.0)
    assert Settings.from_env({}).impulse_locator == "kuulo_server.impact.coarse:CoarseLocator"


def test_events_survive_restart_and_detail_lists_reports(tmp_path):
    settings = Settings(db_path=tmp_path / "k.db", clock=FakeClock(T0), tick_interval_s=None)
    with TestClient(create_app(settings)) as c:
        post_bang(c, register_ring(c), ORIGIN, T0)
    with TestClient(create_app(settings)) as c:
        [e] = events(c)
        detail = c.get(f"/v1/impulse-events/{e['event_id']}").json()
        assert len(detail["reports"]) == 6
        assert c.get(f"/v1/impulse-events/{uuid4()}").status_code == 404


def test_impulse_events_are_published_live(client):
    ring = register_ring(client, 3)
    with client.websocket_connect("/v1/live") as ws:
        post_bang(client, ring, ORIGIN, T0, only=[0])
        assert '"impulse_event"' in ws.receive_text()
```

- [ ] **Step 2: Run** `uv run --no-sync pytest server/tests/test_impulses.py` — Expected: fail (import errors).

- [ ] **Step 3: Implement.**

`db.py` adds three tables:

```python
class ImpulseReportRow(Base):
    __tablename__ = "impulse_reports"
    report_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(64), index=True)
    onset_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_json: Mapped[str] = mapped_column(Text)


class ImpulseEventRow(Base):
    __tablename__ = "impulse_events"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_json: Mapped[str] = mapped_column(Text)


class ImpulseEventReportRow(Base):
    __tablename__ = "impulse_event_reports"
    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    report_id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
```

`ingest.py` adds `ingest_impulse(session, report, now, settings) -> IngestResult`: known node
(else 401), valid signature (else 401), onset not more than `future_tolerance_s` ahead (else 400),
duplicate `report_id` → `IngestResult(status="duplicate")`, otherwise store and return `accepted`
with `late` = onset older than `late_after_s`. **Late reports are not rejected.**

`impact/base.py`:

```python
"""The locator plug-in interface, plus grouping helpers any locator can use (spec §3, §5)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from kuulo_protocol.geo import distance_m, from_local, to_local
from kuulo_protocol.impulses import (
    CLOCK_SIGMA_S,
    Ellipse,
    ImpulseEvent,
    ImpulseKind,
    ImpulseReport,
    LocationQuality,
)
from kuulo_protocol.models import Track


@dataclass(frozen=True)
class LocatorConfig:
    air_temperature_c: float = 10.0
    model_sigma_s: float = 0.005
    max_array_m: float = 5000.0
    coarse_radius_m: float = 1500.0
    confidence: float = 0.95
    lookback_s: float = 60.0
    max_extrapolation_s: float = 30.0
    margin_m: float = 300.0
    max_track_speed_mps: float = 80.0


class ImpulseContext(Protocol):
    now: datetime

    def reports_between(self, start: datetime, end: datetime) -> list[ImpulseReport]: ...

    def event_ids_for(self, report_ids: list[UUID]) -> dict[UUID, UUID]: ...

    def tracks_seen_between(self, start: datetime, end: datetime) -> list[Track]: ...


class ImpulseLocator(Protocol):
    def on_report(self, report: ImpulseReport, ctx: ImpulseContext) -> ImpulseEvent | None: ...


def report_sigma(r: ImpulseReport, cfg: LocatorConfig) -> float:
    """1-sigma timing uncertainty: node clock, the node's own onset picking, propagation floor."""
    return math.sqrt(CLOCK_SIGMA_S[r.time_quality] ** 2 + r.onset_sigma_s ** 2
                     + cfg.model_sigma_s ** 2)


def consistent(a: ImpulseReport, b: ImpulseReport, c: float, cfg: LocatorConfig) -> bool:
    """Could one impulse have reached both nodes with this onset difference?"""
    d = distance_m(a.sensor_location, b.sensor_location)
    dt = abs((a.onset_at - b.onset_at).total_seconds())
    return dt <= d / c + 3 * math.hypot(report_sigma(a, cfg), report_sigma(b, cfg)) + 0.05


def nearby_reports(report, ctx, c, cfg) -> list[ImpulseReport]:
    window = timedelta(seconds=cfg.max_array_m / c + 1.0)
    return [r for r in ctx.reports_between(report.onset_at - window, report.onset_at + window)
            if r.report_id != report.report_id]


def group_reports(report, nearby, c, cfg) -> list[ImpulseReport]:
    group = [report]
    for cand in sorted(nearby, key=lambda r: abs((r.onset_at - report.onset_at).total_seconds())):
        if any(m.source.id == cand.source.id for m in group):
            continue  # at most one report per node: the nearest in time wins
        if all(consistent(cand, m, c, cfg) for m in group):
            group.append(cand)
    return sorted(group, key=lambda r: r.onset_at)


def event_id_for(group, ctx) -> UUID:
    known = ctx.event_ids_for([r.report_id for r in group])
    return Counter(known.values()).most_common(1)[0][0] if known else uuid4()


def coarse_event(event_id, group, cfg, now) -> ImpulseEvent:
    origin = group[0].sensor_location
    local = [to_local(origin, r.sensor_location) for r in group]
    cx = sum(x for x, _ in local) / len(local)
    cy = sum(y for _, y in local) / len(local)
    spread = max(math.hypot(x - cx, y - cy) for x, y in local)
    radius = min(cfg.coarse_radius_m + spread, 100_000.0)
    return ImpulseEvent(
        event_id=event_id, kind=ImpulseKind.UNASSOCIATED, quality=LocationQuality.COARSE,
        position=from_local(origin, cx, cy),
        ellipse=Ellipse(semi_major_m=radius, semi_minor_m=radius, bearing_deg=0.0,
                        confidence=cfg.confidence),
        occurred_at=group[0].onset_at, node_ids=[r.source.id for r in group],
        report_ids=[r.report_id for r in group], updated_at=now,
    )
```

`impact/coarse.py`:

```python
"""Public baseline locator: which nodes heard the same impulse, and a circle around them.

A precise location and drone-impact attribution come from a private locator loaded through
Settings.impulse_locator (spec §5).
"""

from __future__ import annotations

from kuulo_protocol.impulses import ImpulseEvent, ImpulseReport, speed_of_sound

from .base import (
    ImpulseContext,
    LocatorConfig,
    coarse_event,
    event_id_for,
    group_reports,
    nearby_reports,
)


class CoarseLocator:
    def __init__(self, config: LocatorConfig | None = None):
        self.config = config or LocatorConfig()
        self.c = speed_of_sound(self.config.air_temperature_c)

    def on_report(self, report: ImpulseReport, ctx: ImpulseContext) -> ImpulseEvent | None:
        group = group_reports(report, nearby_reports(report, ctx, self.c, self.config), self.c,
                              self.config)
        return coarse_event(event_id_for(group, ctx), group, self.config, ctx.now)
```

`impact/context.py` — `DbImpulseContext(sessions, now)`: `reports_between` parses `raw_json` of
rows whose `onset_at` is in range; `event_ids_for` reads `impulse_event_reports`;
`tracks_seen_between` parses `TrackRow.raw_json` with `last_seen` in range (closed tracks too:
they keep their last position and velocity).

`impulses.py` (mirrors `tracks.py`): `persist_impulse_event` (upsert the row, then replace its
link rows), `recent_impulse_events(session, since)` (newest first), and
`impulse_event_detail(session, event_id) -> ImpulseEventDetail | None`.

`fusion/loader.py`:

```python
def load_plugin(spec: str, *args):
    """Instantiate "module:Class" with args, so a private engine or locator can be dropped in."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"plugin must look like 'module:Class', got {spec!r}")
    return getattr(import_module(module_name), class_name)(*args)


def load_engine(spec: str) -> FusionEngine:
    return load_plugin(spec)
```

`config.py`: `impulse_locator: str = "kuulo_server.impact.coarse:CoarseLocator"`,
`air_temperature_c: float = 10.0`, and:

```python
    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> Settings:
        """Deployment overrides: a private engine or locator is selected without code changes."""
        overrides: dict = {}
        if value := environ.get("KUULO_FUSION_ENGINE"):
            overrides["fusion_engine"] = value
        if value := environ.get("KUULO_IMPULSE_LOCATOR"):
            overrides["impulse_locator"] = value
        if value := environ.get("KUULO_AIR_TEMPERATURE_C"):
            overrides["air_temperature_c"] = float(value)
        return cls(**overrides)
```

`main.py`: `app = create_app(Settings.from_env())`.

`app.py`: build the locator at startup, naming the setting when it cannot be imported:

```python
    config = LocatorConfig(air_temperature_c=settings.air_temperature_c)
    try:
        locator = load_plugin(settings.impulse_locator, config)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"impulse_locator {settings.impulse_locator!r} cannot be loaded: "
                          f"{exc}") from exc

    def run_locator(report: ImpulseReport) -> None:
        try:
            event = locator.on_report(report, DbImpulseContext(sessions, settings.clock()))
        except Exception:
            log.exception("impulse locator failed; ingest continues")
            return
        if event is None:
            return
        with sessions() as session:
            persist_impulse_event(session, event)
        hub.publish(LiveEvent(type="impulse_event", data=event))
```

and the three endpoints:

```python
    @app.post("/v1/impulses")
    async def post_impulse(report: ImpulseReport) -> IngestResult:
        with sessions() as session:
            result = ingest_impulse(session, report, settings.clock(), settings)
        if result.status == "accepted":
            run_locator(report)
        return result

    @app.get("/v1/impulse-events")
    async def get_impulse_events(hours: float = 24.0) -> list[ImpulseEvent]:
        with sessions() as session:
            return recent_impulse_events(session, settings.clock() - timedelta(hours=hours))

    @app.get("/v1/impulse-events/{event_id}")
    async def get_impulse_event(event_id: str) -> ImpulseEventDetail:
        with sessions() as session:
            detail = impulse_event_detail(session, event_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="unknown impulse event")
        return detail
```

`testing.py` gets the shared helpers (add imports for `UUID`, `GeoPoint`, `distance_m`,
`from_local`, `speed_of_sound` and `make_impulse_report`):

```python
IMPULSE_ORIGIN = GeoPoint(lat=60.17, lon=24.94)
RING_OFFSETS = [(-800, -600), (900, -500), (700, 800), (-600, 900), (0, -1000), (1000, 200)]


@dataclass(frozen=True)
class RingNode:
    keys: NodeKeys
    location: GeoPoint

    @property
    def node_id(self) -> str:
        return self.keys.node_id

    @property
    def private_key(self) -> str:
        return self.keys.private_key


def register_ring(client: TestClient, n: int = 6, prefix: str = "s") -> list[RingNode]:
    """n nodes at fixed offsets (metres) around IMPULSE_ORIGIN."""
    out = []
    for i, (x, y) in enumerate(RING_OFFSETS[:n]):
        p = from_local(IMPULSE_ORIGIN, x, y)
        out.append(RingNode(register(client, f"{prefix}{i}", lat=p.lat, lon=p.lon), p))
    return out


def post_bang(client: TestClient, ring: list[RingNode], source: GeoPoint, at: datetime, *,
              only: list[int] | None = None, temp_c: float = 10.0) -> list[UUID]:
    """Post every (or `only` the listed) node's signed, GPS-timed report of an impulse."""
    c = speed_of_sound(temp_c)
    ids = []
    for node in ring if only is None else [ring[i] for i in only]:
        onset = at + timedelta(seconds=distance_m(node.location, source) / c)
        r = make_impulse_report(node.node_id, onset_at=onset, lat=node.location.lat,
                                lon=node.location.lon)
        response = post_signed(client, "/v1/impulses", r, node.private_key)
        assert response.status_code == 200, response.text
        ids.append(r.report_id)
    return ids
```

- [ ] **Step 4: Run** `uv run --no-sync pytest server` and `uv run --no-sync ruff check .` — Expected: pass.
- [ ] **Step 5: Commit** `feat(server): impulse ingest, physical grouping, pluggable locator with public coarse baseline`.

---

### Task 4: CAP 1.2 export

**Files:** Create `server/src/kuulo_server/impact/cap.py`, `server/tests/test_cap.py`; modify
`config.py` (`cap_status: str = "Test"`, `cap_sender: str = "kuulo@localhost"`), `app.py`.

**Interfaces (Produces):** `to_cap_xml(event: ImpulseEvent, *, sender: str, status: str, track_callsign: str | None = None) -> bytes`;
`GET /v1/impulse-events/{id}/cap` → `application/cap+xml` (a registered media type).

- [ ] **Step 1: Failing tests:**

```python
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from kuulo_protocol.geo import distance_m
from kuulo_protocol.impulses import Ellipse, ImpulseEvent, ImpulseKind, LocationQuality
from kuulo_protocol.models import GeoPoint
from kuulo_server.impact.cap import NS, to_cap_xml

T = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def event(kind=ImpulseKind.DRONE_IMPACT):
    return ImpulseEvent(
        event_id=uuid4(), kind=kind, quality=LocationQuality.MULTILATERATED,
        position=GeoPoint(lat=60.17, lon=24.94),
        ellipse=Ellipse(semi_major_m=120, semi_minor_m=40, bearing_deg=30, confidence=0.95),
        occurred_at=T, node_ids=["a", "b", "c", "d"], report_ids=[uuid4()], updated_at=T,
        associated_track_id=uuid4() if kind is ImpulseKind.DRONE_IMPACT else None,
    )


def parse(xml: bytes):
    return ET.fromstring(xml)


def text(root, path):
    return root.find(path, {"cap": NS}).text


def test_required_cap_elements_are_present():
    root = parse(to_cap_xml(event(), sender="kuulo@localhost", status="Test"))
    assert root.tag == f"{{{NS}}}alert"
    for path in ("cap:identifier", "cap:sender", "cap:sent", "cap:status", "cap:msgType",
                 "cap:scope", "cap:info/cap:category", "cap:info/cap:event",
                 "cap:info/cap:urgency", "cap:info/cap:severity", "cap:info/cap:certainty",
                 "cap:info/cap:area/cap:areaDesc", "cap:info/cap:area/cap:polygon"):
        assert root.find(path, {"cap": NS}) is not None, path
    assert text(root, "cap:status") == "Test"
    assert text(root, "cap:info/cap:category") == "Security"


def test_sent_uses_a_numeric_offset_not_z():
    root = parse(to_cap_xml(event(), sender="s", status="Test"))
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", text(root, "cap:sent"))


def test_polygon_is_closed_and_follows_the_ellipse():
    e = event()
    root = parse(to_cap_xml(e, sender="s", status="Test"))
    pairs = [tuple(map(float, p.split(","))) for p in
             text(root, "cap:info/cap:area/cap:polygon").split()]
    assert len(pairs) >= 4 and pairs[0] == pairs[-1]
    for lat, lon in pairs:
        assert 39 <= distance_m(e.position, GeoPoint(lat=lat, lon=lon)) <= 121


def test_severity_and_event_name_follow_the_kind():
    impact = parse(to_cap_xml(event(), sender="s", status="Test"))
    other = parse(to_cap_xml(event(ImpulseKind.UNASSOCIATED), sender="s", status="Test"))
    assert text(impact, "cap:info/cap:severity") == "Severe"
    assert text(other, "cap:info/cap:severity") == "Moderate"
    assert "impact" in text(impact, "cap:info/cap:event").lower()


def test_invalid_status_is_rejected():
    with pytest.raises(ValueError):
        to_cap_xml(event(), sender="s", status="Real")


def test_endpoint_serves_cap_and_404s_unknown_events(client):
    from kuulo_server.testing import IMPULSE_ORIGIN, T0, post_bang, register_ring

    post_bang(client, register_ring(client), IMPULSE_ORIGIN, T0)
    [e] = client.get("/v1/impulse-events").json()
    response = client.get(f"/v1/impulse-events/{e['event_id']}/cap")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/cap+xml")
    assert parse(response.content).tag == f"{{{NS}}}alert"
    assert client.get(f"/v1/impulse-events/{uuid4()}/cap").status_code == 404
```

- [ ] **Step 2: Run** — Expected: import error.
- [ ] **Step 3: Implement** `cap.py`:

```python
"""CAP 1.2 (OASIS Common Alerting Protocol) rendering of an impulse event, for alerting systems."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from kuulo_protocol.geo import ellipse_ring
from kuulo_protocol.impulses import ImpulseEvent, ImpulseKind, LocationQuality

NS = "urn:oasis:names:tc:emergency:cap:1.2"
STATUSES = ("Actual", "Exercise", "System", "Test", "Draft")


def _sent(dt) -> str:
    return dt.replace(microsecond=0).isoformat()  # "+00:00": CAP forbids "Z"


def to_cap_xml(event: ImpulseEvent, *, sender: str, status: str,
               track_callsign: str | None = None) -> bytes:
    if status not in STATUSES:
        raise ValueError(f"CAP status must be one of {STATUSES}, got {status!r}")
    ET.register_namespace("", NS)
    q = lambda tag: f"{{{NS}}}{tag}"  # noqa: E731
    alert = ET.Element(q("alert"))

    def add(parent, tag, value):
        el = ET.SubElement(parent, q(tag))
        el.text = value
        return el

    impact = event.kind is ImpulseKind.DRONE_IMPACT
    add(alert, "identifier", f"kuulo-{event.event_id}")
    add(alert, "sender", sender)
    add(alert, "sent", _sent(event.updated_at))
    add(alert, "status", status)
    add(alert, "msgType", "Alert")
    add(alert, "scope", "Restricted")
    add(alert, "restriction", "Emergency services and authorised counter-UAS operators")
    info = ET.SubElement(alert, q("info"))
    add(info, "category", "Security")
    add(info, "event", "Drone impact" if impact else "Unexplained acoustic impulse")
    add(info, "urgency", "Immediate")
    add(info, "severity", "Severe" if impact else "Moderate")
    located = event.quality is LocationQuality.MULTILATERATED
    add(info, "certainty", "Likely" if impact and located else "Possible")
    add(info, "senderName", "Kuulo acoustic sensor network")
    e = event.ellipse
    add(info, "headline", f"{'Drone impact' if impact else 'Acoustic impulse'} located "
                          f"to ±{e.semi_major_m:.0f} m by {len(event.node_ids)} sensors")
    track_ref = track_callsign or event.associated_track_id
    track = f" Associated drone track: {track_ref}." if impact else ""
    add(info, "description",
        f"Location quality: {event.quality.value}. {int(e.confidence * 100)} % region: ellipse "
        f"{e.semi_major_m:.0f} m x {e.semi_minor_m:.0f} m, major axis bearing "
        f"{e.bearing_deg:.0f} deg. Estimated time {event.occurred_at.isoformat()}.{track}")
    area = ET.SubElement(info, q("area"))
    add(area, "areaDesc", f"{int(e.confidence * 100)} % confidence region of the impulse")
    ring = ellipse_ring(event.position, e.semi_major_m, e.semi_minor_m, e.bearing_deg, steps=32)
    add(area, "polygon", " ".join(f"{p.lat:.6f},{p.lon:.6f}" for p in ring))
    return ET.tostring(alert, encoding="utf-8", xml_declaration=True)
```

and the endpoint:

```python
    @app.get("/v1/impulse-events/{event_id}/cap")
    async def get_impulse_event_cap(event_id: str) -> Response:
        with sessions() as session:
            detail = impulse_event_detail(session, event_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="unknown impulse event")
        body = to_cap_xml(detail.event, sender=settings.cap_sender, status=settings.cap_status)
        return Response(content=body, media_type="application/cap+xml")
```

Validate `cap_status` at `create_app` time (raise `ValueError` for anything not in `STATUSES`).

- [ ] **Step 4: Run** `uv run --no-sync pytest server` — Expected: pass.
- [ ] **Step 5: Commit** `feat(server): CAP 1.2 export of impulse events (status Test by default)`.

---

### Task 5: Node — impulse detector

**Files:** Create `node/src/kuulo_node/impulse.py`, `node/tests/test_impulse.py`.

**Interfaces (Produces):**
```python
@dataclass(frozen=True) class ImpulseConfig:
    enabled: bool = True; sta_ms: float = 2.0; lta_s: float = 5.0; trigger_db: float = 20.0
    min_peak_dbfs: float = -40.0; min_active_bands: int = 6; max_rise_ms: float = 50.0
    min_decay_db: float = 10.0; post_s: float = 1.0; dead_time_s: float = 2.0
    highpass_hz: float = 100.0
@dataclass(frozen=True) class ImpulseCandidate: onset_utc: datetime; onset_sigma_s: float; features: ImpulseFeatures
def aic_pick(x: np.ndarray) -> int
class ImpulseDetector: __init__(cfg: ImpulseConfig = ImpulseConfig()); push(block: AudioBlock, utc: datetime) -> list[ImpulseCandidate]; reset()
SIGMA_RISE_FACTOR = 0.25   # onset sigma = clamp(SIGMA_RISE_FACTOR * rise time, 2 samples, 20 ms); recalibrated in Task 9
```

- [ ] **Step 1: Failing tests** — `node/tests/test_impulse.py`:

```python
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kuulo_node.audio import SAMPLE_RATE as SR
from kuulo_node.audio import AudioBlock
from kuulo_node.impulse import ImpulseConfig, ImpulseDetector, aic_pick

T0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def noise(seconds, level=0.003, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def burst(amp=0.5, tau=0.03, seconds=0.4, seed=1):
    n = int(seconds * SR)
    env = np.exp(-np.arange(n) / (tau * SR))
    return (np.random.default_rng(seed).standard_normal(n) * amp * env).astype(np.float32)


def place(bg, event, at_s):
    x = bg.copy()
    i = int(at_s * SR)
    x[i:i + event.size] += event[: x.size - i]
    return np.clip(x, -1, 1)


def run(x, block_s=0.1, cfg=None, utc0=T0):
    det = ImpulseDetector(cfg or ImpulseConfig())
    n = int(block_s * SR)
    out = []
    for i in range(0, x.size, n):
        out += det.push(AudioBlock(x[i:i + n], i / SR), utc0 + timedelta(seconds=i / SR))
    return out


def test_aic_finds_the_change_point():
    x = np.concatenate([noise(0.05), burst(seconds=0.05)])
    assert abs(aic_pick(x.astype(np.float64)) - int(0.05 * SR)) <= 8


def test_one_bang_one_report_with_sub_millisecond_onset():
    [c] = run(place(noise(6), burst(), 3.0))
    assert abs((c.onset_utc - (T0 + timedelta(seconds=3.0))).total_seconds()) < 0.001
    assert 0 < c.onset_sigma_s <= 0.02
    assert c.features.peak_dbfs > -10 and c.features.rise_time_ms < 5 and not c.features.clipped


def test_stationary_noise_gives_nothing():
    assert run(noise(20)) == []


def test_faded_in_tone_is_narrowband_and_rejected():
    t = np.arange(int(4 * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 1000 * t) * np.clip((t - 2.0) / 0.05, 0, 1)
    assert run(noise(4) + tone.astype(np.float32)) == []


def test_step_to_sustained_noise_is_not_an_impulse_and_does_not_retrigger():
    x = np.concatenate([noise(3), noise(12, level=0.1, seed=5)])
    assert run(x) == []


def test_echo_within_dead_time_is_not_reported_again():
    x = place(place(noise(6), burst(), 2.0), burst(amp=0.25, seed=2), 2.3)
    assert len(run(x)) == 1


def test_two_bangs_three_seconds_apart_are_two_reports():
    x = place(place(noise(8), burst(), 2.0), burst(seed=3), 5.0)
    assert len(run(x)) == 2


def test_clipping_is_flagged():
    [c] = run(place(noise(6), burst(amp=3.0), 3.0))
    assert c.features.clipped and c.features.peak_dbfs == pytest.approx(0, abs=0.1)


def test_block_size_does_not_change_the_onset():
    x = place(noise(6), burst(), 3.0)
    a, b = run(x, block_s=0.1), run(x, block_s=0.0137)
    assert abs((a[0].onset_utc - b[0].onset_utc).total_seconds()) < 1 / SR + 1e-9


def test_onset_uses_the_block_utc_anchor_not_the_audio_clock():
    x = place(noise(6), burst(), 3.0)
    [c] = run(x, utc0=T0 + timedelta(hours=1))
    assert abs((c.onset_utc - (T0 + timedelta(hours=1, seconds=3))).total_seconds()) < 0.001


def test_restart_resets_warm_up():
    det = ImpulseDetector()
    x = place(noise(3), burst(), 0.2)
    det.push(AudioBlock(noise(2), 0.0), T0)
    assert det.push(AudioBlock(x, 10.0, restart=True), T0 + timedelta(seconds=10)) == []


def test_quiet_click_below_min_peak_is_ignored():
    # burst peaks near 4 sigma = 0.004 (-48 dBFS), below the -40 dBFS default
    assert run(place(noise(6, level=0.0001), burst(amp=0.001), 3.0)) == []
```

- [ ] **Step 2: Run** — Expected: import error.
- [ ] **Step 3: Implement** `node/src/kuulo_node/impulse.py`:

```python
"""Impulse detector: STA/LTA trigger plus AIC onset picker, both borrowed from seismology.

Runs on the node's 16 kHz audio beside the drone classifier and reports only an onset time,
its uncertainty and summary features: no waveform leaves the node (spec §4).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from kuulo_protocol.features import extract_frames
from kuulo_protocol.impulses import ImpulseFeatures

from .audio import SAMPLE_RATE as SR
from .audio import AudioBlock

FRAME = 32  # 2 ms energy frames
PRE_S = 0.25  # audio kept from before the trigger, for onset picking
WARM_S = 0.5
SIGMA_RISE_FACTOR = 0.25


@dataclass(frozen=True)
class ImpulseConfig:
    enabled: bool = True
    sta_ms: float = 2.0
    lta_s: float = 5.0
    trigger_db: float = 20.0
    min_peak_dbfs: float = -40.0
    min_active_bands: int = 6
    max_rise_ms: float = 50.0
    min_decay_db: float = 10.0
    post_s: float = 1.0
    dead_time_s: float = 2.0
    highpass_hz: float = 100.0


@dataclass(frozen=True)
class ImpulseCandidate:
    onset_utc: datetime
    onset_sigma_s: float
    features: ImpulseFeatures


def aic_pick(x: np.ndarray) -> int:
    """Onset index in x (Maeda 1985): argmin of k*log var(x[:k]) + (n-k-1)*log var(x[k:])."""
    n = x.size
    if n < 16:
        return 0
    c1, c2 = np.cumsum(x), np.cumsum(x * x)
    k = np.arange(4, n - 4)
    v1 = c2[k - 1] / k - (c1[k - 1] / k) ** 2
    v2 = (c2[-1] - c2[k - 1]) / (n - k) - ((c1[-1] - c1[k - 1]) / (n - k)) ** 2
    aic = k * np.log(np.maximum(v1, 1e-20)) + (n - k - 1) * np.log(np.maximum(v2, 1e-20))
    return int(k[np.argmin(aic)])


def _db(x: float) -> float:
    return 10 * math.log10(max(x, 1e-20))


class ImpulseDetector:
    def __init__(self, cfg: ImpulseConfig | None = None) -> None:
        self.cfg = cfg or ImpulseConfig()
        self._sos = butter(2, self.cfg.highpass_hz, btype="highpass", fs=SR, output="sos")
        self._n = 0  # absolute index of the next incoming sample
        self._anchors: deque[tuple[int, datetime]] = deque(maxlen=256)
        self.reset()

    def reset(self) -> None:
        self._zi = sosfilt_zi(self._sos) * 0.0
        self._raw = np.zeros(0, np.float32)
        self._hp = np.zeros(0, np.float64)
        self._start = self._n  # absolute index of _raw[0]
        self._next_frame = self._n
        self._lta: float | None = None
        self._warm = 0
        self._dead_until = -1
        self._pending: int | None = None

    def _utc_of(self, idx: int) -> datetime:
        for a_idx, a_utc in reversed(self._anchors):
            if a_idx <= idx:
                return a_utc + timedelta(seconds=(idx - a_idx) / SR)
        a_idx, a_utc = self._anchors[0]
        return a_utc + timedelta(seconds=(idx - a_idx) / SR)

    def push(self, block: AudioBlock, utc: datetime) -> list[ImpulseCandidate]:
        if block.restart:
            self.reset()
        x = np.asarray(block.samples, np.float32)
        self._anchors.append((self._n, utc))
        hp, self._zi = sosfilt(self._sos, x.astype(np.float64), zi=self._zi)
        self._raw = np.concatenate([self._raw, x])
        self._hp = np.concatenate([self._hp, hp])
        self._n += x.size
        self._scan()
        out = []
        if self._pending is not None and self._n >= self._pending + int(self.cfg.post_s * SR):
            cand = self._analyse(self._pending)
            self._pending = None
            if cand is not None:
                out.append(cand)
        self._trim()
        return out

    def _scan(self) -> None:
        cfg = self.cfg
        warm_frames = int(WARM_S * SR / FRAME)
        alpha = FRAME / (cfg.lta_s * SR)
        while self._next_frame + FRAME <= self._n:
            i = self._next_frame - self._start
            e = float(np.mean(self._hp[i:i + FRAME] ** 2)) + 1e-20
            s = self._next_frame
            self._next_frame += FRAME
            if self._lta is None:
                self._lta = e
            if (self._warm >= warm_frames and self._pending is None and s >= self._dead_until
                    and _db(e / self._lta) >= cfg.trigger_db):
                self._pending = s
                self._dead_until = s + int(cfg.dead_time_s * SR)
            elif s >= self._dead_until:
                rate = 1.0 / (self._warm + 1) if self._warm < warm_frames else alpha
                self._lta += (e - self._lta) * rate
            self._warm += 1

    def _trim(self) -> None:
        keep_from = self._n - int((PRE_S + self.cfg.post_s + 0.1) * SR)
        if self._pending is not None:
            keep_from = min(keep_from, self._pending - int(PRE_S * SR))
        keep_from = min(keep_from, self._next_frame)
        cut = keep_from - self._start
        if cut > 0:
            self._raw, self._hp = self._raw[cut:], self._hp[cut:]
            self._start = keep_from

    def _reject(self, tail_energy: float) -> None:
        self._lta = max(tail_energy, 1e-20)  # re-baseline: sustained sound is the new normal

    def _analyse(self, trigger: int) -> ImpulseCandidate | None:
        cfg = self.cfg
        lo = max(trigger - int(PRE_S * SR), self._start)
        hi = min(trigger + int(cfg.post_s * SR), self._n)
        raw = self._raw[lo - self._start: hi - self._start]
        hp = self._hp[lo - self._start: hi - self._start]
        pk_lo = max(trigger - int(0.01 * SR), lo) - lo
        peak = pk_lo + int(np.argmax(np.abs(hp[pk_lo:])))
        tail = float(np.mean(hp[-int(0.1 * SR):] ** 2))
        peak_amp = float(np.max(np.abs(raw[pk_lo:])))
        peak_dbfs = min(0.0, 20 * math.log10(max(peak_amp, 1e-9)))
        if peak_dbfs < cfg.min_peak_dbfs:
            self._reject(tail)
            return None
        a_lo = max(0, (trigger - lo) - int(0.1 * SR))
        onset = a_lo + aic_pick(hp[a_lo: peak + 1])
        env = np.sqrt(np.convolve(hp ** 2, np.ones(16) / 16, mode="same"))
        top = float(env[onset: peak + 16].max())
        rise_idx = np.nonzero(env[onset:] >= 0.9 * top)[0]
        start_idx = np.nonzero(env[onset:] >= 0.1 * top)[0]
        rise_s = ((rise_idx[0] - start_idx[0]) / SR) if rise_idx.size and start_idx.size else 0.0
        frames = hp[onset: onset + (hp.size - onset) // FRAME * FRAME].reshape(-1, FRAME)
        energies = np.mean(frames ** 2, axis=1) if frames.size else np.array([1e-20])
        decay_db = _db(float(energies.max()) / max(tail, 1e-20))
        env10 = np.sqrt(np.convolve(hp ** 2, np.ones(160) / 160, mode="same"))
        above = np.nonzero(env10[peak:] >= math.sqrt(self._lta) * 10 ** (10 / 20))[0]
        duration_s = ((peak + (above[-1] if above.size else 0)) - onset) / SR
        centre = max(0, min(peak - 160, raw.size - 320))
        bands = extract_frames(raw[centre: centre + 320]).band_db[0]
        active = int(np.sum(bands >= bands.max() - 20))
        if (rise_s * 1000 > cfg.max_rise_ms or decay_db < cfg.min_decay_db
                or active < cfg.min_active_bands):
            self._reject(tail)
            return None
        sigma = min(0.02, max(2 / SR, SIGMA_RISE_FACTOR * rise_s))
        features = ImpulseFeatures(
            peak_dbfs=round(peak_dbfs, 2),
            snr_db=round(min(150.0, max(-50.0, _db(float(energies.max()) / self._lta))), 2),
            rise_time_ms=round(rise_s * 1000, 3), duration_ms=round(max(duration_s, 0) * 1000, 1),
            clipped=bool(np.sum(np.abs(raw) >= 0.99) >= 3),
            band_db=[round(float(b), 2) for b in bands],
        )
        return ImpulseCandidate(self._utc_of(lo + onset), sigma, features)
```

- [ ] **Step 4: Run** `uv run --no-sync pytest node/tests/test_impulse.py` — Expected: pass. If a
  threshold test fails, fix the detector rather than the test, and record any threshold change
  as a ruling.
- [ ] **Step 5: Commit** `feat(node): impulse detector (STA/LTA trigger, AIC onset picker)`.

---

### Task 6: Node — UTC-stamped audio, runner wiring, config, outbox priority

**Files:** Modify `node/src/kuulo_node/{audio,runner,config,outbox,cli}.py`, `node/config.example.toml`;
tests `node/tests/{test_audio,test_runner,test_outbox,test_config_keys,test_e2e}.py`.

**Interfaces (Produces):** `AudioBlock.utc: datetime | None = None` (UTC of the first sample);
`adc_utc(now, current_time, adc_time, frames, rate, latency) -> datetime`;
`wav_source(path, *, speed=1.0, block_s=0.1, sleep=..., start_utc: datetime | None = None)`;
`NodeConfig.impulse: ImpulseConfig`; `RunStats.impulses: int`; the runner posts to `/v1/impulses`.

- [ ] **Step 1: Failing tests.**

```python
# test_audio.py  (add: from datetime import UTC, datetime, timedelta; import soundfile as sf)
from kuulo_node.audio import adc_utc, wav_source

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def test_adc_timestamp_is_used_when_valid():
    assert adc_utc(NOW, 100.050, 100.010, 1600, 16000, 0.02) == NOW - timedelta(seconds=0.04)


def test_missing_adc_timestamp_falls_back_to_buffer_length_plus_latency():
    assert adc_utc(NOW, 100.0, 0.0, 1600, 16000, 0.02) == NOW - timedelta(seconds=0.12)


def test_wav_source_stamps_blocks_from_start_utc(tmp_path):
    path = tmp_path / "s.wav"
    sf.write(path, np.zeros(16000, np.float32), 16000)
    blocks = list(wav_source(path, speed=0, start_utc=NOW))
    assert blocks and all(b.utc == NOW + timedelta(seconds=b.t) for b in blocks)


# test_outbox.py
def test_full_outbox_drops_heartbeats_then_observations_then_impulses():
    box = Outbox(cap=2)
    box.put("/v1/impulses", "i1")
    box.put("/v1/observations", "o1")
    box.put("/v1/impulses", "i2")
    assert [b for _, _, b in box.peek(5)] == ["i1", "i2"]


# test_runner.py  (add: from dataclasses import replace; from datetime import UTC, datetime,
# timedelta; from kuulo_node.impulse import ImpulseConfig; from kuulo_protocol.signing import verify)
UTC0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def bang_blocks(seconds, at_s, utc0=UTC0):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(seconds * SAMPLE_RATE)) * 0.003).astype(np.float32)
    n, i = int(0.4 * SAMPLE_RATE), int(at_s * SAMPLE_RATE)
    decay = np.exp(-np.arange(n) / (0.03 * SAMPLE_RATE))
    x[i:i + n] += (rng.standard_normal(n) * 0.5 * decay).astype(np.float32)
    step = SAMPLE_RATE // 10
    for k in range(0, x.size, step):
        yield AudioBlock(x[k:k + step], k / SAMPLE_RATE,
                         utc=utc0 + timedelta(seconds=k / SAMPLE_RATE))


def test_a_bang_in_the_audio_is_reported_with_its_utc_onset(tmp_path):
    r, up = runner(tmp_path, lambda i: {PROPELLER: 0.0})
    r.run(bang_blocks(6, 3.0))
    [report] = [m for p, m in up.sent if p == "/v1/impulses"]
    assert verify(report, r.keys.public_key)
    assert abs((report.onset_at - (UTC0 + timedelta(seconds=3))).total_seconds()) < 0.001
    assert r.stats.impulses == 1


def test_impulse_detection_can_be_disabled(tmp_path):
    cfg = replace(make_test_config(tmp_path), impulse=ImpulseConfig(enabled=False))
    up = RecordingUplink()
    NodeRunner(cfg, load_or_create_keys(cfg.key_file), FakeClassifier(lambda i: {}), up
               ).run(bang_blocks(6, 3.0))
    assert not [p for p, _ in up.sent if p == "/v1/impulses"]


# test_config_keys.py
def test_impulse_table_overrides_defaults(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(EXAMPLE.read_text() + "\n[impulse]\ntrigger_db = 25\n")
    assert load_config(path).impulse.trigger_db == 25


def test_impulse_trigger_must_be_positive(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text(EXAMPLE.read_text() + "\n[impulse]\ntrigger_db = -1\n")
    with pytest.raises(ConfigError, match="trigger_db"):
        load_config(path)


# test_e2e.py
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
```

(`runner`, `RecordingUplink`, `make_test_config`, `PROPELLER` and `SAMPLE_RATE` are the helpers
already used in those files.)

- [ ] **Step 2: Run** — Expected: fail.
- [ ] **Step 3: Implement.**
  - `audio.py`: add `utc` to `AudioBlock`, add `adc_utc` (below). In `_open_stream`, the callback
    signature becomes `callback(indata, frames, time_info, status)`; it enqueues
    `(indata[:, 0].copy(), adc_utc(datetime.now(UTC), time_info.currentTime,
    time_info.inputBufferAdcTime, frames, rate, holder["latency"]))`, where `holder["latency"]` is
    set to `stream.latency` right after the stream is created. `mic_source` unpacks the pair and
    yields `AudioBlock(samples, t, restart, utc=utc)`. `wav_source` sets `utc = start_utc + t`
    (`start_utc` defaults to `datetime.now(UTC)` at the call). Document that timestamps are
    audio time for `speed != 1`.

```python
def adc_utc(now: datetime, current_time: float, adc_time: float, frames: int, rate: float,
            latency: float) -> datetime:
    """UTC of a block's first sample from PortAudio's capture timestamp, when it is usable.

    Some host APIs report inputBufferAdcTime as 0; then fall back to the buffer's length plus the
    stream's reported input latency.
    """
    age = current_time - adc_time
    if adc_time > 0 and 0 <= age < 1.0:
        return now - timedelta(seconds=age)
    return now - timedelta(seconds=frames / rate + max(latency, 0.0))
```

  - `runner.py`: `self._impulses = ImpulseDetector(cfg.impulse) if cfg.impulse.enabled else None`.
    At the start of `_process`, `utc = block.utc or self._fallback_utc(block)`, where the fallback
    anchors `self._now() - timedelta(seconds=block.t)` on the first block and adds `block.t` after.
    Then send each candidate as a signed `ImpulseReport` (source `acoustic_node`, the config
    location and `time_quality`) to `/v1/impulses`, increment `stats.impulses`, and log a WARNING
    line `IMPULSE peak %.1f dBFS onset %s`.
  - `config.py`: parse the `[impulse]` table into `ImpulseConfig` (enabled, trigger_db, min_peak_dbfs,
    dead_time_s, post_s). Validate `trigger_db > 0`, `post_s > 0`, `dead_time_s >= post_s`.
    `config.example.toml`: document the table as comments, like `[traces]`.
  - `outbox.py`: in `put`, when full, drop in the order `("/v1/heartbeats", "/v1/observations")`,
    then the oldest row of any path.
- [ ] **Step 4: Run** `uv run --no-sync pytest node` — Expected: pass.
- [ ] **Step 5: Commit** `feat(node): report impulses with driver-timestamped UTC onsets`.

---

### Task 7: Simulator — blasts, detonating drones, blocked sound paths

**Files:** Modify `sim/src/kuulo_sim/{scenario,engine,runner,cli}.py`; create
`sim/src/kuulo_sim/scenarios/{impact_strike,firework}.yaml`, `sim/tests/test_impulses.py`; modify
`server/tests/conftest.py`, `server/tests/test_scenarios.py`.

**Interfaces (Produces):**
```python
class ImpactSpec(_Spec): source_db: float = 150.0
class ImpulseSpec(_Spec): lat: float; lon: float; at_s: float (>= 0); source_db: float = 150.0; label: str = "impulse"
DroneSpec.impact: ImpactSpec | None = None      # detonates at the end of its route
Scenario.impulses: list[ImpulseSpec] = []; Scenario.nlos_delay_s: dict[str, float] = {}
Scenario.air_temperature_c: float = 10.0
def route_end_s(drone) -> float; def route_end_point(drone) -> GeoPoint
SimMessage.kind: Literal["observation", "heartbeat", "impulse"]
@dataclass(frozen=True) class ImpulseTruth: label: str; at: datetime; position: GeoPoint
SimulationRun.impulse_truth() -> list[ImpulseTruth]
```

- [ ] **Step 1: Failing tests** — `sim/tests/test_impulses.py`:

```python
from datetime import UTC, datetime

from kuulo_protocol.geo import distance_m
from kuulo_protocol.impulses import speed_of_sound
from kuulo_sim.engine import SimulationRun
from kuulo_sim.scenario import Scenario, load_scenario, resolve_scenario

T0 = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)


def reports(run):
    return [m.payload for m in run.messages() if m.kind == "impulse"]


def scenario(name, **update):
    return load_scenario(resolve_scenario(name)).model_copy(update=update)


def test_impact_strike_reports_the_detonation_from_every_node_in_range():
    run = SimulationRun(scenario("impact_strike"), T0)
    [truth] = run.impulse_truth()
    rs = reports(run)
    assert truth.label.startswith("impact:") and len(rs) == len(run.scenario.nodes)


def test_arrival_differences_match_geometry_for_gps_nodes():
    run = SimulationRun(scenario("impact_strike", nlos_delay_s={}), T0)
    [truth] = run.impulse_truth()
    c = speed_of_sound(run.scenario.air_temperature_c)
    gps = [r for r in reports(run) if r.time_quality.value == "gps"]
    a, b = gps[0], gps[1]
    expected = (distance_m(a.sensor_location, truth.position)
                - distance_m(b.sensor_location, truth.position)) / c
    assert abs((a.onset_at - b.onset_at).total_seconds() - expected) < 0.003


def test_speed_factor_does_not_compress_arrival_differences():
    slow = reports(SimulationRun(scenario("impact_strike"), T0))
    fast = reports(SimulationRun(scenario("impact_strike"), T0, time_scale=4.0))
    diff = lambda rs: [(r.onset_at - rs[0].onset_at).total_seconds() for r in rs]  # noqa: E731
    assert [round(d, 6) for d in diff(slow)] == [round(d, 6) for d in diff(fast)]


def test_nlos_node_hears_it_late():
    open_path = {r.source.id: r.onset_at for r in reports(SimulationRun(
        scenario("impact_strike", nlos_delay_s={}), T0))}
    blocked = {r.source.id: r.onset_at for r in reports(SimulationRun(
        scenario("impact_strike"), T0))}
    late = sorted(n for n in open_path if (blocked[n] - open_path[n]).total_seconds() > 0.1)
    assert late == sorted(scenario("impact_strike").nlos_delay_s) == ["s06"]


def test_weak_impulse_is_not_heard():
    sc = scenario("firework")
    weak = sc.model_copy(update={"impulses": [sc.impulses[0].model_copy(update={"source_db": 60})]})
    assert reports(SimulationRun(weak, T0)) == []  # 60 dB at 1 m is below every node's floor


def test_impulses_do_not_change_existing_scenario_messages():
    plain = SimulationRun(scenario("helsinki_pass"), T0).messages()
    base = scenario("helsinki_pass").model_dump()
    extra = Scenario.model_validate({**base, "impulses": [
        {"lat": 60.17, "lon": 24.94, "at_s": 50, "source_db": 150, "label": "bang"}]})
    messages = SimulationRun(extra, T0).messages()
    assert any(m.kind == "impulse" for m in messages)  # the bang really is heard
    other = [m for m in messages if m.kind != "impulse"]
    assert [m.payload.model_dump_json() for m in plain] == \
           [m.payload.model_dump_json() for m in other]


def test_deterministic():
    a = [r.model_dump_json() for r in reports(SimulationRun(scenario("impact_strike"), T0))]
    b = [r.model_dump_json() for r in reports(SimulationRun(scenario("impact_strike"), T0))]
    assert a == b
```

In `server/tests/test_scenarios.py`:

```python
def test_impact_strike_is_one_coarse_event_around_the_blast(run_scenario):
    result = run_scenario("impact_strike")  # public CoarseLocator: grouping, no precise fix
    [event] = result.impulse_events
    [truth] = result.impulse_truth
    assert event.quality.value == "coarse"
    assert set(event.node_ids) == {n.id for n in result.run.scenario.nodes}
    assert distance_m(event.position, truth.position) <= event.ellipse.semi_major_m


def test_firework_far_from_drones_is_an_unassociated_impulse(run_scenario):
    result = run_scenario("firework")
    [event] = result.impulse_events
    assert event.kind.value == "unassociated"
```

- [ ] **Step 2: Run** — Expected: fail.
- [ ] **Step 3: Implement.**
  - `scenario.py`: the specs above; a validator that `nlos_delay_s` keys are node ids and values
    lie in [0, 5]. Factor the route-building code of `drone_position` into
    `_route(drone) -> (origin, points)` and add `route_end_s` (start + total length / speed) and
    `route_end_point`.
  - `engine.py`: in `_build`, store the per-node clock offsets as `self._offsets`. After the
    existing loop, call `self._impulse_messages()`, which uses its **own**
    `random.Random(sc.seed ^ 0x1A1A5)` so existing streams are untouched:

```python
    def _impulse_sources(self) -> list[tuple[str, float, GeoPoint, float]]:
        sc = self.scenario
        out = [(i.label, i.at_s, GeoPoint(lat=i.lat, lon=i.lon), i.source_db) for i in sc.impulses]
        for d in sc.drones:
            if d.impact is not None:
                end = (route_end_s(d), route_end_point(d), d.impact.source_db)
                out.append((f"impact:{d.id}", *end))
        return sorted(out, key=lambda s: s[1])

    def _impulse_messages(self) -> list[SimMessage]:
        sc = self.scenario
        rng = random.Random(sc.seed ^ 0x1A1A5)
        c = speed_of_sound(sc.air_temperature_c)
        out = []
        for label, t, pos, source_db in self._impulse_sources():
            for node in sorted(sc.nodes, key=lambda n: n.id):
                if node_offline(sc, node.id, t):
                    continue
                r = distance_m(node, pos)
                level = received_level_db(source_db, r)
                snr = level - node.noise_floor_db
                if snr < IMPULSE_DETECT_SNR_DB:
                    continue
                sigma = min(0.01, max(1 / 16_000, 0.0005 * 10 ** ((40 - snr) / 20)))
                # Propagation, clock error and picking noise are real seconds: only the emission
                # instant follows the scenario's time scale (Review Focus 1).
                delay = (r / c + sc.nlos_delay_s.get(node.id, 0.0) + self._offsets[node.id]
                         + rng.gauss(0, sigma))
                onset = self._at(t) + timedelta(seconds=delay)
                peak = min(0.0, level + DB_SPL_TO_DBFS)
                report = ImpulseReport(
                    source=Source(type=SourceType.SIMULATED_NODE, id=node.id), onset_at=onset,
                    onset_sigma_s=sigma, time_quality=node.time_quality,
                    sensor_location=SensorLocation(lat=node.lat, lon=node.lon, accuracy_m=10.0),
                    features=ImpulseFeatures(
                        peak_dbfs=round(peak, 2), snr_db=round(min(snr, 150.0), 2),
                        rise_time_ms=1.0, duration_ms=150.0, clipped=level + DB_SPL_TO_DBFS >= 0,
                        band_db=[round(peak - 15.0, 2)] * 32),
                    report_id=UUID(int=rng.getrandbits(128), version=4),
                )
                signed = sign(report, self.node_keys[node.id][0])
                out.append(SimMessage(onset + timedelta(seconds=1.0), "impulse", signed))
        return out
```

  (`IMPULSE_DETECT_SNR_DB = 20.0` in `engine.py`; `DB_SPL_TO_DBFS` from `.traces`.) Merge these
  into the sorted message list; `impulse_truth()` returns one `ImpulseTruth` per source.
  - `runner.py` / `cli.py` / `server/tests/conftest.py`: map `kind → path` with
    `{"observation": "/v1/observations", "heartbeat": "/v1/heartbeats", "impulse": "/v1/impulses"}`.
    `write_truth` also writes `truth_impulses.json` beside the truth file. `ScenarioResult` gains
    `impulse_events: list[ImpulseEvent]` (read from `/v1/impulse-events` at the end) and
    `impulse_truth`.
  - `impact_strike.yaml`: nodes `s01`–`s08` on a grid around lat 60.165–60.177 and
    lon 24.915–24.955, `s01`–`s04` with `time_quality: gps` and `s05`–`s08` with `ntp`. One drone
    (speed 25 m/s, start 5 s) flies from (60.1712, 24.9050) to (60.1712, 24.9352) with
    `impact: {source_db: 150}`. `nlos_delay_s: {s06: 0.12}`, duration 120 s.
  - `firework.yaml`: the same nodes, no drones, one impulse
    `{lat: 60.1740, lon: 24.9200, at_s: 20, source_db: 140, label: firework}`, duration 60 s.
- [ ] **Step 4: Run** `uv run --no-sync pytest sim server` — Expected: pass, and all existing
  scenario tests unchanged.
- [ ] **Step 5: Commit** `feat(sim): detonating drones, standalone blasts, blocked sound paths`.

---

### Task 8: Dashboard — impulse layer, list, detail, CAP link

**Files:** Modify `dashboard/src/api/{client,types}.ts`, `state/reducer.ts`, `map/{geo,layers}.ts`,
`components/{MapView,NodePanel,TopBar}.tsx`, `App.tsx`, `styles.css`; create
`components/ImpulseDetail.tsx`; tests `state/reducer.test.ts`, `map/map.test.ts`.

**Interfaces (Produces):** `State.impulses: Record<string, ImpulseEvent>`, `State.selectedImpulseId: string | null`;
actions `{type: "selectImpulse"; id: string | null}`; `snapshot` gains `impulses: ImpulseEvent[]`;
`ellipsePolygon(lat, lon, a, b, bearingDeg, steps?)`; `impulsesToGeoJSON`, `impulseEllipsesToGeoJSON`;
`fetchSnapshot()` also loads `/v1/impulse-events`; `fetchImpulseDetail(id)`.

- [ ] **Step 1: Failing tests.**

```ts
// state/reducer.test.ts fixture
const impact: ImpulseEvent = {
  event_id: "e1", kind: "drone_impact", quality: "multilaterated",
  position: { lat: 60.17, lon: 24.94 },
  ellipse: { semi_major_m: 120, semi_minor_m: 40, bearing_deg: 30, confidence: 0.95 },
  alternatives: [], occurred_at: "2026-09-26T12:00:00Z", node_ids: ["a", "b", "c", "d"],
  excluded_node_ids: [], report_ids: ["r1"], residuals_ms: {}, rms_residual_ms: 1.2,
  associated_track_id: "t1", updated_at: "2026-09-26T12:00:01Z",
};

// map/map.test.ts
it("ellipse polygon has the right axes and orientation", () => {
  const ring = ellipsePolygon(60.17, 24.94, 400, 100, 90, 64);
  expect(ring.length).toBe(65);
  expect(ring[0]).toEqual(ring[64]);
  expect(haversineM(60.17, 24.94, ring[0][1], ring[0][0])).toBeCloseTo(400, -1);
  expect(ring[0][0]).toBeGreaterThan(24.94);                       // major axis points east
  expect(haversineM(60.17, 24.94, ring[16][1], ring[16][0])).toBeCloseTo(100, -1);
});

// state/reducer.test.ts
it("impulse events upsert, log IMPACT vs IMPULSE, and selection is exclusive with tracks", () => {
  let s = reducer(initialState, { type: "select", trackId: "t1" });
  s = reducer(s, { type: "live", receivedAt: 1, event: { type: "impulse_event", data: impact } });
  expect(s.impulses[impact.event_id]).toEqual(impact);
  expect(s.events[0].message).toMatch(/^IMPACT/);
  s = reducer(s, { type: "selectImpulse", id: impact.event_id });
  expect(s.selectedTrackId).toBeNull();
  s = reducer(s, { type: "select", trackId: "t1" });
  expect(s.selectedImpulseId).toBeNull();
});
it("snapshot replaces impulses", () => {
  let s = reducer(initialState, { type: "live", receivedAt: 1,
    event: { type: "impulse_event", data: impact } });
  s = reducer(s, { type: "snapshot", nodes: [], tracks: [], impulses: [{ ...impact, event_id: "e2" }] });
  expect(Object.keys(s.impulses)).toEqual(["e2"]);
});
```

Make `impulses` optional on the `snapshot` action (default `[]`) so the existing reducer and
`live.ts` tests keep compiling; `live.ts` passes `snap.impulses` through.

- [ ] **Step 2: Run** `cd dashboard && npx vitest run` — Expected: fail.
- [ ] **Step 3: Implement.**
  - `geo.ts`:

```ts
/** Closed ring of [lon, lat] on an ellipse; bearing = major axis, degrees clockwise from north. */
export function ellipsePolygon(lat: number, lon: number, a: number, b: number, bearingDeg: number,
  steps = 64): [number, number][] {
  const th = rad(bearingDeg);
  const mLat = 110_540, mLon = 111_320 * Math.cos(rad(lat));
  const ring: [number, number][] = [];
  for (let i = 0; i < steps; i++) {
    const t = (2 * Math.PI * i) / steps;
    const x = a * Math.cos(t) * Math.sin(th) + b * Math.sin(t) * Math.cos(th);
    const y = a * Math.cos(t) * Math.cos(th) - b * Math.sin(t) * Math.sin(th);
    ring.push([lon + x / mLon, lat + y / mLat]);
  }
  ring.push(ring[0]);
  return ring;
}
```

  (It uses the same flat-earth projection as `protocol/geo.py`.)
  - `reducer.ts`: `impulses` in state and snapshot; the `impulse_event` live case upserts and
    logs `IMPACT n SENSORS ±a M` (tone danger) or `IMPULSE …` (tone warning), with source
    `I-XXXX` (use `shortTrackId`, swapping the `T-` prefix for `I-`). Log only when the event is
    new or its quality changes. Selecting a track clears the impulse selection and vice versa.
  - `layers.ts`: points (`id`, `kind`, `quality`) and ellipse polygons (kind). For AMBIGUOUS
    events, also emit the `alternatives` as points with `alt: 1`.
  - `MapView.tsx`: sources `impulses` and `impulse-ellipses`. Layers: an ellipse fill (opacity
    0.12) and a dashed outline, coloured by kind (`drone_impact` → `PALETTE.danger`, otherwise
    `PALETTE.warning`). Impulse points are circles of radius 8 with a thick stroke, drawn above
    tracks; alternatives are hollow circles. Clicking dispatches `selectImpulse`.
  - `NodePanel.tsx`: an "Impulses" section above Sensors. Rows: `I-XXXX`, ±semi-major, and a tag
    IMPACT (danger) or IMPULSE (warning); clicking selects the event.
  - `ImpulseDetail.tsx` (shown in the right panel when an impulse is selected): KIND, QUALITY,
    POS, 95 % ELLIPSE `a × b M @ bearing°`, TIME, SENSORS n (excluded listed), RMS residual ms, a
    per-node residual table, the associated track callsign (click → select track), and a
    `CAP XML` link to `/v1/impulse-events/{id}/cap` (`target="_blank"`).
  - `TopBar.tsx`: stat `IMPACTS n`, with a danger tag when any event is `drone_impact`.
- [ ] **Step 4: Run** `npx vitest run && npm run build` — Expected: pass. Then run `make dev` and
  `make sim SCENARIO=impact_strike SPEED=2` in a browser. Confirm: the track is confirmed, then an
  orange IMPULSE circle around the sensors appears, and its detail lists all 8 nodes and a CAP
  link. `make sim SCENARIO=firework` shows another. (The red IMPACT view with an ellipse and the
  track link is checked with the private locator in the private plan, task F4.) Stop `make dev`
  afterwards.
- [ ] **Step 5: Commit** `feat(dashboard): impulse events with uncertainty ellipses, detail and CAP link`.

---

### Task 9: Real-audio evaluation of the node detector

**Files:** Create `ml/src/kuulo_ml/impulses.py`, `ml/tests/test_impulses.py`; modify
`ml/src/kuulo_ml/cli.py` (`impulses` command), `Makefile` (`impulse-eval`); output `ml/IMPULSE_RESULTS.md`.

**Interfaces (Produces):**
```python
IMPULSIVE = ("fireworks", "glass_breaking", "door_wood_knock")
CONTINUOUS = ("rain", "wind", "sea_waves", "engine", "train", "helicopter", "airplane",
              "chainsaw", "vacuum_cleaner", "crackling_fire", "insects", "crickets")
def mix_at(background: np.ndarray, event: np.ndarray, offset: int, snr_db: float, ref: int) -> np.ndarray
    # scales `event` so its energy over 50 ms after sample `ref` is snr_db above the background's mean energy
def detect(x: np.ndarray) -> list[tuple[int, float]]   # (onset sample index, onset sigma s)
def evaluate(positives: list[np.ndarray], backgrounds: list[np.ndarray],
             negatives: list[np.ndarray], rng: np.random.Generator,
             snrs=(30, 20, 10)) -> dict
    # keys: clean_detected (fraction), false_per_hour,
    #       snr[s] = {detected, median_error_ms, p95_error_ms, within_2sigma}
def report_markdown(results: dict) -> str
```

- [ ] **Step 1: Failing tests** (synthetic audio only, so CI can run them):

```python
import numpy as np

from kuulo_ml.impulses import evaluate, mix_at

SR = 16_000


def burst(seed, amp=0.5):
    n = int(0.4 * SR)
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * amp * np.exp(-np.arange(n) / (0.03 * SR))).astype(np.float32)


def noise(seed, seconds=8.0, level=0.003):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * level).astype(np.float32)


def test_mix_places_the_event_at_the_offset_and_snr():
    bg = noise(0)
    ev = np.concatenate([np.zeros(100, np.float32), burst(1)])
    added = mix_at(bg, ev, offset=32_000, snr_db=20, ref=100) - bg
    assert np.all(added[:32_000] == 0)
    seg = added[32_100: 32_100 + int(0.05 * SR)].astype(np.float64)
    snr = 10 * np.log10(np.mean(seg ** 2) / np.mean(bg.astype(np.float64) ** 2))
    assert abs(snr - 20) < 0.2


def test_evaluate_on_synthetic_bursts_detects_them_with_small_error():
    positives = [np.concatenate([np.zeros(1600, np.float32), burst(s)]) for s in range(5)]
    backgrounds = [noise(10 + s) for s in range(3)]
    negatives = [noise(20 + s, seconds=30) for s in range(3)]
    r = evaluate(positives, backgrounds, negatives, np.random.default_rng(2), snrs=(30,))
    assert r["clean_detected"] == 1.0 and r["false_per_hour"] == 0.0
    assert r["snr"][30]["detected"] == 1.0 and r["snr"][30]["median_error_ms"] < 1.0
```

`evaluate` pads each clean positive with 3 s of white noise at 1e-4 before detecting (so the
detector's 0.5 s warm-up never overlaps the event). Its reference onset is the first detection
minus the padding. In each mix, the report nearest the expected onset (within ±0.5 s) counts as
the detection.

- [ ] **Step 2: Run** — Expected: fail.
- [ ] **Step 3: Implement.**
  - **Clean detection:** for each ESC-50 clip in `IMPULSIVE`, the fraction of clips with at
    least one report (clips padded to 3 s of leading silence so the detector warms up).
  - **False triggers:** reports per hour over every clip in `CONTINUOUS` and every DroneNoise
    recording (drones must not look like bangs).
  - **Onset error vs SNR:** a reference onset is the detector's first onset on the clean,
    padded positive (only clips where it fires). Each positive is mixed into a random 8 s
    background from `CONTINUOUS` or DroneNoise, at offset 2.0 s + U(0, 0.5) s, at each SNR in
    (30, 20, 10) dB. Record the detection rate, the median and 95th-percentile absolute onset
    error in ms, and the fraction of errors within 2× the reported `onset_sigma_s` (calibration).
  - `report_markdown` writes the tables plus the caveat: "fireworks and knocks are proxies; no
    licensed blast recordings were used".
  - **Calibrate:** if, at 20 dB and above, fewer than 90 % of errors fall within 2σ, raise
    `SIGMA_RISE_FACTOR` in `node/src/kuulo_node/impulse.py` until they do. Re-run, and record
    the final value and why in the results and as a ruling.
  - Run `make impulse-eval` for real (it needs the ESC-50 and DroneNoise downloads already in
    `data/datasets/`).
- [ ] **Step 4: Run** `uv run --no-sync pytest ml` and `make impulse-eval` — Expected: tests pass;
  `ml/IMPULSE_RESULTS.md` written.
- [ ] **Step 5: Commit** `feat(ml): real-audio evaluation of the impulse detector; calibrated onset sigma`.

---

### Task 10: Documentation and final checks

- [ ] README:
  - **How it works:** an impulse paragraph, including that precise localization and drone-impact
    attribution come from an optional private locator selected with `KUULO_IMPULSE_LOCATOR`.
    **Do not name the private package.**
  - **Quick start:** `make sim SCENARIO=impact_strike`.
  - **How well it works:** Task 9's detector numbers.
  - **Known limits:** the public locator is coarse; laptop-mic timing; the fireworks proxy; CAP
    status Test; no authentication.
  - **Roadmap:** impulse events done.
- [ ] Spec status → "Implemented"; add a `docs/superpowers/plans/PROGRESS.md` entry.
- [ ] No code or config names the private package:
  `git grep -n -E "kuulo[_-]fusion" -- '*.py' '*.toml' '*.ts' '*.tsx' Makefile` returns nothing
  (case-sensitive on purpose: the public `KUULO_FUSION_ENGINE` variable is fine).
- [ ] `make test`, `cd dashboard && npm run build` and `make prepublish` are all green. The new
  scenarios' coordinates are simulated Helsinki grid points.
- [ ] Final whole-branch review per the executing skill; fix Critical/Important findings with
  tests first; list rulings and deferred minors for the user.
- [ ] Commit `docs: impulse events in README, spec status, progress`.

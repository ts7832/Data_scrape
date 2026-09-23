# Kuulo — Milestone 1 Design Spec

**Date:** 2026-09-24
**Status:** Draft, awaiting review
**Working name:** Kuulo (Finnish: "hearing")

## 1. Purpose and context

Kuulo is a civic acoustic drone-detection network, starting in Helsinki, with a
verification/fusion layer that turns many cheap sensor observations into
trustworthy drone tracks.

**Why:** Long- and medium-range radars lose small, low, slow drones. Nordic–Baltic
governments are covering their *borders* (Latvia acoustic border network, Estonia
"drone wall", Finnish Border Guard trials), but *interiors* — cities, ports,
energy sites — are thinly covered, and false alarms are costly (Vilnius airport
closed for a flock of birds, Sept 2026). No civilian sensing network exists in
Finland; Finland's planned 112 Suomi drone alerts (by 2027) are one-way.

**Competitive position:** Drone Radar (Mainline, Lithuania, launched June 2026)
runs a crowd-sourced Android-phone network targeting the Baltics and Poland.
Kuulo differs by (a) outdoor, weatherproof, GPS-timeable hardware nodes as a
trusted backbone, (b) a source-agnostic fusion/verification layer that can
ingest other networks' data, and (c) Finland first.

**Two layers (both started in this milestone):**
- **Layer A — Sensing network:** nodes that detect drones acoustically.
- **Layer B — Fusion/verification:** combines observations from any source into
  confirmed / tentative / downgraded tracks with traceable evidence.

**Goals of Milestone 1:** a complete, locally runnable pipeline on a laptop,
suitable as a public GitHub portfolio project and as the foundation for the
startup. No hardware and no cloud.

**Non-goals of Milestone 1:** Raspberry Pi deployment, node provisioning (hotspot
setup, claim codes), OTA updates, internet deployment, user accounts, citizen
reporting app, Remote ID / ADS-B ingestion, TDOA localization, the private
fusion engine, military-drone (Shahed-class) accuracy.

## 2. Open-core boundary

| Public repo (`kuulo`, AGPL-3.0) | Private (later, `kuulo-fusion`) |
|---|---|
| protocol, node software, baseline model, training scripts | advanced fusion engine (TDOA, Kalman/MHT, trust scores) |
| server, basic fusion, dashboard, simulator | models trained on network data |
| build guide, docs | aggregated detection data, node coverage maps |

Rules:
- The repository contains **download scripts only — never audio files or model weights**.
- **Raw audio never leaves a node.**
- **The live coverage map is never public.**

## 3. Architecture

```
kuulo/
├── protocol/   Pydantic models + generated JSON Schema + Ed25519 signing
├── node/       mic/wav → features → classifier → smoothing → signed events → uplink
├── ml/         dataset download, training, evaluation, ONNX export
├── server/     FastAPI: ingest → store (SQLite) → fusion plugin → WebSocket
├── dashboard/  React + TypeScript + Vite + MapLibre (OpenFreeMap tiles)
├── sim/        scenario-driven fake nodes using the real API
└── docs/
```

Data flow:
```
laptop mic ─► node ─┐
                    ├─► server ─► fusion ─► tracks ─► WebSocket ─► dashboard
sim (N nodes) ──────┘     └─► SQLite
```

Tooling: Python 3.12 managed by `uv` (one workspace, one package per component);
Node.js LTS + npm for the dashboard; `pytest`, `ruff`; `vitest` for the dashboard.
One command (`make dev`) starts server + dashboard.

## 4. Protocol (`protocol/`)

All components exchange four message types, defined once as Pydantic v2 models.
JSON Schema is generated from them; the dashboard's TypeScript types are generated
from that schema.

### 4.1 Observation — "a source noticed something"
| Field | Type | Notes |
|---|---|---|
| `schema_version` | str | `"1.0"` |
| `observation_id` | UUID | unique; server uses it for idempotency |
| `source` | `{type, id}` | type ∈ `acoustic_node`, `simulated_node`, `citizen_report`, `remote_id`, `adsb`, `external_network` (only the first two used in M1) |
| `observed_at` | UTC datetime, ms precision | |
| `time_quality` | enum | `gps` \| `ntp` \| `manual` |
| `sensor_location` | `{lat, lon, accuracy_m}` | location of the **sensor**, not the target |
| `detection` | `{label, confidence, bearing_deg?}` | label ∈ `drone_multirotor`, `drone_fixedwing_engine`, `aircraft`, `bird`, `unknown`; confidence 0–1; bearing null for single mic |
| `event` | `{detection_id, phase}` | groups one pass; phase ∈ `start`, `update`, `end` |
| `acoustic` | `{snr_db, peak_freq_hz}` optional | |
| `signature` | str | Ed25519 over canonical JSON (sorted keys, no whitespace, signature field excluded) |

### 4.2 Heartbeat — "I'm alive"
`node_id`, `sent_at`, `software_version`, `mic_ok`, `cpu_temp_c?`, `queue_depth`, `signature`. Sent every 60 s.

### 4.3 Track — produced by the server
`track_id`, `status` (`tentative` \| `confirmed` \| `downgraded` \| `closed`),
`label`, `confidence`, `position {lat, lon}`, `uncertainty_m`,
`velocity {speed_mps, heading_deg}?`, `first_seen`, `last_seen`,
`observation_ids[]`, `silent_neighbour_ids[]`.

### 4.4 FeatureTrace — detailed record of a detection
Header: `trace_id`, `node_id`, `detection_id`, `segment_index`, `time_quality`,
`frame_period_ms` (20), `start_at`, `band_edges_hz[32]`, `signature`.
Body: per frame — `t_offset_ms`, `band_db[32]` (log-mel energies), `rms_db`,
`peak_freq_hz`. Stored as compressed binary (NumPy `.npz`), ~6 KB/s.
Coarse 32-band energies are used so the trace cannot be turned back into
intelligible speech.

### 4.5 Identity
Each node generates an Ed25519 keypair on first run, stored in `~/.kuulo/` (0600).
The public key is registered with the server; every message is signed.

## 5. Node (`node/`)

Pipeline:
```
audio source → 16 kHz mono ring buffer → 0.96 s windows, 50 % hop
  → classifier → smoothing → Observation builder → sign → uplink (with offline queue)
                    └─► FeatureTrace recorder (active only during detections)
```

- **Audio sources:** `sounddevice` microphone; `--input file.wav` replay (same
  code path, replayed at real-time or accelerated speed).
- **Classifier interface:** `score(window) -> {label: prob}`. Two implementations:
  - **Step A (zero-training):** YAMNet (TF-Hub/TFLite or ONNX port) class scores
    for aircraft-related classes (Propeller/airscrew, Aircraft, Helicopter, Engine,
    Drone where present), mapped to a drone score.
  - **Step B (trained):** frozen YAMNet 1024-d embeddings → small head
    (logistic regression or 1-hidden-layer MLP) trained in `ml/`, exported to ONNX,
    run with `onnxruntime`.
  Selected by config; A vs. B compared on held-out data.
- **Smoothing:** a detection starts when ≥3 of the last 5 windows exceed the
  threshold (default 0.5); ends after 5 s below threshold. It emits `start`,
  then `update` every 5 s, then `end`. All values are configurable.
- **Uplink:** HTTP POST (`httpx`) with retry and exponential backoff.
  - Offline queue is a local SQLite file capped at 10 000 messages.
  - When the queue is full, heartbeats are dropped first; observations are dropped
    only after the cap and never silently (the drop is logged and counted).
- **Heartbeat:** every 60 s, including `mic_ok` (the mic is reopened every 10 s on failure).
- **Privacy:** audio exists only in the in-memory ring buffer.
  `--debug-save-clips` (off by default) writes clips locally only.
- **Config:** one TOML file (node id, location, server URL, thresholds, classifier choice).
- **Clock:** `time_quality=ntp` on a laptop by default; `manual` if the OS clock is
  not NTP-synced.

### 5.1 FeatureTrace recording rules
- Recording runs only while a detection is active.
- Traces are written in **60 s segments**; a long event produces multiple segments.
- Local store: a node-side disk budget (default 500 MB) with **priority eviction**:
  1. uncorroborated segments older than 7 days
  2. oldest uncorroborated segments
  3. corroborated segments already uploaded (last)
- **Upload policy:**
  - **Server pull:** the server requests segments for any detection that
    contributes to a `confirmed` track. **No length cap**: all segments of a
    confirmed event are uploaded, so long events such as swarms are kept in full.
  - **Auto-push:** single-node detections with confidence ≥0.8 sustained ≥20 s.
  - **Hard-negative sampling:** 1 % of uncorroborated detections, capped at
    3 min per detection.
  - Otherwise traces stay local and expire.
- **Upload budget:** 50 MB/day per node for auto-push and sampling. Server-pull
  for confirmed tracks bypasses the budget.

Known limit: separating individual drones within a dense swarm from acoustic
energy alone is not solved in M1. The traces preserve the data so this can be
attempted later.

## 6. Server (`server/`)

FastAPI + Uvicorn. SQLite (WAL mode) via SQLAlchemy 2.0. Binds to `127.0.0.1` only.

| Endpoint | Purpose |
|---|---|
| `POST /v1/nodes/register` | node id, public key, location (open registration on localhost in M1) |
| `POST /v1/observations` | one or a batch of signed Observations |
| `POST /v1/heartbeats` | signed Heartbeat |
| `POST /v1/traces` | multipart: signed header + `.npz` body |
| `GET /v1/traces/requests?node_id=` | node polls for trace segments the server wants |
| `GET /v1/nodes`, `GET /v1/tracks` | current state for the dashboard |
| `GET /v1/tracks/{id}` | a track with its evidence (observations, silent neighbours) |
| `WS /v1/live` | pushes `observation`, `track`, `node_status` events |

Ingest pipeline for each message:
1. **Validate** against the schema → 400 with the reason if invalid.
2. **Verify signature** → 401 if it fails.
3. **Idempotency:** a duplicate `observation_id` returns 200 and is ignored.
4. **Time sanity:** reject if more than 30 s in the future; accept older messages
   but mark them `late=true` (stored as evidence, never trigger live tracks).
5. **Store** the message.
6. **Fusion:** call the fusion plugin. Exceptions are caught and logged, and
   ingest continues.
7. **Broadcast** over the WebSocket.

**Node status** is computed from heartbeat age: `online` <2 min, `stale` 2–5 min,
`offline` >5 min.

**Noisy-node metric:** per node, the rate of uncorroborated detections over 24 h,
exposed in `GET /v1/nodes`.

**Tables:** `nodes`, `observations`, `heartbeats_latest`, `tracks`,
`track_observations`, `traces`, `trace_requests`.
Trace bodies are stored on disk under `data/traces/`; the database holds metadata.

**Fusion tick:** runs once per second (ages, closes and downgrades tracks).

**Security note (README):** M1 is localhost-only, with no auth on read endpoints.
It must not be exposed to the internet. Auth, claim codes and TLS come in the
deployment milestone.

## 7. Basic fusion (public)

Plugin interface:
```python
class FusionEngine(Protocol):
    def on_observation(self, obs: Observation, ctx: FusionContext) -> list[TrackUpdate]: ...
    def on_tick(self, now: datetime, ctx: FusionContext) -> list[TrackUpdate]: ...
```
`FusionContext` gives read access to the node registry, node status, and recent
observations. The engine is selected by a config entry point, so the private
engine drops in without changes to the public code.

**BasicFusion algorithm:**
1. **Grouping:** non-late, drone-labelled observations from nodes within
   `group_radius_m` (default 2000) and `group_window_s` (default 10).
2. **Location:** a weighted centroid of the detecting nodes' positions, with
   weight = confidence × 10^(snr_db/20).
   Uncertainty = max(node spread, `base_range_m`, default 300).
   This is coarse, on the order of hundreds of metres, and documented as such.
3. **Association:** a group within `gate_m` (default 1000) and 10 s of an existing
   track updates it. Position uses exponential smoothing; speed and heading come
   from the last positions. Otherwise the group starts a new track.
   A track with no updates for 30 s → `closed`.
4. **Status:**
   - `tentative`: one node.
   - `confirmed`: ≥2 distinct nodes within the window.
   - `downgraded`: one node detecting while ≥1 **online** neighbour within
     `expected_hearing_m` (default 500) reports nothing. Confidence is multiplied
     by 0.5 per silent neighbour, and such a track cannot be confirmed by that
     node alone.
5. When a track becomes `confirmed`, trace requests are created for all
   contributing detections.

## 8. Dashboard (`dashboard/`)

React + TypeScript + Vite; MapLibre GL with OpenFreeMap tiles (no API key);
centred on Helsinki.
- **Layers:**
  - nodes, coloured by status (online / stale / offline)
  - observation pulses
  - tracks: marker, uncertainty circle and trail, coloured by status
    (tentative amber, confirmed red, downgraded grey)
- **Side panel:**
  - live track list
  - track detail showing the evidence (contributing observations, confidences,
    silent neighbours)
  - node list with the noisy-node metric
  - a connection indicator
- **Data:** initial REST load, then WebSocket updates; on reconnect (with backoff)
  it performs a full REST resync.
- **Types** are generated from the protocol JSON Schema (`json-schema-to-typescript`).
- **Out of scope:** timeline replay and authentication.

## 9. Simulator (`sim/`)

- **Scenario YAML:**
  - nodes (id, lat/lon, time_quality, noise floor)
  - drones (label, waypoints, speed, source level dB)
  - false-alarm events (position, time, duration, level)
  - node failures (offline windows)
  - random seed
- **Physics:**
  - received level = source level − 20·log10(r) − α·r
  - SNR = received − noise floor
  - detection probability is a logistic function of SNR, and confidence is
    derived from SNR with noise added
  - arrival time = emission + r/343 m/s + clock error sampled per `time_quality`
    (gps ±1 µs, ntp ±20 ms, manual ±500 ms)
- **Outputs:** signed Observations, Heartbeats and FeatureTraces (synthetic band
  energies consistent with the received level and Doppler), all through the real HTTP API.
- **Modes:** real-time or accelerated; mixed mode alongside a real laptop node.
- **Ground truth:** true drone positions are written to a JSON file for evaluation.
- **Bundled scenarios:**
  - `single_node`
  - `helsinki_pass` (10 nodes, 1 drone)
  - `false_alarm` (truck near one node)
  - `two_drones` (5 km apart)
  - `node_failure`
  - `long_event` (10-minute loiter: exercises multi-segment traces)

## 10. ML (`ml/`)

- `ml/datasets/`: a download script per dataset, each with a documented licence;
  data lands in `data/` (gitignored). Candidates:
  - drone positives: DroneAudioDataset (GitHub), DADS (Hugging Face)
  - negatives: ESC-50 and similar sets of traffic, wind, rain, birds and engines
  - **Each licence is verified before use and recorded in `ml/DATASETS.md`,
    including whether commercial use is permitted.**
- **No YouTube or other scraped audio.**
- Splits are by recording/source, not by window, to avoid leakage.
- `ml/train.py`: YAMNet embeddings → head → ONNX export.
- `ml/evaluate.py`: precision/recall/F1 and a confusion matrix for Step A and
  Step B on the same held-out set; results go into the README.

## 11. Error handling

| Where | Failure | Behaviour |
|---|---|---|
| Node | mic missing or fails | `mic_ok=false` in heartbeat; retry every 10 s |
| Node | server unreachable | queue + exponential backoff (max 60 s) |
| Node | model file missing | exit with a clear message and the download command |
| Node | disk budget reached | priority eviction (§5.1) |
| Server | malformed or bad signature | 400/401 with a reason; logged |
| Server | fusion exception | logged, isolated; ingest continues |
| Server | SQLite busy | WAL + short retry |
| Dashboard | WebSocket drop | "reconnecting" indicator, backoff, full resync |
| Sim | invalid scenario | validation error naming the field |

## 12. Testing

- **protocol:**
  - round-trip serialisation
  - canonical JSON stability
  - sign/verify, and tamper detection
- **node:**
  - smoothing on synthetic confidence sequences
  - synthetic signals generated at test time (tones, harmonics, noise; no audio
    files committed) produce the expected events through the wav-replay path
  - offline queue and retry against a mock server
  - trace segmentation and eviction order
- **server:**
  - API tests for valid, malformed and bad-signature messages
  - idempotency
  - late marking
  - node status ageing
  - WebSocket receives events
  - trace request / upload flow
- **fusion:** scenario tests via the simulator with ground truth:
  - `single_node` → only tentative
  - `helsinki_pass` → confirmed, with mean location error below a documented bound
  - `false_alarm` → never confirmed
  - `two_drones` → two tracks
  - `node_failure` → offline node's silence not counted
- **sim:**
  - seed determinism
  - physics monotonicity (farther means quieter)
- **dashboard:** vitest for the state reducer and WebSocket resync logic.
- **End-to-end:** start the server and run `helsinki_pass` → a confirmed track
  exists via REST.
- **CI:** GitHub Actions runs ruff, pytest and vitest on push.

## 13. Milestone 1 definition of done

1. `make dev` starts the server and dashboard; `kuulo-node` runs against the laptop mic.
2. Playing a drone recording near the laptop produces an observation on the map
   within 3 s.
3. The simulator's `helsinki_pass` produces a confirmed track, with the location
   error measured and reported; `false_alarm` never confirms.
4. The README contains the Step A vs. Step B classifier metrics.
5. Feature traces are recorded, pulled on confirmation, and stored; `long_event`
   yields multiple segments.
6. All tests pass locally and in CI.
7. The README includes:
   - purpose
   - demo GIF
   - architecture
   - quick start
   - known limitations (localhost-only, coarse location, dataset licences, swarm separation)
   - open-core note
   - AGPL-3.0 licence

## 14. Build order

1. Repo skeleton, uv workspace, protocol + schema + signing
2. Server ingest, storage, API tests
3. Simulator (data flows before any ML)
4. Dashboard live map
5. Basic fusion + scenario tests
6. Node with Step A classifier (mic + wav replay)
7. Feature traces + server-pull upload
8. ML Step B training + evaluation
9. README, demo GIF, CI

## 15. Prerequisites (user, one-time)

`brew install uv node`. Everything else installs inside the project folder.

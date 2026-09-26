# Kuulo Part 3 — Milestone 1 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish everything Milestone 1's definition of done still lacks: a trained Step B classifier with
honest Step A vs Step B metrics, FeatureTraces end to end (node, simulator, server pull), a durable
node queue with batch ingest, the deferred review fixes, the `long_event` scenario, a real-audio
end-to-end check, the demo GIF, and a merged `main`.

**Architecture:** No new services. `protocol/` gains the FeatureTrace header; `server/` gains trace
storage, trace requests created when a track confirms, and batch ingest; `node/` gains a trace
recorder with a disk-budgeted store, a trace uploader, a SQLite outbox and a Step B classifier
(frozen YAMNet embedding → small ONNX head); `sim/` synthesises traces; a new `ml/` workspace
member downloads licensed data, extracts embeddings, trains, exports ONNX and evaluates.

**Tech Stack:** Python 3.12 / uv workspace, FastAPI, SQLAlchemy, NumPy/SciPy, ai-edge-litert
(YAMNet TFLite), scikit-learn + skl2onnx (training only), onnxruntime (node inference),
React/TypeScript/Vitest (dashboard).

**Spec:** `docs/superpowers/specs/2026-09-24-kuulo-milestone1-design.md` (§4.4, §5, §5.1, §6, §9,
§10, §12, §13) and `docs/superpowers/specs/2026-09-24-kuulo-part2-demo-node-design.md`.

---

## Under the hood (read this first)

- **FeatureTrace.** While a detection is active the node computes, every 20 ms, the energy in
  32 mel-spaced frequency bands (a "log-mel" frame: a coarse spectrogram column), plus overall
  loudness (RMS dB) and the strongest frequency. 32 bands at 50 frames/s is enough to see a
  drone's rotor harmonics and Doppler shift, but far too coarse to reconstruct speech — that is
  the privacy argument. Frames are grouped into 60 s segments and saved as compressed `.npz`
  (NumPy's zip-of-arrays format) with a signed JSON header.
- **Why sign a hash of the body.** Ed25519 signs the header JSON. Putting `body_sha256` inside the
  header means the one signature also covers the binary body: change a byte of the body and the
  hash no longer matches.
- **Server pull.** When fusion confirms a track (≥2 nodes agree), the server records a *trace
  request* for every detection that contributed. Nodes poll `GET /v1/traces/requests` and upload
  those segments. Confirmed events are exactly the data worth keeping to train better models.
- **Priority eviction.** The node has a disk budget (500 MB). When full it deletes the least
  valuable segments first: old uncorroborated ones, then any uncorroborated, and only last the
  corroborated ones it has already uploaded.
- **Outbox pattern (SQLite queue).** Messages go into a local SQLite table before they are sent,
  and are deleted only after the server accepts them. A crash, a reboot or a network outage
  loses nothing; order is preserved because rows are read by an autoincrement id.
- **Batch ingest.** Draining a backlog one HTTP request per message is slow; the server now accepts
  a JSON array of observations in one request and returns one result per item.
- **Step A vs Step B.** Step A uses YAMNet's own 521 AudioSet class scores (no training). Step B
  keeps YAMNet frozen but reads its internal 1024-number *embedding* (the layer just before the
  classifier) and trains a small head (logistic regression or a one-hidden-layer MLP) on drone vs
  not-drone. This is *transfer learning*: YAMNet already knows how sounds differ; we only learn
  the drone boundary. The head is exported to ONNX, a portable model format run by onnxruntime.
- **Leakage.** One flight is heard by 9 microphones at once. If mic 1 of a flight is in training
  and mic 2 of the same flight in test, the test score is inflated. Splits are therefore by
  *flight*, never by window or file. ESC-50 ships with 5 folds built the same way.
- **Augmentation.** Training positives are recorded close-ish and clean; real nodes hear drones far
  away behind traffic. On the fly we simulate distance (gain loss plus high-frequency roll-off),
  mix in background noise at −5…+20 dB SNR, add synthetic reverb, apply a slow ±3 % pitch drift
  (a Doppler pass) and random gain/shift. Only the training set is augmented; evaluation is clean.
- **C++.** Nothing in this milestone genuinely needs it: the per-frame DSP is a few vectorised NumPy
  calls at 50 frames/s, and inference runs in LiteRT/onnxruntime, which are already C++ inside.
  The natural C++ candidate is a Raspberry Pi node's audio front-end, which is a later milestone.

## Data sources (licences verified 2026-09-26)

| Dataset | Licence | Commercial | Use |
|---|---|---|---|
| DroneNoise Database, Univ. of Salford (figshare 22133411) | CC BY 4.0 | Yes | drone positives (outdoor overflights, 4 drones × 9 mics, 50 kHz) |
| ESC-50 (Piczak) | CC BY-NC 3.0 | **No** | negatives + hard-negative classes |
| DroneAudioDataset (Al-Emadi) | none stated | — | **rejected**: no licence means all rights reserved |
| DADS (Hugging Face) | claims MIT | — | **rejected**: repackages the unlicensed and NC sets above; card says to verify per source; no recording ids, so no leak-free split |

Consequence (documented in DATASETS.md and README): a head trained with ESC-50 is research-only;
it must be retrained on commercially licensed negatives before any commercial use.

## Global Constraints

- Python `>=3.12,<3.13`; every package is a uv workspace member; `make` targets use `uv run --no-sync`.
- No audio, model weights, datasets, keys or databases are ever committed (`data/` is gitignored;
  `make prepublish` must stay green).
- No YouTube or scraped audio. Every dataset's licence is recorded in `ml/DATASETS.md`.
- Raw audio never leaves the node; traces are 32-band log-mel only.
- FeatureTrace: `frame_period_ms` 20, 32 bands, 60 s segments, node disk budget default 500 MB,
  upload budget 50 MB/day for auto-push + sampling; server pull bypasses the budget.
- Auto-push: single-node detection with confidence ≥ 0.8 sustained ≥ 20 s. Hard-negative
  sampling: 1 % of uncorroborated detections, capped at 3 min per detection.
- Node outbox: SQLite, capped at 10 000 messages; heartbeats dropped before observations; drops logged and counted.
- Server binds 127.0.0.1 only; fusion exceptions never break ingest.
- CI never downloads models or datasets.

## Review Focus

1. A malformed trace upload (truncated zip, wrong array shapes, a pickled object array, body not matching `body_sha256`) must return 400, never 500, and must never unpickle.
2. A node restarted after an outage must still hold its unsent messages, in order, with heartbeats dropped first when full.
3. Disk budget reached while a detection is recording: eviction must never delete the segment being written or a segment the server has requested but not yet received.
4. A trace request for segments the node has already evicted must be closed (`unavailable`), not left open forever.
5. A server restart between confirmation and upload must keep the requests (they live in SQLite), and the node picks them up on its next poll.

---

## File structure

```
protocol/src/kuulo_protocol/traces.py        FeatureTraceHeader, TraceRequest, band layout constants
server/src/kuulo_server/traces.py            store/validate traces, create/close trace requests
server/src/kuulo_server/db.py                + TraceRow, TraceRequestRow, NodeRow.last_heartbeat_sent_at
server/src/kuulo_server/app.py               + /v1/traces, /v1/traces/requests, batch ingest, WS origin
node/src/kuulo_node/features.py              20 ms log-mel frame extractor (shared with sim)
node/src/kuulo_node/tracestore.py            segment recorder + disk store + priority eviction
node/src/kuulo_node/traceupload.py           server pull, auto-push, sampling, daily budget
node/src/kuulo_node/outbox.py                SQLite outbox
node/src/kuulo_node/classify.py              + YamnetEmbedder, HeadClassifier (Step B)
sim/src/kuulo_sim/traces.py                  synthetic band energies (level + Doppler)
sim/src/kuulo_sim/scenarios/long_event.yaml
ml/pyproject.toml, ml/src/kuulo_ml/{datasets,embed,augment,train,evaluate}.py, ml/tests/
dashboard/src/state/live.ts, components/TrackDetail.tsx
```

`features.py` lives in `kuulo_protocol` rather than `node` because the simulator must produce the
same band layout: `protocol/src/kuulo_protocol/features.py` (NumPy becomes a protocol dependency).

---

### Task 1: Server hardening and batch ingest

**Files:** Modify `server/src/kuulo_server/{app,ingest,db}.py`, `protocol/src/kuulo_protocol/models.py`;
Test `server/tests/test_ingest.py`, `server/tests/test_hardening.py` (new).

**Interfaces:**
- Produces: `POST /v1/observations` accepts an `Observation` or `list[Observation]` (1–100);
  list responses are `list[IngestResult | {"status":"rejected","reason":str}]` with HTTP 200.
- Produces: `NodeRegistration.public_key` validated as base64 of exactly 32 bytes.

- [ ] Step 1: Failing tests:
  - registration with `public_key="not-base64!"` or 31 bytes → 400.
  - replayed heartbeat: post HB(sent_at=T), then HB(sent_at=T-10s) validly signed → 409 `"stale heartbeat"`; node's `last_heartbeat_at` unchanged.
  - WebSocket with `Origin: http://evil.example` is closed with code 1008; `Origin: http://127.0.0.1:5173` and no Origin header are accepted.
  - batch of 3 observations (one with bad signature) → 200 with `[accepted, rejected(bad signature), accepted]`; the two good ones are stored and fused.
  - batch of 101 → 400; empty list → 400.
  - `run_tick` computes `uncorroborated_rate` only for nodes whose status changed (spy counts calls).
- [ ] Step 2: Run, confirm failures.
- [ ] Step 3: Implement: `field_validator("public_key")` decoding base64 and checking `len == 32`;
  `NodeRow.last_heartbeat_sent_at` column (added by an idempotent `ALTER TABLE` in
  `make_session_factory` for existing DBs); `ingest_heartbeat` raises `IngestError(409, ...)` when
  `hb.sent_at <= last_heartbeat_sent_at`; WS origin allow-list from `Settings.allowed_origins`
  (default the dashboard and server hosts on 127.0.0.1/localhost); observations endpoint reads the
  raw JSON body, validates via `TypeAdapter(Observation | list[Observation])`, and for lists runs
  each item through ingest, catching `IngestError`/`ValidationError` per item; `run_tick` builds
  views without the rate and fetches the rate only for a changed node.
- [ ] Step 4: `uv run --no-sync pytest server protocol` green.
- [ ] Step 5: Commit `fix(server): key validation, heartbeat replay, WS origin, batch ingest, cheaper tick`.

### Task 2: Dashboard live-connection races

**Files:** `dashboard/src/state/live.ts`, `dashboard/src/state/live.test.ts`.

- [ ] Step 1: Failing vitest cases: (a) a message arriving on a superseded socket after reconnect is
  ignored; (b) `onclose` of a superseded socket does not schedule a second reconnect; (c) a
  snapshot fetch failure keeps the backoff growing (attempt not reset until a snapshot succeeds);
  (d) `stop()` during a pending snapshot prevents dispatch.
- [ ] Step 2: Implement guards `if (socket !== ws) return;` in `onmessage`/`onclose`, reset
  `attempt = 0` only after a successful snapshot.
- [ ] Step 3: `npx vitest run` green, `npm run build` green. Commit.

### Task 3: FeatureTrace protocol and shared feature extractor

**Files:** Create `protocol/src/kuulo_protocol/traces.py`, `protocol/src/kuulo_protocol/features.py`;
modify `protocol/pyproject.toml` (+numpy), `schema.py`; Test `protocol/tests/test_traces.py`.

**Interfaces (Produces):**
```python
N_BANDS = 32; FRAME_PERIOD_MS = 20; FRAME_SAMPLES = 320; SAMPLE_RATE = 16_000
BAND_EDGES_HZ: tuple[float, ...]            # 33 mel-spaced edges, 50 Hz .. 8000 Hz
class FeatureTraceHeader(WireModel):
    schema_version: Literal["1.0"]; trace_id: UUID; node_id: str; detection_id: UUID
    segment_index: int (>=0); final: bool; time_quality: TimeQuality
    frame_period_ms: Literal[20]; start_at: datetime; frame_count: int (1..3000)
    band_edges_hz: list[float] (len 33); body_sha256: str (64 hex); signature: str = ""
class TraceRequest(WireModel): request_id: UUID; node_id: str; detection_id: UUID; created_at: datetime
class TraceUnavailable(WireModel): node_id: str; detection_id: UUID; sent_at: datetime; signature: str = ""
def extract_frames(samples: np.ndarray) -> FrameBlock   # float32 16 kHz -> per 20 ms frame
@dataclass class FrameBlock: band_db: np.ndarray[F,32] float32; rms_db: [F]; peak_freq_hz: [F]
def encode_body(t_offset_ms, band_db, rms_db, peak_freq_hz) -> bytes   # np.savez_compressed
def decode_body(data: bytes, frame_count: int) -> dict[str, np.ndarray]  # allow_pickle=False, shape checks, raises ValueError
```
- [ ] Step 1: Tests: band edges strictly increasing, 33 values, 50…8000; a 1 kHz tone puts its
  maximum energy in the band containing 1 kHz and `peak_freq_hz` within ±50 Hz of 1000; 1 s of
  audio → 50 frames; `encode_body`/`decode_body` round-trip; `decode_body` raises `ValueError` on
  truncated bytes, wrong shapes, object arrays and frame_count mismatch; header sign/verify.
- [ ] Step 2: Implement (Hann-windowed rfft per 320-sample frame, triangular mel filterbank,
  `10*log10(energy+1e-10)`), export new models in `schema.py`, regenerate dashboard types
  (`make types`).
- [ ] Step 3: Tests green; commit.

### Task 4: Server trace storage and requests

**Files:** Create `server/src/kuulo_server/traces.py`; modify `db.py`, `app.py`, `config.py`
(`traces_dir`, `max_trace_bytes = 4 MB`), `protocol/api.py` (`TrackDetail.trace_segments: int`);
Test `server/tests/test_traces.py`.

**Interfaces (Produces):**
- `POST /v1/traces` multipart: field `header` (JSON FeatureTraceHeader), file `body` (.npz).
  400 invalid/hash mismatch/bad body/too large; 401 unknown node/bad signature; 200 `{"status":"stored"|"duplicate"}`.
- `GET /v1/traces/requests?node_id=` → `list[TraceRequest]` (open only).
- `POST /v1/traces/unavailable` signed `TraceUnavailable` → closes the request with reason `unavailable`.
- `create_requests_for_track(session, track, now)` called from `apply_updates` when status is CONFIRMED;
  one request per `(node_id, detection_id)`, idempotent. A request closes when a segment with `final=True` arrives.
- [ ] Step 1: Tests: confirmed track (two nodes) → two open requests; tentative → none; upload
  segments 0 and 1(final) → request closed, files at `traces_dir/<node>/<trace_id>.npz`, rows
  present; duplicate trace_id → duplicate; tampered body → 400; pickled object array → 400 and no
  unpickling; oversize → 400; bad signature → 401; unavailable closes request; requests survive
  `create_app` restart on the same DB; `GET /v1/tracks/{id}` reports `trace_segments`.
- [ ] Step 2: Implement; commit `feat(server): feature trace upload, trace requests on confirmation`.

### Task 5: Node SQLite outbox

**Files:** Create `node/src/kuulo_node/outbox.py`; modify `uplink.py`, `config.py`
(`state_dir`, default `<config dir>/.state/<node_id>/`), `cli.py`; Test `node/tests/test_outbox.py`, update `test_uplink.py`.

**Interfaces (Produces):**
```python
class Outbox:
    def __init__(self, path: Path, cap: int = 10_000): ...
    def put(self, path: str, body: str) -> None       # drops oldest heartbeat first, else oldest message; counts drops
    def peek(self, limit: int) -> list[tuple[int, str, str]]   # (row_id, path, body) in order
    def delete(self, ids: list[int]) -> None
    def __len__(self) -> int; dropped: int
```
`Uplink(client, outbox=Outbox(...))` keeps its public API (`send`, `flush`, `ensure_registered`,
`pending_count`, `dropped`). `flush` sends consecutive observations as one batch (≤100) and
heartbeats singly; per-item rejections are dropped and counted; 5xx/transport errors back off
without deleting.
- [ ] Step 1: Tests: order preserved across reopen; cap drops heartbeat before observation;
  batch posted as JSON array; per-item rejection deleted and counted; transport error keeps rows.
- [ ] Step 2: Implement (sqlite3, WAL, `CREATE TABLE outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, body TEXT)`); in-memory `:memory:` default for tests. Commit.

### Task 6: Node trace recorder, store and uploader

**Files:** Create `node/src/kuulo_node/tracestore.py`, `node/src/kuulo_node/traceupload.py`;
modify `runner.py`, `detector.py` (expose active detection id and confidence), `config.py`
(`[traces]` table: `budget_mb=500`, `upload_mb_per_day=50`, `sample_rate=0.01`), `cli.py`;
Tests `node/tests/test_tracestore.py`, `node/tests/test_traceupload.py`.

**Interfaces (Produces):**
```python
class TraceStore:
    def __init__(self, root: Path, node_id: str, private_key: str, time_quality, *, budget_bytes, now): ...
    def feed(self, samples_16k: np.ndarray, t_audio: float, detection_id: UUID | None) -> None
        # keeps a 3 s pre-roll; while detection_id is set, appends frames; closes a segment every 60 s
        # and on detection end (final=True); writes <root>/<detection>/<segment>.npz + .json
    def segments(self, detection_id) -> list[SegmentInfo]
    def mark(self, detection_id, *, corroborated=None, uploaded_segment=None) -> None
    def evict_to_budget(self) -> int   # order: uncorroborated >7 d, uncorroborated oldest, corroborated+uploaded oldest; never the open segment or requested-not-uploaded
class TraceUploader:
    def __init__(self, client, store, keys, *, now, rng, upload_bytes_per_day, sample_rate): ...
    def on_detection_end(self, detection_id, max_conf_sustained_s: float, sustained_conf: float) -> None
        # auto-push if conf>=0.8 sustained>=20 s; else with probability sample_rate queue first 3 min
    def poll(self) -> None   # GET requests, mark corroborated, upload all segments (bypass budget) or POST unavailable
    def pump(self) -> None   # upload queued pushes/samples within the daily budget
```
- [ ] Step 1: Tests (synthetic audio, fake clock, httpx MockTransport): 130 s detection → 3 segments,
  last `final`; pre-roll included; eviction order and protections; server request → all segments
  uploaded with valid signatures and request closed; evicted → unavailable posted; auto-push rule;
  sampling with seeded rng honours 3-minute cap; budget blocks the 51st MB but not a pull.
- [ ] Step 2: Implement and wire into `NodeRunner` (store fed per audio block; uploader polled every 10 s). Commit.

### Task 7: Simulator traces and `long_event`

**Files:** Create `sim/src/kuulo_sim/traces.py`, `sim/src/kuulo_sim/scenarios/long_event.yaml`;
modify `engine.py` (record per-detection received level/Doppler per tick), `runner.py` (poll
requests and upload), `sim/tests/test_traces.py`, `server/tests/test_scenarios.py`.

- `long_event`: 3 nodes 400 m apart, one drone loitering on a 150 m circle for 600 s.
- Synthetic band energies: rotor harmonic comb (fundamental 180 Hz × Doppler factor
  `c/(c - v_radial)`) at the received level, over a pink-ish noise floor at the node's
  `noise_floor_db`; deterministic from the scenario seed.
- [ ] Step 1: Tests: synthetic trace peak band shifts up while approaching and down while receding;
  `long_event` run through the in-process server confirms a track and stores ≥ 10 segments for a
  single detection; `helsinki_pass` stores traces only for confirmed detections.
- [ ] Step 2: Implement; commit.

### Task 8: ML package — datasets, embeddings, augmentation

**Files:** Create `ml/pyproject.toml` (workspace member `kuulo-ml`: numpy, scipy, soundfile,
scikit-learn, skl2onnx, onnxruntime, ai-edge-litert, kuulo-node), `ml/src/kuulo_ml/{__init__,datasets,embed,augment}.py`,
`ml/tests/test_{datasets,augment,embed}.py`; root `pyproject.toml` workspace + testpaths; `Makefile`
(`datasets`, `embed`, `train`, `evaluate`).

- `datasets.py`: `download_dronenoise(root)` via figshare API (skip `Calib_*`, `.xlsx`), resumable,
  sha-free but size-checked; `download_esc50(root)` from the GitHub archive zip; `build_manifest(root) -> list[Clip]`
  with `Clip(path, label: 0|1, group: str, kind: str, split: "train"|"val"|"test")`.
  DroneNoise group = file stem without `_M<n>` (one flight); per drone type the flights are sorted
  and every 4th goes to test, every 4th+1 to val. ESC-50: fold 5 test, fold 4 val. `kind` is the
  ESC-50 category or `drone:<type>`. Hard negatives: chainsaw, engine, airplane, helicopter,
  hand_saw, vacuum_cleaner, insects.
- `augment.py`: `augment(x, rng, backgrounds) -> x'` with distance (gain −0…−30 dB + 1st-order
  low-pass 2–8 kHz), background mix at SNR ∈ [−5, 20] dB, synthetic exponential-decay reverb
  (RT60 0.2–1.0 s) with p = 0.5, time-varying pitch ±3 %, random gain ±6 dB and circular shift.
- `embed.py`: `YamnetEmbedder` (LiteRT with preserved tensors, reads tensor
  `tower0/network/layer28/reduce_mean` → 1024-d, and the 521 scores) and `embed_clips(...)` writing
  `data/ml/features_{split}.npz` (embeddings, Step A drone score using node default weights,
  label, group, kind, clip id). DroneNoise windows are kept only when their RMS is within 20 dB of
  the recording's loudest window (drops windows where the far drone is inaudible). Training split
  gets 2 augmented copies per positive window, 1 per negative.
- [ ] Step 1: Tests with synthetic data only: manifest split has no group in two splits; augment
  preserves length/finite/bounded, SNR mixing hits the requested SNR ±0.5 dB; embedder skipped if
  model absent. Implement; commit.

### Task 9: Train, export ONNX, evaluate, node Step B

**Files:** Create `ml/src/kuulo_ml/{train,evaluate}.py`, `ml/tests/test_train.py`; modify
`node/src/kuulo_node/classify.py` (+`HeadClassifier`), `config.py` (`classifier = "yamnet"|"head"`,
`head_path`), `cli.py`, `node/pyproject.toml` (+onnxruntime), `node/config.example.toml`.

- `train.py`: standardise → fit LogisticRegression (C grid) and MLPClassifier(128, early stopping);
  choose by validation average precision; pick the threshold maximising F1 on validation; export
  `StandardScaler+model` pipeline with skl2onnx to `data/models/drone_head.onnx` and
  `drone_head.json` (threshold, metrics, training date, dataset licences).
- `evaluate.py`: on the clean test split, for Step A (node weights, threshold 0.5) and Step B (its
  threshold): precision, recall, F1, confusion matrix, false-positive rate per hard-negative kind;
  writes `ml/RESULTS.md`.
- `HeadClassifier.score(window) -> {"drone": p, **top YAMNet scores}`; one LiteRT invoke gives both.
  With `classifier = "head"` the weights default to `{"drone": 1.0}`.
- [ ] Step 1: Test on synthetic separable embeddings: training yields AP > 0.95, ONNX output
  matches sklearn `predict_proba` to 1e-5, `HeadClassifier` with a fake embedder uses it.
- [ ] Step 2: Run `make datasets embed train evaluate` for real; commit code and `ml/RESULTS.md`.

### Task 10: Real-audio end-to-end, README, demo GIF, merge

- [ ] Replay a held-out DroneNoise test flight through `kuulo-node --input` (Step B) against a live
  server; measure the audio time of the first `start` observation (DoD: < 3 s after the drone is
  audible) and confirm a TENTATIVE track via REST. Replay ESC-50 hard negatives; record outcomes.
- [ ] README: real metrics table (Step A vs Step B), dataset licences, traces, outbox, updated known
  limits; DATASETS.md complete; PROGRESS.md updated.
- [ ] Demo GIF: `make dev` + `make sim SCENARIO=helsinki_pass SPEED=4` recorded in Chrome to `docs/demo.gif`; restore the README image line.
- [ ] `make test`, `make prepublish`, `npm run build` all green; CI workflow also runs `ml/tests`.
- [ ] Merge `dashboard-ops-redesign` into local `main` (fast-forward). Pushing is left to the user
  (the project's settings deny `git push`).

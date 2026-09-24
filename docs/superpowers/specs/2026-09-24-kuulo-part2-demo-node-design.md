# Kuulo Part 2: Demo Node Design Spec

**Date:** 2026-09-24
**Status:** Approved in conversation, awaiting written-spec review
**Parent spec:** `2026-09-24-kuulo-milestone1-design.md` (sections 5 and 10 are the full design;
this document is the slice being built now)

## 1. Purpose

A public portfolio demo for the Junction (Espoo) hackathon application, to be finished
today. The repo is published briefly, then made private again.

**Success:** play or fly a hobby drone near the laptop, and a track appears on the
dashboard within seconds. Everyday sounds (speech, typing, traffic, a fan) do not
trigger it. A README with a demo GIF shows this.

**Consequences of publishing:** making the repo private again does not un-publish
it. Clones, forks and archives persist. Only content that is safe to be public forever
is committed (section 7).

## 2. Scope

**In:**
- `node/` package: laptop microphone or wav-file input, YAMNet zero-training
  classifier (the parent spec's "Step A"), smoothing, signed observations and heartbeats, HTTP uplink.
- A committed demo config located at the University of Helsinki city centre campus.
- Full README, AGPL-3.0 LICENSE, GitHub Actions CI, a pre-publish checklist.

**Deferred:** Step B trained head and evaluation, FeatureTraces (parent section 5.1),
SQLite offline queue, Raspberry Pi deployment and multi-node hardware, Shahed-class
training data.

## 3. Node pipeline

```
AudioSource (mic | wav) -> 16 kHz mono -> Windower (0.96 s, 50 % hop)
  -> Classifier.score(window) -> drone score -> DetectionSmoother -> phase?
  -> Observation (signed) -> Uplink (HTTP, retry)      Heartbeat every 60 s
```

- **AudioSource.** `MicSource` uses `sounddevice` at 16 kHz mono. `WavSource`
  reads a file, resamples to 16 kHz mono, and replays at real-time speed, or
  accelerated with `--speed`. Both yield timestamped float32 blocks.
- **Windower.** A ring buffer that emits 15 600-sample windows every 7 800 samples,
  each with its start time.
- **Classifier interface.** `score(window: np.ndarray) -> dict[str, float]`, so that Step B
  can be dropped in later.
  - `YamnetClassifier` loads the YAMNet `.tflite` model with `ai-edge-litert`, not full
    TensorFlow; `tflite-runtime` is abandoned. It returns the 521 class scores. ONNX via
    `onnxruntime` is the fallback if LiteRT does not run on Python 3.12 or macOS arm64.
  - `FakeClassifier` is scripted, and is used in tests and CI.
- **Drone score.** A weighted maximum over AudioSet classes, with "Propeller, airscrew"
  weighted highest, then "Helicopter", then "Aircraft"/"Aircraft engine". Buzz, insect
  and electric-shaver classes are excluded, because they cause false positives. Weights
  live in config and are tuned by hand with `--print-scores`. The label is
  `drone_multirotor`.
- **Smoothing.** Reuses `kuulo_protocol.smoothing.DetectionSmoother`: start when
  3 of the last 5 windows are ≥ 0.5, end after 5 s below threshold, and send an update every 5 s.
  **Reset rule:** when the audio stream restarts or has a gap longer than one hop,
  any open detection is ended (an `end` is emitted) and a new smoother is created.
  Windows from before a dropout never count toward a new detection.
- **Observation.** `source.type=acoustic_node`, the config location, `time_quality=ntp`,
  and an `acoustic` block with a rough SNR (window RMS against a running noise floor)
  and the peak frequency.
- **Signing and registration.** On first run the node generates an Ed25519 keypair
  and stores it in a gitignored key file next to its config. It registers via
  `POST /v1/nodes/register`, and signs every message with `kuulo_protocol.signing`.
- **Uplink.** `httpx` POSTs with bounded exponential backoff and an in-memory
  retry list capped at 1 000 messages. Drops are logged and counted, never silent.
- **Heartbeat.** Sent every 60 s with `mic_ok` and `queue_depth`.
- **Mic health.** If the mic stream fails, or delivers only digital silence for 10 s,
  the node logs an actionable message (macOS: System Settings → Privacy →
  Microphone), sets `mic_ok=false`, and retries opening the mic every 10 s.
- **Privacy.** Audio exists only in memory. Nothing is written to disk.

## 4. Configuration and model file

- `node/config.example.toml` (committed): node id `demo-laptop`, location
  60.1694, 24.9490 (University of Helsinki main building, city centre campus),
  accuracy 50 m, server `http://127.0.0.1:8000`, thresholds, class weights.
- Real configs (`node/*.local.toml`) and key files are gitignored.
- Only the configured location is ever reported: no GPS, no IP geolocation. Traffic
  goes to localhost, so no IP address appears in any data.
- `node/scripts/download_model.py` fetches YAMNet `.tflite` (Apache-2.0) and the
  class map into `data/models/` (gitignored). It is never committed. Its source and
  licence are recorded in `ml/DATASETS.md`, a new file.

## 5. CLI and demo

- `kuulo-node run [--config PATH] [--input clip.wav] [--speed X] [--print-scores]`
- `make node` (live mic) and `make node INPUT=clip.wav`
- One laptop is one node, so basic fusion shows the drone as **TENTATIVE**. This is
  intended, and the README explains it. Multi-node confirmation is shown with the
  simulator.

## 6. Testing

- Unit tests: windowing (sizes, hop, timestamps), drone-score mapping, reset on
  gap, observation building and signing, uplink retry and cap. All use `FakeClassifier`
  and synthetic audio, so no model is needed.
- End to end: replay a synthetic wav through `WavSource` with a scripted
  `FakeClassifier` into the FastAPI test client, and assert that a track is created.
- A manual check with the real model: `--print-scores` with a licensed drone clip,
  speech and a fan. The observed behaviour is recorded in the README.
- CI (GitHub Actions): ruff, pytest, vitest, dashboard build. CI never downloads the model.

## 7. Publishing

- README: what Kuulo is, an architecture diagram, quick start (including the macOS mic
  permission and Linux `libportaudio2`), how detection works, known limits
  (single-node demo is tentative only; Step A also fires on propeller aircraft and
  some engines; location is coarse), the security note, and a GIF slot.
- `LICENSE`: AGPL-3.0.
- Pre-publish checklist in the README: no real coordinates, no audio, no model
  weights, no keys, `data/` ignored. Verified with a `git ls-files` and grep pass before the repo goes public.

## 8. Risks

- Runtime compatibility of `ai-edge-litert` on Python 3.12 or macOS arm64. The first
  plan task is a spike; ONNX is the fallback.
- The model download host (`storage.googleapis.com`) may need adding to the sandbox
  network allowlist. The user approves the download (file, source, size) first.
- Step A accuracy on real drones through a laptop mic is unknown until tuned.

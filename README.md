# Kuulo

Civic acoustic drone detection for Finnish cities: cheap microphones that hear hobby drones,
and a fusion layer that turns many noisy detections into tracks you can trust.

<!-- ![demo](docs/demo.gif) -- add this back once docs/demo.gif is recorded (node + dashboard
     running together); see the Quick start section below to record it. -->

## Why

Long- and medium-range radars lose small, low, slow drones. Nordic–Baltic governments are
covering their *borders* (Latvia's acoustic border network, Estonia's "drone wall", Finnish
Border Guard trials), but *interiors* — cities, ports, energy sites — are thinly covered, and
false alarms are costly (Vilnius airport closed for a flock of birds, Sept 2026). No civilian
sensing network exists in Finland; Finland's planned 112 Suomi drone alerts (by 2027) are
one-way.

## How it works

```text
mic / wav ─► kuulo-node ─► POST /v1/observations (Ed25519-signed, batched) ─► server ─► fusion ─► dashboard
             YAMNet embedding ─► trained head ─► 3-of-5 smoothing            SQLite    tracks    MapLibre + Blueprint
             SQLite outbox · 32-band feature traces ◄── trace requests ◄── confirmed track
```

**Node** (`node/`) listens to a microphone or replays a wav file. Each 0.975 s window goes once
through YAMNet (a pretrained audio model); a small trained head reads YAMNet's internal
embedding and outputs a drone probability (Step B). A 3-of-5 rule turns window scores into
start/update/end detections, which are signed and delivered through a durable SQLite outbox,
so an outage or reboot loses nothing. While a detection is active the node also records a
*feature trace*: 32 coarse frequency bands every 20 ms, enough to see rotor harmonics and
Doppler, far too coarse to reconstruct speech.

**Server** (`server/`) verifies every signature, stores observations, heartbeats and traces in
SQLite, and streams live updates to the dashboard over a WebSocket.

**Fusion** (`server/`, basic implementation) turns observations into tracks. A single sensor
can only ever produce a **TENTATIVE** track; agreement between multiple sensors produces
**CONFIRMED**; nearby sensors that heard nothing lower a track's confidence as silent
neighbours. When a track confirms, the server asks every contributing node for that
detection's feature traces: confirmed events are the data worth keeping.

**Dashboard** (`dashboard/`) is a dark, dense, Blueprint-based ops console: a live map, a
sensor and track list, per-track evidence (including how many trace segments arrived), and an
event log.

**ML** (`ml/`) downloads licensed data, extracts YAMNet embeddings, trains and exports the Step
B head to ONNX, and evaluates Step A against Step B on held-out recordings.

**Impulse events.** Alongside the drone classifier, each node runs a second, independent
detector (an STA/LTA trigger plus an AIC onset picker, both borrowed from seismology) for sharp
acoustic impulses — a detonation, a hard crash — and reports only an onset time, its
uncertainty and coarse summary features; no waveform ever leaves the node. The server groups reports from multiple nodes into events and resolves a location with a
built-in coarse locator — a circle around the centroid of every node that heard it, sized to
their spread — then exports each event as a CAP 1.2 alert (`GET
/v1/impulse-events/{id}/cap`, `status="Test"` by default) for emergency-alerting systems. This
baseline always reports an unattributed **IMPULSE** (orange). Precise multilateration from
onset-time differences, and attributing an event to the drone track that caused it (shown as a
red **IMPACT**), can come from an optional, more capable locator plugged in through the
`KUULO_IMPULSE_LOCATOR` environment variable (the same pattern `KUULO_FUSION_ENGINE` uses for
the drone tracker) — this repository ships only the coarse baseline.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Node.js 20+.

```bash
make setup
(cd dashboard && npm install)
make model     # downloads the ~4 MB YAMNet model into data/models/; model files are never committed
make ml        # optional but recommended: ~1.3 GB of licensed data, then a few minutes of CPU
               # to train Step B (see "How well it works"); without it the node uses Step A
make dev       # server on 127.0.0.1:8000, dashboard on http://127.0.0.1:5173
```

In a second terminal:

```bash
make node                        # live microphone
make node INPUT=path/to/clip.wav # replay a wav file instead
```

Platform notes:

- **macOS:** the first run asks for microphone access for your terminal app; if you denied it,
  enable it in System Settings → Privacy & Security → Microphone.
- **Linux:** `sudo apt-get install libportaudio2`.

`make sim` plays a scripted multi-node scenario instead of a real microphone, useful for seeing
a CONFIRMED track without hardware. `make sim SCENARIO=long_event` loiters a drone for ten
minutes, so its feature traces span many 60 s segments. `make sim SCENARIO=impact_strike` plays
a drone track that ends in an acoustic impulse, so an IMPACT (or IMPULSE, with only the coarse
locator) appears on the map; `make sim SCENARIO=firework` plays a standalone bang with no drone
nearby, for the unattributed case.

## How well it works

Measured on held-out data no model saw during training or validation: DroneNoise flights
(outdoor overflights of four small drones, each heard by up to nine microphones) against ESC-50
negatives. Full tables, per-sound false-positive rates and method notes are in
[`ml/RESULTS.md`](ml/RESULTS.md); `make ml` reproduces them.

Per 0.975 s window (5207 windows, 1861 drone):

| Classifier | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|
| Step A: YAMNet class scores, zero training | 0.048 | 0.001 | 0.001 | 0.006 |
| **Step B: trained head on YAMNet embeddings** | **0.931** | **0.902** | **0.916** | **0.037** |

Per recording, replayed through the node's own smoothing (what a node would actually report):

| Classifier | Drone recordings detected | Median / worst time to first detection | False detections: drone-like sounds / other |
|---|---|---|---|
| Step A | 0 / 46 | — | 2 / 56, 0 / 344 |
| **Step B** | **46 / 46** | **1.95 s / 3.41 s** | **1 / 56, 7 / 344** |

Step A fails because YAMNet hears distant drones in quiet field recordings as "Silence". That
is why Step B exists. Step B's worst drone-like-sound false-positive rate per window is 5.6 %
(insects); airplanes, helicopters and engines are at 0 %. Scaling every negative to the drones'
loudness moves Step B's false-positive rate only from 1.2 % to 2.6 % on drone-like sounds,
so the head is not simply keying on loudness.

End to end, replaying a held-out DroneNoise flight in real time through `kuulo-node` against a
live server gave the first detection 2.1 s after playback started and a TENTATIVE track on the
server; 12 held-out chainsaw and helicopter clips gave no detection.

**Impulse detector**, evaluated against ESC-50 fireworks/glass-breaking/door-knock clips (as a
proxy — no licensed blast recordings exist) mixed into ESC-50 continuous sounds and DroneNoise
recordings at controlled SNR. Full tables in [`ml/IMPULSE_RESULTS.md`](ml/IMPULSE_RESULTS.md);
`make impulse-eval` reproduces them.

| SNR | Detected | Median onset error | Within reported 2σ | Gross mis-picks |
|---|---|---|---|---|
| 30 dB | 82.3 % | 1.50 ms | 91.1 % | 13.8 % |
| 20 dB | 74.7 % | 2.94 ms | 95.7 % | 22.0 % |
| 10 dB | 26.6 % | 4.62 ms | 100.0 % | 33.3 % |

Clean detection on the proxy clips is 65.8 %, with 21.18 false triggers per hour on continuous
sounds and drone recordings — the detector's STA/LTA trigger and rise/decay/bandwidth gates are
tuned to reject anything that isn't a genuinely sharp, decaying, broadband event, and often do.
The node's own reported uncertainty (`onset_sigma_s`) comes from how sharply its AIC picker pins
the onset — a genuinely per-event, SNR-dependent confidence, not a fixed constant — calibrated
so that at 20 dB SNR and above, at least 90 % of *non-gross* onset errors fall within twice that
uncertainty. "Gross mis-picks" (>20 ms, usually a different crackle or knock elsewhere in a
multi-event proxy clip) are a detection-correctness failure, not a timing-precision one, so a
downstream locator combining several nodes handles them with its own outlier rejection rather
than a wider sigma; see [`ml/IMPULSE_RESULTS.md`](ml/IMPULSE_RESULTS.md)'s Calibration section.

## Known limits

- **One recording campaign.** Every drone positive comes from one outdoor campaign (Edzell,
  Scotland, 2022) recorded with measurement microphones. Performance through a laptop microphone
  in a city is not measured, and a recording-site bias cannot be ruled out.
- **Training licence.** Step B's negatives are ESC-50 (CC BY-NC): the trained head is research
  only and must be retrained on commercially licensed negatives before commercial use. See
  [`ml/DATASETS.md`](ml/DATASETS.md).
- A single-laptop demo can only ever produce a TENTATIVE track; CONFIRMED needs agreement
  between multiple sensors (see `make sim`).
- Location is the node's configured position, reported at whatever accuracy its config states
  — hundreds of metres is typical, not a precise fix.
- The silent-neighbour rule uses free-field hearing ranges; buildings and wind make a silent
  neighbour weaker evidence than it looks, so its penalty is deliberately gentle and configurable.
- Separating individual drones in a dense swarm is not attempted; feature traces keep the data
  so it can be later.
- **The public impulse locator is coarse.** Without a `KUULO_IMPULSE_LOCATOR` plugged in, every
  event is a wide circle around the nodes that heard it, and never attributed to a drone track.
- **Impulse onset timing rides on the node's own clock**, typically a laptop's; it is not a
  GPS-disciplined clock, so cross-node onset-time differences carry whatever jitter that
  introduces, on top of the detector's own calibrated uncertainty.
- **No licensed blast recordings.** The impulse detector's real-audio evaluation uses ESC-50
  fireworks, glass-breaking and door-knock clips as the closest available proxy for a blast's
  sharp, broadband signature — see [`ml/IMPULSE_RESULTS.md`](ml/IMPULSE_RESULTS.md).
- CAP alerts are exported with `status="Test"`, not `"Actual"` — deliberately, until a real
  deployment reviews that switch.
- Localhost only, no authentication. Do not expose the server to the internet.

## Roadmap

- **Impulse events: done.** Nodes report sharp acoustic impulses (a detonation, a hard crash)
  with precise timestamps and calibrated uncertainty. The server groups them into events on the
  map and exports them as CAP 1.2 alerts for emergency services and counter-UAS operators.
  Precise location and attribution to the drone that caused the event come from an optional
  private fusion engine, through the `KUULO_IMPULSE_LOCATOR` plug-in; this repository ships the
  coarse public baseline. Design:
  [`docs/superpowers/specs/2026-09-26-kuulo-impulse-events-design.md`](docs/superpowers/specs/2026-09-26-kuulo-impulse-events-design.md);
  plan: [`docs/superpowers/plans/2026-09-26-kuulo-impulse-events.md`](docs/superpowers/plans/2026-09-26-kuulo-impulse-events.md).

## Privacy

Raw audio exists only in the node's memory and is never written to disk or uploaded, unless you
explicitly pass `--debug-save-clips DIR`, which writes each detection's audio to that local
folder and nowhere else. Feature traces are 32-band spectra, not audio. The node reports only
its configured location — the demo config points at the University of Helsinki main building,
not a real deployment. No model weights, audio, datasets or private keys are committed to this
repository, in the current tree or anywhere in its history; `make prepublish` checks both
before every publish.

## Development

```bash
make test        # ruff, pytest, vitest
make ml          # datasets -> embed -> train -> evaluate (writes ml/RESULTS.md)
make prepublish  # fails if audio, models, keys, databases or local configs are tracked
                  # (current tree and full git history); also prints every tracked
                  # coordinate for you to read -- that part is not an automated check
```

Before making this repository public, in order: record the demo GIF, run `make prepublish`
and read its coordinate list yourself, push, then change the repository's visibility.

Repository layout: `protocol/` (shared message types, signing, feature-trace format),
`server/` (ingest, storage, fusion, trace requests), `sim/` (multi-node scenario player with
synthetic traces), `node/` (microphone/wav listener, outbox, trace store), `dashboard/` (the ops
console), `ml/` (datasets, training, evaluation). Design specs and implementation plans live in
`docs/superpowers/`.

## Licence

AGPL-3.0. See LICENSE. Models and datasets: see ml/DATASETS.md. Drone recordings: DroneNoise
Database, University of Salford, CC BY 4.0. Negatives: ESC-50, K. J. Piczak, CC BY-NC 3.0.

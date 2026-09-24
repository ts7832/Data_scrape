# Kuulo

Civic acoustic drone detection for Finnish cities: cheap microphones that hear hobby drones,
and a fusion layer that turns many noisy detections into tracks you can trust.

<!-- record with the node + dashboard running; see "Demo" below -->
![demo](docs/demo.gif)

## Why

Long- and medium-range radars lose small, low, slow drones. Nordic–Baltic governments are
covering their *borders* (Latvia's acoustic border network, Estonia's "drone wall", Finnish
Border Guard trials), but *interiors* — cities, ports, energy sites — are thinly covered, and
false alarms are costly (Vilnius airport closed for a flock of birds, Sept 2026). No civilian
sensing network exists in Finland; Finland's planned 112 Suomi drone alerts (by 2027) are
one-way.

## How it works

```text
laptop mic / wav ─► kuulo-node ─► POST /v1/observations (Ed25519-signed) ─► server ─► fusion ─► dashboard
                   YAMNet (AudioSet) ─► drone score ─► 3-of-5 smoothing        SQLite   tracks   MapLibre + Blueprint
```

**Node** (`node/`) listens to a microphone or replays a wav file, scores each window with a
pretrained sound classifier (YAMNet, zero training), smooths the raw scores into
start/update/end detections, and signs and uploads them as observations.

**Server** (`server/`) ingests observations and heartbeats over HTTP, stores them, and streams
live updates to the dashboard over a WebSocket.

**Fusion** (`server/`, basic implementation) turns observations into tracks. A single sensor
can only ever produce a **TENTATIVE** track; agreement between multiple sensors produces
**CONFIRMED**; nearby sensors that heard nothing lower a track's confidence as silent
neighbours.

**Dashboard** (`dashboard/`) is a dark, dense, Blueprint-based ops console: a live map, a
sensor and track list, per-track evidence, and an event log.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Node.js 20+.

```bash
make setup
(cd dashboard && npm install)
make model     # downloads the ~4 MB YAMNet model into data/models/; model files are never committed
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
a CONFIRMED track without hardware.

## How well it works

Step A (YAMNet, zero training) has not yet been tuned against a real microphone and real drone
audio — that trial is the next step before this demo is recorded, and the class weights below
are the untuned defaults from `node/config.example.toml`.

| Class | Weight |
| --- | --- |
| Propeller, airscrew | 1.0 |
| Helicopter | 0.8 |
| Aircraft | 0.6 |
| Aircraft engine | 0.6 |

Step A is zero-training YAMNet. It also responds to propeller aircraft and some engines; a
trained head (Step B) is the planned fix.

## Known limits

- A single-laptop demo can only ever produce a TENTATIVE track; CONFIRMED needs agreement
  between multiple sensors (see `make sim`).
- Location is the node's configured position, reported at whatever accuracy its config states
  — hundreds of metres is typical, not a precise fix.
- Step A (YAMNet, zero training) also fires on propeller aircraft and some other engine
  sounds; see "How well it works" above.
- Localhost only, no authentication. Do not expose the server to the internet.

## Privacy

Raw audio exists only in the node's memory and is never written to disk or uploaded. The node
reports only its configured location — the demo config points at the University of Helsinki
main building, not a real deployment. No model weights, audio, or private keys are committed
to this repository; `make prepublish` checks that before every publish.

## Development

```bash
make test        # ruff, pytest, vitest
make prepublish  # verifies no audio, models, keys, databases or local configs are tracked
```

Repository layout: `protocol/` (shared message types and signing), `server/` (ingest, storage,
fusion), `sim/` (multi-node scenario player), `node/` (microphone/wav listener), `dashboard/`
(the ops console), `ml/` (model and dataset provenance). Design specs and implementation plans
live in `docs/superpowers/`.

## Licence

AGPL-3.0. See LICENSE. Models and datasets: see ml/DATASETS.md.

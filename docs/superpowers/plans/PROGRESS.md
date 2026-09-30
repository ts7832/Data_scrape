# Kuulo progress

## Part 4 (2026-09-29): Impulse events

Plan: `docs/superpowers/plans/2026-09-26-kuulo-impulse-events.md`. Spec:
`docs/superpowers/specs/2026-09-26-kuulo-impulse-events-design.md` (status: Implemented).

- **Wire protocol and node detector:** `kuulo_protocol.impulses` (`ImpulseReport`,
  `ImpulseEvent`, `Ellipse`, `ImpulseKind`, `LocationQuality`, `speed_of_sound`); node-side
  `ImpulseDetector` — an STA/LTA trigger plus a Maeda-1985 AIC onset picker, reporting only an
  onset time, its calibrated uncertainty and coarse summary features. No waveform leaves the
  node.
- **Server:** ingest and grouping (`server/src/kuulo_server/impact/`), a coarse public locator
  (centroid-of-sensors circle, always `UNASSOCIATED`), and a `KUULO_IMPULSE_LOCATOR` plug-in
  point — the same pattern as `KUULO_FUSION_ENGINE` — for an optional, more capable private
  locator to do real multilateration and drone-impact attribution.
- **CAP 1.2 export:** `GET /v1/impulse-events/{id}/cap`, `status="Test"` by default.
- **Simulator:** `impact_strike` (a drone track ending in an impulse) and `firework` (a
  standalone, unattributed bang) bundled scenarios.
- **Dashboard:** IMPACT (red, drone-attributed) vs IMPULSE (orange, unassociated) map markers
  and detail panel.
- **Real-audio evaluation of the node detector** (`ml/src/kuulo_ml/impulses.py`,
  `make impulse-eval` → `ml/IMPULSE_RESULTS.md`): ESC-50 fireworks/glass-breaking/door-knock
  clips as a blast proxy (no licensed blast recordings exist), mixed into ESC-50 continuous
  sounds and DroneNoise recordings at controlled SNR. `onset_sigma_s` is a genuine per-event
  confidence from the AIC onset picker's own curvature (tighter for a clean onset, looser for a
  noisy one — measurably SNR-dependent), scaled by a calibration constant (`SIGMA_SCALE`) to
  match real error; a first version instead derived it from a fixed multiple of rise time,
  which passed the same coverage target only by inflating every report toward one ceiling
  regardless of onset clarity — a final-review finding, replaced during that review's fix pass.
  At ≥20 dB SNR, ≥90 % of *non-gross* onset errors (i.e. excluding wrong-event mis-picks, which
  a per-event sigma cannot represent and a downstream locator must reject separately) fall
  within the reported 2σ (91.1 % at 30 dB, 95.7 % at 20 dB, 100 % at 10 dB). Full reasoning is
  in the SDD ledger's Task 9 and final-review rulings and in `ml/IMPULSE_RESULTS.md`'s own
  Calibration section.
- **Private engine split:** the precise TDOA multilateration solver and drone-attribution logic
  live in a separate, never-published private repository, loaded only through the
  `KUULO_IMPULSE_LOCATOR` plug-in point — this repository never imports it and contains no
  reference to its package name.
- 331 Python tests, 31 dashboard tests, all built test-first per the executing-plans skill.
- **Final whole-branch review** (fresh reviewer, most capable model): one Critical finding (a
  private-repo name accidentally left in this file's own first draft) and four Important
  findings (the sigma-calibration issue above; the impulse locator not isolating persist/publish
  from a misbehaving plug-in, so ingest could get stuck; the dashboard not logging an IMPACT
  escalation when an event's quality doesn't also change; the private repo's name pre-existing
  in a Part-1 spec doc) were all fixed in one pass, each with a failing test first. 13 Minor
  findings were deferred — see the ledger and this branch's final message for the full list.

Still needs a person: the dashboard demo GIF, pushing this branch and creating/pushing the
private repository (both denied to the agent by repository/`gh` settings), and a real-microphone
trial of the impulse detector against a real bang in a city.

---

## Part 3 (2026-09-26): Milestone 1 completion

Plan: `docs/superpowers/plans/2026-09-26-kuulo-part3-milestone1-completion.md`.

- **Step B classifier:** licensed data (DroneNoise CC BY 4.0 + ESC-50 CC BY-NC), leak-free
  flight-level splits, training-only augmentation, YAMNet embedding → MLP head → ONNX.
  Held-out: F1 0.916 per window; 46/46 drone recordings detected, median 1.95 s to START.
  Step A (zero-training YAMNet) detects 0/46: it hears distant drones as "Silence".
  `make ml` reproduces everything; results in `ml/RESULTS.md`.
- **Feature traces end to end:** node recorder (3 s pre-roll, 60 s segments, 500 MB budget with
  priority eviction), server pull on confirmation, auto-push (≥0.8 for ≥20 s), 1 % hard-negative
  sampling capped at 3 min, 50 MB/day budget, "unavailable" replies; simulator synthesises
  traces with Doppler; `long_event` scenario uploads ≥10 segments for one detection.
- **Node:** durable SQLite outbox (10 000 cap, heartbeats dropped first), batched observation
  delivery, `classifier = "auto"`, opt-in `--debug-save-clips`.
- **Server:** batch ingest, trace storage and requests, Ed25519 key validation, heartbeat replay
  guard, WebSocket Origin check, cheaper tick, additive column migration.
- **Dashboard:** reconnect races fixed; evidence panel shows trace segment count.
- **Real-audio end to end:** a held-out flight replayed in real time through `kuulo-node` gave
  START after 2.1 s and a TENTATIVE track on a live server.

Still needs a person: the dashboard demo GIF recorded in a browser, pushing to GitHub (the
project settings deny `git push` to the agent), and a real-microphone trial in a city.

---

# Kuulo Part 1 — overnight build: morning summary

Plan: `docs/superpowers/plans/2026-09-24-kuulo-part1-core-pipeline.md` — all 9 tasks complete,
final review complete, fix pass complete. **Ready to merge.**

## What got built

The core pipeline from the design spec, as a working local system:

- **`protocol/`** — Pydantic wire models, Ed25519 signing, API views, JSON Schema export
- **`server/`** — FastAPI ingest server, SQLite storage, node health, live WebSocket, a
  pluggable fusion-engine interface with the `BasicFusion` baseline (weighted-centroid
  location, silent-neighbour downgrade logic, track lifecycle)
- **`sim/`** — deterministic scenario simulator with real acoustic physics, 5 bundled
  scenarios, a realtime runner, the `kuulo-sim` CLI
- **`dashboard/`** — React + TypeScript + Vite + MapLibre live map with a resilient
  WebSocket (backoff, resync-safe buffering) and an evidence side panel
- **`Makefile` / `README.md`** — `make dev`, `make sim`, `make test`

## Test results

**91 pytest + 13 vitest passing, ruff clean.** Real end-to-end verification, not just unit
tests: the simulator's `helsinki_pass` scenario run against a live `uvicorn` server produces a
confirmed track with a measured **mean location error of 176 m** (well inside the 750 m bound).

## Final review

Dispatched to a fresh Opus reviewer against the whole branch. Verdict: **ready to merge, with
fixes** — no Critical findings, all five of the plan's Review Focus items (signed-message edge
cases, node re-registration conflict, restart closing open tracks, a stalled WebSocket client,
snapshot/live-event races) confirmed implemented and genuinely tested, all 10 implementation
rulings held up under scrutiny. Full report and every ruling in
`.superpowers/sdd/2026-09-24-kuulo-part1-core-pipeline/progress.md`.

**3 Important findings were fixed** (each with a failing test written first, then full-suite
green):
1. A closed track stayed on the dashboard map until the next resync (undid the restart-safety
   fix client-side). Fixed in the reducer.
2. The README's own two-command quick start crashed the second time (`make sim` then
   `make sim SCENARIO=false_alarm`) with an uncaught 409, because two scenarios sharing a node
   id like "n01" got different signing keys. Fixed by deriving keys from node id alone, plus
   friendly CLI error messages. **Verified live** against a running server with the exact
   README sequence — both commands now succeed.
3. One observation with an extreme or non-finite `snr_db` (e.g. `1e4`, or a raw `Infinity`
   JSON literal) could silence fusion for every nearby node, or in the `Infinity` case,
   network-wide. Fixed by bounding the field and disallowing inf/nan on every wire model —
   which surfaced and let us fix a second real bug: the validation-error handler crashed with
   an unhandled 500 instead of a clean 400 when echoing an invalid non-finite value back.

**1 Important finding was a plan-vs-spec gap, not a code bug:** the spec allows batching
multiple observations per request; only single-observation ingest was built. Ruled: genuinely
deferred to Part 2, where the node's offline-queue flush is the natural first caller. No code
written for it now — Part 2's plan should say so explicitly.

**Deferred minors** (11 items — replayed-heartbeat liveness spoofing, unvalidated registration
public keys, no WebSocket Origin check, a few `live.ts` reconnect races, a StrictMode cleanup
gap in `MapView`, a couple of test-strength gaps, a performance nit in `run_tick`, one README
correction) are listed in full in the SDD ledger and were intentionally left for you to
prioritize — per the executing-plans skill, minors never enter the fix pass.

## Known limitation: local `uv` environment bug

Repeatedly, any `uv sync` that (re)builds a workspace member's editable install broke Python's
import of some *other* already-installed member (`ModuleNotFoundError: No module named
'kuulo_protocol'` and similar), on this machine specifically. **Fix, every time it happens:**
```
uv sync --reinstall
```
then confirm with `uv run python -c "import kuulo_protocol, kuulo_server, kuulo_sim"` before
trusting further `uv run` commands. Added `pythonpath` in `pyproject.toml`'s pytest config as a
safety net for test runs, but it doesn't cover `uvicorn`/`kuulo-sim` directly. Not shipped-code
related; this is worth mentioning to whoever else sets up this repo, in case it's a broader
uv/macOS interaction rather than something local to this machine. Once, this also corrupted
`.venv` outright (a killed background process mid-write) — fixed with a full `rm -rf .venv &&
uv sync`; never kill a `uv sync`/`uv run pytest` process while it's mid-install again.

## What's still pending (needs you)

1. **The visual browser check.** I have no browser in this session. Run:
   ```
   make dev            # server + dashboard
   make sim            # in a second terminal
   ```
   Open http://127.0.0.1:5173 and confirm: nodes appear green, a pulse flashes at each
   detecting node as `helsinki_pass` plays, a track appears (amber → red as it confirms) and
   moves west→east with a trail and an uncertainty circle, clicking it shows the evidence list,
   and — now that the fix pass landed — a closed track actually disappears rather than
   lingering.
2. **Push to GitHub / add the LICENSE (AGPL-3.0 per the spec).** Not done — pushing and
   licensing are yours to decide when.
3. **Decide on the deferred minors and the batch-ingest gap** — prioritize or explicitly wave
   off any of them for Part 2's plan.

## Next: Part 2

Node software (real mic/wav detector), FeatureTraces, ML training/evaluation, CI, and the full
README were explicitly out of scope for Part 1 and are next.

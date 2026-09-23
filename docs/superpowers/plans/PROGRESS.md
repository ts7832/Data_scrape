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

# Kuulo Part 1 — overnight build progress

Live notes while `docs/superpowers/plans/2026-09-24-kuulo-part1-core-pipeline.md` executes.
A full summary replaces this section when the run finishes.

## Notable events

- **Tasks 4, 5, 6 (recurring, structural — not a code/plan bug):** every `uv sync` that
  rebuilds a workspace member's editable install (adding a new member, or `uv run pytest`
  auto-rebuilding after source changes) reliably breaks Python's import of some OTHER
  already-installed workspace member (`kuulo_protocol`/`kuulo_server`/`kuulo_sim`), causing
  `ModuleNotFoundError` — confirmed both for pytest and for the `kuulo-sim` console script.
  **Fix every time:** `uv sync --reinstall`, then smoke-test with
  `uv run python -c "import kuulo_protocol, kuulo_server, kuulo_sim"` before trusting further
  `uv run` commands. Also added `pythonpath = ["protocol/src", "server/src", "sim/src"]` to
  pytest config as a safety net (protects pytest only, not `uvicorn`/`kuulo-sim`). This will
  be called out as a known limitation in the README. Full investigation in the SDD ledger at
  `.superpowers/sdd/2026-09-24-kuulo-part1-core-pipeline/progress.md`.

- **Task 8:** the venv got corrupted once by a killed background `uv run pytest` (was hung
  waiting on a WebSocket event fusion wasn't wired to send yet — expected mid-task, but killing
  it mid-write left a garbage `pygments-2.21.0 2.dist-info` directory). Fixed with a full
  `rm -rf .venv && uv sync && uv sync --reinstall` (verified `.venv` is gitignored first; no
  source was touched). From here on, `uv`/pytest commands run in the foreground only.
  All 5 scenario tests (single_node, helsinki_pass, false_alarm, two_drones, node_failure)
  passed on the first try. **helsinki_pass mean location error: 176 m** (bound was <750 m).

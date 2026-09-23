# Kuulo Part 1 — overnight build progress

Live notes while `docs/superpowers/plans/2026-09-24-kuulo-part1-core-pipeline.md` executes.
A full summary replaces this section when the run finishes.

## Notable events

- **Task 4:** hit an environment bug (not a code/plan bug): the `.venv`'s editable installs
  of `kuulo_protocol`/`kuulo_server` intermittently vanished from `sys.path`, causing
  `ModuleNotFoundError` on otherwise-passing code. Fixed with `uv sync --reinstall`, verified
  stable across repeated runs, and added `pythonpath = ["protocol/src", "server/src"]` to
  pytest config as a safety net for the rest of the run. If a *non-test* `uv run` command
  (e.g. `uvicorn`, `kuulo-sim`) ever fails with `ModuleNotFoundError: No module named 'kuulo_*'`
  later tonight, the fix is the same: `uv sync --reinstall`. Full detail in the SDD ledger at
  `.superpowers/sdd/2026-09-24-kuulo-part1-core-pipeline/progress.md`.

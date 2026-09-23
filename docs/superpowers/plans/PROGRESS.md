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

# Kuulo

Civic acoustic drone-detection network, starting in Helsinki. This repository is the open core:
protocol, server, basic fusion, simulator and dashboard. See
`docs/superpowers/specs/2026-09-24-kuulo-milestone1-design.md` for the design.

> Status: Milestone 1, part 1 (core pipeline). The node software, ML detector and full README
> arrive in part 2. Localhost only — do not expose the server to the internet.

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Node.js 20+.

```bash
uv sync
(cd dashboard && npm install)
make dev            # server on 127.0.0.1:8000, dashboard on http://127.0.0.1:5173
make sim            # in a second terminal: plays the helsinki_pass scenario
make sim SCENARIO=false_alarm SPEED=5
```

## Tests

```bash
make test
```

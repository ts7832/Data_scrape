.PHONY: dev server dashboard sim test types node model prepublish

SCENARIO ?= helsinki_pass
SPEED ?= 1

# --no-sync everywhere: `uv run`'s own automatic re-sync-on-change has been
# confirmed (see docs/superpowers/plans/PROGRESS.md) to intermittently break
# the editable install of one workspace package while rebuilding another.
# Skipping it here removes the trigger entirely. Run `make setup` once after
# cloning, and again after changing any pyproject.toml or lock-affecting file.

setup:
	uv sync --reinstall

server:
	uv run --no-sync uvicorn kuulo_server.main:app --host 127.0.0.1 --port 8000

dashboard:
	cd dashboard && npm run dev

dev:
	$(MAKE) -j2 server dashboard

sim:
	uv run --no-sync kuulo-sim run $(SCENARIO) --speed $(SPEED)

types:
	cd dashboard && npm run gen:types

test:
	uv run --no-sync ruff check .
	uv run --no-sync pytest
	cd dashboard && npx vitest run

NODE_CONFIG ?= node/config.local.toml

node/config.local.toml:
	cp node/config.example.toml $@

model:
	uv run --no-sync python node/scripts/download_model.py

node: $(NODE_CONFIG)
	uv run --no-sync kuulo-node run --config $(NODE_CONFIG) $(if $(INPUT),--input $(INPUT)) $(if $(SPEED),--speed $(SPEED)) $(if $(SCORES),--print-scores)

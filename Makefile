.PHONY: dev server dashboard sim test types

SCENARIO ?= helsinki_pass
SPEED ?= 1

server:
	uv run uvicorn kuulo_server.main:app --host 127.0.0.1 --port 8000

dashboard:
	cd dashboard && npm run dev

dev:
	$(MAKE) -j2 server dashboard

sim:
	uv run kuulo-sim run $(SCENARIO) --speed $(SPEED)

types:
	cd dashboard && npm run gen:types

test:
	uv run ruff check .
	uv run pytest
	cd dashboard && npx vitest run

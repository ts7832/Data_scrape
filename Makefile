.PHONY: dev server dashboard sim test types node model prepublish datasets embed train evaluate ml

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

# Step B (see ml/DATASETS.md for licences): ~1.3 GB of downloads, then a few minutes of CPU.
datasets:
	uv run --no-sync kuulo-ml datasets

embed: model
	uv run --no-sync kuulo-ml embed

train:
	uv run --no-sync kuulo-ml train

evaluate:
	uv run --no-sync kuulo-ml evaluate

ml: datasets embed train evaluate

node: $(NODE_CONFIG)
	uv run --no-sync kuulo-node run --config $(NODE_CONFIG) $(if $(INPUT),--input $(INPUT)) $(if $(SPEED),--speed $(SPEED)) $(if $(SCORES),--print-scores)

# Publishing shares the whole git history, not just the current tree, so this checks both:
# a forbidden file committed and later deleted would pass a tree-only check. The coordinate
# scan below stays informational, not a hard failure -- read it yourself before publishing.
prepublish:
	@! git ls-files | grep -E '\.(wav|flac|mp3|tflite|onnx|db|key)$$|(^|/)data/|\.local\.toml$$|-state/' \
		|| (echo "FAIL: forbidden files are tracked (see above)"; exit 1)
	@! git log --all --name-only --pretty=format: | sort -u \
		| grep -E '\.(wav|flac|mp3|tflite|onnx|db|key)$$|(^|/)data/|\.local\.toml$$|-state/' \
		|| (echo "FAIL: forbidden files exist somewhere in git history (see above)"; exit 1)
	@echo "Tracked coordinates -- read this yourself, it is not an automated check. Every"
	@echo "entry must be simulated, a test fixture, or the demo location 60.1694, 24.9490:"
	@git grep -nE 'lat[" =:]+[0-9]{2}\.[0-9]{3}' -- ':!docs' | cut -c1-120
	@echo "OK: no audio, models, keys, databases or local configs are tracked, now or in history."

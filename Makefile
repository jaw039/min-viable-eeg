# Common commands. Override the interpreter with PY=python.
PY ?= .venv/bin/python
SHARD ?= 0
NUM_SHARDS ?= 4

.PHONY: help test audit analyze tables data cache figures artifacts manifest smoke shard kaggle-bundle

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## / - /'

test: ## run the test suite
	$(PY) -m pytest -q

audit: ## protocol audit of every result row (exit 1 on violation)
	$(PY) scripts/audit_results.py --results 'results/*.jsonl'

analyze: ## k*, budget curve, controls, coverage -> results/kstar_report.json
	$(PY) scripts/analyze.py --results 'results/*.jsonl'

tables: ## electrode tables for the paper -> paper/channel_tables.md
	$(PY) scripts/channel_tables.py

data: ## download EDFs from PhysioNet and inventory them (~1.5 GB)
	$(PY) scripts/download_data.py --subjects $$(seq 1 109)
	$(PY) -m src.inventory

cache: ## preprocess to data/processed/S###/{X,y}.npy
	$(PY) scripts/cache_preprocessed.py

figures: ## data-verification figures (needs matplotlib)
	$(PY) scripts/make_figures.py

artifacts: ## regenerate the frozen artifacts (writers refuse to overwrite; see docs/reproducing.md)
	$(PY) -m src.splits && $(PY) -m src.ranking && $(PY) -m src.budget && $(PY) -m src.stability

manifest: ## write manifests/manifest_val.jsonl (committed already)
	$(PY) scripts/run_sweep.py --write-manifest

smoke: ## one short training run to time the hardware
	$(PY) scripts/run_sweep.py --smoke-test

shard: ## run one shard locally: make shard SHARD=0
	$(PY) scripts/run_sweep.py --manifest --shard-id $(SHARD) --num-shards $(NUM_SHARDS)

kaggle-bundle: ## build the code and cache zips for Kaggle under dist/
	$(PY) scripts/make_kaggle_bundle.py --code --cache

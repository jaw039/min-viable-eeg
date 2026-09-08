# Reproducing this work

Four levels, from cheapest to most expensive. Each level's expected output is
stated so that a mismatch is detectable.

## Level 0: verify the paper numbers from the committed rows

Needs Python 3.12 and the pinned requirements. No EEG data, no GPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                                   # 100 passed, 6 skipped
python scripts/audit_results.py --results 'results/*.jsonl' # RESULT: PASS
python scripts/analyze.py --results 'results/*.jsonl'       # writes results/kstar_report.json
python scripts/channel_tables.py --out paper/channel_tables.md
```

The six skipped tests read raw EDF files and run once the data are
downloaded. The audit re-derives every row's channel set from the frozen
ranking, checks the evaluated subjects against the split file, checks ranking
provenance and config hash, and reports coverage per arm.

Hashes of the inputs that every number depends on (`shasum -a 256`, first 16
hex digits):

| File | sha256 |
| --- | --- |
| `config.yaml` | `af8fbd091329044e` (the row field `config_sha256` is its first 12 digits) |
| `artifacts/splits.json` | `b5b370b649920612` |
| `artifacts/channel_ranking.json` | `2a78bb38ef66d85c` |
| `artifacts/budgets.json` | `2a344260d4d05b58` |
| `artifacts/stability.json` | `4d7c7a95a98800b6` |
| `manifests/manifest_val.jsonl` | `28d4dd698d481cd8` |
| `results/shard_0_of_4_val.jsonl` | `e71b31bc9b5be8d4` |
| `results/shard_1_of_4_val.jsonl` | `5e9902ef98ed8715` |

`protocol_hash()` in `src/utils.py` hashes only the config sections that can
change a frozen artifact (`dataset`, `preprocess`, `splits`, `budgets`,
`reduction_mode`) and evaluates to `e9f48c5483a2`. The artifacts record the
config hash of the file as it was when they were generated (`f31815c177f9`);
the whole-file hash moved to `af8fbd091329` when training hyperparameters were
added, and `tests/test_protocol_integrity.py` checks that none of those
additions touch a protocol key.

## Level 1: rebuild the cache and the frozen artifacts from raw EEG

About 1.5 GB of downloads; CPU only; an hour or two.

```bash
python scripts/download_data.py --subjects $(seq 1 109)   # skips S088/S092/S100
python -m src.inventory                                    # every file 160 Hz, 64 channels
python scripts/cache_preprocessed.py                       # data/processed/S###/{X,y}.npy
python scripts/make_figures.py                             # figures/*.png (needs matplotlib)
```

The artifact writers refuse to overwrite an existing file, so regenerate into
a scratch copy and compare content, ignoring the provenance block (commit and
timestamp legitimately differ):

```bash
mkdir -p /tmp/regen && cp -r artifacts /tmp/regen/orig
rm artifacts/splits.json artifacts/channel_ranking.json artifacts/budgets.json artifacts/stability.json
python -m src.splits && python -m src.ranking && python -m src.budget && python -m src.stability
python - <<'PY'
import json
for name in ("splits", "channel_ranking", "budgets", "stability"):
    a = json.load(open(f"artifacts/{name}.json")); b = json.load(open(f"/tmp/regen/orig/{name}.json"))
    a.pop("provenance"); b.pop("provenance")
    print(f"{name:<18}", "identical" if a == b else "DIFFERENT")
PY
git checkout -- artifacts/      # restore the committed, provenance-stamped originals
```

Everything is seeded (split seed 42, bootstrap seed 42, 200 resamples), so
all four should print `identical`.

## Level 2: re-run the validation sweep

Needs a GPU. The manifest is committed, so nothing has to be regenerated:

```bash
python scripts/run_sweep.py --smoke-test                                  # one short k=8 run
python scripts/run_sweep.py --manifest --shard-id 0 --num-shards 4        # ~70 min on T4 ×2
```

Repeat for shards 1–3, or use Kaggle as described in [kaggle.md](kaggle.md).
Rows append to `results/shard_<i>_of_4_val.jsonl` and completed conditions
are skipped on a re-run. Expect channel sets, splits and seeds to match the
committed rows exactly (the audit checks this) and κ to match closely but not
bit-for-bit across GPU models and CUDA versions.

Do not edit `config.yaml` between shards: every ratio divides by κ_full, and a
different `config_sha256` in one shard makes the audit fail on purpose.

## Level 3: the test split

Only after all four validation shards are in and k\* has been chosen:

```bash
python scripts/analyze.py --results 'results/*.jsonl'          # reports k*
python scripts/run_sweep.py --write-manifest --split test --budget <k*>
python scripts/run_sweep.py --manifest --split test --shard-id 0 --num-shards 1
python scripts/analyze.py --results 'results/*.jsonl' --test-report
```

The test manifest writer refuses to run without `--budget`, so the test
split cannot be swept.

## Environment used for the committed rows

Kaggle, GPU T4 ×2, image
`gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`,
`python3.12.13 torch2.10.0+cu128 numpy2.0.2` (stamped in every row as
`environment`). Local analysis: Python 3.12.7 with `requirements.txt`.

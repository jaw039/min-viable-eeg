# Result rows and run log

Each `shard_<i>_of_4_val.jsonl` holds one provenance-stamped row per condition
of `manifests/manifest_val.jsonl` (every fourth condition, offset `i`). These
files are the evidence behind every number in the paper and are tracked in
git, as is `kstar_report.json`, the analysis they were read into.
`single.jsonl` (ad-hoc runs) and `teachers/` (cached distillation teachers)
are not tracked.

Audit before analysing:

```bash
python scripts/audit_results.py --results 'results/*.jsonl'
```

## Validation sweep

All shards: Kaggle notebook `jackiewang2323/notebookeb3bd89e45`, GPU T4 ×2,
code dataset `mve-code` v1 (commit `e35a506`), cache dataset
`mve-eegmmidb-cache` v1, `config_sha256 = af8fbd091329`, environment
`python3.12.13 torch2.10.0+cu128 numpy2.0.2`. Wall time is the sum of the
rows' `runtime_sec`.

| Shard | Kaggle version | Date (UTC) | Rows | Errors | Wall time | sha256 (first 16) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | not recorded | 2026-09-05 | 173 | 0 | 70.5 min | `e71b31bc9b5be8d4` |
| 1 | Version 5 | 2026-09-08, 00:04–01:01 | 173 | 0 | 56.5 min | `5e9902ef98ed8715` |
| 2 | Version 6 | 2026-09-08, 01:09–02:16 | 173 | 0 | 68.4 min | `34ff025cfa12c128` |
| 3 | Version 7 | 2026-09-08, 02:25–04:12 | 172 | 0 | 57.0 min | `e1db29deab737975` |

Discarded runs: a Version 4 of shard 1 was cancelled 55 minutes in, before
its output was saved; nothing from it is used. Interactive sessions do not
persist output and none of their rows exist.

The validation sweep is complete: 691 of 691 conditions, 0 errors,
`audit_results.py --require-complete` passes. The label-shuffle control
scores κ = 0.007 on permuted labels (no leakage). `kstar_report.json` is the
analysis of these rows and is committed; regenerate it with `make analyze`.
Headline on validation: k\*(0.90) = 32 (κ_full 0.443 ± 0.063, κ_32 0.420 ±
0.040 over 5 seeds), but k\* per seed is 16/32/64/32/64 and the threshold
sweep gives 32/32/64, so the report's verdict is that the budget curve with
its uncertainty is the result, not the integer.

## Adding a shard

1. Wait for the Version to show COMPLETE (`kaggle kernels status ...`).
2. `kaggle kernels output jackiewang2323/notebookeb3bd89e45 -p /tmp/kaggle_out`
3. `cp /tmp/kaggle_out/results/shard_<i>_of_4_val.jsonl results/`
4. Run the audit; it must print `RESULT: PASS` and one `config_sha256`.
5. Add the row to the table above with `shasum -a 256`.

## Test split

Not run. `manifests/manifest_test.jsonl` was written from the validation
report: k\* = 32 and the 64-channel reference, ranked, scratch, at all five
seeds (10 conditions). Running it needs a new code-dataset version built
from the merged commit (`make kaggle-bundle`), and one Kaggle Version with
`SPLIT = "test"` and `KSTAR = 32` in the notebook's configuration cell;
the notebook checks the validation report before writing the manifest. It
will be recorded here with the same fields:

```bash
python scripts/run_sweep.py --write-test-manifest --kstar-report results/kstar_report.json
python scripts/run_sweep.py --manifest --split test --shard-id 0 --num-shards 1
python scripts/audit_results.py --results 'results/*.jsonl'
python scripts/analyze.py --results 'results/*.jsonl' --test-report
```

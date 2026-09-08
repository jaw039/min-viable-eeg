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

`manifests/manifest_test.jsonl` was written from the validation report:
k\* = 32 and the 64-channel reference, ranked, scratch, at all five seeds
(10 conditions). It runs as one Kaggle Version with `SPLIT = "test"` and
`KSTAR = 32` in the notebook's configuration cell; the notebook checks the
validation report before writing the manifest. Every attempt is listed.

| Attempt | Kaggle version | Date (UTC) | Code dataset | GPU | Outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | Version 8 | 2026-09-08, 18:55–18:57 | `mve-code` v4 (commit `8f91320`) | Tesla P100 | failed before any metric: the image's torch 2.10 cu128 build has no kernels for the P100 (sm_60); 10 error rows, discarded, not in `results/` |
| 2 | Version 9 | 2026-09-08, 19:08–19:18 | `mve-code` v5 (commit `5ad044b`) | Tesla T4 ×2 | 10/10 conditions, 0 errors, audit PASS; rows in `shard_0_of_1_test.jsonl` |

Run record, same fields as the validation shards. Every row stamps
`config_sha256 = af8fbd091329`, environment
`python3.12.13 torch2.10.0+cu128 numpy2.0.2` and
`git_commit = 5ad044b67c5fdffad83ceb329abf5de7b2f796c1` (read from the
`COMMIT` file in the code snapshot). Launched with
`kaggle kernels push --accelerator NvidiaTeslaT4`.

| Shard | Kaggle version | Date (UTC) | Rows | Errors | Wall time | sha256 (first 16) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 of 1 (test) | Version 9 | 2026-09-08, 19:08–19:18 | 10 | 0 | 9.6 min | `8aa365f62143db80` |

Result (`kstar_report.json`, section `test_report`), evaluated once at the
validation-chosen k\* = 32; 16 test subjects, 718 trials:

| Budget | Test κ (mean ± sd, 5 seeds) | Per-seed κ (42, 123, 456, 789, 101112) |
| --- | --- | --- |
| 32 (k\*) | 0.292 ± 0.061 | 0.236, 0.358, 0.313, 0.332, 0.221 |
| 64 (full) | 0.326 ± 0.032 | 0.309, 0.322, 0.359, 0.355, 0.283 |

Retained fraction κ₃₂ / κ₆₄ = **0.897** (ratio of means). Per matched seed
it is 0.76, 1.11, 0.87, 0.93, 0.78 (mean 0.89 ± 0.14), so the point estimate
sits just below the 0.90 target with a wide interval, consistent with the
validation verdict that k\* is unstable. The test subjects are harder than
the validation subjects (κ₆₄ 0.326 vs 0.443); no test subject is near
chance at 64 channels (per-subject κ 0.06–0.66, median 0.28). No other
budget was evaluated on test.

Commands (the first two ran inside the notebook on Kaggle):

```bash
python scripts/run_sweep.py --write-test-manifest --kstar-report results/kstar_report.json
python scripts/run_sweep.py --manifest --split test --shard-id 0 --num-shards 1
python scripts/audit_results.py --results 'results/*.jsonl'
python scripts/analyze.py --results 'results/*.jsonl' --test-report
```

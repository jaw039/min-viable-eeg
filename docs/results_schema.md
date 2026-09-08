# Result row schema

One JSON object per line in `results/shard_<i>_of_<n>_<split>.jsonl`. A
condition that raised is written as `{"error": ..., ...condition}` instead and
is excluded from analysis but counted by the coverage report.

| Field | Meaning |
| --- | --- |
| `budget_k` | Number of electrodes the model was trained on |
| `selection` | `ranked`, `random` or `sensorimotor` |
| `selection_seed` | Seed of the random subset; `null` for the deterministic arms |
| `training` | `scratch`, `distill` or `distill_shuffled_teacher` |
| `train_seed` | Seed for weight initialisation, batching and the inner holdout |
| `split` | `val` or `test` |
| `shuffle_labels` | `true` only on the negative-control row |
| `channels` | Electrodes trained on, in ranking (or draw) order |
| `channels_montage_order` | The same electrodes in data-array order; ranked rows must equal `artifacts/budgets.json` |
| `n_channels` | `len(channels)`, must equal `budget_k` |
| `kappa` | Cohen's κ pooled over all trials of the evaluated split: the primary metric |
| `kappa_macro_subject`, `kappa_macro_subject_std` | Mean and SD of per-subject κ, the alternative aggregation |
| `kappa_per_subject` | κ for each evaluated subject, keyed `S###`; never averaged away |
| `accuracy`, `macro_f1` | Secondary metrics on the same trials |
| `n_trials`, `n_subjects`, `n_trials_per_subject` | What was evaluated; subjects must equal the split in `artifacts/splits.json` |
| `n_fit_subjects`, `n_inner_holdout_subjects` | Training subjects used for fitting and for early stopping (63 and 11 of the 74) |
| `inner_kappa`, `best_epoch` | Early-stopping diagnostics on the inner holdout |
| `aggregation_primary` | `pooled`: which κ the headline uses |
| `splits_seed` | Seed of the split file the row was evaluated under |
| `ranking_provenance` | Provenance block of the `channel_ranking.json` the channel set was derived from |
| `config_sha256` | First 12 hex digits of the sha256 of `config.yaml` as used |
| `git_commit` | Commit of the code, or `unknown` when run from a snapshot without `.git` |
| `environment`, `device` | Python, torch and numpy versions; `cuda` or `cpu` |
| `runtime_sec` | Wall time for training plus evaluation |

`scripts/audit_results.py` checks the invariants stated above for every row.

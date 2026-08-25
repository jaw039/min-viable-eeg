# Project Summary — min-viable-eeg data pipeline

Research question: how few EEG electrodes are needed for motor imagery
decoding (PhysioNet EEGMMIDB, 2-class left/right fist, runs 4/8/12).
This log covers the data-pipeline work to date (2026-08-16).

## Environment

- System Python 3.9.6 only (no conda/uv/homebrew) → `.venv` at repo root,
  pinned in `requirements.txt`: `mne==1.8.0` (last MNE supporting 3.9),
  `numpy==2.0.2`, `scipy==1.13.1`, `PyYAML==6.0.2`, `pytest==8.3.5`.
- Run everything from repo root with `.venv/bin/python`.

## Session 1 — download, inventory, loader (commit 5bd80e2)

- `scripts/download_data.py` — downloads runs 4/8/12 per subject via
  `eegbci.load_data`, skips excluded subjects (88/92/100), appends
  provenance (git commit + config hash) to `data/download_log.jsonl`.
- `src/inventory.py` (`python -m src.inventory`) — writes
  `data/inventory.csv` (subject, run, sfreq, n_channels, T0/T1/T2 counts);
  WARNs on sfreq ≠ 160 or channels ≠ 64. First 15 files: all clean.
- `src/loader.py` — `load_subject(subject_id, config, runs=None)` →
  `X (n_trials, 64, 641) float32`, `y (0=left/1=right)`, standardized
  channel names. Per file: read EDF → `eegbci.standardize` →
  standard_1005 montage → 8–30 Hz FIR on continuous data → epoch 0–4 s
  post-cue (T1/T2 only, `baseline=None`) → stack runs. Labels derived
  from post-drop `epochs.events`, so X/y stay aligned if an epoch is
  dropped at a truncated recording end.
- `src/utils.py` — config loading, EDF path helpers, provenance
  (git commit + sha256 config hash).
- `tests/test_loader.py` — label mapping proven against raw annotations
  (independent of the loader), shapes/dtype, exact run-concatenation
  consistency.

## Code review (deferred fixes)

A review of session-1 code confirmed no loader-logic bugs but flagged
test-coverage gaps (label oracle covers only S001R04; skip-guard checks
one file; epoch-drop path and excluded-subject guard untested; config
`label_map` key is unread). Notable dataset fact: **S104R08.edf is
truncated (~106 s)** — the loader handles it (drops 1 epoch, ~42 trials
for S104), but the covering test is still to be written.

## Session 2 — splits, normalization, cache (commits 5abbfa6, 6b33efb)

- `src/splits.py` (`python -m src.splits`) — deterministic subject-wise
  split of IDs 1–109 minus exclusions (106 subjects), seed 42 →
  train 74 / val 16 / test 16. Wrote `splits.json` (committed; protocol:
  generated once). Refuses to overwrite; delete manually to regenerate
  (regeneration is deterministic → identical content).
- `src/normalize.py` — `fit_stats(X_train)` per-channel mean/std
  (train split only, per protocol), `apply_stats(X, mu, sd)` → float32.
  Raises on zero-std channels and shape mismatches.
- `scripts/cache_preprocessed.py` — caches `data/processed/S###/X.npy`
  (float32) + `y.npy` per downloaded, non-excluded subject; skips
  already-cached and not-yet-downloaded subjects; provenance appended to
  `data/processed/cache_log.jsonl`. S001–S005 cached (X(45, 64, 641) each).
- `tests/test_splits_norm.py` — splits disjoint/cover all 106, excluded
  absent, same-seed determinism, sizes 74/16/16, overwrite refusal;
  normalization stats match manual computation, are provably unaffected
  by val/test data, and val normalized with train stats stays off-center.

**Test suite: 12/12 passing.**

## Full-dataset status (complete)

- All 106 included subjects downloaded (318 EDFs) and cached to
  `data/processed/` — 4750 trials total. Refreshed inventory: 318 rows,
  zero protocol warnings (every file 160 Hz / 64 channels).
- Epoch drops on real data: 20 runs across 10 subjects (S034, S037, S041,
  S064, S072, S073, S074, S076, S102, S104) each dropped the final epoch
  because its 4-s window overran the recording end; S104R08 is an outright
  truncated recording (~106 s). The loader handles this by design (labels
  come from post-drop events, X/y stay aligned), so these subjects simply
  have 42–44 trials instead of 45.

## Session 3 — channel ranking + budget utility (commit 45fdb9a)

- `src/ranking.py` (`python -m src.ranking`) — per-channel Fisher score of
  8–30 Hz log-bandpower (per-trial log-variance of the bandpassed epochs),
  computed per train-split subject and averaged over the 74 train subjects
  only (no val/test influence, tested). Wrote `channel_ranking.json`
  (committed; generated once, refuses overwrite like splits.json).
- Ranking result: right sensorimotor cortex dominates — top 5 are C4, CP4,
  C6, CP6, FC4; C2 rank 7, CP3 rank 17, C3 rank 23, Cz rank 49. The
  hemispheric asymmetry is a property of the data/method (lateralized ERD),
  worth noting when interpreting small-k budgets: k=4 selects
  right-hemisphere channels only.
- `src/budget.py` — `top_k_channels` + `reduce_channels`/`apply_budget`:
  physically subsets to the top-k ranked channels, returned in original
  montage order; k=64 is the identity; `mask` mode raises until needed.
- `tests/test_ranking_budget.py` — 9 tests: known-discriminative-channel
  recovery, determinism, train-only leakage guard, selection correctness,
  budget validation, overwrite refusal. Full suite: 28 passing.

## Session 4 — ranking verification + frozen budget sets (commits 3c7a2af..b1be5db)

Week 3 task: "verify the final electrode ranking and provide the exact
selected channel sets used at each budget so every experiment is
reproducible."

- **Ranking verified reproducible.** `channel_ranking.json` was regenerated
  from a clean checkout and compared to the committed file: channel order
  and all 64 Fisher scores byte-identical (max |Δ| = 0.0). The old file's
  provenance read `ac98052-dirty` (generated before `src/ranking.py` was
  committed); the regenerated file (5e0fbda) is stamped with clean commit
  `3c7a2af`.
- **Provenance fix (3c7a2af).** `get_git_commit()` marked the tree dirty
  whenever `git status` was non-empty — which regenerating a tracked
  generated-once file always is. Writers now pass their own output path as
  `ignore_paths`, so only unrelated changes count as dirty.
- **`budgets.json` (committed, generated once via `python -m src.budget`).**
  Exact channel set for every budget, in ranking order (best-first) and
  montage order (the order `reduce_channels` emits data). Stamped with its
  own provenance plus the ranking's. `test_committed_budgets_match_committed_ranking`
  guards the two files against drift. Full suite: 31 passing.

| k | channels added (ranking order, cumulative) |
|---|---|
| 4 | C4, CP4, C6, CP6 |
| 6 | + FC4, F7 |
| 8 | + C2, AF7 |
| 12 | + PO7, O1, P5, PO3 |
| 16 | + CP2, P6, F8, P4 |
| 32 | + CP3, CP5, FC2, P7, FC6, P3, C3, P8, O2, TP8, Iz, PO4, Fp1, F4, Oz, PO8 |
| 64 | all |

**Caveat for interpretation.** C3 (left motor cortex) enters only at rank
23, so no budget ≤ 16 contains it, and several non-motor channels (F7, AF7,
PO7, O1) outrank it. The C4-side asymmetry is real (the contralateral ERD
check in `figures/erd_check.png` shows the C4 effect ≈ 2× C3), but the
frontal/occipital channels in the top 12 may reflect lateralized
gaze/attention correlates of the cue rather than motor ERD. This does not
affect the decoding objective (any class-discriminative channel counts) but
should be stated when reporting "minimum electrodes for motor imagery".
The ranking method is frozen; a bootstrap stability check over train
subjects is the natural follow-up, not a method change.

## Next steps

- Repo migration: DONE — `origin` now points at
  github.com/jaw039/min-viable-eeg (private); the IBM remote was removed
  locally, so pushes can no longer reach github.ibm.com. The accidental
  repo at github.ibm.com/JackieWang/min-viable-eeg still exists on IBM's
  side (initial commit only) — delete via its web UI Settings if desired.
- Pipeline scope (loader, splits, normalization, cache, ranking, budgets):
  COMPLETE and frozen. Downstream consumers: `splits.json`,
  `channel_ranking.json`, `budgets.json`, `data/processed/S###/{X,y}.npy`.
- Optional robustness: bootstrap the 74 train subjects and report top-k
  membership frequency; per-subject vs shared-montage overlap table
  (mentor request). Neither changes the frozen sets.
- Open decision to confirm with Kiran: `reduction_mode: reduce` (physical
  channel subsetting) is in config.yaml but not in the locked protocol list.

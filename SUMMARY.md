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

## Next steps

- Repo migration: DONE — `origin` now points at
  github.com/jaw039/min-viable-eeg (private); the IBM remote was removed
  locally, so pushes can no longer reach github.ibm.com. The accidental
  repo at github.ibm.com/JackieWang/min-viable-eeg still exists on IBM's
  side (initial commit only) — delete via its web UI Settings if desired.
- Deferred review fixes (test coverage) listed above.
- Remaining pipeline scope: channel ranking and reduced-channel budget
  utility (budgets 4/6/8/12/16/32/64; headline = smallest k with
  κ_k/κ_full ≥ 0.90).

# Locked protocol

Set in Kiran's pipeline specification before any data were processed. Every
result row was produced under these values. `config.yaml` is the
machine-readable copy; the first 12 hex digits of its sha256
(`af8fbd091329`) are stamped into every row as `config_sha256`. Changing
anything in sections 1–6 invalidates the frozen artifacts and every result
row, and needs sign-off from the whole team. Sections 7–9 are fixed for the
sweep but are not protocol: they may be revised between sweeps, never within
one.

## 1. Dataset

- PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB): 109 subjects,
  64 channels, 160 Hz. Runs 4, 8 and 12 (imagined opening and closing of the
  left or right fist).
- Excluded: S088, S092, S100, recorded at 128 Hz. 106 subjects remain.
- Channel names standardised with MNE's `eegbci.standardize` and the
  `standard_1005` montage.

## 2. Task and labels

- Two classes from the annotations: T1 = imagined left fist → 0, T2 =
  imagined right fist → 1. T0 (rest) is discarded.
- 4,750 trials in total, 42–45 per subject. Ten subjects lose one final epoch
  per affected run because the 4-second window overruns the recording; labels
  are taken from the surviving epochs so `X` and `y` stay aligned.

## 3. Preprocessing

- FIR band-pass 8–30 Hz applied to the continuous recording, before epoching.
- Epochs 0–4 s after the cue: 641 samples per channel. No baseline
  correction, no ICA, no artifact rejection in this version.
- Cached per subject as `data/processed/S###/X.npy` `(n_trials, 64, 641)
  float32` and `y.npy`.

## 4. Splits

- Subject-wise, seed 42, ratios 0.7 / 0.15 / 0.15 of the 106 subjects:
  74 train / 16 validation / 16 test. Generated once into
  `artifacts/splits.json`; the writer refuses to overwrite it.
- Validation subjects: S007, S012, S017, S031, S035, S045, S050, S055, S059, S068, S075, S089, S093, S096, S098, S108.
- Test subjects: S009, S013, S014, S015, S020, S023, S036, S037, S042, S048, S054, S065, S066, S067, S078, S101.
- No subject appears in two splits (tested).

## 5. Normalisation

- Per-channel mean and standard deviation fitted on training trials only and
  applied unchanged to validation and test (tested for leakage).

## 6. Channel ranking, budgets and stability

- Ranking method `fisher_log_bandpower_subject_mean`: trials are already
  band-limited, so each trial's per-channel log-variance is its 8–30 Hz log
  band-power. Per training subject and channel, the Fisher score
  (mean_left − mean_right)² / (var_left + var_right) is computed over that
  subject's trials; the channel's final score is the mean across the 74
  training subjects. Ties break by channel index, so the ranking is
  deterministic. Validation and test subjects are never read (tested).
- Budgets k ∈ {4, 6, 8, 12, 16, 32, 64}. The ranked set for each k is the
  top-k of the frozen ranking, frozen in `artifacts/budgets.json` with a drift
  guard that runs before every ranked training run.
- `reduction_mode: reduce`: the unselected channels are removed from the
  input rather than zero-masked. **[UNRESOLVED]** Not on the original locked
  list; awaiting sign-off.
- Stability (`artifacts/stability.json`): top-k inclusion frequency over 200
  bootstrap resamples of the training subjects (seed 42), and the overlap of
  each subject's own top-k with the shared top-k.

## 7. Model and training

- EEGNet-style CNN (`src/eegnet.py`): temporal convolution, depthwise
  spatial convolution over the selected channels, separable convolution,
  dropout 0.5. Input `(channels, 641)`, output 2 classes. Not a faithful port
  of the published EEGNet (no max-norm constraints; temporal kernel 64 rather
  than fs/2).
- Hyperparameters from `config.yaml`: learning rate 0.001, batch size 32, up
  to 100 epochs, early-stopping patience 10. The training loop is
  `src/training.py`.
- Early stopping uses an inner holdout of 15% of the *training* subjects
  (11 of 74), drawn per train seed. The validation split is never used for
  model selection.
- Train seeds: 42, 123, 456, 789, 101112. Every condition is run at all five.
- Distillation (`src/distillation.py`): the student at budget k learns from a
  64-channel teacher trained with the same seed (α 0.5, temperature 4.0);
  teachers are cached in `teachers/`, never committed.

## 8. Metric and headline rule

- Cohen's κ pooled over all trials of the evaluated split is the primary
  metric. The macro average over per-subject κ and the full per-subject
  distribution are stored alongside it in every row.
- κ_full is the mean κ of the ranked k = 64 rows over the five seeds.
- k*(τ) = min{ k : κ_k ≥ τ · κ_full }, τ = 0.90, with sensitivity at 0.85 and
  0.95. Chosen on the validation split only, enforced by exception in
  `src/kstar.py`. The test split is evaluated once, at that k*.
- If k* differs across seeds or thresholds, the budget curve with its
  uncertainty is the result and the report says so (`verdict` field).

## 9. Controls

All controls are rows of the manifest, not separate scripts.

- **Random subsets**: 20 seeded subsets (selection seeds 0–19) per budget
  below 64, each trained at all five seeds. Ranked is compared with the
  empirical distribution of random κ; with 20 subsets the smallest reportable
  p-value is 1/21, and the analysis prints that floor.
- **Sensorimotor restriction**: the frozen ranking restricted to a declared
  17-electrode motor strip, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4,
  C6, CP3, CP1, CPz, CP2, CP4, at budgets 4–16. Separates motor signal from
  cue-correlated gaze or attention. **[UNRESOLVED]** The pool is a proposal
  awaiting sign-off.
- **Label shuffle**: one ranked k = 64 run with permuted training labels;
  expected κ ≈ 0. The analysis reports it before anything else.
- **Shuffled-teacher distillation**: distillation from a teacher trained on
  permuted labels, at the same budgets and seeds as real distillation. A gain
  over scratch that also appears here is regularisation, not transfer.

## 10. Provenance and determinism

- Every writer stamps the git commit and config hash into its output. Frozen
  artifacts carry `provenance`; result rows carry `config_sha256`,
  `git_commit`, `environment`, `splits_seed` and `ranking_provenance`.
- Nothing random is unseeded: split seed, bootstrap seed, selection seeds and
  train seeds are all explicit.
- `protocol_hash()` covers only sections 1–6 of the config and equals
  `e9f48c5483a2`; the artifacts record config hash `f31815c177f9`, from
  before the training sections were appended, and
  `tests/test_protocol_integrity.py` checks that the later additions touch
  no protocol key.
- `config.yaml` must not change between shards of one sweep: `config_sha256`
  is part of the audit and a mixed sweep fails it.

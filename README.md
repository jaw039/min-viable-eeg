# Minimum Viable EEG

How few scalp electrodes does motor-imagery decoding need? This repository is
the complete, tested pipeline behind that question: a frozen preprocessing and
channel-ranking stage, a 691-condition experiment written down before it ran,
a runner that stamps every result row with its provenance, and the analysis
and audit tools that turn those rows into paper numbers.

**Status (2026-09-08).** The pipeline and its frozen artifacts are final. The
validation sweep ran on Kaggle in four shards; all 691 conditions are in
`results/` and pass the audit with complete coverage. k\* was chosen on
validation and then evaluated once on the test split: 32 electrodes retain
0.897 of the 64-electrode κ there, against a 0.90 target. The run log is
[results/README.md](results/README.md).

| Stage | State |
| --- | --- |
| Download, preprocessing, epoch cache | done, tested |
| Subject-wise split, channel ranking, budgets, stability | done, frozen in [`artifacts/`](artifacts/) |
| Experiment manifest, 691 validation conditions | committed in [`manifests/`](manifests/) |
| Validation sweep | complete: 691/691 conditions, 0 errors, audit PASS |
| k\* selection on validation | k\*(0.90) = 32; unstable across seeds (16–64), so the curve is the result |
| Test split, evaluated once at k\* | done: 10/10 conditions, 0 errors, audit PASS; κ₃₂ / κ₆₄ = 0.897 on test |
| Manuscript | drafts in `paper/`; every unmeasured number stays bracketed |

## Research question

We measure the **effective channel budget** of scalp motor-imagery decoding:
how far the electrode count can fall while retaining a stated fraction of
full-montage performance.

This is deliberately not the claim that motor imagery *requires* k electrodes.
Volume conduction smears each cortical source across a wide patch of scalp, so
neighbouring channels partly re-measure the same signal and 64 channels carry
far fewer than 64 independent signals. k\* is a property of the instrument
under volume conduction, not a count of brain regions.

### Definition

```
k*(τ) = min{ k : κ_k ≥ τ · κ_full }
```

- Primary threshold τ = 0.90; sensitivity at τ ∈ {0.85, 0.90, 0.95}.
- κ is Cohen's kappa, pooled over validation trials (the per-subject
  distribution is stored alongside and never averaged away).
- **k\* is chosen on the validation split. The test split is evaluated once,
  at the already-chosen k\*.** This is enforced in code: `select_kstar()`
  raises on a test row, and the test report takes k\* as an argument so it has
  no search over budgets.

The full protocol, with every locked value and the two items still awaiting
sign-off, is in [docs/protocol.md](docs/protocol.md).

## Data

PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB), 64 channels at
160 Hz, runs 4, 8 and 12. Two classes from the imagery cues: T1 = left fist,
T2 = right fist; rest is discarded. Three subjects recorded at 128 Hz (S088,
S092, S100) are excluded, leaving 106 subjects, split subject-wise at seed 42
into 74 train / 16 validation / 16 test. Epochs are 0–4 s after the cue,
band-passed 8–30 Hz, about 45 trials per subject, 4,750 in total.

Raw data are not in the repository. `scripts/download_data.py` fetches them
from PhysioNet and `scripts/cache_preprocessed.py` writes one
`data/processed/S###/{X,y}.npy` pair per subject; `X` is
`(n_trials, 64, 641) float32`, `y` is 0 for left and 1 for right.

## The experiment

All 691 validation conditions are listed in `manifests/manifest_val.jsonl`,
generated once from `config.yaml` and split into four shards by taking every
fourth line, so each shard is a representative quarter of every arm.

| Arm | Conditions | Purpose |
| --- | --- | --- |
| Ranked, trained from scratch | 7 budgets × 5 seeds = 35 | The headline curve, using the frozen top-k sets |
| Random subsets | 6 budgets × 20 subsets × 5 seeds = 600 | The baseline ranked must beat; 20 subsets set the resolution floor of the test |
| Sensorimotor-restricted | 5 budgets × 5 seeds = 25 | Ranking restricted to a declared 17-electrode motor strip, to separate motor signal from cue-correlated gaze or attention |
| Distillation from the 64-channel model | 3 budgets × 5 seeds = 15 | Extension: does a full-montage teacher help a small montage |
| Distillation from a label-shuffled teacher | 3 budgets × 5 seeds = 15 | Control: a gain that also appears here is regularisation, not transfer |
| Label-shuffle control | 1 | The pipeline must score at chance on permuted labels |

The model is a compact EEGNet-style CNN retrained per condition, with early
stopping on an inner holdout of the *training* subjects so that the
validation split is used for nothing but choosing k\*.

## Leakage prevention

Each of these is enforced somewhere a test can fail.

- Subject-wise splits: no subject appears in two splits.
- Normalisation statistics and the channel ranking come from training subjects
  only.
- `artifacts/budgets.json` fixes the exact channel set per budget, and a drift
  guard runs before every ranked training run.
- Model selection (early stopping) and budget selection (k\*) never share
  data.
- The test split reaches the analysis only through a report that takes k\* as
  an input.

## Repository layout

```
artifacts/     frozen, generated-once protocol outputs; see artifacts/README.md
config.yaml    the one configuration file; its hash is stamped into every row
manifests/     the experiment, written before it ran
results/       provenance-stamped result rows, one file per shard, plus the run log
src/           library code (loader, splits, ranking, channel selection, model, runner, k*, analysis)
scripts/       command-line entry points; scripts/legacy/ holds superseded prototypes
tests/         pytest suite, including protocol-integrity and leakage tests
notebooks/     the Kaggle sweep notebook
docs/          protocol, reproduction guide, Kaggle runbook, result schema, development log
figures/       data-verification figures
paper/         manuscript drafts
```

## Quick start

```bash
git clone https://github.com/jaw039/min-viable-eeg && cd min-viable-eeg
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # all pass; six loader tests skip until the raw EDFs are downloaded
```

Everything below the sweep works without the raw data, because the frozen
artifacts and the result rows are committed:

```bash
python scripts/audit_results.py --results 'results/*.jsonl'   # protocol audit of every row
python scripts/analyze.py --results 'results/*.jsonl'         # k*, curves, controls, coverage
python scripts/channel_tables.py                              # electrode tables for the paper
```

To rebuild from raw EEG, run the sweep, or verify the frozen artifacts
byte-for-byte, follow [docs/reproducing.md](docs/reproducing.md). A `Makefile`
wraps the common commands (`make test`, `make audit`, `make analyze`).

## Reproducing a paper number

Every number traces back through its result row (schema in
[docs/results_schema.md](docs/results_schema.md)):

1. `config_sha256` names the exact configuration, `git_commit` or the Kaggle
   code-dataset version names the exact code.
2. `channels` is the exact electrode list trained on, verifiable against
   `artifacts/budgets.json`.
3. `train_seed`, `selection_seed`, `split`, `budget_k`, `selection` and
   `training` identify the condition.

`scripts/audit_results.py` performs that trace mechanically for every row: it
re-derives the channel set from the frozen ranking, checks the evaluated
subjects against `artifacts/splits.json`, checks ranking provenance and config
hash, and reports coverage against the manifest. It exits non-zero on any
violation. k\* claims additionally cite `results/kstar_report.json`, which
records the budget curve, per-seed k\*, threshold sensitivity and coverage,
and flags the report as provisional while any budget lacks a planned seed.

The test split is run from `manifests/manifest_test.jsonl`, written by
`scripts/run_sweep.py --write-test-manifest --kstar-report results/kstar_report.json`
(or `--kstar <k>`). It holds the chosen budget and the 64-channel reference
at every planned seed and nothing else, so no other budget is ever evaluated
on test and the retained fraction is computable there. Controls stay on
validation.

## Running the sweep on Kaggle

The runbook, including the datasets, notebook and mount paths the team
actually used, is [docs/kaggle.md](docs/kaggle.md). The short version: one
private dataset holds the preprocessed cache and the frozen artifacts, a
second pins the code to a commit, and `notebooks/kaggle_sweep.ipynb` runs one
shard per saved Version. Only a saved Version's output persists.

## Validation results

Validation split, complete sweep, every number from provenance-stamped rows
(`results/kstar_report.json`). The test result follows in the next section.

- **The negative control passes.** Permuted labels give κ = 0.007.
- **k\*(0.90) = 32 on validation, and it is not stable.** κ_full is
  0.443 ± 0.063 over five seeds and κ at 32 electrodes is 0.420 ± 0.040.
  Per seed, k\* is 16, 32, 64, 32 and 64; across thresholds it is 32 at
  τ = 0.85 and 0.90 and 64 at 0.95. The analysis reports the budget curve
  with its uncertainty as the result rather than the integer.
- **The frozen ranked k=4 set (C4, CP4, C6, CP6) scores at chance** (κ =
  0.002). All four are adjacent right-hemisphere sites, so the set carries
  no left/right contrast. Univariate ranking selects redundant neighbours;
  this is a finding about the selection method.
- **Ranked never beats random significantly.** At every budget the ranked
  set sits inside the distribution of 20 random subsets; the smallest
  empirical p is 0.14 (k = 6 and k = 32) against a floor of 0.048.
- **The sensorimotor restriction is mixed by budget**: the motor strip wins
  at k = 4 and 8, the unrestricted ranking at 6 and 16, and they agree at 12.
- **Distillation helps only at k = 4** (+0.031 over scratch, with the
  shuffled-label teacher at −0.004), and not at 6 or 8.
- **Subjects differ widely**: per-subject κ at 64 channels ranges 0.21–0.90.

## Limitations

- **Subject heterogeneity.** Most training subjects share no electrode with
  the shared top-4, and with about 45 trials per subject genuine heterogeneity
  cannot be separated from estimation noise.
- **Possible cue or gaze contamination.** C3 enters the ranking only at rank
  23 while frontal and occipital channels outrank it. The sensorimotor arm
  addresses this; a closed-loop replication would settle it.
- **The small-k montage is not uniquely determined.** Only C4 is stable under
  bootstrap resampling; its three companions are interchangeable with frontal
  and occipital sites.
- **One dataset, one paradigm, offline evaluation.** κ here is an upper bound
  on closed-loop performance and may be a property of EEGMMIDB as much as of
  scalp EEG.
- **EEGNet-style, not a faithful port.** No max-norm constraints; kernel
  length 64 rather than fs/2.

## Test result (confirmatory, evaluated once)

The test split was run once, at the validation-chosen k\* = 32 and the
64-channel reference, five training seeds each
(`results/shard_0_of_1_test.jsonl`; `results/kstar_report.json`, section
`test_report`).

| Budget | Test κ (mean ± sd over 5 seeds) |
| --- | --- |
| 32 (k\*) | 0.292 ± 0.061 |
| 64 (full montage) | 0.326 ± 0.032 |

- **32 electrodes retain 0.897 of the full-montage κ on test**, just below
  the 0.90 target k\* was chosen against. Per matched seed the ratio runs
  from 0.76 to 1.11 (mean 0.89 ± 0.14): a point estimate with a wide
  interval, in line with the validation verdict that k\* is unstable.
- **The test subjects are harder.** κ at 64 channels is 0.326 on test
  against 0.443 on validation; per-subject κ spans 0.06–0.66 (median 0.28).
  The budget claim is about the retained fraction, not the absolute κ.
- No other budget was evaluated on test, so these rows allow no post-hoc
  choice of k\*.

## Open items before submission

- `reduction_mode: reduce` and the 17-electrode sensorimotor pool are not on
  the locked protocol list and need sign-off; both are marked `[UNRESOLVED]`
  where they appear.
- The validation rows from Kaggle record `git_commit: "unknown"`: `mve-code`
  v1 predates the `COMMIT` file the bundle now places in the snapshot, so
  their pin is the code-dataset version (v1 = `e35a506`). Later rows stamp
  the commit from that file.
- Choose a licence.

## Citation

Dataset:

> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R.
> (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
> *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.

> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
> *Circulation*, 101(23), e215–e220.

Architecture:

> Lawhern, V.J., Solon, A.J., Waytowich, N.R., Gordon, S.M., Hung, C.P., Lance,
> B.J. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based
> Brain-Computer Interfaces. *Journal of Neural Engineering*, 15(5), 056013.

This repository: see `CITATION.cff`.

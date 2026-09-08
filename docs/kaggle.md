# Running the sweep on Kaggle

The README's "Running on Kaggle" section is the summary. This is the runbook.

Budget: **30 GPU-hours per person per week**, resetting **Saturday 00:00 UTC**.
Four accounts is far more than this sweep needs. The Algoverse compute policy
requires free tiers before anything paid, so this is also the path that keeps
you eligible for AWS credit or an A100 grant later.

---

## A · Once per person (5 minutes)

1. Create a free account at [kaggle.com](https://www.kaggle.com).
2. **Settings → Phone verification.** This is what unlocks GPUs *and* internet
   access; without it the accelerator menu stays greyed out. If SMS fails (VoIP
   numbers, some regions), use the manual activation form linked there.
3. Send your Kaggle username to whoever creates the dataset in step B.

## B · Once for the team

### The data

Whoever has `data/processed/` populated uploads it as a **private** dataset:

- Build the zip with `python scripts/make_kaggle_bundle.py --cache`. It
  contains `cache/processed/S###/{X,y}.npy` plus `cache/splits.json`,
  `cache/channel_ranking.json` and `cache/budgets.json` (copied from
  `artifacts/`). Datasets → New Dataset → upload it. Kaggle unzips archives
  on its side and keeps the zip's top-level folder, which is why the mount
  path ends in `/cache`.
- Name it something stable, e.g. `mve-eegmmidb-cache`. Keep it **Private** and
  add the other three as collaborators, so every account reads the *same
  versioned copy* and no number depends on whose local cache produced it.
- ~0.8 GB against a 200 GB private-dataset limit.
- Note the dataset **version number** and pin it in every run.

### The code

Either option works; pick one and be consistent.

- **Zip snapshot** (simplest, what the team uses):
  `python scripts/make_kaggle_bundle.py --code` writes a `git archive` of
  HEAD (it refuses a dirty tree) under a `code/` folder; upload it as a
  second private dataset (`mve-code`). The dataset version pins the code.
  Result rows from a snapshot record `git_commit: "unknown"` because the
  archive has no `.git`; the dataset version number is the pin.
- **Git clone**: Add-ons → Secrets → add `GITHUB_TOKEN` (a fine-grained,
  read-only PAT for this repo), then clone and `git checkout <full-sha>` in the
  notebook.

Never run off "latest main". A sweep whose code moved mid-run is not one
experiment.

### The manifest

Generate it once and commit it, so the experiment is written down before it
runs:

```bash
python scripts/run_sweep.py --write-manifest
```

That writes `manifests/manifest_val.jsonl` — 691 conditions at the configured
5 seeds and 20 random subsets. Everyone runs a shard of the same file.

## C · Per run

1. New Notebook → upload `notebooks/kaggle_sweep.ipynb`.
2. Settings pane:
   - **Accelerator → GPU T4 ×2.** T4 ×2 and P100 draw from the same 30-hour
     meter, so T4 ×2 gives two GPUs per quota-hour.
   - **Environment → Original.** "Latest" silently upgrades packages
     mid-project, and `mne==1.8.0` breaks against newer scipy.
   - Internet on (only needed if cloning with a token).
3. Add Input → attach `mve-eegmmidb-cache` and `mve-code`.
4. Set `SHARD_ID` in the config cell. Person *i* takes shard *i*.
5. Run through the smoke test and **stop there**. Post the projected hours to
   the channel. If one condition is seconds, carry on; if it is an hour,
   re-plan before spending anyone's quota.
6. Set `SHARD_ID`, then **Save Version → Save & Run All (Commit)**. That runs
   the whole notebook headless as "Version N" and persists `/kaggle/working`
   as that version's output. Do **not** run a shard with the toolbar's Run
   All: an interactive session's output is discarded when the session ends.
   Completed conditions are skipped on a re-run, but only if the earlier
   rows are attached as input.

**Stop the session from Active Events when you finish.** The meter runs while a
session is open even when idle, and closing the browser tab does not stop it.

## D · Merging and analysing

Each saved Version writes `/kaggle/working/results/shard_<i>_of_<n>_val.jsonl`,
which persists as that version's output (20 GB limit), can be attached as
*input* to another notebook, and downloads with
`kaggle kernels output <owner>/<notebook> -p <dir>`. Whoever does the analysis attaches all four, or downloads
them, then:

```bash
python scripts/analyze.py --results 'results/*.jsonl' --emit-followup
```

Before that, audit the rows. The audit re-derives every row's channel set,
split membership and ranking provenance from the frozen artifacts and exits
non-zero on any protocol violation:

```bash
python scripts/audit_results.py --results 'results/*.jsonl'
```

`analyze.py` prints the negative control first (and warns while no
label-shuffle row exists yet), then k\*, the budget curve, ranked-vs-random
with its resolution floor, the sensorimotor comparison, distillation, subject
heterogeneity, and a coverage report. It writes `results/kstar_report.json`.

Run the test split **once**, at the k\* the validation curve chose:

```bash
python scripts/run_sweep.py --write-test-manifest --kstar-report results/kstar_report.json
cat manifests/manifest_test.jsonl        # k* and k=64, ranked, scratch, every planned seed; nothing else
python scripts/run_sweep.py --manifest --split test --shard-id 0 --num-shards 1
python scripts/audit_results.py --results 'results/*.jsonl'
python scripts/analyze.py --results 'results/*.jsonl' --test-report
```

Writing a test manifest through `--write-manifest` is refused on purpose: the
test manifest comes only from `--write-test-manifest` (or `--kstar <k>`),
which holds k\* and the 64-channel reference and nothing else, so the
retained fraction is computable on test and no other budget is evaluated
there. Controls (random, sensorimotor, distillation, label shuffle) stay on
validation.

---

## Gotchas that cost a day

- **The quota meter runs on idle sessions.** The single most common way free
  hours disappear. Active Events panel, bottom left.
- **Only a saved Version keeps its output.** "Save & Run All (Commit)" runs
  a fresh headless copy in its own session, alongside any interactive session
  you still have open. Stop the *interactive* session from Active Events to
  save quota; never cancel the Version. Shard 1 was lost once, 55 minutes in,
  by cancelling the wrong one.
- **Sessions cap at 12 hours** and `/kaggle/working` is wiped when the session
  ends — which is why the runner is resumable.
- **No shell.** Prefix shell commands with `!` in a cell.
- **Changing the accelerator restarts the session** and clears your variables.
- **Do not edit `config.yaml` mid-sweep.** Every ratio in the paper divides by
  κ_full; a config change between the full-montage runs and the reduced-budget
  runs makes those ratios incomparable. The config hash in each row will show
  it, but by then the runs are wasted.

## What not to do

**Do not put this sweep on the Algoverse A100.** It is free and available, but
grants are time-boxed to 1–2 days and *everything on the machine is deleted at
expiry* — the wrong home for provenance-stamped result rows. EEGNet on 4,750
trials has no use for 40 GB of VRAM either. Keep the A100 in reserve for
genuine overflow, and treat the L4 backup machines the same way.

---

## Running without Kaggle

The notebook is a thin wrapper. Everything works locally:

```bash
python scripts/run_sweep.py --smoke-test
python scripts/run_sweep.py --budget 8 --selection ranked --split val
python scripts/run_sweep.py --budget 4 --selection random --selection-seed 3
python scripts/run_sweep.py --budget 6 --selection sensorimotor
python scripts/run_sweep.py --budget 4 --training distill
python scripts/run_sweep.py --budget 64 --shuffle-labels     # expect κ ≈ 0
python scripts/run_sweep.py --manifest --shard-id 0 --num-shards 4
```

`MVE_DATA_ROOT` points the runner at a cache outside the repo:

```bash
MVE_DATA_ROOT=/kaggle/input/datasets/<owner>/mve-eegmmidb-cache/cache/processed python scripts/run_sweep.py --smoke-test
```

---

## E · What the team actually ran

Recorded so that the sweep can be re-executed, or its rows traced, without
anyone's memory.

| Item | Value |
| --- | --- |
| Cache dataset | `jackiewang2323/mve-eegmmidb-cache`, version 1, 724 MB, uploaded 2026-09-05 |
| Code dataset | `jackiewang2323/mve-code`, version 1 = commit `e35a506`, uploaded 2026-09-05 |
| Notebook | `jackiewang2323/notebookeb3bd89e45` (this repository's `notebooks/kaggle_sweep.ipynb`) |
| Accelerator | GPU T4 ×2 |
| Image | `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` |
| Environment stamped in rows | `python3.12.13 torch2.10.0+cu128 numpy2.0.2` |
| Config hash stamped in rows | `af8fbd091329` |
| Mount paths | `/kaggle/input/datasets/jackiewang2323/mve-eegmmidb-cache/cache`, `/kaggle/input/datasets/jackiewang2323/mve-code/code` |

Per-shard outcomes (version numbers, wall time, row counts, file hashes) are
in [results/README.md](../results/README.md).

Checking and pulling a shard from the command line (Kaggle CLI 2.x needs
Python ≥ 3.11 and a token at `~/.kaggle/access_token`):

```bash
kaggle kernels status jackiewang2323/notebookeb3bd89e45
kaggle kernels output jackiewang2323/notebookeb3bd89e45 -p /tmp/kaggle_out
cp /tmp/kaggle_out/results/shard_2_of_4_val.jsonl results/
python scripts/audit_results.py --results 'results/*.jsonl'
python scripts/analyze.py --results 'results/*.jsonl'
```

The notebook in this repository matches the current layout (`artifacts/`).
`mve-code` v1 predates that move, so all four validation shards run from
v1 unchanged; a later run (the test split) needs a new dataset version
built with `scripts/make_kaggle_bundle.py --code` from the merged commit.

Two rules learned the hard way: editing a notebook cell does not change the
kernel's variables until the cell is re-run, and only a saved Version's
output survives the end of a session.

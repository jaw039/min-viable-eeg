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

- Datasets → New Dataset → upload a zip containing `processed/` plus
  `splits.json`, `channel_ranking.json` and `budgets.json`. Kaggle unzips
  archives on its side.
- Name it something stable, e.g. `mve-eegmmidb-cache`. Keep it **Private** and
  add the other three as collaborators, so every account reads the *same
  versioned copy* and no number depends on whose local cache produced it.
- ~0.8 GB against a 200 GB private-dataset limit.
- Note the dataset **version number** and pin it in every run.

### The code

Either option works; pick one and be consistent.

- **Zip snapshot** (simplest): zip the repo at a specific commit, upload as a
  second private dataset (`mve-code`). The dataset version pins the code.
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
6. Run the sweep cell. It is resumable — completed conditions are skipped, so a
   12-hour session limit or a dropped connection costs nothing.

**Stop the session from Active Events when you finish.** The meter runs while a
session is open even when idle, and closing the browser tab does not stop it.

## D · Merging and analysing

Each notebook writes `/kaggle/working/results/shard_<i>_of_<n>_val.jsonl`,
which persists as notebook output (20 GB limit) and can be attached as *input*
to another notebook. Whoever does the analysis attaches all four, or downloads
them, then:

```bash
python scripts/analyze.py --results 'results/*.jsonl' --emit-followup
```

That prints the negative control first — if the label-shuffle floor is not near
zero, it stops, because nothing else is interpretable until that is explained.
Then k\*, the budget curve, ranked-vs-random with its resolution floor, the
sensorimotor comparison, distillation, subject heterogeneity, and a coverage
report. It writes `kstar_report.json`.

Run the test split **once**, at the k\* the validation curve chose:

```bash
python scripts/run_sweep.py --write-manifest --split test --budget <k*>
python scripts/run_sweep.py --manifest --split test --shard-id 0 --num-shards 1
python scripts/analyze.py --results 'results/*.jsonl' --test-report
```

`--write-manifest --split test` refuses to run without `--budget`, on purpose.

---

## Gotchas that cost a day

- **The quota meter runs on idle sessions.** The single most common way free
  hours disappear. Active Events panel, bottom left.
- **"Save & Run All" starts a *separate* session.** Two concurrent sessions
  burn two meters. Cancel duplicates.
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
MVE_DATA_ROOT=/kaggle/input/mve-eegmmidb-cache/processed python scripts/run_sweep.py --smoke-test
```

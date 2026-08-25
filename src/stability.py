"""Channel-stability check: do the same electrodes stay important when the
training subjects are resampled, and how much does each subject's own
ranking agree with the shared montage?

Run as: python -m src.stability [--n-bootstrap 200] [--seed 42]

Uses TRAIN-split subjects only, via the same per-subject Fisher scores the
shared ranking is built from (src.ranking.train_subject_scores). Two views:

  bootstrap    Resample the train subjects with replacement n_bootstrap times,
               rebuild the shared ranking each time, and record how often each
               channel lands in the top-k for every budget k. Frequency near 1
               means the frozen top-k does not depend on which subjects were
               drawn; frequency near 0.5 means it is a coin flip.
  per_subject  Rank channels for each subject from its OWN scores alone, and
               record (a) how often each channel is in a subject's personal
               top-k and (b) the overlap |personal top-k ∩ shared top-k| / k.
               This is the subject-specific vs shared-montage comparison.

Writes stability.json at the repo root (deterministic given --seed; may be
regenerated freely — nothing downstream selects channels from it) and prints
a summary table. Frozen inputs: channel_ranking.json, budgets in config.yaml.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.ranking import RANKING_PATH, aggregate_ranking, train_subject_scores
from src.utils import REPO_ROOT, load_config, provenance

STABILITY_PATH = REPO_ROOT / "stability.json"
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_SEED = 42


def _topk(scores: np.ndarray, ch_names: List[str], k: int) -> List[str]:
    """Top-k channel names from one score vector (same tie-break as ranking)."""
    ranked, _ = aggregate_ranking([scores], ch_names)
    return ranked[:k]


def bootstrap_topk_frequency(
    scores: np.ndarray,
    ch_names: List[str],
    budgets: Sequence[int],
    n_bootstrap: int,
    seed: int,
) -> Dict[int, Dict[str, float]]:
    """For each budget k: fraction of bootstrap resamples (over subjects) in
    which each channel appears in the recomputed shared top-k.

    scores: (n_subjects, n_channels) per-subject Fisher scores.
    """
    if scores.ndim != 2 or scores.shape[1] != len(ch_names):
        raise ValueError("scores must be (n_subjects, {})".format(len(ch_names)))
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    rng = np.random.default_rng(seed)
    n_subj = scores.shape[0]
    counts = {k: np.zeros(len(ch_names), dtype=np.int64) for k in budgets}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_subj, size=n_subj)
        mean_scores = scores[idx].mean(axis=0)
        for k in budgets:
            for ch in _topk(mean_scores, ch_names, k):
                counts[k][ch_names.index(ch)] += 1
    return {
        int(k): {ch: float(counts[k][i]) / n_bootstrap for i, ch in enumerate(ch_names)}
        for k in budgets
    }


def per_subject_topk(
    scores: np.ndarray,
    ch_names: List[str],
    budgets: Sequence[int],
    shared_ranked: List[str],
) -> Dict[int, dict]:
    """For each budget k: how often each channel is in a subject's personal
    top-k, and the per-subject overlap with the shared top-k.
    """
    if scores.ndim != 2 or scores.shape[1] != len(ch_names):
        raise ValueError("scores must be (n_subjects, {})".format(len(ch_names)))
    if sorted(shared_ranked) != sorted(ch_names):
        raise ValueError("shared_ranked must be a permutation of ch_names")
    out = {}
    for k in budgets:
        shared = set(shared_ranked[:k])
        counts = np.zeros(len(ch_names), dtype=np.int64)
        overlaps = []
        for s in range(scores.shape[0]):
            own = _topk(scores[s], ch_names, k)
            for ch in own:
                counts[ch_names.index(ch)] += 1
            overlaps.append(len(shared.intersection(own)) / float(k))
        overlaps_arr = np.array(overlaps)
        out[int(k)] = {
            "frequency": {
                ch: float(counts[i]) / scores.shape[0] for i, ch in enumerate(ch_names)
            },
            "overlap": {
                "mean": float(overlaps_arr.mean()),
                "std": float(overlaps_arr.std()),
                "min": float(overlaps_arr.min()),
                "max": float(overlaps_arr.max()),
                "per_subject": [float(v) for v in overlaps],
            },
        }
    return out


def summarize(
    shared_ranked: List[str],
    boot: Dict[int, Dict[str, float]],
    per_subj: Dict[int, dict],
    budgets: Sequence[int],
) -> str:
    """Plain-text table: for each budget, the frozen set with both frequencies."""
    lines = []
    for k in budgets:
        if k >= len(shared_ranked):
            continue
        ov = per_subj[k]["overlap"]
        lines.append(
            "k={:>2}  mean personal/shared overlap {:.2f} (min {:.2f}, max {:.2f})".format(
                k, ov["mean"], ov["min"], ov["max"]
            )
        )
        lines.append("      {:<6} {:>10} {:>12}".format("chan", "bootstrap", "per-subject"))
        for ch in shared_ranked[:k]:
            lines.append(
                "      {:<6} {:>10.2f} {:>12.2f}".format(
                    ch, boot[k][ch], per_subj[k]["frequency"][ch]
                )
            )
    return "\n".join(lines)


def write_stability(
    config: dict,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    ranking_path: Path = RANKING_PATH,
    out_path: Path = STABILITY_PATH,
) -> None:
    with open(ranking_path) as f:
        ranking = json.load(f)
    shared_ranked = ranking["channels"]

    scores, ch_names, train_subjects = train_subject_scores(config)
    if sorted(ch_names) != sorted(shared_ranked):
        raise ValueError("Ranking channels do not match loader montage channels")
    budgets = [k for k in config["budgets"] if k < len(ch_names)]  # full montage is trivial

    boot = bootstrap_topk_frequency(scores, ch_names, budgets, n_bootstrap, seed)
    per_subj = per_subject_topk(scores, ch_names, budgets, shared_ranked)

    payload = {
        "provenance": provenance(ignore_paths=[out_path]),
        "ranking_provenance": ranking["provenance"],
        "n_train_subjects": len(train_subjects),
        "train_subjects": train_subjects,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "budgets": budgets,
        "shared_ranked": shared_ranked,
        "bootstrap_topk_frequency": {str(k): v for k, v in boot.items()},
        "per_subject": {str(k): v for k, v in per_subj.items()},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print("Wrote {} ({} bootstrap resamples, {} train subjects)".format(
        out_path, n_bootstrap, len(train_subjects)))
    print(summarize(shared_ranked, boot, per_subj, budgets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    write_stability(load_config(), n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()

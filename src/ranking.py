"""Channel ranking from train-split subjects only (leakage-free).

Run as: python -m src.ranking

Writes channel_ranking.json at the repo root with the same generated-once
policy as splits.json (refuses to overwrite; delete manually to regenerate —
regeneration is deterministic).

Method: fisher_log_bandpower_subject_mean. Trials are already bandpassed to
the protocol band (8-30 Hz), so each trial/channel log-variance is its band
log-power. Per subject, each channel scores
(mean_left - mean_right)^2 / (var_left + var_right) over that subject's
trials; the final score is the mean across train subjects. Ties break by
original channel index, so the ranking is fully deterministic.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.utils import REPO_ROOT, data_root, load_config, provenance

RANKING_PATH = REPO_ROOT / "channel_ranking.json"
SPLITS_PATH = REPO_ROOT / "splits.json"

METHOD = "fisher_log_bandpower_subject_mean"
_EPS = 1e-12  # keeps the Fisher denominator finite on degenerate input


def fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-channel Fisher score of log-bandpower for ONE subject.

    X: (n_trials, n_channels, n_samples) bandpassed epochs; y: 0/1 labels.
    Returns float64 (n_channels,).
    """
    if X.ndim != 3:
        raise ValueError("Expected (n_trials, n_channels, n_samples), got {}".format(X.shape))
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        raise ValueError("Need both classes (0 and 1) present, got {}".format(classes))
    # Band-limited signal -> per-trial variance is bandpower in the band.
    log_power = np.log(X.astype(np.float64).var(axis=2))  # (n_trials, n_channels)
    left = log_power[y == 0]
    right = log_power[y == 1]
    num = (left.mean(axis=0) - right.mean(axis=0)) ** 2
    den = left.var(axis=0) + right.var(axis=0) + _EPS
    return num / den


def aggregate_ranking(
    per_subject_scores: List[np.ndarray], ch_names: List[str]
) -> Tuple[List[str], np.ndarray]:
    """Mean score across subjects -> channels ranked best-first.

    Returns (ranked channel names, their scores in ranked order).
    """
    scores = np.mean(np.stack(per_subject_scores, axis=0), axis=0)
    if scores.shape != (len(ch_names),):
        raise ValueError(
            "Scores shape {} does not match {} channels".format(scores.shape, len(ch_names))
        )
    # Descending by score; stable tie-break by original channel index.
    order = np.lexsort((np.arange(len(ch_names)), -scores))
    return [ch_names[i] for i in order], scores[order]


def compute_ranking(config: dict) -> Tuple[List[str], np.ndarray, List[int]]:
    """Rank channels using cached epochs of the TRAIN split only."""
    with open(SPLITS_PATH) as f:
        train_subjects = json.load(f)["train"]

    # Channel names are protocol-constant; read them once via the loader.
    from src.loader import load_subject

    _, _, ch_names = load_subject(train_subjects[0], config)

    processed = data_root(config) / "processed"
    per_subject = []
    for subject in train_subjects:
        sub_dir = processed / "S{:03d}".format(subject)
        X = np.load(sub_dir / "X.npy")
        y = np.load(sub_dir / "y.npy")
        if X.shape[1] != len(ch_names):
            raise ValueError("S{:03d} cache has {} channels".format(subject, X.shape[1]))
        per_subject.append(fisher_scores(X, y))
    ranked, scores = aggregate_ranking(per_subject, ch_names)
    return ranked, scores, train_subjects


def write_ranking(config: dict, out_path: Path = RANKING_PATH) -> None:
    if out_path.exists():
        print("REFUSING to overwrite existing {}".format(out_path))
        print("The ranking feeds the headline result and is generated once. If you")
        print("really intend to regenerate it, delete the file manually first:")
        print("    rm {}".format(out_path))
        raise SystemExit(1)
    ranked, scores, train_subjects = compute_ranking(config)
    payload = {
        "provenance": provenance(),
        "method": METHOD,
        "band_hz": config["preprocess"]["bandpass"],
        "n_train_subjects": len(train_subjects),
        "channels": ranked,
        "scores": [float(s) for s in scores],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print("Wrote {} ({} channels, {} train subjects)".format(out_path, len(ranked), len(train_subjects)))
    print("Top 16: {}".format(ranked[:16]))


def main() -> None:
    write_ranking(load_config())


if __name__ == "__main__":
    main()

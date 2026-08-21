"""Data-verification figures for the processed EEGMMIDB cache.

Run as: .venv/bin/python scripts/make_figures.py

Writes to figures/:
  sample_trials.png  Four raw preprocessed trials from S001 (left vs right
                     fist, at C3 and C4) to eyeball amplitude scale and band.
  erd_check.png      Contralateral ERD check across all cached subjects:
                     per-subject log-bandpower difference at C3 (right minus
                     left trials) and C4 (left minus right). Correct labels
                     predict both distributions shift negative.

Requires matplotlib (not in the pinned requirements.txt; install with
.venv/bin/pip install matplotlib).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loader import load_subject
from src.utils import REPO_ROOT, data_root, load_config

FIG_DIR = REPO_ROOT / "figures"


def sample_trials_figure(config: dict) -> None:
    X, y, ch = load_subject(1, config)
    c3, c4 = ch.index("C3"), ch.index("C4")
    t = np.arange(X.shape[2]) / float(config["dataset"]["sfreq"])
    left_i = int(np.where(y == 0)[0][0])
    right_i = int(np.where(y == 1)[0][0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    rows = [(left_i, "LEFT fist (y=0)", "tab:blue"), (right_i, "RIGHT fist (y=1)", "tab:red")]
    cols = [(c3, "C3 (left motor cortex)"), (c4, "C4 (right motor cortex)")]
    for r, (trial, label, color) in enumerate(rows):
        for c, (idx, name) in enumerate(cols):
            ax = axes[r][c]
            ax.plot(t, X[trial, idx] * 1e6, lw=0.7, color=color)
            ax.set_title("Trial {} — {} — {}".format(trial, label, name), fontsize=10)
            ax.grid(alpha=0.3)
    axes[1][0].set_xlabel("Time after cue (s)")
    axes[1][1].set_xlabel("Time after cue (s)")
    axes[0][0].set_ylabel("µV")
    axes[1][0].set_ylabel("µV")
    fig.suptitle("S001 — raw preprocessed trials (8–30 Hz bandpassed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_trials.png", dpi=130)
    plt.close(fig)


def erd_check_figure(config: dict) -> None:
    _, _, ch = load_subject(1, config)
    c3, c4 = ch.index("C3"), ch.index("C4")
    processed = data_root(config) / "processed"
    subjects = sorted(processed.glob("S*"))

    d_c3, d_c4 = [], []
    for sub in subjects:
        X = np.load(sub / "X.npy")
        y = np.load(sub / "y.npy")
        lp = np.log(X.astype(np.float64).var(axis=2))  # (trials, 64) log-bandpower
        d_c3.append(lp[y == 1, c3].mean() - lp[y == 0, c3].mean())
        d_c4.append(lp[y == 0, c4].mean() - lp[y == 1, c4].mean())
    d_c3, d_c4 = np.array(d_c3), np.array(d_c4)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(d_c3, bins=30, alpha=0.6, label="C3: right-trials minus left-trials")
    ax.hist(d_c4, bins=30, alpha=0.6, label="C4: left-trials minus right-trials")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Log-bandpower difference (negative = expected ERD direction)")
    ax.set_ylabel("Number of subjects (of {})".format(len(subjects)))
    ax.set_title("Contralateral ERD check across all cached subjects")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "erd_check.png", dpi=130)
    plt.close(fig)

    n = len(subjects)
    print("C3 suppression during right-fist imagery: {}/{} subjects".format(int((d_c3 < 0).sum()), n))
    print("C4 suppression during left-fist imagery:  {}/{} subjects".format(int((d_c4 < 0).sum()), n))


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    config = load_config()
    sample_trials_figure(config)
    erd_check_figure(config)
    print("Wrote {}".format(FIG_DIR / "sample_trials.png"))
    print("Wrote {}".format(FIG_DIR / "erd_check.png"))


if __name__ == "__main__":
    main()

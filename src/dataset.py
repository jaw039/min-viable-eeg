"""Subject-indexed split assembly from the preprocessed cache.

Everything downstream needs three things kept together: the trials, the labels,
and which subject each trial came from. Subject identity is carried all the way
to the metrics so per-subject kappa stays recoverable -- BCI populations are
bimodal, and a mean over 16 test subjects can describe nobody.

Nothing here touches an EDF or imports mne; the experiment path reads only the
cached arrays, so a Kaggle run needs the cache and the frozen artifacts and
nothing else.
"""

import os
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

import numpy as np

from src.utils import REPO_ROOT


class SplitData(NamedTuple):
    X: np.ndarray            # (n_trials, n_channels, n_samples) float32
    y: np.ndarray            # (n_trials,) int64
    subjects: np.ndarray     # (n_trials,) int32 -- subject id per trial
    subject_ids: List[int]   # unique subjects, in split order


def data_root() -> Path:
    """Where the preprocessed per-subject cache lives.

    MVE_DATA_ROOT lets a Kaggle notebook point at a mounted dataset without
    editing code or copying the cache into the repo.
    """
    env = os.environ.get("MVE_DATA_ROOT")
    return Path(env) if env else REPO_ROOT / "data" / "processed"


def subject_dir(subject: int, root: Optional[Path] = None) -> Path:
    return (root or data_root()) / "S{:03d}".format(int(subject))


def available_subjects(root: Optional[Path] = None) -> List[int]:
    """Subjects with a complete cache entry, ascending."""
    root = root or data_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("S"):
            if (d / "X.npy").exists() and (d / "y.npy").exists():
                try:
                    out.append(int(d.name[1:]))
                except ValueError:
                    continue
    return out


def load_subject_cache(subject: int, root: Optional[Path] = None):
    d = subject_dir(subject, root)
    x_path, y_path = d / "X.npy", d / "y.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Missing cache for subject {} at {}.\n"
            "Run scripts/cache_preprocessed.py, or set MVE_DATA_ROOT to a "
            "mounted copy (e.g. /kaggle/input/<dataset>/processed).".format(subject, d)
        )
    X = np.load(x_path)
    y = np.load(y_path)
    if len(X) != len(y):
        raise ValueError(
            "Cache for S{:03d} has {} trials but {} labels".format(
                int(subject), len(X), len(y)
            )
        )
    return X, y


def load_split(subjects: Sequence[int], root: Optional[Path] = None) -> SplitData:
    """Concatenate cached trials for these subjects, keeping subject identity."""
    if not subjects:
        raise ValueError("No subjects given")
    xs, ys, ss = [], [], []
    for s in subjects:
        X, y = load_subject_cache(s, root)
        xs.append(X)
        ys.append(y)
        ss.append(np.full(len(y), int(s), dtype=np.int32))
    return SplitData(
        X=np.concatenate(xs).astype(np.float32),
        y=np.concatenate(ys).astype(np.int64),
        subjects=np.concatenate(ss),
        subject_ids=[int(s) for s in subjects],
    )


def trials_per_subject(data: SplitData) -> dict:
    return {
        "S{:03d}".format(int(s)): int((data.subjects == s).sum())
        for s in sorted(set(data.subjects.tolist()))
    }

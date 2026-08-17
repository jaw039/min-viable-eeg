"""Cache preprocessed epochs for every downloaded, non-excluded subject.

Usage: python scripts/cache_preprocessed.py

Runs the loader per subject and saves data/processed/S###/X.npy (float32)
and y.npy. Skips subjects already cached and subjects whose EDFs are not
downloaded yet (printing which ones). Appends provenance to
data/processed/cache_log.jsonl.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.loader import load_subject
from src.utils import N_SUBJECTS, data_root, edf_path, load_config, provenance


def main() -> None:
    config = load_config()
    excluded = set(config["dataset"]["exclude_subjects"])
    runs = config["dataset"]["runs"]
    processed_root = data_root(config) / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    cached = []
    already = []
    missing = []
    for subject in range(1, N_SUBJECTS + 1):
        if subject in excluded:
            continue
        out_dir = processed_root / "S{:03d}".format(subject)
        x_path = out_dir / "X.npy"
        y_path = out_dir / "y.npy"
        if x_path.exists() and y_path.exists():
            already.append(subject)
            continue
        if not all(edf_path(config, subject, run).exists() for run in runs):
            missing.append(subject)
            continue
        X, y, _ = load_subject(subject, config)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(x_path, X)  # float32 straight from the loader
        np.save(y_path, y)
        cached.append(subject)
        print("cached S{:03d}: X{} y{}".format(subject, X.shape, y.shape))

    if already:
        print("already cached ({}): {}".format(len(already), already))
    if missing:
        print("EDFs not downloaded yet ({}): {}".format(len(missing), missing))

    log_entry = provenance()
    log_entry.update({"cached": cached, "already_cached": already, "missing": missing})
    log_path = processed_root / "cache_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print("Logged provenance to {}".format(log_path))


if __name__ == "__main__":
    main()

"""Subject-wise train/val/test splits, generated once (see CLAUDE.md).

Run as: python -m src.splits

Writes splits.json at the repo root and refuses to overwrite an existing
file. Operates on the subject ID list only — no EEG data required.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils import N_SUBJECTS, REPO_ROOT, load_config, provenance

SPLITS_PATH = REPO_ROOT / "splits.json"


def make_splits(config: dict) -> Dict[str, List[int]]:
    """Deterministic subject-wise split of IDs 1..109 minus exclusions."""
    ratios = config["splits"]["ratios"]
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("splits.ratios must sum to 1, got {}".format(ratios))
    excluded = set(config["dataset"]["exclude_subjects"])
    subjects = [s for s in range(1, N_SUBJECTS + 1) if s not in excluded]
    rng = np.random.default_rng(config["splits"]["seed"])
    shuffled = [subjects[i] for i in rng.permutation(len(subjects))]
    n = len(shuffled)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def write_splits(config: dict, out_path: Path = SPLITS_PATH) -> None:
    if out_path.exists():
        print("REFUSING to overwrite existing {}".format(out_path))
        print("The protocol locks splits once generated (CLAUDE.md). If you really")
        print("intend to regenerate them, delete the file manually first:")
        print("    rm {}".format(out_path))
        raise SystemExit(1)
    splits = make_splits(config)
    payload = {
        "provenance": provenance(),
        "strategy": config["splits"]["strategy"],
        "seed": config["splits"]["seed"],
        "ratios": config["splits"]["ratios"],
        "n_subjects": sum(len(v) for v in splits.values()),
    }
    payload.update(splits)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        "Wrote {} (train={}, val={}, test={})".format(
            out_path, len(splits["train"]), len(splits["val"]), len(splits["test"])
        )
    )


def main() -> None:
    write_splits(load_config())


if __name__ == "__main__":
    main()

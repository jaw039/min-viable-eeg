"""Fingerprints of the inputs a result depends on: frozen artifacts and the cache.

Used to key teacher checkpoints, so a cached teacher is reused only when the
data, the split, the ranking and the configuration that produced it are the
ones in force now (see src.checkpoints).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.dataset import data_root
from src.utils import ARTIFACTS_DIR

ARTIFACT_FILES = {
    "splits_sha256": "splits.json",
    "ranking_sha256": "channel_ranking.json",
    "budgets_sha256": "budgets.json",
    "stability_sha256": "stability.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(repo_root: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """sha256 of each frozen artifact; None for one that is missing."""
    base = Path(repo_root) / "artifacts" if repo_root is not None else ARTIFACTS_DIR
    return {
        key: (sha256_file(base / name) if (base / name).exists() else None)
        for key, name in ARTIFACT_FILES.items()
    }


def cache_identity(cfg: Optional[dict] = None, root: Optional[Path] = None) -> Dict[str, object]:
    """Shape-level fingerprint of the preprocessed cache.

    Every cached subject with its array shapes, dtypes and byte sizes, read
    from the .npy headers only, so it costs milliseconds and changes whenever
    the cache is regenerated with different trials, channels or samples. It
    does not hash the samples themselves.
    """
    root = Path(root) if root is not None else data_root()
    entries = []
    for d in sorted(root.glob("S*")):
        x, y = d / "X.npy", d / "y.npy"
        if not (x.exists() and y.exists()):
            continue
        xa = np.load(x, mmap_mode="r")
        ya = np.load(y, mmap_mode="r")
        entries.append([d.name, list(xa.shape), str(xa.dtype), list(ya.shape), str(ya.dtype),
                        x.stat().st_size, y.stat().st_size])
    blob = json.dumps(entries, separators=(",", ":")).encode()
    name = ((cfg or {}).get("dataset") or {}).get("name", "eegmmidb")
    return {
        "dataset_name": name,
        "cache_manifest_sha256": hashlib.sha256(blob).hexdigest(),
        "n_subjects_cached": len(entries),
        "cache_root": str(root),
    }

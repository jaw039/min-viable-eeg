"""Fingerprints of the inputs a result depends on: frozen artifacts and the cache.

Used to key teacher checkpoints, so a cached teacher is reused only when the
data, the split, the ranking and the configuration that produced it are the
ones in force now (see src.checkpoints).
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from src.dataset import data_root
from src.utils import ARTIFACTS_DIR, REPO_ROOT

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


def source_identity() -> str:
    """Identify library code even in an archive without .git or in a dirty tree."""
    files = sorted((REPO_ROOT / "src").glob("*.py"))
    if not files:
        raise ValueError("No library source files to identify")
    entries = [(p.name, sha256_file(p)) for p in files]
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def cache_identity(cfg: Optional[dict] = None, root: Optional[Path] = None,
                   subjects: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Content fingerprint of the cache entries used by a teacher.

    Hash both complete .npy files, including the samples and labels. A cache
    regenerated with the same shapes can still contain different EEG or labels.
    Teacher callers pass only fitting and inner-holdout subjects; validation
    and test entries do not participate in teacher identity.
    """
    root = Path(root) if root is not None else data_root()
    entries = []
    directories = (sorted(root / "S{:03d}".format(int(s)) for s in set(subjects))
                   if subjects is not None else sorted(root.glob("S*")))
    for d in directories:
        x, y = d / "X.npy", d / "y.npy"
        if not (x.exists() and y.exists()):
            raise FileNotFoundError("Incomplete cache entry at {}".format(d))
        entries.append([d.name, sha256_file(x), sha256_file(y)])
    if not entries:
        raise ValueError("No cache entries to identify at {}".format(root))
    blob = json.dumps(entries, separators=(",", ":")).encode()
    name = ((cfg or {}).get("dataset") or {}).get("name", "eegmmidb")
    return {
        "dataset_name": name,
        "cache_manifest_sha256": hashlib.sha256(blob).hexdigest(),
        "n_subjects_cached": len(entries),
        "cache_root": str(root),
    }

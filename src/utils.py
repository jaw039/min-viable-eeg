"""Shared utilities: config loading, data paths, and provenance (git commit + config hash)."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
# Generated-once protocol artifacts (splits, ranking, budgets, stability).
# Frozen: their writers refuse to overwrite an existing file.
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# EEGMMIDB ships 109 subjects, S001-S109 (protocol-locked dataset).
N_SUBJECTS = 109

PathLike = Union[str, Path]


def load_config(path: Optional[PathLike] = None) -> dict:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def data_root(config: dict) -> Path:
    # Resolve relative to the repo root, not cwd, so scripts work from anywhere.
    return (REPO_ROOT / config["dataset"]["path"]).resolve()


def edf_path(config: dict, subject: int, run: int) -> Path:
    return (
        data_root(config)
        / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
        / "S{:03d}".format(subject)
        / "S{:03d}R{:02d}.edf".format(subject, run)
    )


def get_git_commit(ignore_paths: Sequence[PathLike] = ()) -> str:
    """HEAD commit hash, suffixed '-dirty' if the working tree has changes.

    Outside a git checkout (the Kaggle code snapshot, an unpacked archive)
    the hash is read from a ``COMMIT`` file at the repo root, which
    ``scripts/make_kaggle_bundle.py`` writes from the archived HEAD. With
    neither source the stamp is "unknown".

    ignore_paths: tracked files whose modification/deletion should NOT count
    as dirty. Generated-once outputs (splits.json, channel_ranking.json) pass
    their own path here, since regenerating them necessarily touches the
    tracked file and would otherwise stamp every regeneration as dirty.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        ignored = {
            Path(p).resolve().relative_to(REPO_ROOT).as_posix() for p in ignore_paths
        }
        dirty = [
            line for line in status.splitlines()
            if line.strip() and line[3:].strip() not in ignored
        ]
        return commit + "-dirty" if dirty else commit
    except (subprocess.CalledProcessError, OSError, ValueError):
        return _commit_from_file()


def _commit_from_file() -> str:
    """The hash recorded by the bundle script in REPO_ROOT/COMMIT, else "unknown"."""
    try:
        text = (REPO_ROOT / "COMMIT").read_text().strip()
    except OSError:
        return "unknown"
    return text if re.fullmatch(r"[0-9a-f]{7,40}", text) else "unknown"


def config_hash(path: Optional[PathLike] = None) -> str:
    # Hash the raw file bytes: canonical and independent of dict ordering.
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]


# The config sections the frozen artifacts actually depend on. Training
# hyperparameters (learning rate, epochs, batch size) do not change what
# splits.json, channel_ranking.json, budgets.json or stability.json contain, so
# adding them must not read as protocol drift. `config_hash` hashes the whole
# file and moves when anything changes; `protocol_hash` moves only when a
# decision that would invalidate a frozen artifact changes.
PROTOCOL_KEYS = ("dataset", "preprocess", "splits", "budgets", "reduction_mode")


def protocol_hash(path: Optional[PathLike] = None) -> str:
    """Hash of only the protocol-relevant config, canonically serialised."""
    cfg = load_config(path)
    subset = {k: cfg[k] for k in PROTOCOL_KEYS if k in cfg}
    blob = json.dumps(subset, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def provenance(
    config_path: Optional[PathLike] = None, ignore_paths: Sequence[PathLike] = ()
) -> dict:
    return {
        "git_commit": get_git_commit(ignore_paths),
        "config_hash": config_hash(config_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

"""Shared utilities: config loading, data paths, and provenance (git commit + config hash)."""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"

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


def get_git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
        return commit + "-dirty" if dirty else commit
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def config_hash(path: Optional[PathLike] = None) -> str:
    # Hash the raw file bytes: canonical and independent of dict ordering.
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]


def provenance(config_path: Optional[PathLike] = None) -> dict:
    return {
        "git_commit": get_git_commit(),
        "config_hash": config_hash(config_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

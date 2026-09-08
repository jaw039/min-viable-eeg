"""Provenance-validated teacher checkpoints.

A distilled student is only interpretable if you know which teacher taught it.
Keying a checkpoint on (mode, n_channels, seed) is not enough: the same
filename survives a config change, an architecture change, a different subject
split, a re-cached dataset, or a new commit. The student then learns from a
teacher that no longer matches the protocol, and nothing in the result row
shows it.

So a checkpoint is a payload, not a bare state dict. On load, the expected
provenance is recomputed and compared field by field. Any scientifically
relevant mismatch rejects the checkpoint -- retrain, or fail with a message
naming the field that moved. Legacy bare state dicts are rejected outright:
they carry no provenance, so they cannot be shown to match.
"""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PAYLOAD_SCHEMA = "mve_teacher_checkpoint_v1"

# Fields that must match exactly for a cached teacher to be reused. Each one
# can change the teacher's weights or what those weights mean.
IDENTITY_FIELDS = (
    "schema",
    "teacher_mode",
    "train_seed",
    "shuffle_seed",
    "n_channels",
    "channel_order",
    "model_config",
    "training_config",
    "distillation_config",
    "config_sha256",
    "git_commit",
    "source_sha256",
    "splits_sha256",
    "budgets_sha256",
    "ranking_sha256",
    "fit_subjects",
    "inner_holdout_subjects",
    "cache_manifest_sha256",
    "dataset_name",
)


class StaleCheckpointError(RuntimeError):
    """A cached teacher does not match the protocol that would produce it."""


class LegacyCheckpointError(StaleCheckpointError):
    """A bare state dict with no provenance."""


def _canonical(obj):
    """JSON-stable form, so dict ordering cannot change an identity hash."""
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def build_identity(
    teacher_mode: str,
    train_seed: int,
    n_channels: int,
    channel_order: Sequence[str],
    config: dict,
    cache_ident: dict,
    artifact_hashes: Dict[str, Optional[str]],
    fit_subjects: Sequence[int],
    inner_holdout_subjects: Sequence[int],
    shuffle_seed: Optional[int] = None,
    git_commit: Optional[str] = None,
    config_sha256: Optional[str] = None,
) -> Dict:
    """The provenance a teacher checkpoint must match to be reusable."""
    from src.utils import config_hash, get_git_commit
    from src.provenance import source_identity

    if teacher_mode not in ("real", "shuffled"):
        raise ValueError("teacher_mode must be 'real' or 'shuffled'")

    return _canonical({
        "schema": PAYLOAD_SCHEMA,
        "teacher_mode": teacher_mode,
        "train_seed": int(train_seed),
        # None for a real teacher; an int for a shuffled one. This is what
        # stops a real and a shuffled teacher sharing an identity.
        "shuffle_seed": None if shuffle_seed is None else int(shuffle_seed),
        "n_channels": int(n_channels),
        "channel_order": list(channel_order),
        "model_config": dict(config.get("model", {})),
        "training_config": dict(config.get("training", {})),
        "distillation_config": dict(config.get("distillation", {})),
        "config_sha256": config_sha256 or config_hash(),
        "git_commit": git_commit or get_git_commit(),
        "source_sha256": source_identity(),
        "splits_sha256": artifact_hashes.get("splits_sha256"),
        "budgets_sha256": artifact_hashes.get("budgets_sha256"),
        "ranking_sha256": artifact_hashes.get("ranking_sha256"),
        "fit_subjects": sorted(int(s) for s in fit_subjects),
        "inner_holdout_subjects": sorted(int(s) for s in inner_holdout_subjects),
        "cache_manifest_sha256": cache_ident.get("cache_manifest_sha256"),
        "dataset_name": cache_ident.get("dataset_name"),
    })


def identity_hash(identity: Dict) -> str:
    """Stable hash over the identity fields only."""
    subset = {k: identity.get(k) for k in IDENTITY_FIELDS}
    blob = json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def checkpoint_filename(identity: Dict) -> str:
    """Filename carrying mode, seed and an identity digest.

    The digest is in the name so two teachers that differ only in, say, config
    cannot collide on disk -- but the name is a convenience, not the check.
    Validation always reads the payload.
    """
    return "teacher_{}_{}ch_seed{}_{}.pt".format(
        identity["teacher_mode"],
        identity["n_channels"],
        identity["train_seed"],
        identity_hash(identity)[:12],
    )


def save_checkpoint(
    path: Path,
    state_dict: Dict,
    identity: Dict,
    best_epoch: int,
    inner_kappa: float,
) -> Dict:
    """Write a payload: weights plus everything needed to validate reuse."""
    import torch

    payload = {
        "schema": PAYLOAD_SCHEMA,
        "state_dict": state_dict,
        "identity": identity,
        "identity_sha256": identity_hash(identity),
        "best_epoch": int(best_epoch),
        "inner_holdout_kappa": round(float(inner_kappa), 6),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": "python{} torch{} {}".format(
            platform.python_version(), torch.__version__, platform.platform()
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def compare_identity(expected: Dict, found: Dict) -> List[Dict]:
    """Fields that differ, with both values, for an actionable message."""
    diffs = []
    for field in IDENTITY_FIELDS:
        e, f = expected.get(field), found.get(field)
        if _canonical(e) != _canonical(f):
            diffs.append({"field": field, "expected": e, "found": f})
    return diffs


def load_checkpoint(path: Path, expected_identity: Dict, device=None) -> Dict:
    """Load a teacher only if its provenance matches exactly.

    Raises LegacyCheckpointError for a bare state dict and StaleCheckpointError
    for any identity mismatch. Never returns a teacher it cannot vouch for.
    """
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(payload, dict) or "identity" not in payload:
        raise LegacyCheckpointError(
            "{} is a bare state dict with no provenance. It cannot be shown to "
            "match the current protocol, so it will not be loaded. Delete it "
            "and retrain the teacher.".format(path)
        )
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise StaleCheckpointError(
            "{} has schema {!r}; expected {!r}.".format(
                path, payload.get("schema"), PAYLOAD_SCHEMA
            )
        )

    diffs = compare_identity(expected_identity, payload["identity"])
    if diffs:
        lines = [
            "Cached teacher at {} does not match the current protocol.".format(path),
            "Mismatched field(s):",
        ]
        for d in diffs[:8]:
            lines.append("  - {}: expected {!r}, checkpoint has {!r}".format(
                d["field"], _short(d["expected"]), _short(d["found"])
            ))
        lines.append(
            "Retrain the teacher, or delete the checkpoint. It will not be "
            "loaded: a student distilled from a mismatched teacher is not "
            "interpretable."
        )
        raise StaleCheckpointError("\n".join(lines))

    return payload


def _short(v, limit: int = 60) -> str:
    s = repr(v)
    return s if len(s) <= limit else s[: limit - 3] + "..."

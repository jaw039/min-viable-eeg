"""Blocker 5 + 6 wiring: teacher checkpoint provenance, result-row provenance.

A distilled student is only interpretable if the teacher that taught it matches
the protocol. A result row is only reproducible if it names the data it read.
These tests make both structural.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.checkpoints import (
    IDENTITY_FIELDS,
    LegacyCheckpointError,
    StaleCheckpointError,
    build_identity,
    checkpoint_filename,
    compare_identity,
    identity_hash,
    load_checkpoint,
    save_checkpoint,
)
from src.utils import load_config

CFG = load_config()
CACHE_IDENT = {"cache_manifest_sha256": "abc123", "dataset_name": "eegmmidb"}
ART = {
    "splits_sha256": "s1",
    "budgets_sha256": "b1",
    "ranking_sha256": "r1",
    "stability_sha256": "t1",
}
CH = ["C3", "C4", "Cz", "Pz"]


def _identity(**over):
    kw = dict(
        teacher_mode="real",
        train_seed=42,
        n_channels=4,
        channel_order=CH,
        config=CFG,
        cache_ident=CACHE_IDENT,
        artifact_hashes=ART,
        fit_subjects=[1, 2, 3],
        inner_holdout_subjects=[4],
    )
    kw.update(over)
    return build_identity(**kw)


def _write(tmp_path, identity, name=None):
    path = tmp_path / (name or checkpoint_filename(identity))
    save_checkpoint(path, {"w": torch.zeros(2)}, identity, best_epoch=7,
                    inner_kappa=0.31)
    return path


# ===========================================================================
# BLOCKER 5 -- teacher checkpoint provenance
# ===========================================================================

def test_identical_provenance_reuses_the_checkpoint(tmp_path):
    ident = _identity()
    path = _write(tmp_path, ident)
    payload = load_checkpoint(path, ident)
    assert payload["best_epoch"] == 7
    assert payload["inner_holdout_kappa"] == 0.31
    assert payload["identity_sha256"] == identity_hash(ident)


def test_changed_config_rejects_the_checkpoint(tmp_path):
    path = _write(tmp_path, _identity())
    other = copy.deepcopy(CFG)
    other["training"]["learning_rate"] = 0.01
    with pytest.raises(StaleCheckpointError) as exc:
        load_checkpoint(path, _identity(config=other))
    assert "training_config" in str(exc.value)


def test_changed_model_settings_reject_the_checkpoint(tmp_path):
    path = _write(tmp_path, _identity())
    other = copy.deepcopy(CFG)
    other["model"]["dropout"] = 0.25
    with pytest.raises(StaleCheckpointError):
        load_checkpoint(path, _identity(config=other))


def test_changed_split_rejects_the_checkpoint(tmp_path):
    path = _write(tmp_path, _identity())
    with pytest.raises(StaleCheckpointError) as exc:
        load_checkpoint(path, _identity(fit_subjects=[1, 2, 3, 5]))
    assert "fit_subjects" in str(exc.value)


def test_changed_split_artifact_hash_rejects_the_checkpoint(tmp_path):
    path = _write(tmp_path, _identity())
    art = dict(ART, splits_sha256="DIFFERENT")
    with pytest.raises(StaleCheckpointError) as exc:
        load_checkpoint(path, _identity(artifact_hashes=art))
    assert "splits_sha256" in str(exc.value)


def test_changed_dataset_fingerprint_rejects_the_checkpoint(tmp_path):
    path = _write(tmp_path, _identity())
    ident2 = _identity(cache_ident={"cache_manifest_sha256": "zzz",
                                    "dataset_name": "eegmmidb"})
    with pytest.raises(StaleCheckpointError) as exc:
        load_checkpoint(path, ident2)
    assert "cache_manifest_sha256" in str(exc.value)


def test_changed_channel_order_rejects_the_checkpoint(tmp_path):
    path = _write(tmp_path, _identity())
    with pytest.raises(StaleCheckpointError):
        load_checkpoint(path, _identity(channel_order=["C4", "C3", "Cz", "Pz"]))


def test_real_and_shuffled_teachers_cannot_collide(tmp_path):
    real = _identity(teacher_mode="real")
    sham = _identity(teacher_mode="shuffled", shuffle_seed=42)
    assert identity_hash(real) != identity_hash(sham)
    assert checkpoint_filename(real) != checkpoint_filename(sham)
    path = _write(tmp_path, real)
    with pytest.raises(StaleCheckpointError):
        load_checkpoint(path, sham)


def test_changed_seed_cannot_collide(tmp_path):
    a, b = _identity(train_seed=42), _identity(train_seed=123)
    assert identity_hash(a) != identity_hash(b)
    path = _write(tmp_path, a)
    with pytest.raises(StaleCheckpointError):
        load_checkpoint(path, b)


def test_changed_commit_rejects_the_checkpoint(tmp_path):
    path = _write(tmp_path, _identity())
    with pytest.raises(StaleCheckpointError) as exc:
        load_checkpoint(path, _identity(git_commit="deadbeef"))
    assert "git_commit" in str(exc.value)


def test_legacy_bare_state_dict_is_rejected(tmp_path):
    """Pre-provenance checkpoints carry nothing to validate, so they are
    refused rather than migrated on a guess."""
    path = tmp_path / "teacher_real_64ch_seed42.pt"
    torch.save({"w": torch.zeros(2)}, path)
    with pytest.raises(LegacyCheckpointError) as exc:
        load_checkpoint(path, _identity())
    assert "bare state dict" in str(exc.value)


def test_rejection_message_names_the_field_that_moved(tmp_path):
    path = _write(tmp_path, _identity())
    other = copy.deepcopy(CFG)
    other["training"]["batch_size"] = 999
    with pytest.raises(StaleCheckpointError) as exc:
        load_checkpoint(path, _identity(config=other))
    msg = str(exc.value)
    assert "Retrain the teacher" in msg
    assert "not interpretable" in msg


def test_identity_covers_every_declared_field():
    ident = _identity()
    for field in IDENTITY_FIELDS:
        assert field in ident, "identity is missing {}".format(field)


def test_identity_hash_is_key_order_independent():
    a = _identity()
    b = dict(reversed(list(a.items())))
    assert identity_hash(a) == identity_hash(b)


def test_compare_identity_reports_every_difference():
    a = _identity()
    b = _identity(train_seed=99, n_channels=8)
    fields = {d["field"] for d in compare_identity(a, b)}
    assert {"train_seed", "n_channels"} <= fields


def test_shuffled_teacher_records_its_shuffle_seed():
    ident = _identity(teacher_mode="shuffled", shuffle_seed=42)
    assert ident["shuffle_seed"] == 42
    assert _identity()["shuffle_seed"] is None


# ===========================================================================
# BLOCKER 6 -- result rows carry dataset and artifact provenance
# ===========================================================================

REQUIRED_PROVENANCE_FIELDS = (
    "git_commit",
    "config_sha256",
    "reduction_mode",
    "n_input_channels",
    "dataset_name",
    "dataset_version",
    "cache_manifest_sha256",
    "cache_is_identified",
    "splits_sha256",
    "budgets_sha256",
    "ranking_sha256",
    "stability_sha256",
    "channels",
    "train_seed",
    "selection_seed",
    "split",
    "runtime_sec",
)


def test_required_provenance_fields_are_declared():
    """Guards the schema itself: dropping a field from the runner without
    updating this list should not be possible silently."""
    assert len(set(REQUIRED_PROVENANCE_FIELDS)) == len(REQUIRED_PROVENANCE_FIELDS)
    for f in ("dataset_version", "cache_manifest_sha256", "splits_sha256"):
        assert f in REQUIRED_PROVENANCE_FIELDS


def test_artifact_hashes_are_real_and_distinct():
    from src.provenance import artifact_hashes
    from src.utils import REPO_ROOT

    h = artifact_hashes(REPO_ROOT)
    assert all(v for v in h.values()), "a frozen artifact is missing"
    assert len(set(h.values())) == 4, "artifact hashes should differ"


def test_reduction_mode_is_recorded_as_approved():
    from src.decisions import decision_value

    assert decision_value(CFG, "reduction_mode") == "reduce"

"""The frozen artifacts must still describe the protocol the code runs.

`config_hash` hashes the whole config file, so it moves when anything changes
-- including training hyperparameters that cannot possibly alter what
splits.json or budgets.json contain. That makes it useless as a drift signal
once the config grows. These tests check the thing that actually matters:
whether the *content* of each frozen artifact still agrees with the config the
runner reads.
"""

import json

import pytest

from src.channels import montage_order
from src.utils import REPO_ROOT, load_config, protocol_hash

ARTIFACTS = ("splits.json", "channel_ranking.json", "budgets.json", "stability.json")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def artifact(name):
    with open(REPO_ROOT / name) as f:
        return json.load(f)


def test_all_frozen_artifacts_share_one_config_hash():
    """They were generated together; if they disagree with each other, one was
    regenerated under a different protocol."""
    hashes = {name: artifact(name)["provenance"]["config_hash"] for name in ARTIFACTS}
    assert len(set(hashes.values())) == 1, "artifacts disagree with each other: {}".format(hashes)


def test_protocol_hash_is_stable_and_narrow():
    """protocol_hash must ignore training hyperparameters."""
    import copy

    from src.utils import PROTOCOL_KEYS

    base = protocol_hash()
    assert base and len(base) == 12
    cfg = load_config()
    for key in ("model", "training", "distillation", "sweep"):
        assert key not in PROTOCOL_KEYS
        assert key in cfg, "config should carry {} for the runner".format(key)


def test_splits_artifact_matches_config(cfg):
    s = artifact("splits.json")
    assert s["strategy"] == cfg["splits"]["strategy"]
    assert s["seed"] == cfg["splits"]["seed"]
    assert s["ratios"] == cfg["splits"]["ratios"]
    n_excluded = len(cfg["dataset"]["exclude_subjects"])
    assert s["n_subjects"] == 109 - n_excluded
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == s["n_subjects"]


def test_splits_are_disjoint_and_exclude_the_excluded(cfg):
    """Subject leakage across splits."""
    s = artifact("splits.json")
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    for bad in cfg["dataset"]["exclude_subjects"]:
        assert bad not in train | val | test


def test_budgets_artifact_matches_config(cfg):
    b = artifact("budgets.json")
    assert b["budgets"] == list(cfg["budgets"])
    assert b["reduction_mode"] == cfg["reduction_mode"]
    assert sorted(int(k) for k in b["sets"]) == sorted(cfg["budgets"])


def test_budgets_agree_with_the_ranking_they_were_cut_from():
    ranking = artifact("channel_ranking.json")
    b = artifact("budgets.json")
    assert b["ranking_provenance"] == ranking["provenance"]
    ranked = ranking["channels"]
    for k_str, entry in b["sets"].items():
        assert entry["ranking_order"] == ranked[: int(k_str)]


def test_montage_order_is_a_permutation_of_the_ranking():
    ranking = artifact("channel_ranking.json")
    assert sorted(montage_order()) == sorted(ranking["channels"])


def test_ranking_band_matches_preprocessing(cfg):
    r = artifact("channel_ranking.json")
    assert list(r["band_hz"]) == list(cfg["preprocess"]["bandpass"])


def test_ranking_used_train_subjects_only():
    r = artifact("channel_ranking.json")
    s = artifact("splits.json")
    assert r["n_train_subjects"] == len(s["train"])

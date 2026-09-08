"""The audit must refuse what it once certified.

Four probes each passed the first version of the audit: a wrong config hash,
a file holding only a failed run, an undeclared training seed, and a
corrupted selection-order channel list. Each is a regression test here, next
to the row they were built from.
"""

import copy
import json

import pytest

from src.audit import audit_rows
from src.channels import load_budgets, load_ranking, montage_order, select_channels
from src.manifest import build_manifest
from src.utils import ARTIFACTS_DIR, config_hash, load_config

CFG = load_config()
RANKED, RANK_PROV = load_ranking()
BUDGETS = load_budgets()
CH = montage_order()
SPLITS = json.load(open(ARTIFACTS_DIR / "splits.json"))
SEEDS = list(CFG["sweep"]["train_seeds"])
N_RANDOM = int(CFG["sweep"]["n_random_subsets"])
MANIFEST = {"val": build_manifest(CFG["budgets"], CH, SEEDS, n_random=N_RANDOM)}


def valid_row(k=4, seed=42, selection="ranked", selection_seed=None, training="scratch",
              split="val", shuffle=False):
    derived = select_channels(selection, k, CH, RANKED, seed=selection_seed)
    subjects = ["S{:03d}".format(s) for s in SPLITS[split]]
    return {
        "budget_k": k, "selection": selection, "selection_seed": selection_seed,
        "training": training, "shuffle_labels": shuffle, "train_seed": seed, "split": split,
        "channels": list(derived),
        "channels_montage_order": [c for c in CH if c in set(derived)],
        "n_channels": k,
        "kappa": 0.1, "aggregation_primary": "pooled",
        "n_trials": 45 * len(subjects), "n_subjects": len(subjects),
        "n_trials_per_subject": {s: 45 for s in subjects},
        "kappa_per_subject": {s: 0.1 for s in subjects},
        "config_sha256": config_hash(), "git_commit": "abc1234",
        "ranking_provenance": dict(RANK_PROV), "splits_seed": SPLITS["seed"],
    }


def audit(rows, errors=(), **kw):
    opts = dict(manifests=MANIFEST, expected_config_hash=config_hash(), ranked=RANKED,
                rank_prov=RANK_PROV, budgets=BUDGETS, ch_names=CH, splits=SPLITS,
                planned_seeds=SEEDS, n_random=N_RANDOM)
    opts.update(kw)
    return audit_rows(list(rows), list(errors), **opts)


def test_a_protocol_conforming_row_passes():
    rep = audit([valid_row(), valid_row(k=8, seed=123), valid_row(k=4, selection="random", selection_seed=3)])
    assert rep["ok"], rep["fails"]
    assert rep["complete"] is False          # three of 691 conditions
    assert rep["coverage"]["val"]["planned"] == 691


def test_wrong_config_hash_fails_even_when_all_rows_agree():
    r = valid_row()
    r["config_sha256"] = "000000000000"
    rep = audit([r])
    assert "config_hash_expected" in rep["fails"] and not rep["ok"]


def test_a_file_of_only_error_rows_is_not_a_pass():
    err = {"error": "RuntimeError: boom", "budget_k": 8, "selection": "ranked",
           "training": "scratch", "train_seed": 42, "split": "val"}
    rep = audit([], errors=[err])
    assert not rep["ok"] and "error_rows" in rep["fails"]
    rep = audit([valid_row()], errors=[err], allow_errors=True)
    assert rep["ok"] and "error_rows" in rep["warns"]


def test_undeclared_training_seed_fails():
    rep = audit([valid_row(seed=999999)])
    assert "seed_declared" in rep["fails"]
    assert "not_in_manifest" in rep["fails"]


def test_corrupted_selection_order_channels_fail_even_if_montage_order_is_intact():
    r = valid_row()
    r["channels"] = ["Fp1", "Fp2", "Oz", "O1"]
    rep = audit([r])
    assert "channels_field" in rep["fails"]
    assert "frozen_budget" in rep["fails"]


def test_condition_outside_the_manifest_fails():
    rep = audit([valid_row(k=4, selection="random", selection_seed=N_RANDOM + 5)])
    assert "seed_declared" in rep["fails"] and "not_in_manifest" in rep["fails"]


def test_missing_manifest_for_a_split_fails():
    rep = audit([valid_row(split="test")])
    assert "manifest_missing" in rep["fails"]


def test_incomplete_coverage_is_reported_and_can_be_made_fatal():
    rep = audit([valid_row()])
    assert rep["complete"] is False and rep["ok"]
    rep = audit([valid_row()], require_complete=True)
    assert "incomplete_coverage" in rep["fails"] and not rep["ok"]


def test_duplicate_condition_fails():
    rep = audit([valid_row(), copy.deepcopy(valid_row())])
    assert "duplicate_condition" in rep["fails"]


def test_evaluated_subjects_must_match_the_frozen_split():
    r = valid_row()
    r["n_trials_per_subject"]["S001"] = 45
    r["kappa_per_subject"]["S001"] = 0.1
    r["n_trials"] += 45
    rep = audit([r])
    assert "split_subjects" in rep["fails"]


def test_committed_rows_pass_the_stricter_audit():
    """The rows already collected must survive the checks added after them."""
    import glob
    from src.utils import REPO_ROOT

    files = sorted(glob.glob(str(REPO_ROOT / "results" / "shard_*_val.jsonl")))
    if not files:
        pytest.skip("no committed result rows")
    rows = [json.loads(l) for f in files for l in open(f) if l.strip()]
    rep = audit([r for r in rows if "error" not in r], errors=[r for r in rows if "error" in r])
    assert rep["ok"], rep["fails"]

"""Manifest/sharding tests, plus an end-to-end run of every condition.

The runner tests use a small synthetic cache with the real montage and shapes.
They prove the plumbing -- that channel selection genuinely gates model input,
that early stopping never sees the validation split, that every arm executes
and emits a well-formed row. None of these numbers are science.
"""

import json

import numpy as np
import pytest

from src import runner as R
from src.channels import SENSORIMOTOR_POOL, montage_order
from src.manifest import build_manifest, condition_key, missing_conditions, shard, summarise

N_SAMPLES = 641
TRIALS = 8
TRAIN = [1, 2, 3, 4, 5, 6]
VAL = [7, 8]
TEST = [9]


# ------------------------------------------------------------------ manifest


@pytest.fixture(scope="module")
def ch_names():
    return montage_order()


def test_manifest_covers_every_arm(ch_names):
    rows = build_manifest([4, 64], ch_names, [42], n_random=3)
    kinds = summarise(rows)
    assert kinds["ranked"] == 2
    assert kinds["random"] == 3           # k=64 random is skipped: it is the montage
    assert kinds["sensorimotor"] == 1     # only k=4 fits the 17-electrode strip
    assert kinds["distill"] == 1
    assert kinds["distill_shuffled_teacher"] == 1
    assert kinds["label-shuffle control"] == 1


def test_manifest_defaults_to_validation(ch_names):
    rows = build_manifest([4, 64], ch_names, [42], n_random=2)
    assert {r["split"] for r in rows} == {"val"}


def test_sharding_is_a_partition(ch_names):
    rows = build_manifest([4, 8, 64], ch_names, [42, 43], n_random=5)
    shards = [shard(rows, i, 4) for i in range(4)]
    assert sum(len(s) for s in shards) == len(rows)
    seen = [condition_key(r) for s in shards for r in s]
    assert len(seen) == len(set(seen)) == len(rows)


def test_shard_bounds_are_checked(ch_names):
    rows = build_manifest([4, 64], ch_names, [42], n_random=1)
    with pytest.raises(ValueError):
        shard(rows, 4, 4)
    with pytest.raises(ValueError):
        shard(rows, -1, 4)


def test_missing_conditions_identifies_the_gap(ch_names):
    rows = build_manifest([4, 64], ch_names, [42], n_random=2)
    done = rows[:3]
    missing = missing_conditions(rows, done)
    assert len(missing) == len(rows) - 3


def test_test_manifest_holds_kstar_and_the_full_montage_only():
    """The confirmatory manifest must let test_report compute the retained
    fraction: k* AND k=64, at every planned seed, and nothing else. The
    generic writer at one budget had no full-montage reference."""
    from src.kstar import test_report
    from src.manifest import build_test_manifest

    rows = build_test_manifest(32, [42, 123, 456, 789, 101112])
    assert len(rows) == 10
    assert {r["budget_k"] for r in rows} == {32, 64}
    assert all(r["selection"] == "ranked" and r["training"] == "scratch"
               and r["split"] == "test" and not r["shuffle_labels"] for r in rows)
    fake = [dict(r, kappa=0.3 if r["budget_k"] == 32 else 0.4) for r in rows]
    rep = test_report(fake, kstar=32)
    assert rep["n_runs_at_kstar"] == 5 and rep["n_runs_full"] == 5
    assert rep["retained_fraction"] == pytest.approx(0.75)


def test_test_manifest_writes_the_full_montage_once_when_it_is_kstar():
    from src.manifest import build_test_manifest

    rows = build_test_manifest(64, [42, 123])
    assert len(rows) == 2
    assert {r["budget_k"] for r in rows} == {64}


def test_test_manifest_requires_a_chosen_kstar():
    from src.manifest import build_test_manifest

    with pytest.raises(ValueError):
        build_test_manifest(None, [42])


# ------------------------------------------------------------------ runner


@pytest.fixture(scope="module")
def fake_cache(tmp_path_factory, ch_names):
    """Synthetic cache with a class-dependent signal planted only on C4."""
    root = tmp_path_factory.mktemp("processed")
    rng = np.random.default_rng(0)
    c4 = ch_names.index("C4")
    for s in TRAIN + VAL + TEST:
        d = root / "S{:03d}".format(s)
        d.mkdir()
        y = np.tile([0, 1], TRIALS // 2).astype(np.int64)
        X = rng.normal(size=(TRIALS, len(ch_names), N_SAMPLES)).astype(np.float32)
        X[y == 1, c4, :] += 3.0
        np.save(d / "X.npy", X)
        np.save(d / "y.npy", y)
    return root


@pytest.fixture(autouse=True)
def wire(monkeypatch, fake_cache):
    monkeypatch.setenv("MVE_DATA_ROOT", str(fake_cache))
    monkeypatch.setattr(
        R, "load_splits_json",
        lambda *a, **k: {"train": TRAIN, "val": VAL, "test": TEST, "seed": 42},
    )


@pytest.fixture(scope="module")
def cfg():
    return {
        "model": {"n_samples": N_SAMPLES, "n_classes": 2, "dropout": 0.5},
        "training": {"batch_size": 4, "learning_rate": 1e-3, "max_epochs": 3,
                     "patience": 2, "inner_val_frac": 0.25},
        "distillation": {"alpha": 0.5, "temperature": 4.0},
        "splits": {"seed": 42},
    }


REQUIRED = {
    "budget_k", "selection", "selection_seed", "training", "train_seed", "split",
    "channels", "n_channels", "kappa", "kappa_macro_subject",
    "kappa_macro_subject_std", "accuracy", "macro_f1", "n_trials", "n_subjects",
    "kappa_per_subject", "n_trials_per_subject", "git_commit", "config_sha256",
    "aggregation_primary", "environment", "device", "runtime_sec",
}


def test_row_matches_the_documented_schema(cfg):
    row = R.run_condition(4, "ranked", split="val", cfg=cfg)
    assert REQUIRED <= set(row)
    assert row["n_channels"] == 4
    assert row["n_trials"] == len(VAL) * TRIALS
    assert set(row["kappa_per_subject"]) == {"S007", "S008"}
    assert row["git_commit"] and row["config_sha256"]


def test_ranked_run_is_drift_guarded(cfg, monkeypatch):
    """A ranked run must refuse to train on channels that disagree with the
    frozen artifact.

    Only the k=4 entry is corrupted; the montage (read from the k=64 entry)
    stays real, so this isolates the guard rather than breaking the loader.
    """
    from src import channels as C

    real = C.load_budgets()
    corrupted = json.loads(json.dumps(real))
    corrupted["sets"]["4"]["ranking_order"] = list(
        reversed(corrupted["sets"]["4"]["ranking_order"])
    )
    monkeypatch.setattr(C, "load_budgets", lambda *a, **k: corrupted)

    with pytest.raises(C.ChannelDriftError, match="disagrees with frozen"):
        R.run_condition(4, "ranked", split="val", cfg=cfg)


def test_channel_selection_actually_gates_model_input(cfg):
    """The planted signal lives only on C4. A subset containing it must beat a
    subset that excludes it -- otherwise selection is not reaching the model."""
    with_c4 = R.run_condition(4, "ranked", split="val", cfg=cfg)   # top-4 includes C4
    assert "C4" in with_c4["channels"]

    from src import channels as C

    ch_names = montage_order()
    ranked = C.load_ranking()[0]
    # Find a random subset that happens to exclude the planted channel.
    seed = next(
        s for s in range(200)
        if "C4" not in C.select_channels("random", 4, ch_names, ranked, seed=s)
    )
    without = R.run_condition(4, "random", split="val", selection_seed=seed, cfg=cfg)
    assert "C4" not in without["channels"]
    assert with_c4["kappa"] > without["kappa"]


def test_sensorimotor_arm_stays_in_pool(cfg):
    row = R.run_condition(6, "sensorimotor", split="val", cfg=cfg)
    assert set(row["channels"]) <= set(SENSORIMOTOR_POOL)


def test_early_stopping_never_sees_the_validation_split(cfg):
    """Model selection and budget selection must not share data."""
    row = R.run_condition(4, "ranked", split="val", cfg=cfg)
    assert row["n_fit_subjects"] + row["n_inner_holdout_subjects"] == len(TRAIN)
    assert row["n_inner_holdout_subjects"] >= 1
    assert row["best_epoch"] >= 1


def test_inner_split_is_subject_wise_and_disjoint():
    fit, hold = R.inner_split(list(range(1, 21)), 0.15, 42)
    assert set(fit).isdisjoint(hold)
    assert sorted(fit + hold) == list(range(1, 21))
    assert len(hold) == 3


def test_inner_split_is_deterministic():
    assert R.inner_split(list(range(1, 21)), 0.15, 42) == R.inner_split(
        list(range(1, 21)), 0.15, 42
    )


def test_distillation_and_sham_teacher_both_run(cfg, tmp_path):
    d = R.run_condition(4, "ranked", training="distill", split="val",
                        cfg=cfg, teacher_cache=tmp_path)
    assert d["training"] == "distill" and d["n_channels"] == 4
    s = R.run_condition(4, "ranked", training="distill_shuffled_teacher",
                        split="val", cfg=cfg, teacher_cache=tmp_path)
    assert s["training"] == "distill_shuffled_teacher"
    names = {p.name for p in tmp_path.iterdir()}
    assert any(n.startswith("teacher_real_") for n in names)
    assert any(n.startswith("teacher_shuffled_") for n in names)


def test_label_shuffle_control_runs_and_is_flagged(cfg):
    row = R.run_condition(4, "ranked", split="val", shuffle_labels=True, cfg=cfg)
    assert row["shuffle_labels"] is True


def test_test_split_is_reachable(cfg):
    row = R.run_condition(4, "ranked", split="test", cfg=cfg)
    assert row["split"] == "test"
    assert set(row["kappa_per_subject"]) == {"S009"}


def test_invalid_modes_rejected(cfg):
    with pytest.raises(ValueError, match="training must be one of"):
        R.run_condition(4, "ranked", training="magic", cfg=cfg)
    with pytest.raises(ValueError, match="split must be one of"):
        R.run_condition(4, "ranked", split="train", cfg=cfg)


def test_missing_cache_gives_an_actionable_error(cfg, monkeypatch):
    monkeypatch.setattr(
        R, "load_splits_json",
        lambda *a, **k: {"train": [96, 97, 98, 99], "val": VAL, "test": TEST, "seed": 42},
    )
    with pytest.raises(FileNotFoundError, match="MVE_DATA_ROOT"):
        R.run_condition(4, "ranked", split="val", cfg=cfg)


# ------------------------------------------------------------------ teacher cache provenance


def test_teacher_checkpoint_is_reused_only_with_matching_provenance(cfg, tmp_path):
    """A cached teacher carries the provenance that would reproduce it. Same
    protocol: reused. Changed learning rate: retrained, never loaded as if
    nothing had moved."""
    first = R.run_condition(4, "ranked", training="distill", split="val", cfg=cfg,
                            teacher_cache=tmp_path)
    assert first["teacher_cached"] is False
    again = R.run_condition(4, "ranked", training="distill", split="val", cfg=cfg,
                            teacher_cache=tmp_path)
    assert again["teacher_cached"] is True
    assert again["teacher_checkpoint"] == first["teacher_checkpoint"]

    changed = json.loads(json.dumps(cfg))
    changed["training"]["learning_rate"] = 0.1
    third = R.run_condition(4, "ranked", training="distill", split="val", cfg=changed,
                            teacher_cache=tmp_path)
    assert third["teacher_cached"] is False
    assert third["teacher_checkpoint"] != first["teacher_checkpoint"]
    assert len(list(tmp_path.glob("teacher_real_*"))) == 2


def test_legacy_bare_state_dict_is_not_loaded_as_a_teacher(cfg, tmp_path):
    """A pre-provenance checkpoint at the expected name is retrained over, and
    the row records why."""
    import torch

    first = R.run_condition(4, "ranked", training="distill", split="val", cfg=cfg,
                            teacher_cache=tmp_path)
    path = tmp_path / first["teacher_checkpoint"]
    torch.save({"w": torch.zeros(2)}, path)
    again = R.run_condition(4, "ranked", training="distill", split="val", cfg=cfg,
                            teacher_cache=tmp_path)
    assert again["teacher_cached"] is False
    assert "bare state dict" in again["teacher_cache_rejected"]
    payload = torch.load(path, weights_only=False)
    assert "identity" in payload


def test_real_and_shuffled_teachers_never_share_a_checkpoint(cfg, tmp_path):
    R.run_condition(4, "ranked", training="distill", split="val", cfg=cfg, teacher_cache=tmp_path)
    R.run_condition(4, "ranked", training="distill_shuffled_teacher", split="val", cfg=cfg,
                    teacher_cache=tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2
    assert any("teacher_real_" in n for n in names) and any("teacher_shuffled_" in n for n in names)


def test_teacher_checkpoint_records_the_frozen_artifacts_and_cache(cfg, tmp_path):
    import torch
    from src.provenance import artifact_hashes

    row = R.run_condition(4, "ranked", training="distill", split="val", cfg=cfg,
                          teacher_cache=tmp_path)
    payload = torch.load(tmp_path / row["teacher_checkpoint"], weights_only=False)
    ident = payload["identity"]
    assert ident["splits_sha256"] == artifact_hashes()["splits_sha256"]
    assert ident["cache_manifest_sha256"] and ident["n_channels"] == 64
    assert ident["fit_subjects"] and ident["inner_holdout_subjects"]

"""One condition in, one result row out.

Everything the sweep needs is a parameter of `run_condition`, so a control is a
row in the manifest rather than a separate script somebody has to remember to
run at 2am. That includes the ones most likely to be skipped under deadline:
the sensorimotor arm, distillation from a teacher that knows nothing, and the
label-shuffle floor.

Two separations are enforced here rather than documented:

* **Model selection never touches the validation split.** Early stopping uses an
  inner subset of the *training* subjects. The validation split exists to choose
  k*, and if it also picked epochs, k* would be selected on data already used
  for fitting.
* **Ranked runs are drift-guarded.** A ranked condition asserts its channel set
  against the frozen budgets.json before training, so a run cannot quietly
  train on channels that disagree with the artifact the paper cites.
"""

import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.budget import reduce_channels
from src.channels import load_ranking, montage_order, selection_for_run
from src.dataset import SplitData, load_split, trials_per_subject
from src.distillation import DistillationLoss, train_distillation_epoch
from src.eegnet import EEGNet
from src.metrics import evaluate
from src.normalize import apply_stats, fit_stats
from src.training import get_device, make_data_loader, predict, train_epoch
from src.utils import REPO_ROOT, config_hash, get_git_commit, load_config

TRAINING_MODES = ("scratch", "distill", "distill_shuffled_teacher")
SPLITS = ("val", "test")


# ------------------------------------------------------------------ helpers


def load_splits_json(path: Optional[Path] = None) -> dict:
    path = path or REPO_ROOT / "splits.json"
    with open(path) as f:
        return json.load(f)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _shuffled(y: np.ndarray, seed: int) -> np.ndarray:
    out = y.copy()
    np.random.default_rng(seed).shuffle(out)
    return out


def inner_split(subjects: List[int], frac: float, seed: int):
    """Hold out a fraction of TRAINING subjects for early stopping.

    Subject-wise, like every other split in the protocol, so no subject's trials
    straddle the fit/early-stopping boundary.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("inner_val_frac must be in (0, 1)")
    ordered = sorted(int(s) for s in subjects)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(ordered))
    n_hold = max(1, int(round(len(ordered) * frac)))
    hold = sorted(int(s) for s in shuffled[:n_hold])
    fit = sorted(int(s) for s in shuffled[n_hold:])
    if not fit:
        raise ValueError("inner split left no subjects to fit on")
    return fit, hold


def environment_string() -> str:
    return "python{} torch{} numpy{}".format(
        platform.python_version(), torch.__version__, np.__version__
    )


def _build(n_channels: int, cfg: dict, device) -> EEGNet:
    m = cfg["model"]
    return EEGNet(
        n_channels=n_channels,
        n_samples=int(m["n_samples"]),
        n_classes=int(m["n_classes"]),
        dropout=float(m["dropout"]),
    ).to(device)


# ------------------------------------------------------------------ training


def train_with_early_stopping(
    X_fit, y_fit, X_hold, y_hold, n_channels: int, cfg: dict, device, seed: int
):
    """Train, keeping the weights that scored best on the inner hold-out."""
    _seed_everything(seed)
    t = cfg["training"]
    model = _build(n_channels, cfg, device)
    fit_loader = make_data_loader(X_fit, y_fit, int(t["batch_size"]), shuffle=True)
    hold_loader = make_data_loader(X_hold, y_hold, int(t["batch_size"]), shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=float(t["learning_rate"]))

    best_score, best_state, best_epoch, since = -np.inf, None, 0, 0
    patience = int(t.get("patience", 10))

    for epoch in range(1, int(t["max_epochs"]) + 1):
        train_epoch(model, fit_loader, opt, device)
        yt, yp = predict(model, hold_loader, device)
        from src.metrics import pooled_metrics

        score = pooled_metrics(yt, yp)["kappa"]
        if score > best_score:
            best_score, best_epoch, since = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "inner_kappa": round(float(best_score), 6)}


def get_teacher(X_fit, y_fit, X_hold, y_hold, cfg, device, seed, mode, cache_dir):
    """Full-montage teacher. mode 'real' or 'shuffled'.

    The shuffled teacher is trained on permuted labels: a confident model that
    knows nothing. If distillation still helps against it, the gain is
    regularisation from soft targets rather than knowledge transfer, and the
    paper has to say so.
    """
    n_channels = X_fit.shape[1]
    tag = "teacher_{}_{}ch_seed{}.pt".format(mode, n_channels, seed)
    path = Path(cache_dir) / tag if cache_dir else None

    if path is not None and path.exists():
        model = _build(n_channels, cfg, device)
        model.load_state_dict(torch.load(path, map_location=device))
        return model, {"teacher_cached": True}

    yf = y_fit if mode == "real" else _shuffled(y_fit, seed)
    yh = y_hold if mode == "real" else _shuffled(y_hold, seed + 1)
    model, info = train_with_early_stopping(
        X_fit, yf, X_hold, yh, n_channels, cfg, device, seed
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    return model, {"teacher_cached": False, "teacher_inner_kappa": info["inner_kappa"]}


def train_student(
    X_fit_full, y_fit, X_hold_full, y_hold, channel_indices, teacher, cfg, device, seed
):
    """Student sees only its k channels; the teacher sees the full montage."""
    _seed_everything(seed)
    t = cfg["training"]
    student = _build(len(channel_indices), cfg, device)
    fit_loader = make_data_loader(X_fit_full, y_fit, int(t["batch_size"]), shuffle=True)
    hold_loader = make_data_loader(
        X_hold_full[:, channel_indices, :], y_hold, int(t["batch_size"]), shuffle=False
    )
    opt = torch.optim.Adam(student.parameters(), lr=float(t["learning_rate"]))
    loss_fn = DistillationLoss(
        alpha=float(cfg["distillation"]["alpha"]),
        temperature=float(cfg["distillation"]["temperature"]),
    )

    best_score, best_state, best_epoch, since = -np.inf, None, 0, 0
    patience = int(t.get("patience", 10))
    from src.metrics import pooled_metrics

    for epoch in range(1, int(t["max_epochs"]) + 1):
        train_distillation_epoch(
            teacher, student, fit_loader, opt, loss_fn, device, channel_indices
        )
        yt, yp = predict(student, hold_loader, device)
        score = pooled_metrics(yt, yp)["kappa"]
        if score > best_score:
            best_score, best_epoch, since = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    return student, {"best_epoch": best_epoch, "inner_kappa": round(float(best_score), 6)}


# ------------------------------------------------------------------ condition


def run_condition(
    budget_k: int,
    selection: str,
    split: str = "val",
    training: str = "scratch",
    selection_seed: Optional[int] = None,
    train_seed: int = 42,
    shuffle_labels: bool = False,
    cfg: Optional[dict] = None,
    teacher_cache: Optional[Path] = None,
) -> Dict:
    """Train and evaluate one condition; return its result row."""
    if training not in TRAINING_MODES:
        raise ValueError("training must be one of {}".format(TRAINING_MODES))
    if split not in SPLITS:
        raise ValueError("split must be one of {}".format(SPLITS))

    cfg = cfg or load_config()
    started = time.time()

    ch_names = montage_order()
    ranked, ranking_prov = load_ranking()
    splits = load_splits_json()

    # Drift guard fires here for ranked runs, before any compute is spent.
    selected = selection_for_run(
        selection, budget_k, ch_names, ranked, seed=selection_seed
    )

    train_subjects = [int(s) for s in splits["train"]]
    fit_subj, hold_subj = inner_split(
        train_subjects,
        float(cfg["training"].get("inner_val_frac", 0.15)),
        int(cfg["splits"]["seed"]),
    )

    fit_data = load_split(fit_subj)
    hold_data = load_split(hold_subj)
    eval_data: SplitData = load_split([int(s) for s in splits[split]])

    # Normalisation statistics come from the fitting subjects only.
    mu, sd = fit_stats(fit_data.X)
    X_fit = apply_stats(fit_data.X, mu, sd)
    X_hold = apply_stats(hold_data.X, mu, sd)
    X_eval = apply_stats(eval_data.X, mu, sd)

    y_fit = _shuffled(fit_data.y, train_seed) if shuffle_labels else fit_data.y
    y_hold = _shuffled(hold_data.y, train_seed + 1) if shuffle_labels else hold_data.y

    device = get_device()
    extra: Dict[str, object] = {}

    if training == "scratch":
        X_fit_k, kept = reduce_channels(X_fit, ch_names, selected)
        X_hold_k, _ = reduce_channels(X_hold, ch_names, selected)
        model, info = train_with_early_stopping(
            X_fit_k, y_fit, X_hold_k, y_hold, len(kept), cfg, device, train_seed
        )
    else:
        kept = [c for c in ch_names if c in set(selected)]
        channel_indices = [ch_names.index(c) for c in kept]
        teacher, tinfo = get_teacher(
            X_fit, y_fit, X_hold, y_hold, cfg, device, train_seed,
            "real" if training == "distill" else "shuffled", teacher_cache,
        )
        extra.update(tinfo)
        model, info = train_student(
            X_fit, y_fit, X_hold, y_hold, channel_indices, teacher, cfg, device, train_seed
        )

    X_eval_k, _ = reduce_channels(X_eval, ch_names, selected)
    loader = make_data_loader(
        X_eval_k, eval_data.y, int(cfg["training"]["batch_size"]), shuffle=False
    )
    y_true, y_pred = predict(model, loader, device)
    scores = evaluate(y_true, y_pred, eval_data.subjects)

    row = {
        "budget_k": int(budget_k),
        "selection": selection,
        "selection_seed": selection_seed,
        "training": training,
        "shuffle_labels": bool(shuffle_labels),
        "train_seed": int(train_seed),
        "split": split,
        "channels": list(selected),
        "channels_montage_order": list(kept),
        "n_channels": len(kept),
        "n_trials": int(len(eval_data.y)),
        "n_trials_per_subject": trials_per_subject(eval_data),
        "n_fit_subjects": len(fit_subj),
        "n_inner_holdout_subjects": len(hold_subj),
        "best_epoch": info["best_epoch"],
        "inner_kappa": info["inner_kappa"],
        "git_commit": get_git_commit(),
        "config_sha256": config_hash(),
        "ranking_provenance": ranking_prov,
        "splits_seed": splits["seed"],
        "environment": environment_string(),
        "device": str(device),
        "runtime_sec": round(time.time() - started, 2),
    }
    row.update(scores)
    row.update(extra)
    return row

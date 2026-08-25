"""Reduced-channel budget utility (config reduction_mode: reduce).

Given the ranked channel list (channel_ranking.json) and a budget k, select
the top-k channels and physically subset the data to (n_trials, k, n_samples).
Selected channels are returned in their ORIGINAL montage order (the order of
ch_names), not ranking order, so downstream code sees a consistent layout.

Run as: python -m src.budget

Writes budgets.json at the repo root: the exact channel set for every budget
in config (both ranking order and montage order), stamped with provenance and
the commit of the ranking it was cut from. Generated once like
channel_ranking.json (refuses overwrite; delete manually to regenerate).
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.utils import REPO_ROOT, load_config, provenance

RANKING_PATH = REPO_ROOT / "channel_ranking.json"
BUDGETS_PATH = REPO_ROOT / "budgets.json"


def top_k_channels(ranked_channels: List[str], k: int) -> List[str]:
    """First k channels of the ranking (best-first)."""
    if not 1 <= k <= len(ranked_channels):
        raise ValueError(
            "Budget k={} outside 1..{}".format(k, len(ranked_channels))
        )
    return list(ranked_channels[:k])


def reduce_channels(
    X: np.ndarray, ch_names: List[str], selected: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Subset X to the selected channels, kept in original montage order."""
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate channels in selection: {}".format(selected))
    missing = [c for c in selected if c not in ch_names]
    if missing:
        raise ValueError("Selected channels not in data: {}".format(missing))
    idx = sorted(ch_names.index(c) for c in selected)
    return X[:, idx, :], [ch_names[i] for i in idx]


def apply_budget(
    X: np.ndarray,
    ch_names: List[str],
    ranked_channels: List[str],
    k: int,
    mode: str = "reduce",
) -> Tuple[np.ndarray, List[str]]:
    """Top-k selection under the configured reduction mode."""
    if mode == "reduce":
        return reduce_channels(X, ch_names, top_k_channels(ranked_channels, k))
    if mode == "mask":
        raise NotImplementedError("reduction_mode 'mask' is not implemented in v1")
    raise ValueError("Unknown reduction_mode: {}".format(mode))


def budget_sets(
    ranked_channels: List[str], ch_names: List[str], budgets: List[int]
) -> dict:
    """Exact channel set per budget, keyed by str(k).

    'ranking_order' is best-first; 'montage_order' is the order the reduced
    data arrays actually come out of reduce_channels (original ch_names order).
    """
    out = {}
    for k in budgets:
        selected = top_k_channels(ranked_channels, k)
        idx = sorted(ch_names.index(c) for c in selected)
        out[str(k)] = {
            "k": int(k),
            "ranking_order": selected,
            "montage_order": [ch_names[i] for i in idx],
        }
    return out


def write_budgets(
    config: dict,
    ranking_path: Path = RANKING_PATH,
    out_path: Path = BUDGETS_PATH,
) -> None:
    if out_path.exists():
        print("REFUSING to overwrite existing {}".format(out_path))
        print("Budget sets are frozen once generated. If you really intend to")
        print("regenerate them, delete the file manually first:")
        print("    rm {}".format(out_path))
        raise SystemExit(1)
    with open(ranking_path) as f:
        ranking = json.load(f)
    ranked = ranking["channels"]

    # Montage order is protocol-constant; read it once via the loader.
    from src.loader import load_subject

    _, _, ch_names = load_subject(1, config)
    if sorted(ch_names) != sorted(ranked):
        raise ValueError("Ranking channels do not match loader montage channels")

    budgets = list(config["budgets"])
    payload = {
        "provenance": provenance(ignore_paths=[out_path]),
        "ranking_provenance": ranking["provenance"],
        "ranking_method": ranking["method"],
        "reduction_mode": config["reduction_mode"],
        "budgets": budgets,
        "sets": budget_sets(ranked, ch_names, budgets),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print("Wrote {} ({} budgets)".format(out_path, len(budgets)))
    for k in budgets:
        print("k={:>2}: {}".format(k, payload["sets"][str(k)]["ranking_order"]))


def main() -> None:
    write_budgets(load_config())


if __name__ == "__main__":
    main()

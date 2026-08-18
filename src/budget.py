"""Reduced-channel budget utility (config reduction_mode: reduce).

Given the ranked channel list (channel_ranking.json) and a budget k, select
the top-k channels and physically subset the data to (n_trials, k, n_samples).
Selected channels are returned in their ORIGINAL montage order (the order of
ch_names), not ranking order, so downstream code sees a consistent layout.
"""

from typing import List, Tuple

import numpy as np


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

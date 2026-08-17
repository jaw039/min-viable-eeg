"""Per-channel normalization.

Protocol (CLAUDE.md / config preprocess.normalize = per_channel_train_stats):
stats MUST be fit on the training split only, then applied unchanged to
val/test. fit_stats therefore takes exactly one array — pass it train data
and nothing else.
"""

from typing import Tuple

import numpy as np


def fit_stats(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over trials and time, from TRAIN data only.

    X_train: (n_trials, n_channels, n_samples). Returns (mu, sd), each a
    float64 array of shape (n_channels,).
    """
    if X_train.ndim != 3:
        raise ValueError("Expected (n_trials, n_channels, n_samples), got shape {}".format(X_train.shape))
    X64 = X_train.astype(np.float64)
    mu = X64.mean(axis=(0, 2))
    sd = X64.std(axis=(0, 2))
    if np.any(sd == 0):
        dead = np.where(sd == 0)[0].tolist()
        raise ValueError("Zero std in channel index(es) {} — dead channel?".format(dead))
    return mu, sd


def apply_stats(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Normalize X with precomputed per-channel stats (from fit_stats on train).

    Returns float32 of the same shape as X.
    """
    if X.ndim != 3:
        raise ValueError("Expected (n_trials, n_channels, n_samples), got shape {}".format(X.shape))
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    if mu.shape != (X.shape[1],) or sd.shape != (X.shape[1],):
        raise ValueError(
            "Stats shape {}/{} does not match n_channels={}".format(mu.shape, sd.shape, X.shape[1])
        )
    Z = (X.astype(np.float64) - mu[None, :, None]) / sd[None, :, None]
    return Z.astype(np.float32)

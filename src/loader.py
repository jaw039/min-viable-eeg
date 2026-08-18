"""Deterministic epoch loader for EEGMMIDB motor-imagery trials.

Per file: read raw EDF -> eegbci.standardize -> standard_1005 montage ->
bandpass (config) on continuous data -> epoch 0-4 s post-cue (T1/T2 only) ->
stack runs. Rest (T0) is dropped per protocol.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np
from mne.datasets import eegbci

from src.utils import edf_path

# Locked protocol: 64 channels; 0-4 s inclusive at 160 Hz -> 641 samples.
N_CHANNELS = 64
N_SAMPLES = 641

# Event ids start at 1 because id 0 means "no event" in MNE events arrays.
# T1 = left fist, T2 = right fist; labels are id - 1 (0 = left, 1 = right).
_EVENT_ID = {"T1": 1, "T2": 2}

# The mapping this module implements. config dataset.label_map must agree,
# or provenance (config hash) would assert a mapping the code never applied.
_LABEL_MAP = {"T1": "left", "T2": "right"}


def _load_run(path: Path, config: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)  # in place
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    l_freq, h_freq = config["preprocess"]["bandpass"]
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")

    events, _ = mne.events_from_annotations(raw, event_id=_EVENT_ID, verbose="ERROR")
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"left": _EVENT_ID["T1"], "right": _EVENT_ID["T2"]},
        tmin=config["preprocess"]["tmin"],
        tmax=config["preprocess"]["tmax"],
        baseline=None,  # default (None, 0) would subtract the t=0 sample
        proj=False,
        preload=True,
        verbose="ERROR",
    )
    if len(epochs) != len(events):
        # Epochs whose window overruns the recording are dropped; y comes from
        # epochs.events, so X and y stay aligned regardless.
        print(
            "WARNING: {} dropped {} of {} epochs".format(
                path.name, len(events) - len(epochs), len(events)
            )
        )

    y = (epochs.events[:, 2] - 1).astype(np.int64)
    X = epochs.get_data(copy=True).astype(np.float32)
    return X, y, list(epochs.ch_names)


def load_subject(
    subject_id: int, config: dict, runs: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load bandpassed, epoched trials for one subject.

    Returns X of shape (n_trials, 64, 641) float32, y of shape (n_trials,)
    with 0 = left fist (T1) and 1 = right fist (T2), and the standardized
    channel names. ``runs`` defaults to the runs listed in the config.
    """
    if subject_id in config["dataset"]["exclude_subjects"]:
        raise ValueError("Subject {} is excluded by protocol".format(subject_id))
    if config["dataset"].get("label_map") != _LABEL_MAP:
        raise ValueError(
            "config dataset.label_map {} does not match the locked protocol "
            "mapping {} implemented by this loader".format(
                config["dataset"].get("label_map"), _LABEL_MAP
            )
        )
    if runs is None:
        runs = config["dataset"]["runs"]

    xs = []
    ys = []
    ch_names = None  # type: Optional[List[str]]
    for run in runs:
        path = edf_path(config, subject_id, run)
        if not path.exists():
            raise FileNotFoundError(
                "{} not found — run scripts/download_data.py first".format(path)
            )
        X_run, y_run, names = _load_run(path, config)
        if ch_names is None:
            ch_names = names
        elif names != ch_names:
            raise ValueError(
                "Channel names differ between runs for subject {}".format(subject_id)
            )
        xs.append(X_run)
        ys.append(y_run)

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if X.shape[1:] != (N_CHANNELS, N_SAMPLES):
        raise ValueError(
            "Unexpected trial shape {}; expected (n, {}, {})".format(
                X.shape, N_CHANNELS, N_SAMPLES
            )
        )
    return X, y, ch_names

"""Unified channel selection for the three experimental arms, plus a drift guard.

    ranked        top-k of the frozen channel_ranking.json (the headline arm)
    random        seeded random subset of the same size (the control arm)
    sensorimotor  the ranking restricted to a declared motor strip

Volume conduction means neighbouring scalp channels partly re-measure the same
cortical sources, so a spread random subset is already a strong baseline.
"Informed selection beats random" therefore has to be won at equal k, not
assumed -- which is why `random` is a first-class selection mode rather than a
separate script.

The sensorimotor arm exists because of a finding in stability.json: C3, the
canonical left motor electrode, does not enter the unrestricted ranking until
rank 23, while F7, AF7, PO7 and O1 outrank it. Those are plausibly lateralised
gaze or attention correlates of the cue. Restricting to the motor strip at
equal k separates motor ERD from cue-correlated activity.

UNRESOLVED: the 17-electrode pool below is a proposal awaiting sign-off. It is
declared here in one place so the Methods can cite it exactly and
`test_sensorimotor_selection_stays_inside_declared_pool` can enforce it.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils import REPO_ROOT

SELECTION_MODES: Tuple[str, ...] = ("ranked", "random", "sensorimotor")

# FC/C/CP rows over sensorimotor cortex. Temporal chains (FT7/8, T7/8, TP7/8)
# are excluded: they sit outside sensorimotor cortex and are a common route for
# muscle artifact.
SENSORIMOTOR_POOL: Tuple[str, ...] = (
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
)


class ChannelDriftError(RuntimeError):
    """A ranked selection disagreed with the frozen budgets.json."""


# ------------------------------------------------------------------ artifacts


def load_ranking(path: Optional[Path] = None) -> Tuple[List[str], dict]:
    """Frozen channel ranking, best-first, with its provenance."""
    path = path or REPO_ROOT / "channel_ranking.json"
    with open(path) as f:
        r = json.load(f)
    return list(r["channels"]), r["provenance"]


def load_budgets(path: Optional[Path] = None) -> dict:
    path = path or REPO_ROOT / "budgets.json"
    with open(path) as f:
        return json.load(f)


def montage_order(path: Optional[Path] = None) -> List[str]:
    """Channel names in data-array (axis 1) order.

    Read from the frozen budgets.json rather than an EDF, so nothing in the
    experiment path needs mne or the raw recordings.
    """
    b = load_budgets(path)
    full = str(max(int(k) for k in b["sets"]))
    return list(b["sets"][full]["montage_order"])


# ------------------------------------------------------------------ selection


def sensorimotor_channels(ch_names: Sequence[str]) -> List[str]:
    """Declared strip members present in this montage, montage order."""
    pool = set(SENSORIMOTOR_POOL)
    return [c for c in ch_names if c in pool]


def max_k(mode: str, ch_names: Sequence[str]) -> int:
    """Largest budget this mode can serve for the given montage."""
    if mode == "sensorimotor":
        return len(sensorimotor_channels(ch_names))
    return len(ch_names)


def select_channels(
    mode: str,
    k: int,
    ch_names: Sequence[str],
    ranked_channels: Sequence[str],
    seed: Optional[int] = None,
) -> List[str]:
    """Channels for one condition, returned in ranking order (best-first).

    `seed` is required for mode='random' and ignored otherwise.
    """
    if mode not in SELECTION_MODES:
        raise ValueError(
            "Unknown selection mode {!r}; expected one of {}".format(
                mode, ", ".join(SELECTION_MODES)
            )
        )

    limit = max_k(mode, ch_names)
    if not 1 <= k <= limit:
        raise ValueError(
            "Budget k={} outside 1..{} for selection={!r}".format(k, limit, mode)
        )

    missing = [c for c in ranked_channels if c not in set(ch_names)]
    if missing:
        raise ValueError(
            "Ranking references channels absent from the montage: {}".format(missing)
        )

    if mode == "ranked":
        return list(ranked_channels[:k])

    if mode == "sensorimotor":
        pool = set(sensorimotor_channels(ch_names))
        return [c for c in ranked_channels if c in pool][:k]

    if seed is None:
        raise ValueError("selection='random' requires an explicit seed")
    rng = random.Random(seed)
    sampled = set(rng.sample(list(ch_names), k))
    # Reported in ranking order so every mode returns a comparable list.
    return [c for c in ranked_channels if c in sampled]


# ------------------------------------------------------------------ drift guard


def assert_matches_frozen_budgets(
    k: int, selected: Sequence[str], budgets: Optional[dict] = None
) -> None:
    """A ranked run must train on exactly the frozen channel set for its budget.

    Without this, a change to the ranking code or the artifact could leave the
    paper describing a montage that was never trained. Called on every ranked
    run, not just in tests.
    """
    b = budgets if budgets is not None else load_budgets()
    entry = b["sets"].get(str(k))
    if entry is None:
        raise ChannelDriftError(
            "budgets.json has no frozen set for k={}; budgets are {}".format(
                k, sorted(int(x) for x in b["sets"])
            )
        )
    expected = list(entry["ranking_order"])
    if list(selected) != expected:
        raise ChannelDriftError(
            "Ranked selection for k={} disagrees with frozen budgets.json.\n"
            "  frozen:   {}\n  computed: {}".format(k, expected, list(selected))
        )


def selection_for_run(
    mode: str,
    k: int,
    ch_names: Sequence[str],
    ranked_channels: Sequence[str],
    seed: Optional[int] = None,
    budgets: Optional[dict] = None,
) -> List[str]:
    """select_channels + the drift guard on ranked runs."""
    selected = select_channels(mode, k, ch_names, ranked_channels, seed=seed)
    if mode == "ranked":
        assert_matches_frozen_budgets(k, selected, budgets)
    return selected


def declared_pool_report(ch_names: Sequence[str]) -> Dict[str, object]:
    """What the sensorimotor arm is, for the Methods section."""
    present = sensorimotor_channels(ch_names)
    return {
        "declared_pool": list(SENSORIMOTOR_POOL),
        "declared_size": len(SENSORIMOTOR_POOL),
        "present_in_montage": present,
        "max_k": len(present),
        "status": "UNRESOLVED: pool awaiting sign-off",
    }

import random
from typing import List, Tuple
import numpy as np
from src.budget import reduce_channels


def random_channels(ch_names: List[str],k: int, seed: int) -> List[str]:
    if not 1 <= k <= len(ch_names):
        raise ValueError(
            "Channel budget {} outside 1..{}".format(k, len(ch_names))
        )

    rng = random.Random(seed)
    sampled = rng.sample(ch_names, k)
    sampled_set = set(sampled)

    return [ch for ch in ch_names if ch in sampled_set]


def apply_random_budget(X: np.ndarray,ch_names: List[str], k: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    selected = random_channels(ch_names=ch_names, k=k, seed=seed)

    return reduce_channels(X, ch_names, selected)
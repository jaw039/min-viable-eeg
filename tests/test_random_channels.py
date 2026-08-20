import numpy as np

from src.random_channels import (
    random_channels,
    apply_random_budget,
)


def test_random_channels_reproducible():
    channels = [
        "A", "B", "C", "D",
        "E", "F", "G", "H",
    ]

    result1 = random_channels(
        channels,
        k=4,
        seed=42,
    )

    result2 = random_channels(
        channels,
        k=4,
        seed=42,
    )

    assert result1 == result2


def test_random_channels_unique():
    channels = [
        "A", "B", "C", "D",
        "E", "F", "G", "H",
    ]

    selected = random_channels(
        channels,
        k=4,
        seed=42,
    )

    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_random_channels_preserve_order():
    channels = [
        "A", "B", "C", "D",
        "E", "F", "G", "H",
    ]

    selected = random_channels(
        channels,
        k=4,
        seed=42,
    )

    positions = [
        channels.index(ch)
        for ch in selected
    ]

    assert positions == sorted(positions)


def test_apply_random_budget_shape():
    X = np.zeros(
        (10, 8, 100),
        dtype=np.float32,
    )

    channels = [
        "A", "B", "C", "D",
        "E", "F", "G", "H",
    ]

    reduced_X, reduced_names = apply_random_budget(
        X,
        channels,
        k=4,
        seed=42,
    )

    assert reduced_X.shape == (10, 4, 100)
    assert len(reduced_names) == 4
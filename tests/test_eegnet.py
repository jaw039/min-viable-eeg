import torch

from src.eegnet import EEGNet


def test_full_channel_eegnet_output_shape():
    model = EEGNet(
        n_channels=64,
        n_samples=641,
        n_classes=2,
    )

    X = torch.randn(
        4,
        64,
        641,
    )

    output = model(X)

    assert output.shape == (
        4,
        2,
    )


def test_reduced_channel_eegnet_output_shape():
    model = EEGNet(
        n_channels=8,
        n_samples=641,
        n_classes=2,
    )

    X = torch.randn(
        4,
        8,
        641,
    )

    output = model(X)

    assert output.shape == (
        4,
        2,
    )


def test_eegnet_rejects_wrong_number_of_channels():
    model = EEGNet(
        n_channels=8,
        n_samples=641,
        n_classes=2,
    )

    X = torch.randn(
        4,
        64,
        641,
    )

    try:
        model(X)
        assert False, "Expected ValueError"

    except ValueError:
        pass
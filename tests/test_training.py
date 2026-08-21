import numpy as np
import torch

from src.eegnet import EEGNet
from src.training import (
    make_data_loader,
    train_epoch,
    predict,
)


def test_make_data_loader():
    X = np.random.randn(
        10,
        8,
        641,
    ).astype(np.float32)

    y = np.array(
        [
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
        ]
    )

    loader = make_data_loader(
        X,
        y,
        batch_size=4,
        shuffle=False,
    )

    X_batch, y_batch = next(
        iter(loader)
    )

    assert X_batch.shape == (
        4,
        8,
        641,
    )

    assert y_batch.shape == (4,)


def test_train_epoch():
    device = torch.device("cpu")

    X = np.random.randn(
        12,
        8,
        641,
    ).astype(np.float32)

    y = np.array(
        [
            0, 1, 0, 1,
            0, 1, 0, 1,
            0, 1, 0, 1,
        ]
    )

    loader = make_data_loader(
        X,
        y,
        batch_size=4,
        shuffle=False,
    )

    model = EEGNet(
        n_channels=8,
        n_samples=641,
        n_classes=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    loss = train_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        device=device,
    )

    assert isinstance(
        loss,
        float,
    )

    assert loss >= 0


def test_predict():
    device = torch.device("cpu")

    X = np.random.randn(
        8,
        8,
        641,
    ).astype(np.float32)

    y = np.array(
        [
            0, 1, 0, 1,
            0, 1, 0, 1,
        ]
    )

    loader = make_data_loader(
        X,
        y,
        batch_size=4,
        shuffle=False,
    )

    model = EEGNet(
        n_channels=8,
        n_samples=641,
        n_classes=2,
    ).to(device)

    y_true, y_pred = predict(
        model=model,
        data_loader=loader,
        device=device,
    )

    assert len(y_true) == 8

    assert len(y_pred) == 8

    assert all(
        prediction in [0, 1]
        for prediction in y_pred
    )
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


def make_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """
    Convert NumPy EEG data into a PyTorch DataLoader.

    Expected X shape:
        (n_trials, channels, samples)

    Expected y shape:
        (n_trials,)
    """

    if X.ndim != 3:
        raise ValueError(
            "Expected X with shape "
            "(trials, channels, samples)"
        )

    if y.ndim != 1:
        raise ValueError(
            "Expected y with shape (trials,)"
        )

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same "
            "number of trials"
        )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train a model normally for one epoch.

    This is used for:
        - the full-channel teacher
        - the student trained from scratch
    """

    model.train()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for X, labels in data_loader:

        X = X.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(X)

        loss = loss_fn(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise ValueError(
                "Training loss is NaN or infinite"
            )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        num_batches += 1

    if num_batches == 0:
        raise ValueError(
            "data_loader contained no batches"
        )

    return total_loss / num_batches


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
):
    """
    Generate predicted class labels.

    Returns:
        y_true
        y_pred
    """

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for X, labels in data_loader:

            X = X.to(device)

            logits = model(X)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    return (
        all_labels,
        all_predictions,
    )


def get_device() -> torch.device:
    """
    Choose the best available device.

    Apple Silicon Mac:
        MPS

    NVIDIA GPU:
        CUDA

    Otherwise:
        CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
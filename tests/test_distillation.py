import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.distillation import (
    DistillationLoss,
    select_student_channels,
    freeze_teacher,
    train_distillation_epoch,
    check_distillation_gradients,
)


class DummyEEGModel(nn.Module):
    """
    Tiny model used only for testing.

    Accepts input shaped:
        (batch, channels, samples)

    and outputs:
        (batch, 2)
    """

    def __init__(self, n_channels: int, n_samples: int = 641):
        super().__init__()

        self.flatten = nn.Flatten()

        self.classifier = nn.Linear(
            n_channels * n_samples,
            2,
        )

    def forward(self, X):
        X = self.flatten(X)
        return self.classifier(X)


def test_distillation_loss_is_finite():
    loss_fn = DistillationLoss(
        alpha=0.5,
        temperature=4.0,
    )

    student_logits = torch.tensor(
        [
            [1.0, 0.5],
            [0.2, 1.3],
        ],
        requires_grad=True,
    )

    teacher_logits = torch.tensor(
        [
            [1.4, 0.2],
            [0.1, 1.8],
        ],
    )

    labels = torch.tensor([0, 1])

    loss = loss_fn(
        student_logits,
        teacher_logits,
        labels,
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_select_student_channels():
    X = torch.randn(
        4,
        64,
        641,
    )

    channel_indices = [
        0,
        2,
        5,
        10,
        20,
        30,
        40,
        50,
    ]

    X_student = select_student_channels(
        X,
        channel_indices,
    )

    assert X_student.shape == (
        4,
        8,
        641,
    )


def test_select_student_channels_rejects_wrong_shape():
    X = torch.randn(
        4,
        64,
    )

    channel_indices = [0, 1]

    try:
        select_student_channels(
            X,
            channel_indices,
        )

        assert False, "Expected ValueError"

    except ValueError:
        pass


def test_freeze_teacher():
    teacher = DummyEEGModel(
        n_channels=64,
    )

    freeze_teacher(teacher)

    assert teacher.training is False

    assert all(
        parameter.requires_grad is False
        for parameter in teacher.parameters()
    )


def test_gradient_check():
    device = torch.device("cpu")

    teacher = DummyEEGModel(
        n_channels=64,
    ).to(device)

    student = DummyEEGModel(
        n_channels=8,
    ).to(device)

    X_full = torch.randn(
        4,
        64,
        641,
    )

    labels = torch.tensor(
        [0, 1, 0, 1]
    )

    channel_indices = [
        0,
        2,
        5,
        10,
        20,
        30,
        40,
        50,
    ]

    loss_fn = DistillationLoss(
        alpha=0.5,
        temperature=4.0,
    )

    result = check_distillation_gradients(
        teacher=teacher,
        student=student,
        X_full=X_full,
        labels=labels,
        channel_indices=channel_indices,
        loss_fn=loss_fn,
        device=device,
    )

    assert result["teacher_output_shape"] == (
        4,
        2,
    )

    assert result["student_output_shape"] == (
        4,
        2,
    )

    assert result["teacher_has_gradients"] is False

    assert result["student_has_gradients"] is True

    assert result["loss"] >= 0


def test_train_distillation_epoch():
    device = torch.device("cpu")

    teacher = DummyEEGModel(
        n_channels=64,
    ).to(device)

    student = DummyEEGModel(
        n_channels=8,
    ).to(device)

    X = torch.randn(
        12,
        64,
        641,
    )

    y = torch.tensor(
        [
            0, 1, 0, 1,
            0, 1, 0, 1,
            0, 1, 0, 1,
        ]
    )

    dataset = TensorDataset(
        X,
        y,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )

    channel_indices = [
        0,
        2,
        5,
        10,
        20,
        30,
        40,
        50,
    ]

    loss_fn = DistillationLoss(
        alpha=0.5,
        temperature=4.0,
    )

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=0.001,
    )

    avg_loss = train_distillation_epoch(
        teacher=teacher,
        student=student,
        data_loader=data_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        channel_indices=channel_indices,
    )

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0
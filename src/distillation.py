from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined supervised and teacher-student distillation loss."""

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        self.alpha = alpha
        self.temperature = temperature
        self.supervised_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        hard_loss = self.supervised_loss(
            student_logits,
            labels,
        )

        temperature = self.temperature

        student_log_probs = F.log_softmax(
            student_logits / temperature,
            dim=1,
        )

        teacher_probs = F.softmax(
            teacher_logits / temperature,
            dim=1,
        )

        soft_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        soft_loss = soft_loss * (temperature ** 2)

        return (
            (1.0 - self.alpha) * hard_loss
            + self.alpha * soft_loss
        )


def select_student_channels(
    X: torch.Tensor,
    channel_indices: List[int],
) -> torch.Tensor:
    """Subset full-channel EEG for the reduced-channel student."""

    if X.ndim != 3:
        raise ValueError(
            "Expected X with shape "
            "(batch, channels, samples), "
            "got {}".format(tuple(X.shape))
        )

    return X[:, channel_indices, :]


def freeze_teacher(teacher: nn.Module) -> None:
    """Freeze teacher parameters and put it in evaluation mode."""

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False


def train_distillation_epoch(
    teacher: nn.Module,
    student: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DistillationLoss,
    device: torch.device,
    channel_indices: List[int],
) -> float:
    """Train the student for one knowledge-distillation epoch."""

    freeze_teacher(teacher)
    student.train()

    total_loss = 0.0
    num_batches = 0

    for X_full, labels in data_loader:
        X_full = X_full.to(device)
        labels = labels.to(device)

        X_student = select_student_channels(
            X_full,
            channel_indices,
        )

        with torch.no_grad():
            teacher_logits = teacher(X_full)

        student_logits = student(X_student)

        loss = loss_fn(
            student_logits,
            teacher_logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise ValueError("Distillation loss is NaN or infinite")

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("data_loader contained no batches")

    return total_loss / num_batches


def check_distillation_gradients(
    teacher: nn.Module,
    student: nn.Module,
    X_full: torch.Tensor,
    labels: torch.Tensor,
    channel_indices: List[int],
    loss_fn: DistillationLoss,
    device: torch.device,
) -> dict:
    """Check teacher outputs and student/teacher gradient behavior."""

    freeze_teacher(teacher)
    teacher.zero_grad(set_to_none=True)
    student.zero_grad(set_to_none=True)
    student.train()

    X_full = X_full.to(device)
    labels = labels.to(device)

    X_student = select_student_channels(
        X_full,
        channel_indices,
    )


    with torch.no_grad():
        teacher_logits = teacher(X_full)

    student_logits = student(X_student)

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "Teacher output shape {} does not match "
            "student output shape {}".format(
                tuple(teacher_logits.shape),
                tuple(student_logits.shape),
            )
        )

    loss = loss_fn(
        student_logits,
        teacher_logits,
        labels,
    )
    if not torch.isfinite(loss):
        raise ValueError("Distillation loss is NaN or infinite")

    loss.backward()

    teacher_has_gradients = any(
        parameter.grad is not None
        for parameter in teacher.parameters()
    )

    student_has_gradients = any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0).item()
        for parameter in student.parameters()
    )

    return {
        "teacher_output_shape": tuple(
            teacher_logits.shape
        ),
        "student_output_shape": tuple(
            student_logits.shape
        ),
        "teacher_has_gradients": teacher_has_gradients,
        "student_has_gradients": student_has_gradients,
        "loss": float(loss.item()),
    }
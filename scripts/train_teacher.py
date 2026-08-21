"""Train the full 64-channel EEGNet teacher."""

import json
from pathlib import Path

import numpy as np
import torch

from src.eegnet import EEGNet
from src.evaluation import compute_metrics
from src.loader import load_subject
from src.normalize import fit_stats, apply_stats
from src.training import (
    get_device,
    make_data_loader,
    predict,
    train_epoch,
)
from src.utils import load_config


SPLITS_PATH = Path("splits.json")
CHECKPOINT_PATH = Path(
    "checkpoints/teacher_64ch.pt"
)


def load_split_subjects():
    """
    Load the locked subject-wise train/val/test split.
    """

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            "splits.json not found. "
            "Run `python3 -m src.splits` first."
        )

    with SPLITS_PATH.open("r") as file:
        splits = json.load(file)

    return (
        splits["train"],
        splits["val"],
        splits["test"],
    )


def load_subject_group(
    subject_ids,
    config,
):
    """
    Load and concatenate EEG trials from multiple subjects.

    Returns:
        X:
            (total_trials, 64, 641)

        y:
            (total_trials,)

        ch_names:
            standardized channel names
    """

    X_parts = []
    y_parts = []

    reference_channels = None

    for subject_id in subject_ids:

        print(
            f"Loading subject {subject_id}..."
        )

        X_subject, y_subject, ch_names = (
            load_subject(
                subject_id=subject_id,
                config=config,
            )
        )

        if reference_channels is None:
            reference_channels = ch_names

        elif ch_names != reference_channels:
            raise ValueError(
                "Channel names differ between subjects"
            )

        X_parts.append(X_subject)
        y_parts.append(y_subject)

    X = np.concatenate(
        X_parts,
        axis=0,
    )

    y = np.concatenate(
        y_parts,
        axis=0,
    )

    return X, y, reference_channels


def main():
    config = load_config()

    train_subjects, val_subjects, _ = (
        load_split_subjects()
    )

    print("Training subjects:")
    print(train_subjects)

    print("\nValidation subjects:")
    print(val_subjects)

    # --------------------------------------------------
    # Load full-channel training data
    # --------------------------------------------------

    X_train, y_train, ch_names = (
        load_subject_group(
            train_subjects,
            config,
        )
    )

    X_val, y_val, val_channels = (
        load_subject_group(
            val_subjects,
            config,
        )
    )

    if ch_names != val_channels:
        raise ValueError(
            "Train and validation channel names differ"
        )

    print("\nTraining shape:")
    print(X_train.shape)

    print("Validation shape:")
    print(X_val.shape)

    # --------------------------------------------------
    # Normalization
    #
    # IMPORTANT:
    # fit statistics ONLY on training data.
    # --------------------------------------------------

    mu, sd = fit_stats(
        X_train
    )

    X_train = apply_stats(
        X_train,
        mu,
        sd,
    )

    X_val = apply_stats(
        X_val,
        mu,
        sd,
    )

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    training_config = config.get(
        "training",
        {}
    )

    batch_size = training_config.get(
        "batch_size",
        32,
    )

    learning_rate = training_config.get(
        "learning_rate",
        0.001,
    )

    teacher_epochs = training_config.get(
        "teacher_epochs",
        10,
    )

    model_config = config.get(
        "model",
        {}
    )

    dropout = model_config.get(
        "dropout",
        0.5,
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = make_data_loader(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = make_data_loader(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    device = get_device()

    print(
        "\nUsing device:",
        device,
    )

    teacher = EEGNet(
        n_channels=64,
        n_samples=641,
        n_classes=2,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        teacher.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_kappa = float("-inf")

    for epoch in range(
        teacher_epochs
    ):
        training_loss = train_epoch(
            model=teacher,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        y_true, y_pred = predict(
            model=teacher,
            data_loader=val_loader,
            device=device,
        )

        metrics = compute_metrics(
            y_true,
            y_pred,
        )

        print(
            "\nEpoch {}/{}".format(
                epoch + 1,
                teacher_epochs,
            )
        )

        print(
            "Training loss: {:.4f}".format(
                training_loss
            )
        )

        print(
            "Validation kappa: {:.4f}".format(
                metrics["kappa"]
            )
        )

        print(
            "Validation accuracy: {:.4f}".format(
                metrics["accuracy"]
            )
        )

        print(
            "Validation F1: {:.4f}".format(
                metrics["f1"]
            )
        )

        # Save the teacher with the best
        # validation Cohen's kappa.
        if metrics["kappa"] > best_kappa:

            best_kappa = metrics["kappa"]

            torch.save(
                {
                    "model_state_dict":
                        teacher.state_dict(),

                    "mu": mu,
                    "sd": sd,

                    "channels": ch_names,

                    "validation_kappa":
                        best_kappa,
                },
                CHECKPOINT_PATH,
            )

            print(
                "Saved new best teacher."
            )

    print(
        "\nFinished training."
    )

    print(
        "Best validation kappa: "
        "{:.4f}".format(best_kappa)
    )

    print(
        "Teacher saved to:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()
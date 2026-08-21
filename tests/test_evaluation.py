import csv

from src.evaluation import (
    compute_metrics,
    make_result,
    save_result,
)


def test_compute_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    assert metrics["kappa"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_contains_all_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    assert "kappa" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics


def test_make_result():
    metrics = {
        "kappa": 0.5,
        "accuracy": 0.75,
        "f1": 0.73,
    }

    result = make_result(
        subject=1,
        split="validation",
        model="EEGNet",
        channel_method="random",
        channel_count=8,
        seed=42,
        metrics=metrics,
    )

    assert result["subject"] == 1
    assert result["split"] == "validation"
    assert result["model"] == "EEGNet"
    assert result["channel_method"] == "random"
    assert result["channel_count"] == 8
    assert result["seed"] == 42

    assert result["kappa"] == 0.5
    assert result["accuracy"] == 0.75
    assert result["f1"] == 0.73


def test_save_result(tmp_path):
    output_file = tmp_path / "experiments.csv"

    result = {
        "subject": 1,
        "split": "validation",
        "model": "EEGNet",
        "channel_method": "random",
        "channel_count": 8,
        "seed": 42,
        "kappa": 0.5,
        "accuracy": 0.75,
        "f1": 0.73,
    }

    save_result(
        result,
        output_path=output_file,
    )

    assert output_file.exists()

    with output_file.open("r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert len(rows) == 1

    assert rows[0]["subject"] == "1"
    assert rows[0]["split"] == "validation"
    assert rows[0]["model"] == "EEGNet"
    assert rows[0]["channel_method"] == "random"
    assert rows[0]["channel_count"] == "8"
    assert rows[0]["seed"] == "42"
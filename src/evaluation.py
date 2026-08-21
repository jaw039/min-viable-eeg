from typing import Dict
from sklearn.metrics import (cohen_kappa_score, accuracy_score, f1_score)
import csv
from pathlib import Path


def compute_metrics(y_true, y_pred,) -> Dict[str, float]:

    return {
        "kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

def make_result(subject, split, model, channel_method, channel_count, seed, metrics,):
    return {"subject": subject, "split": split, "model": model, "channel_method": channel_method, "channel_count": channel_count, "seed": seed, "kappa": metrics["kappa"], "accuracy": metrics["accuracy"], "f1": metrics["f1"]}

def save_result(result, output_path="results/experiments.csv"):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = output_path.exists()

    fieldnames = [
        "subject",
        "split",
        "model",
        "channel_method",
        "channel_count",
        "seed",
        "kappa",
        "accuracy",
        "f1",
    ]

    with output_path.open("a", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(result)
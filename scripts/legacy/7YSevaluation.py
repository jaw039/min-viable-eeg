"""
Evaluation metrics and utilities.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score
from typing import Dict, Union, List

def calculate_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Cohen's κ with quadratic weighting.
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy."""
    return accuracy_score(y_true, y_pred)

def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """Calculate F1 score."""
    return f1_score(y_true, y_pred, average=average)

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all metrics."""
    return {
        'kappa': calculate_kappa(y_true, y_pred),
        'accuracy': calculate_accuracy(y_true, y_pred),
        'f1_macro': calculate_f1(y_true, y_pred, average='macro'),
        'f1_weighted': calculate_f1(y_true, y_pred, average='weighted')
    }

def compute_subject_metrics(y_true_per_subject: Dict[str, np.ndarray],
                           y_pred_per_subject: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Compute per-subject metrics."""
    subject_metrics = {}
    
    for subject_id in y_true_per_subject:
        y_true = y_true_per_subject[subject_id]
        y_pred = y_pred_per_subject[subject_id]
        
        subject_metrics[subject_id] = calculate_all_metrics(y_true, y_pred)
    
    return subject_metrics

def format_metric(metric: float, std: float = None) -> str:
    """Format metric for reporting."""
    if std is None:
        return f"{metric:.3f}"
    return f"{metric:.3f} ± {std:.3f}"

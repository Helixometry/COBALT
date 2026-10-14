"""Metrics for TB detection."""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, sensitivity_specificity_support
from typing import Dict


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, confidences: np.ndarray = None) -> Dict[str, float]:
    """Compute TB detection metrics."""
    metrics = {"accuracy": accuracy_score(targets, predictions)}

    if confidences is not None:
        try:
            metrics["auc"] = roc_auc_score(targets, confidences)
        except:
            metrics["auc"] = 0.0

    return metrics

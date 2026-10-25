"""Metrics for Lab 3. Confusion matrix is implemented manually (no sklearn metrics)."""
from __future__ import annotations

import numpy as np


def confusion_matrix_manual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Build a 2x2 confusion matrix; rows/columns are (-1, +1) = (Class A, Class B)."""
    actual = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must contain the same number of samples.")
    matrix = np.zeros((2, 2), dtype=int)
    label_to_index = {-1.0: 0, 1.0: 1}
    for true_label, predicted_label in zip(actual, predicted):
        true_key = float(true_label)
        predicted_key = float(predicted_label)
        if true_key not in label_to_index or predicted_key not in label_to_index:
            raise ValueError("Labels must be encoded as -1 and +1.")
        matrix[label_to_index[true_key], label_to_index[predicted_key]] += 1
    return matrix


def accuracy_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute overall accuracy directly, without a built-in metrics package."""
    actual = np.asarray(y_true).reshape(-1)
    predicted = np.asarray(y_pred).reshape(-1)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Inputs must have equal non-zero lengths.")
    correct = 0
    for true_label, predicted_label in zip(actual, predicted):
        if true_label == predicted_label:
            correct += 1
    return correct / len(actual)


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    """Return overall accuracy and class recalls from a 2x2 matrix."""
    matrix = np.asarray(matrix, dtype=int)
    if matrix.shape != (2, 2):
        raise ValueError("Expected a 2x2 confusion matrix.")
    total = int(matrix.sum())
    correct = int(matrix[0, 0] + matrix[1, 1])
    a_total = int(matrix[0].sum())
    b_total = int(matrix[1].sum())
    return {
        "accuracy": correct / total if total else 0.0,
        "class_a_recall": int(matrix[0, 0]) / a_total if a_total else 0.0,
        "class_b_recall": int(matrix[1, 1]) / b_total if b_total else 0.0,
    }

"""Evaluation helpers for sentiment classification models."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def get_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute standard accuracy."""

    return float(accuracy_score(y_true, y_pred))


def get_macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute macro-averaged F1."""

    return float(f1_score(y_true, y_pred, average="macro"))


def get_weighted_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute weighted F1."""

    return float(f1_score(y_true, y_pred, average="weighted"))


def get_classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    digits: int = 4,
) -> str:
    """Return a text classification report."""

    return classification_report(y_true, y_pred, digits=digits, zero_division=0)


def get_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Return the confusion matrix as a NumPy array."""

    return confusion_matrix(y_true, y_pred, labels=labels)


def serialize_evaluation_results(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable evaluation summary."""

    ordered_labels = list(labels) if labels is not None else sorted({*y_true, *y_pred})
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=ordered_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=ordered_labels)

    return {
        "labels": ordered_labels,
        "accuracy": get_accuracy(y_true, y_pred),
        "macro_f1": get_macro_f1(y_true, y_pred),
        "weighted_f1": get_weighted_f1(y_true, y_pred),
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
    }

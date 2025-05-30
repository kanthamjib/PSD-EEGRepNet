"""
Evaluation Metrics
==================
This module provides standardized evaluation metrics for classification tasks,
particularly suitable for EEG-based motor-imagery classification experiments.
It includes functions to calculate accuracy, precision, recall, F1-score, and
Cohen's Kappa coefficient, along with a comprehensive classification report.

Metrics
-------
- **Accuracy:** The proportion of correctly predicted observations.
- **Precision:** Weighted precision across all classes.
- **Recall:** Weighted recall across all classes.
- **F1-score:** Harmonic mean of precision and recall, weighted across all classes.
- **Cohen's Kappa:** Statistic measuring inter-rater agreement for categorical items,
  accounting for agreement occurring by chance.
"""

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             cohen_kappa_score)
import numpy as np

__all__ = ["compute_metrics", "print_report"]


def compute_metrics(y_true, y_pred):
    """Calculate and return common classification evaluation metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels from the model.

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, weighted precision, weighted recall,
        weighted F1-score, and Cohen's Kappa coefficient.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "kappa": kappa
    }


def print_report(y_true, y_pred):
    """Print the confusion matrix and classification report.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels from the model.

    Notes
    -----
    The classification report provides precision, recall, and F1-score for each class,
    alongside overall accuracy, macro average, and weighted average.
    """
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, digits=4)

    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", rep)

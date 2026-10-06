"""
evaluate.py
-----------
Metrics for imbalanced fraud detection. AUC-PR is the primary metric;
accuracy is deliberately not used.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_recall_curve, precision_score, recall_score, f1_score,
    fbeta_score, confusion_matrix)


def best_threshold(y_true, y_scores, beta: float = 2.0) -> float:
    """Probability cut-off that maximises F-beta (default F2, recall-leaning).
    Tune this on TRAIN/CV scores only, never on the test set."""
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    prec, rec = prec[:-1], rec[:-1]
    fbeta = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(fbeta))])


def evaluate(name, y_true, scores, threshold) -> dict:
    """Return a row of metrics for one model at a given threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "model": name,
        "AUC_PR": average_precision_score(y_true, scores),
        "ROC_AUC": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "threshold": threshold,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }

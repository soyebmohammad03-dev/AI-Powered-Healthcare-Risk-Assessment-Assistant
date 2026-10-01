"""Evaluation metrics for fitted classifiers."""
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

CV_SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
              "f1": "f1", "roc_auc": "roc_auc"}


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),  # [[TN, FP], [FN, TP]]
    }

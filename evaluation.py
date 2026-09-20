"""Shared evaluation helpers: metrics and leakage-safe repeated stratified cross-validation."""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import N_SPLITS

METRICS = ["pr_auc", "roc_auc", "f1", "recall", "precision", "balanced_acc"]


def score_binary(y_true, proba, threshold=0.5):
    """PR-AUC is the headline metric (rare positive class); the rest use the given threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "pr_auc": average_precision_score(y_true, proba) if two_classes else np.nan,
        "roc_auc": roc_auc_score(y_true, proba) if two_classes else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
    }


def _fit_predict(pipeline, X, y, train_idx, valid_idx):
    model = clone(pipeline)                       # fresh, unfitted copy for every fold
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    return valid_idx, model.predict_proba(X.iloc[valid_idx])[:, 1]


def cv_evaluate(pipeline, X, y, n_splits=N_SPLITS, n_repeats=1, random_state=0, n_jobs=None):
    """Repeated stratified K-fold.  Returns (per-fold metrics DataFrame, OOF probabilities [repeats x n]).

    The same random_state gives identical folds for every pipeline, so results of different
    experiments are *paired* fold-by-fold.
    """
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    splits = list(splitter.split(X, y))
    results = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(pipeline, X, y, tr, va) for tr, va in splits)

    rows, oof = [], np.full((n_repeats, len(y)), np.nan)
    for k, (valid_idx, proba) in enumerate(results):
        repeat = k // n_splits
        oof[repeat, valid_idx] = proba
        rows.append({"repeat": repeat, "fold": k % n_splits, **score_binary(y.iloc[valid_idx], proba)})
    return pd.DataFrame(rows), oof


def summarize(folds, metrics=METRICS):
    """mean and std of each metric across folds."""
    out = {}
    for m in metrics:
        out[f"{m}_mean"] = folds[m].mean()
        out[f"{m}_std"] = folds[m].std(ddof=1)
    return out

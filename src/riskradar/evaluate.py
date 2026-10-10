"""Classification metrics used across the project.

* micro-F1 (== accuracy for single-label problems) answers "how often is the label right?"
* macro-F1 averages per-class F1 equally, so rare classes (AI, pandemic, reinsurance)
  count as much as common ones (regulatory) -- the metric that exposes lazy classifiers.
* bootstrap 95% CIs make the uncertainty from a ~200-item gold set explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from .taxonomy import KEYS


def per_class_report(y_true, y_pred, labels=KEYS) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return pd.DataFrame({"label": labels, "precision": p, "recall": r, "f1": f, "support": s})


def summary(y_true, y_pred, labels=KEYS, secondary=None) -> dict:
    """Headline metrics. Macro-F1 is averaged over classes present in the gold set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = [l for l in labels if (y_true == l).any()]
    _, _, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=present, average="macro", zero_division=0)
    _, _, f_micro, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), average="micro", zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f_macro),
        "micro_f1": float(f_micro),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if secondary is not None:
        # lenient accuracy: prediction matches the gold primary OR a gold secondary label
        ok = [p == t or (p in (s or [])) for p, t, s in zip(y_pred, y_true, secondary)]
        out["lenient_accuracy"] = float(np.mean(ok))
    return out


def bootstrap_ci(y_true, y_pred, metric: str = "macro_f1", n_boot: int = 1000, seed: int = 0,
                 labels=KEYS) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(summary(y_true[idx], y_pred[idx], labels)[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(y_true, pred_a, pred_b, metric: str = "macro_f1", n_boot: int = 1000,
                     seed: int = 0, labels=KEYS) -> dict:
    """P(metric(A) > metric(B)) under resampling of gold items, and the mean difference."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs.append(summary(y_true[idx], pred_a[idx], labels)[metric] - summary(y_true[idx], pred_b[idx], labels)[metric])
    diffs = np.asarray(diffs)
    return {"mean_diff": float(diffs.mean()), "p_a_better": float((diffs > 0).mean()),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))}


def confusion(y_true, y_pred, labels=KEYS) -> pd.DataFrame:
    m = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(m, index=list(labels), columns=list(labels))

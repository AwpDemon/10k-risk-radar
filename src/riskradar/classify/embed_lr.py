"""Embeddings + multinomial logistic regression.

Text is embedded once (nomic-embed-text through any OpenAI-compatible endpoint), then a
class-balanced logistic regression is trained. With only a few hundred gold labels we
report *out-of-fold* predictions from stratified 5-fold cross-validation, so every gold
item is predicted by a model that never saw it.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EMBED_PREFIX = "classification: "


def rf_text(heading: str, body: str, max_body: int = 1500) -> str:
    return heading + "\n" + body[:max_body]


def make_model(C: float = 4.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=5000, class_weight="balanced")


def cross_val_predict(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42,
                      C: float = 4.0) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny classes trigger "least populated class" warnings
        for tr, te in skf.split(X, y):
            m = make_model(C).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
    return pred

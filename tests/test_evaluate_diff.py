import numpy as np
import pandas as pd

from riskradar.diff import classify_change, diff_company, text_delta
from riskradar.evaluate import bootstrap_ci, confusion, paired_bootstrap, summary


def test_summary_metrics():
    y = ["a", "a", "b", "c"]
    p = ["a", "b", "b", "c"]
    s = summary(y, p, labels=("a", "b", "c"), secondary=[[], ["b"], [], []])
    assert s["accuracy"] == 0.75 and s["lenient_accuracy"] == 1.0
    assert abs(s["macro_f1"] - np.mean([2 / 3, 2 / 3, 1.0])) < 1e-9


def test_bootstrap_and_confusion():
    y = np.array(["a", "b"] * 20)
    lo, hi = bootstrap_ci(y, y, n_boot=50, labels=("a", "b"))
    assert lo == hi == 1.0
    cm = confusion(["a", "b"], ["b", "b"], labels=("a", "b"))
    assert cm.loc["a", "b"] == 1 and cm.loc["b", "b"] == 1
    d = paired_bootstrap(y, y, np.array(["a"] * 40), n_boot=50, labels=("a", "b"))
    assert d["p_a_better"] == 1.0


def test_text_delta_and_change_labels():
    old = "Cyber attacks could disrupt our systems."
    assert classify_change(text_delta(old, old)) == "unchanged"
    new = old + " " + "Generative AI tools increase the sophistication of phishing attacks. " * 15
    d = text_delta(old, new)
    assert d.added_words > 75 and classify_change(d) == "expanded"


def test_diff_company_new_removed_matched():
    rows = [("r22a", 2022, "Cyber risk", "body"), ("r22b", 2022, "LIBOR transition", "body"),
            ("r23a", 2023, "Cyber risk", "body"), ("r23c", 2023, "AI risk", "body")]
    rfs = pd.DataFrame(rows, columns=["rf_id", "fiscal_year", "heading", "body"]).assign(ticker="T")
    e = {"r22a": np.array([1.0, 0, 0]), "r22b": np.array([0, 1.0, 0]),
         "r23a": np.array([1.0, 0, 0]), "r23c": np.array([0, 0, 1.0])}
    d = diff_company(rfs, e).set_index("rf_id")
    assert d.loc["r23a", "change"] == "unchanged" and d.loc["r23a", "matched_rf_id"] == "r22a"
    assert d.loc["r23c", "change"] == "new"
    assert d.loc["r22b", "change"] == "removed"

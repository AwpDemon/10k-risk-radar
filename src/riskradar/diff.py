"""Year-over-year change detection for risk factors.

For each company and each pair of consecutive fiscal years:

1. embed every risk factor (heading + start of body);
2. solve a one-to-one assignment between last year's and this year's risk factors that
   maximises total cosine similarity (Hungarian algorithm), keeping pairs above a
   similarity threshold;
3. label this year's risk factors ``new`` (no match), ``unchanged``, ``modified`` or
   ``expanded`` (matched, with materially more text), and last year's unmatched ones
   ``removed``.

Embeddings catch reworded headings ("Cyber attacks..." -> "Cybersecurity incidents...")
that exact string diffs would call new; the word-level diff then measures *how much*
the matched text changed.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MATCH_THRESHOLD = 0.80     # cosine similarity; calibrated in docs (see README "Change detection")
UNCHANGED_RATIO = 0.97     # word-level SequenceMatcher ratio
EXPANDED_MIN_WORDS = 75    # net words added
EXPANDED_MIN_GROWTH = 1.25 # length ratio new/old

_WORD = re.compile(r"[A-Za-z0-9']+")


def words(t: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(t)]


@dataclass
class TextDelta:
    ratio: float
    added_words: int
    removed_words: int
    growth: float


def text_delta(old: str, new: str) -> TextDelta:
    a, b = words(old), words(new)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    added = removed = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("insert", "replace"):
            added += j2 - j1
        if op in ("delete", "replace"):
            removed += i2 - i1
    return TextDelta(sm.ratio(), added, removed, len(b) / max(len(a), 1))


def classify_change(delta: TextDelta) -> str:
    if delta.ratio >= UNCHANGED_RATIO:
        return "unchanged"
    if delta.added_words - delta.removed_words >= EXPANDED_MIN_WORDS and delta.growth >= EXPANDED_MIN_GROWTH:
        return "expanded"
    return "modified"


def match_pair(prev_emb: np.ndarray, curr_emb: np.ndarray, threshold: float = MATCH_THRESHOLD):
    """Return list of (curr_idx, prev_idx, sim) for matched pairs above threshold."""
    if len(prev_emb) == 0 or len(curr_emb) == 0:
        return []
    sim = curr_emb @ prev_emb.T
    r, c = linear_sum_assignment(-sim)
    return [(int(i), int(j), float(sim[i, j])) for i, j in zip(r, c) if sim[i, j] >= threshold]


def diff_company(rfs: pd.DataFrame, emb: dict[str, np.ndarray], threshold: float = MATCH_THRESHOLD) -> pd.DataFrame:
    """rfs: risk factors of ONE company (all years). emb: rf_id -> unit vector."""
    out = []
    years = sorted(rfs.fiscal_year.unique())
    for y0, y1 in zip(years, years[1:]):
        if y1 != y0 + 1:
            continue
        prev = rfs[rfs.fiscal_year == y0].reset_index(drop=True)
        curr = rfs[rfs.fiscal_year == y1].reset_index(drop=True)
        pe = np.stack([emb[i] for i in prev.rf_id]) if len(prev) else np.zeros((0, 1))
        ce = np.stack([emb[i] for i in curr.rf_id]) if len(curr) else np.zeros((0, 1))
        pairs = match_pair(pe, ce, threshold)
        matched_curr = {i: (j, s) for i, j, s in pairs}
        matched_prev = {j for _, j, _ in pairs}
        best_sim = (ce @ pe.T).max(axis=1) if len(prev) and len(curr) else np.zeros(len(curr))
        for i, row in curr.iterrows():
            rec = {"rf_id": row.rf_id, "ticker": row.ticker, "fiscal_year": y1, "prev_year": y0,
                   "best_sim": float(best_sim[i]) if len(best_sim) else 0.0}
            if i in matched_curr:
                j, s = matched_curr[i]
                p = prev.iloc[j]
                d = text_delta(p.heading + " " + p.body, row.heading + " " + row.body)
                rec.update(change=classify_change(d), matched_rf_id=p.rf_id, sim=s, text_ratio=d.ratio,
                           added_words=d.added_words, removed_words=d.removed_words, growth=d.growth)
            else:
                rec.update(change="new", matched_rf_id=None, sim=None, text_ratio=None,
                           added_words=None, removed_words=None, growth=None)
            out.append(rec)
        for j, p in prev.iterrows():
            if j not in matched_prev:
                out.append({"rf_id": p.rf_id, "ticker": p.ticker, "fiscal_year": y1, "prev_year": y0,
                            "best_sim": None, "change": "removed", "matched_rf_id": None, "sim": None,
                            "text_ratio": None, "added_words": None, "removed_words": None, "growth": None})
    return pd.DataFrame(out)

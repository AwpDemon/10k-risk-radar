"""Keyword baseline: count taxonomy keywords, heading hits weighted 3x. Deterministic, free,
and a useful floor -- any learned/LLM approach must beat it to justify its cost."""
from __future__ import annotations

import re
from functools import lru_cache

from ..taxonomy import CATEGORIES

HEADING_WEIGHT = 3.0


@lru_cache(maxsize=None)
def _patterns() -> dict[str, list[re.Pattern]]:
    pats = {}
    for c in CATEGORIES:
        # "word*" is a prefix match; everything else must match on word boundaries.
        pats[c.key] = [re.compile(r"\b" + re.escape(k.rstrip("*")) + ("" if k.endswith("*") else r"\b"), re.I)
                       for k in c.keywords]
    return pats


def keyword_scores(heading: str, body: str) -> dict[str, float]:
    scores = {}
    for key, pats in _patterns().items():
        s = 0.0
        for p in pats:
            s += HEADING_WEIGHT * len(p.findall(heading)) + len(p.findall(body))
        scores[key] = s
    return scores


def predict(heading: str, body: str, default: str = "other") -> str:
    scores = keyword_scores(heading, body)
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else default

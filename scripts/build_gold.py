"""Merge the labeling sheet (idx|primary|secondary|segment_ok|note) into data/gold/gold.csv."""
import sys

import pandas as pd

from riskradar import config
from riskradar.taxonomy import KEYS

sample = pd.read_csv(config.GOLD / "gold_sample.csv")
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.rstrip("\n")
    if not line.strip():
        continue
    idx, primary, secondary, ok, note = line.split("|")
    assert primary in KEYS, primary
    for s in filter(None, secondary.split(";")):
        assert s in KEYS, s
    rows.append({"idx": int(idx), "primary": primary, "secondary": secondary, "segment_ok": int(ok), "note": note})
lab = pd.DataFrame(rows).set_index("idx")
assert lab.index.is_unique and len(lab) == len(sample), (len(lab), len(sample))
gold = sample.join(lab)
gold.to_csv(config.GOLD / "gold.csv", index=False)
print(gold.primary.value_counts())
print(gold.groupby("split").primary.value_counts().unstack(0).fillna(0).astype(int))

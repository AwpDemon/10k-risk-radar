"""Draw the gold-set sample: 60 risk factors per sector (240 total), seeded.

The first 15 per sector are the *dev* split (used for prompt iteration); the remaining
45 per sector are the *test* split (used only for final reported numbers).
"""
import pandas as pd

from riskradar import config

SEED = 20261010
PER_SECTOR = 60
DEV_PER_SECTOR = 15

rf = pd.read_parquet(config.INTERIM / "risk_factors.parquet")
parts = []
for _sector, g in rf.groupby("sector"):
    s = g.sample(PER_SECTOR, random_state=SEED).copy()
    s["split"] = ["dev"] * DEV_PER_SECTOR + ["test"] * (PER_SECTOR - DEV_PER_SECTOR)
    parts.append(s)
gold = pd.concat(parts).sample(frac=1, random_state=SEED)  # shuffle so labeling order is random
gold["body_excerpt"] = gold.body.str.slice(0, 1200)
cols = ["rf_id", "split", "ticker", "sector", "fiscal_year", "group", "heading", "body_excerpt"]
gold[cols].to_csv(config.GOLD / "gold_sample.csv", index=False)
print(gold.groupby(["sector", "split"]).size())

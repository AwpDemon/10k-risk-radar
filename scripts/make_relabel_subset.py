"""Draw a random 50-item subset of the gold set, shuffled, with labels hidden, for the
intra-annotator consistency check (second labeling pass)."""
import pandas as pd

from riskradar import config

g = pd.read_csv(config.GOLD / "gold.csv", keep_default_na=False)
sub = g.sample(50, random_state=99)[["rf_id", "heading", "body_excerpt"]].reset_index(drop=True)
sub.to_csv(config.GOLD / "relabel_subset.csv", index=False)
print(len(sub))

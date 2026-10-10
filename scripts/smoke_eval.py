"""CI smoke eval: score the keyword baseline on the committed gold set and the committed
LLM predictions, and fail if either drops below a floor. Needs no network or GPU."""
import sys

import pandas as pd

from riskradar import config
from riskradar.classify import baseline
from riskradar.evaluate import summary

g = pd.read_csv(config.GOLD / "gold.csv", keep_default_na=False)
test = g[(g.split == "test") & (g.segment_ok == 1)]
pred = [baseline.predict(h, b) for h, b in zip(test.heading, test.body_excerpt)]
s = summary(test.primary, pred)
print(f"keyword baseline on test excerpts: accuracy={s['accuracy']:.3f} macro_f1={s['macro_f1']:.3f}")

p = pd.read_csv(config.RESULTS / "gold_predictions.csv", keep_default_na=False)
p = p[p.split == "test"]
best = "zero_shot:qwen2.5:14b-instruct"
s2 = summary(p.gold, p[best])
print(f"committed {best} predictions: accuracy={s2['accuracy']:.3f} macro_f1={s2['macro_f1']:.3f}")
sys.exit(0 if s["accuracy"] >= 0.6 and s2["macro_f1"] >= 0.85 else 1)

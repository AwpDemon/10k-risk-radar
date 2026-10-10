"""Error analysis for the best classifier on the test split.

Buckets every error into:
  * secondary_hit  -- the model chose the gold *secondary* label (a boundary disagreement)
  * flagged_ambiguous -- the annotator had already marked the item as ambiguous in the note
  * other_error    -- a plain miss
and writes data/results/error_analysis_test.csv + .json.
"""
import json
import sys

import pandas as pd

from riskradar import config

model = sys.argv[1] if len(sys.argv) > 1 else "zero_shot:qwen2.5:14b-instruct"
p = pd.read_csv(config.RESULTS / "gold_predictions.csv", keep_default_na=False)
g = pd.read_csv(config.GOLD / "gold.csv", keep_default_na=False)[["rf_id", "heading", "note"]]
d = p[p.split == "test"].merge(g, on="rf_id")
err = d[d[model] != d.gold].copy()
err["pred"] = err[model]
err["secondary_hit"] = err.apply(lambda r: r.pred in r.gold_secondary.split(";"), axis=1)
err["flagged_ambiguous"] = err.note.str.contains("ambiguous|=", regex=True)
err["bucket"] = err.apply(lambda r: "secondary_hit" if r.secondary_hit else
                          ("flagged_ambiguous" if r.flagged_ambiguous else "other_error"), axis=1)
pairs = (err.groupby(["gold", "pred"]).size().sort_values(ascending=False).head(10)
         .reset_index(name="n").to_dict(orient="records"))
out = {"model": model, "n_test": int(len(d)), "n_errors": int(len(err)),
       "buckets": err.bucket.value_counts().to_dict(), "top_confusions": pairs}
err[["rf_id", "gold", "gold_secondary", "pred", "bucket", "heading", "note"]].to_csv(
    config.RESULTS / "error_analysis_test.csv", index=False)
(config.RESULTS / "error_analysis_test.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

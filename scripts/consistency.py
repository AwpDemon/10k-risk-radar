"""Intra-annotator consistency: compare the second (blind, shuffled) labeling pass on the
50-item subset with the first-pass gold labels."""
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from riskradar import config

sub = pd.read_csv(config.GOLD / "relabel_subset.csv")
p2 = {}
for line in open(config.GOLD / "labels_pass2_subset.txt", encoding="utf-8"):
    if line.strip():
        i, lab = line.strip().split("|")
        p2[int(i)] = lab
sub["pass2"] = [p2[i] for i in range(len(sub))]
gold = pd.read_csv(config.GOLD / "gold.csv", keep_default_na=False).set_index("rf_id")
sub["pass1"] = sub.rf_id.map(gold.primary)
sub["pass1_secondary"] = sub.rf_id.map(gold.secondary)
agree = (sub.pass1 == sub.pass2)
lenient = agree | sub.apply(lambda r: r.pass2 in r.pass1_secondary.split(";"), axis=1)
res = {"n": len(sub), "exact_agreement": float(agree.mean()),
       "cohen_kappa": float(cohen_kappa_score(sub.pass1, sub.pass2)),
       "agreement_counting_pass1_secondary": float(lenient.mean()),
       "disagreements": sub[~agree][["rf_id", "heading", "pass1", "pass2"]].assign(
           heading=lambda d: d.heading.str.slice(0, 90)).to_dict(orient="records")}
(config.RESULTS / "consistency.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k != "disagreements"}, indent=2))
for d in res["disagreements"]:
    print(d)

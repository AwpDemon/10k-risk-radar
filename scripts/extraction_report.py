"""Extraction + segmentation quality report.

1. Automatic: share of filings where Item 1A was located (by strategy and sector),
   section length, and risk factors per filing.
2. Manual: `--spotcheck` prints the first/last blocks of 25 randomly chosen sections
   (seed 7) so a human can judge whether the boundaries are right. The judgments are
   recorded in data/results/extraction_spotcheck.csv.
"""
import json
import sys

import pandas as pd

from riskradar import config
from riskradar.pipeline import _blocks_from_json

sec = pd.read_parquet(config.INTERIM / "sections.parquet")
fil = pd.read_parquet(config.INTERIM / "filings.parquet")[["accession", "sector"]]
rf = pd.read_parquet(config.INTERIM / "risk_factors.parquet")
sec = sec.merge(fil, on="accession")
per = rf.groupby("accession").size()

report = {
    "filings": int(len(sec)),
    "item1a_located": int(sec.ok.sum()),
    "success_rate": float(sec.ok.mean()),
    "by_method": sec.method.value_counts().to_dict(),
    "by_sector_success": sec.groupby("sector").ok.mean().round(3).to_dict(),
    "section_chars_median": int(sec[sec.ok].n_chars.median()),
    "section_chars_min": int(sec[sec.ok].n_chars.min()),
    "risk_factors_total": int(len(rf)),
    "risk_factors_per_filing": {k: float(v) for k, v in per.describe().round(1).items()},
    "segmentation_style": rf["style"].value_counts().to_dict(),
}
(config.RESULTS / "extraction_report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

if "--spotcheck" in sys.argv:
    for r in sec.sample(25, random_state=7).itertuples():
        bl = [b for b in _blocks_from_json(r.blocks_json) if not b.is_table]
        print(f"=== {r.ticker} {r.fiscal_year} {r.method}")
        for b in bl[:2]:
            print("   START:", b.text[:110])
        for b in bl[-2:]:
            print("   END  :", b.text[:110])

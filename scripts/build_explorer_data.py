"""Build the compact, committed explorer table (data/sample/explorer.parquet).

It holds every risk factor's heading, a 400-character excerpt, labels and YoY change
status -- enough for the dashboard -- but not the full filing text, which stays in the
gitignored cache and can be rebuilt with `riskradar run`.
"""
import pandas as pd

from riskradar import config
from riskradar.insights import load

df = load()
d = pd.read_parquet(config.INTERIM / "diffs.parquet")
d = d[d.change != "removed"][["rf_id", "change"]]
filings = pd.read_parquet(config.INTERIM / "filings.parquet")[["accession", "url"]]
out = (df.merge(d, on="rf_id", how="left").merge(filings, on="accession", how="left")
       .assign(excerpt=lambda x: x.body.str.slice(0, 400)))
cols = ["rf_id", "accession", "ticker", "company", "sector", "fiscal_year", "group", "heading", "excerpt",
        "primary", "secondary", "severity", "is_ai", "ai_heading", "is_cyber", "is_climate", "change", "n_chars", "url"]
out[cols].to_parquet(config.SAMPLE / "explorer.parquet", index=False, compression="zstd")
print(len(out), "rows ->", config.SAMPLE / "explorer.parquet")

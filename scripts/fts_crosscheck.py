"""Independent cross-check of the AI trend with SEC EDGAR full-text search (EFTS).

For each fiscal year we ask EFTS which of our universe's 10-K filings contain a phrase
*anywhere* in the filing, and compare with how many of the same filings mention it
inside the Item 1A risk factors we extracted. EFTS is SEC's own index, so agreement on
the direction of the trend is a check that the extraction/segmentation is not
manufacturing it.
"""
import json

import pandas as pd

from riskradar import config
from riskradar.edgar import EdgarClient

PHRASES = ['"artificial intelligence"', '"generative AI"']
c = EdgarClient()
filings = pd.read_parquet(config.INTERIM / "filings.parquet")
rf = pd.read_parquet(config.INTERIM / "risk_factors.parquet")
ciks = sorted({int(x) for x in filings.cik})
acc_set = set(filings.accession)
rows = []
for phrase in PHRASES:
    for fy in sorted(filings.fiscal_year.unique()):
        fy_acc = set(filings[filings.fiscal_year == fy].accession)
        start, end = f"{fy + 1}-01-01", f"{fy + 1}-12-31"  # FY t 10-Ks are filed in t+1
        hits = set()
        for page in range(10):
            url_q = phrase
            r = c.full_text_search(url_q, forms="10-K", start=start, end=end, ciks=ciks) if page == 0 else \
                c.get_json(f"https://efts.sec.gov/LATEST/search-index?q={url_q}&forms=10-K&dateRange=custom"
                           f"&startdt={start}&enddt={end}&ciks={','.join(f'{x:010d}' for x in ciks)}&from={100 * page}")
            batch = r["hits"]["hits"]
            hits |= {h["_source"]["adsh"] for h in batch}
            if len(batch) < 100:
                break
        in_filing = hits & fy_acc
        term = phrase.strip('"').lower()
        in_1a = {a for a, g in rf[rf.fiscal_year == fy].groupby("accession")
                 if (g.heading + " " + g.body).str.lower().str.contains(term, regex=False).any()}
        rows.append({"phrase": phrase, "fiscal_year": int(fy), "filings": len(fy_acc),
                     "efts_any_document": len(in_filing), "item1a_extracted": len(in_1a & fy_acc),
                     "item1a_but_not_efts": len((in_1a & fy_acc) - in_filing)})
df = pd.DataFrame(rows)
df.to_csv(config.RESULTS / "fts_crosscheck.csv", index=False)
print(df.to_string(index=False))
(config.RESULTS / "fts_crosscheck.json").write_text(json.dumps(rows, indent=2))

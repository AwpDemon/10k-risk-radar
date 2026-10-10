"""Write data/gold/few_shot.csv: one hand-labeled example per class, drawn from risk factors
that are NOT in the gold set (so few-shot prompting never leaks evaluation items)."""
import pandas as pd

from riskradar import config

PICKS = [  # rf_id, primary, secondary, severity, ai_related, rationale
    ("0000070858-25-000139-013", "cyber_tech", "operational_people", 3, False, "Cybersecurity incidents at the bank or its third parties."),
    ("0000005513-26-000008-011", "ai_model", "regulatory_legal", 3, True, "Use of AI and evolving AI regulation."),
    ("0000899051-25-000015-002", "climate_catastrophe", "", 4, False, "Catastrophe and severe-weather losses."),
    ("0000320335-24-000006-015", "regulatory_legal", "", 3, False, "Heavy regulation and regulatory change."),
    ("0000092230-25-000020-001", "market_interest_rate", "", 3, False, "Interest-rate levels and changes."),
    ("0000035527-26-000124-000", "credit", "", 3, False, "Deteriorating borrower credit quality."),
    ("0001096343-23-000033-004", "reserving_underwriting", "", 3, False, "Actual losses may differ from loss reserves."),
    ("0000092230-25-000020-010", "liquidity_capital", "", 3, False, "Inability to access funding or monetize assets."),
    ("0001558370-25-001301-009", "reinsurance", "", 3, False, "Availability and price of reinsurance protection."),
    ("0000005272-24-000023-028", "pandemic_health", "", 3, False, "Epidemics, pandemics or other health crises."),
    ("0000049196-25-000020-009", "macro_geopolitical", "market_interest_rate", 3, False, "Global economic and geopolitical instability."),
    ("0000004977-23-000055-022", "competition_strategy", "", 2, False, "Competition for market share."),
    ("0000713676-23-000020-023", "operational_people", "", 2, False, "Dependence on skilled labor and talent retention."),
    ("0001281761-24-000010-038", "other", "regulatory_legal", 1, False, "Anti-takeover provisions may depress share value."),
]
rf = pd.read_parquet(config.INTERIM / "risk_factors.parquet").set_index("rf_id")
gold = set(pd.read_csv(config.GOLD / "gold.csv").rf_id)
rows = []
for rid, p, s, sev, ai, why in PICKS:
    assert rid not in gold, rid
    r = rf.loc[rid]
    rows.append({"rf_id": rid, "ticker": r.ticker, "fiscal_year": r.fiscal_year, "heading": r.heading,
                 "body_excerpt": r.body[:600], "primary": p, "secondary": s, "severity": sev,
                 "ai_related": ai, "rationale": why})
pd.DataFrame(rows).to_csv(config.GOLD / "few_shot.csv", index=False)
print(len(rows), "few-shot examples written")

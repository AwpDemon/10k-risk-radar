"""Risk-factor taxonomy: 14 primary categories for insurers, brokers, and banks.

Each risk factor gets exactly one *primary* category (what the risk is fundamentally about,
i.e. the loss driver) and optionally secondary categories. Definitions here are the single
source of truth: they feed the LLM prompts, the keyword baseline, the labeling guidelines
(docs/LABELING_GUIDELINES.md), and the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    name: str
    definition: str
    includes: str
    excludes: str
    keywords: tuple[str, ...]


CATEGORIES: tuple[Category, ...] = (
    Category(
        "cyber_tech", "Cyber & technology",
        "Cyberattacks, data breaches, privacy/data-protection failures, IT system outages, "
        "technology change and digital disruption.",
        "ransomware, hacking, information security, system failures, cloud/vendor IT outages, "
        "data privacy laws when the risk is about protecting data",
        "AI/model-specific risks (-> ai_model); cyber *insurance claims* written by an insurer "
        "are underwriting exposure (-> reserving_underwriting) unless the text is about the "
        "company's own systems",
        ("cyber", "cybersecurity", "data breach", "hack", "ransomware", "malware",
         "information security", "information technology", "systems failure", "technology",
         "privacy", "personal information", "phishing", "denial of service", "computer"),
    ),
    Category(
        "ai_model", "AI & model risk",
        "Artificial intelligence, machine learning, generative AI, and reliance on models, "
        "algorithms, analytics or quantitative assumptions that may be wrong.",
        "AI adoption/competition, AI regulation, model error, actuarial/pricing/credit models "
        "when the text's main point is that models or assumptions may be flawed",
        "reserve adequacy where models are only mentioned in passing (-> reserving_underwriting)",
        ("artificial intelligence", "generative ai", "machine learning", "ai", "algorithm",
         "models", "modeling", "analytics", "model risk", "large language"),
    ),
    Category(
        "climate_catastrophe", "Climate & catastrophe",
        "Natural and man-made catastrophes, severe weather, climate change (physical and "
        "transition risk), terrorism events, and ESG/climate-related scrutiny.",
        "hurricanes, wildfires, earthquakes, convective storms, terrorism losses, climate "
        "regulation and ESG disclosure pressure",
        "pandemics (-> pandemic_health)",
        ("catastrophe", "hurricane", "wildfire", "earthquake", "climate", "weather", "flood",
         "storm", "terrorism", "natural disaster", "esg", "sustainability", "greenhouse"),
    ),
    Category(
        "regulatory_legal", "Regulatory, legal & compliance",
        "Laws, regulation, supervision, capital/solvency rules imposed by regulators, tax law, "
        "litigation, investigations, sanctions and compliance failures.",
        "insurance/bank regulation, Dodd-Frank, Basel, NAIC, stress tests, consumer protection, "
        "AML/sanctions, lawsuits, tax changes, accounting standard changes",
        "regulation that is specifically about AI (-> ai_model) or data privacy (-> cyber_tech)",
        ("regulation", "regulatory", "regulators", "legislation", "law", "laws", "litigation",
         "lawsuit", "compliance", "supervision", "tax", "sanction", "enforcement", "legal",
         "capital requirements", "dodd-frank", "basel", "naic"),
    ),
    Category(
        "market_interest_rate", "Market & interest-rate risk",
        "Changes in interest rates, credit spreads, equity markets, FX, and the resulting value "
        "of the investment portfolio or net interest income.",
        "investment portfolio valuation, unrealized losses, low/high rate environment, "
        "hedging, LIBOR transition, inflation's effect on rates/asset values",
        "borrower default (-> credit); broad recession narrative (-> macro_geopolitical)",
        ("interest rate", "interest rates", "market volatility", "equity market", "investment portfolio",
         "foreign exchange", "currency", "spread", "hedging", "net interest", "libor", "valuation"),
    ),
    Category(
        "credit", "Credit & counterparty risk",
        "Borrowers, issuers, reinsurers or other counterparties failing to pay; loan losses; "
        "allowance for credit losses; concentrations of credit exposure.",
        "loan defaults, commercial real estate credit, consumer credit, counterparty exposure, "
        "investment defaults/impairments",
        "reinsurer credit when the text is mainly about reinsurance availability (-> reinsurance)",
        ("credit risk", "default", "borrower", "counterparty", "loan losses", "credit losses",
         "allowance", "delinquen*", "creditworthiness", "commercial real estate", "impairment"),
    ),
    Category(
        "reserving_underwriting", "Reserving, pricing & underwriting",
        "Insurance-specific risk that claims, benefits, mortality/morbidity, or losses exceed "
        "reserves and pricing; underwriting and actuarial assumptions; emerging claim issues.",
        "loss reserve adequacy, social inflation, mass torts, asbestos, mortality/longevity, "
        "policyholder behavior, product guarantees, emerging coverage issues",
        "catastrophe-driven losses (-> climate_catastrophe)",
        ("reserve", "reserves", "underwriting", "pricing", "claims", "mortality", "morbidity",
         "longevity", "actuarial", "policyholder", "loss costs", "asbestos", "social inflation"),
    ),
    Category(
        "liquidity_capital", "Liquidity, funding, capital & ratings",
        "Ability to fund obligations, access capital markets, maintain capital levels, pay "
        "dividends from subsidiaries, and keep credit / financial-strength ratings.",
        "deposit outflows, holding-company dependence on subsidiary dividends, ratings "
        "downgrades, debt covenants, collateral calls",
        "regulator-imposed capital rules themselves (-> regulatory_legal)",
        ("liquidity", "funding", "capital", "rating", "ratings", "downgrade", "dividends",
         "borrow", "debt", "deposits", "collateral", "holding company"),
    ),
    Category(
        "reinsurance", "Reinsurance",
        "Availability, cost, adequacy and collectibility of reinsurance or retrocession.",
        "reinsurance program limits, reinsurer insolvency, rising reinsurance prices",
        "",
        ("reinsurance", "reinsurer", "reinsurers", "retrocession", "ceded"),
    ),
    Category(
        "pandemic_health", "Pandemic & public health",
        "Pandemics (e.g. COVID-19) and other public-health crises and their business effects.",
        "COVID-19 impacts on operations, claims, mortality, economy",
        "",
        ("pandemic", "covid", "coronavirus", "epidemic", "public health", "outbreak"),
    ),
    Category(
        "macro_geopolitical", "Macroeconomic & geopolitical",
        "General economic conditions, recession, inflation, geopolitical conflict, political "
        "instability, government shutdowns, and country risk.",
        "economic downturn, war (Russia/Ukraine, Middle East), trade policy, sovereign debt, "
        "political uncertainty",
        "rate-specific effects (-> market_interest_rate)",
        ("economic conditions", "recession", "economy", "inflation", "geopolitical", "war",
         "political", "conflict", "tariff", "trade", "sovereign", "global economic", "downturn"),
    ),
    Category(
        "competition_strategy", "Competition, strategy & reputation",
        "Competitive pressure, strategic execution, acquisitions/integration, new products, "
        "brand and reputational damage, distribution relationships, customer retention.",
        "M&A integration, pricing competition, disintermediation, fintech/insurtech "
        "competitors, ESG reputation (when about reputation), client concentration",
        "AI-driven competition (-> ai_model)",
        ("competition", "competitive", "competitors", "acquisition", "acquisitions", "strategy",
         "reputation", "brand", "growth", "market share", "distribution", "integration"),
    ),
    Category(
        "operational_people", "Operational & people",
        "Failures of processes, controls, third-party vendors, outsourcing, fraud, conduct, "
        "and dependence on key personnel / talent; business continuity.",
        "key employees, talent retention, internal controls over financial reporting, "
        "third-party service providers, misconduct, operational errors",
        "IT/cyber failures (-> cyber_tech)",
        ("personnel", "employees", "talent", "key executives", "third-party", "third parties",
         "vendors", "outsourc*", "operational", "fraud", "misconduct", "internal controls", "human capital"),
    ),
    Category(
        "other", "Other / corporate structure",
        "Stock-price, shareholder-rights, anti-takeover, governance, holding-company/foreign "
        "domicile structure, and accounting-estimate risks that fit nowhere else.",
        "share price volatility, controlling shareholder, Bermuda/Irish domicile issues, "
        "goodwill impairment, critical accounting estimates",
        "",
        ("common stock", "share price", "shareholders", "anti-takeover", "takeover", "bermuda",
         "ireland", "goodwill", "controlling", "articles of association", "bye-laws"),
    ),
)

KEYS: tuple[str, ...] = tuple(c.key for c in CATEGORIES)
BY_KEY: dict[str, Category] = {c.key: c for c in CATEGORIES}


def taxonomy_markdown() -> str:
    lines = ["| key | name | definition | includes | excludes / route elsewhere |", "|---|---|---|---|---|"]
    for c in CATEGORIES:
        lines.append(f"| `{c.key}` | {c.name} | {c.definition} | {c.includes} | {c.excludes or '-'} |")
    return "\n".join(lines)


def taxonomy_prompt_block() -> str:
    parts = []
    for c in CATEGORIES:
        parts.append(f"- {c.key}: {c.definition} Includes: {c.includes}."
                     + (f" Not: {c.excludes}." if c.excludes else ""))
    return "\n".join(parts)

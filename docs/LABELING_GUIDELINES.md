# Labeling guidelines (gold set)

These rules were written **before** labeling and applied to every gold item in
`data/gold/gold.csv`. They exist so that labels are reproducible: a second annotator
following this page should reach the same label most of the time (we measure that, see
"Consistency" in the README).

## What is being labeled

One unit = one risk factor as produced by the segmenter: its **heading** (the bold/italic
sub-caption the SEC requires) and its **body** (the paragraphs under it). The labeling view
shows the heading and the first ~1,200 characters of the body, which is also roughly what
the classifiers weigh most.

For each unit record:

| field | values | required |
|---|---|---|
| `primary` | one taxonomy key (below) | yes |
| `secondary` | 0-2 other keys, `;`-separated | no |
| `segment_ok` | `1` if the unit is a real, single risk factor; `0` if the segmenter produced junk (a summary bullet, a fragment, two risk factors glued together) | yes |
| `note` | free text for hard calls | no |

Units with `segment_ok = 0` are kept in the file (they measure segmentation quality) but are
**excluded from classification metrics**: a classifier should not be graded on text that is
not a risk factor.

## The taxonomy

See `src/riskradar/taxonomy.py` for the machine-readable version; the table in
`docs/TAXONOMY.md` is generated from it.

`cyber_tech`, `ai_model`, `climate_catastrophe`, `regulatory_legal`, `market_interest_rate`,
`credit`, `reserving_underwriting`, `liquidity_capital`, `reinsurance`, `pandemic_health`,
`macro_geopolitical`, `competition_strategy`, `operational_people`, `other`.

## Decision rules (in order)

1. **Primary = the loss driver named in the heading.** Ask "what event or condition, if it
   happens, causes the harm?" Headings follow the pattern *[driver] could [harm]*; label the
   driver, not the harm. "Changes in interest rates could reduce our net interest income" is
   `market_interest_rate`, not `liquidity_capital`.
2. **If the heading is generic** ("We face risks in our business"), use the body's first
   two sentences.
3. **Regulation about a specific topic goes to the topic when the topic is AI or data
   privacy** (`ai_model`, `cyber_tech`); all other regulation, supervision, capital rules
   imposed by regulators, tax and litigation go to `regulatory_legal`.
4. **Capital: who is acting?** Regulators changing capital/stress-test rules ->
   `regulatory_legal`. The company being unable to fund itself, keep ratings, or upstream
   dividends -> `liquidity_capital`.
5. **Insurers' losses:** catastrophe/weather/climate-driven -> `climate_catastrophe`;
   reserve adequacy, pricing, mortality/morbidity, social inflation, mass torts ->
   `reserving_underwriting`; reinsurance availability/cost/collectibility -> `reinsurance`.
6. **Investments:** price/rate/spread movements -> `market_interest_rate`; issuer default
   or impairment -> `credit`.
7. **Technology:** cyberattacks, outages, IT change, data privacy -> `cyber_tech`; anything
   whose subject is AI/ML/generative AI, or models/assumptions being wrong -> `ai_model`.
8. **People and process:** key personnel, talent, vendors/outsourcing, controls, fraud,
   misconduct -> `operational_people`.
9. **Economy:** recession, inflation (as an economic condition), war, politics, trade ->
   `macro_geopolitical`. Inflation's effect on *claims costs* -> `reserving_underwriting`.
10. **Competition, M&A, strategy, reputation, distribution, growth** -> `competition_strategy`.
11. **Pandemic** -> `pandemic_health` even when the harm is economic.
12. **Shareholder/structure** (stock price, anti-takeover, foreign domicile, controlling
    holder, goodwill/accounting estimates) -> `other`.
13. **Secondary** labels only for topics that get at least a full sentence of substance,
    not a passing list item.
14. **When torn between two**, pick the one the heading's grammatical subject names and put
    the other in `secondary`; write a `note`.

### Clarifications added during labeling (v1.1)

These came up repeatedly in the first ~100 items and were then applied to every item
(the `note` column records the hard calls):

15. **Accounting estimates, valuation judgments, credit-loss allowance methodology, goodwill
    impairment, dividend policy** -> `other` (the driver is management judgment / reporting),
    with the economic topic as `secondary`. **Changes in accounting standards** imposed by
    FASB/SEC -> `regulatory_legal`.
16. **Intellectual property** (protection or infringement claims) -> `competition_strategy`.
17. **ESG / sustainability scrutiny** -> `climate_catastrophe` (the taxonomy groups
    climate and ESG pressure); reputation-only framing -> `competition_strategy`.
18. **Pension obligations / plan assets, FX translation, mark-to-market** -> `market_interest_rate`.
19. **Technology change, digital capability, IT systems** -> `cyber_tech`; enterprise
    risk-management frameworks, internal controls, business continuity -> `operational_people`.
20. **Resolution regimes / loss absorption by security holders (TLAC)** -> `regulatory_legal`.

## Worked examples (paraphrased headings)

| heading (paraphrased) | primary | secondary | why |
|---|---|---|---|
| Catastrophe losses, including from climate change, could materially reduce our profitability | climate_catastrophe | | rule 5 |
| Our loss reserves may prove inadequate, which could harm results | reserving_underwriting | | rule 5 |
| Reinsurance may be unavailable at acceptable cost, or reinsurers may fail to pay | reinsurance | credit | rule 5; reinsurer credit is secondary |
| A cyberattack or other security breach could disrupt operations | cyber_tech | | rule 7 |
| Our use of AI and machine-learning tools exposes us to new risks | ai_model | regulatory_legal | rule 7 |
| Changes in regulation, including stress-test-driven capital requirements, could restrict our activities | regulatory_legal | liquidity_capital | rules 3-4 |
| As a holding company we depend on subsidiaries' dividends | liquidity_capital | | rule 4 |
| A downgrade of our financial-strength ratings could hurt sales | liquidity_capital | competition_strategy | rule 4 |
| Our success depends on retaining key personnel | operational_people | | rule 8 |
| Russia-Ukraine and Middle East conflicts could hurt markets and our business | macro_geopolitical | | rule 9 |
| Increased competition, including from insurtechs, could reduce margins | competition_strategy | | rule 10 |
| The market price of our common shares may be volatile | other | | rule 12 |

## Consistency check

After labeling all items, a random subset was re-labeled from a shuffled file with the
first-pass labels hidden. Agreement (accuracy and Cohen's kappa) between the two passes is
reported in the README. Because both passes were made by the same annotator, this measures
*intra*-annotator consistency (how stable the guidelines are), not inter-annotator agreement.

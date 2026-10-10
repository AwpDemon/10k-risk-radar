"""Zero-shot and few-shot LLM classification with structured JSON output.

The model returns one JSON object per risk factor. Output is validated against the
taxonomy; anything unparseable or off-taxonomy is counted (we report the invalid-output
rate) and mapped to "other" so metrics are never silently inflated.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..llm import LLMClient, parse_json
from ..taxonomy import KEYS, taxonomy_prompt_block

PROMPT_VERSION = "v3"
MAX_BODY_CHARS = 2500

SYSTEM = """You are a risk analyst who classifies risk factors from SEC 10-K filings of insurers, insurance brokers and banks.

Taxonomy (use these keys exactly):
{taxonomy}

Decision rules for hard cases:
- Label the DRIVER named in the heading (the event or condition that causes harm), not the harm. "Changes in interest rates could reduce our liquidity" -> market_interest_rate.
- Regulation about AI -> ai_model; regulation about data privacy/cybersecurity -> cyber_tech; all other laws, regulation, supervision, tax, litigation, investigations, sanctions, accounting-standard changes and resolution regimes -> regulatory_legal.
- Capital: regulators changing capital/stress-test rules -> regulatory_legal; the company being unable to fund itself, maintain capital, keep its credit/financial-strength ratings, meet debt covenants, or receive subsidiary dividends -> liquidity_capital.
- Investments: price, interest-rate, spread, currency (FX) or equity-market movements, pension-plan asset/obligation values -> market_interest_rate; issuer/borrower/counterparty default -> credit.
- Management's accounting estimates, valuation judgments, goodwill impairment, dividend policy, share price, anti-takeover provisions, shareholder tax -> other.
- Technology change, digital capability, IT systems, data security -> cyber_tech; risk-management processes, internal controls, vendors/outsourcing, business continuity, fraud, key people -> operational_people.
- Intellectual property, reputation, brand, M&A and integration, new products, distribution, competition, strategy execution -> competition_strategy.
- ESG / sustainability scrutiny and climate change -> climate_catastrophe.
- Inflation, recession, geopolitics, government/political actions, country risk -> macro_geopolitical; but inflation's effect on claim costs -> reserving_underwriting.

Rules:
- primary_category: the ONE category describing what fundamentally drives the potential loss in this risk factor. Judge by the heading first, then the body.
- secondary_categories: up to 2 other categories that are substantively discussed (not just mentioned in passing). May be empty.
- severity: 1-5, how severe the potential impact is AS WORDED. 1 = generic/boilerplate with no stated magnitude; 3 = "material adverse effect" on results or financial condition; 5 = threatens solvency/viability or describes losses that are already occurring at material scale.
- ai_related: true only if the text discusses artificial intelligence, machine learning or generative AI (not just "models" or "technology").
- rationale: one short sentence.

Return ONLY a JSON object: {{"primary_category": "<key>", "secondary_categories": ["<key>"], "severity": <1-5>, "ai_related": <true|false>, "rationale": "<sentence>"}}"""


def build_user(heading: str, body: str) -> str:
    body = body[:MAX_BODY_CHARS] + (" [...]" if len(body) > MAX_BODY_CHARS else "")
    return f"Risk factor heading:\n{heading}\n\nRisk factor text:\n{body}"


@dataclass
class FewShotExample:
    heading: str
    body: str
    primary: str
    secondary: list[str] = field(default_factory=list)
    severity: int = 3
    ai_related: bool = False
    rationale: str = ""


def build_messages(heading: str, body: str, shots: list[FewShotExample] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM.format(taxonomy=taxonomy_prompt_block())}]
    for ex in shots or []:
        msgs.append({"role": "user", "content": build_user(ex.heading, ex.body[:600])})
        msgs.append({"role": "assistant", "content": json.dumps({
            "primary_category": ex.primary, "secondary_categories": ex.secondary,
            "severity": ex.severity, "ai_related": ex.ai_related, "rationale": ex.rationale,
        })})
    msgs.append({"role": "user", "content": build_user(heading, body)})
    return msgs


def _coerce_key(v) -> str | None:
    if not isinstance(v, str):
        return None
    k = re.sub(r"[^a-z_]", "", v.strip().lower().replace(" ", "_").replace("-", "_").replace("&", ""))
    if k in KEYS:
        return k
    for key in KEYS:  # tolerate e.g. "cyber" or "reinsurance_risk"
        if k and (k.startswith(key) or key.startswith(k)):
            return key
    return None


@dataclass
class LLMPrediction:
    primary: str
    secondary: list[str]
    severity: int | None
    ai_related: bool | None
    rationale: str
    valid: bool
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    raw: str


def validate(raw: str) -> tuple[dict, bool]:
    obj = parse_json(raw)
    if obj is None:
        return {}, False
    primary = _coerce_key(obj.get("primary_category"))
    sec = [k for k in (_coerce_key(s) for s in (obj.get("secondary_categories") or []) if isinstance(s, str)) if k]
    sev = obj.get("severity")
    try:
        sev = int(sev)
        sev = sev if 1 <= sev <= 5 else None
    except (TypeError, ValueError):
        sev = None
    ai = obj.get("ai_related")
    ai = ai if isinstance(ai, bool) else (str(ai).lower() == "true" if ai is not None else None)
    out = {"primary": primary or "other", "secondary": [s for s in sec if s != primary][:2],
           "severity": sev, "ai_related": ai, "rationale": str(obj.get("rationale", ""))[:300]}
    return out, primary is not None


def classify_one(client: LLMClient, heading: str, body: str,
                 shots: list[FewShotExample] | None = None, model: str | None = None) -> LLMPrediction:
    res = client.chat(build_messages(heading, body, shots), json_mode=True, max_tokens=200, model=model)
    parsed, ok = validate(res.content)
    if not parsed:
        parsed = {"primary": "other", "secondary": [], "severity": None, "ai_related": None, "rationale": ""}
    return LLMPrediction(**parsed, valid=ok, prompt_tokens=res.prompt_tokens,
                         completion_tokens=res.completion_tokens, latency_s=res.latency_s,
                         cached=res.cached, raw=res.content)


def classify_many(client: LLMClient, items: list[tuple[str, str]],
                  shots: list[FewShotExample] | None = None, model: str | None = None,
                  workers: int = 4, progress: bool = True) -> list[LLMPrediction]:
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(classify_one, client, h, b, shots, model) for h, b in items]
        it = tqdm(futs, desc=f"llm:{model or client.settings.model}", disable=not progress)
        return [f.result() for f in it]

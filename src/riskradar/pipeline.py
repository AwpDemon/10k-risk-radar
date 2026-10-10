"""Pipeline stages. Each stage reads/writes parquet tables under data/interim/ so stages
can be re-run independently (and the expensive ones are cached)."""
from __future__ import annotations

import logging
import re
from collections import namedtuple
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import config
from .edgar import EdgarClient, list_10k_filings
from .universe import load_universe

log = logging.getLogger(__name__)

FILINGS = "filings.parquet"
SECTIONS = "sections.parquet"
RISK_FACTORS = "risk_factors.parquet"


def interim(name: str) -> Path:
    return config.INTERIM / name


def stage_fetch(client: EdgarClient | None = None, tickers: list[str] | None = None) -> pd.DataFrame:
    """List 10-K filings for the universe and download each primary document (cached)."""
    config.ensure_dirs()
    client = client or EdgarClient()
    companies, fiscal_years = load_universe(client)
    if tickers:
        wanted = {t.upper() for t in tickers}
        companies = [c for c in companies if c.ticker in wanted]
    rows = []
    for co in tqdm(companies, desc="companies"):
        for f in list_10k_filings(client, co.cik, co.ticker, co.name, co.sector, fiscal_years):
            client.filing_document(f.cik, f.accession, f.primary_document)
            rows.append({**asdict(f), "fiscal_year": f.fiscal_year, "url": f.url})
    df = pd.DataFrame(rows)
    df.to_parquet(interim(FILINGS), index=False)
    log.info("fetched %d filings (%d network requests)", len(df), client.network_requests)
    return df


def _extract_one(client: EdgarClient, row) -> tuple[dict, list]:
    from .extract import exhibit13_candidates, locate_by_heading, locate_risk_factors
    from .html_blocks import parse_blocks

    raw = client.filing_document(int(row.cik), row.accession, row.primary_document)
    blocks = parse_blocks(raw)
    sec = locate_risk_factors(blocks)
    source_doc = row.primary_document
    if not sec.ok:
        try:
            idx = client.filing_index_html(int(row.cik), row.accession)
            for name in exhibit13_candidates(idx):
                ex_blocks = parse_blocks(client.filing_document(int(row.cik), row.accession, name))
                sec13 = locate_by_heading(ex_blocks, method="exhibit13")
                if sec13.ok:
                    sec, source_doc = sec13, name
                    break
        except Exception as exc:  # pragma: no cover - network edge cases
            sec.note += f"; exhibit13 lookup failed: {exc}"
    meta = {
        "accession": row.accession, "ticker": row.ticker, "fiscal_year": int(row.fiscal_year),
        "method": sec.method, "ok": sec.ok, "n_chars": sec.n_chars, "n_blocks": len(sec.blocks),
        "source_document": source_doc, "note": sec.note, "doc_blocks": len(blocks),
    }
    return meta, sec.blocks


_WORKER_CLIENT: EdgarClient | None = None
_Row = namedtuple("_Row", ["cik", "accession", "primary_document", "ticker", "fiscal_year"])


def _extract_worker(args) -> dict:
    """Process-pool worker. Each process gets its own client with a share of the rate limit
    so that the *total* request rate stays under SEC's 10 req/s even when cache misses hit
    the network from several processes at once."""
    import json
    from dataclasses import asdict as _asdict

    global _WORKER_CLIENT
    row, max_rps = args
    if _WORKER_CLIENT is None:
        _WORKER_CLIENT = EdgarClient(max_rps=max_rps)
    meta, blocks = _extract_one(_WORKER_CLIENT, row)
    meta["blocks_json"] = json.dumps([_asdict(b) for b in blocks])
    return meta


def stage_extract(workers: int = 4) -> pd.DataFrame:
    """Locate Item 1A in every fetched filing; store section blocks as JSON."""
    from concurrent.futures import ProcessPoolExecutor

    filings = pd.read_parquet(interim(FILINGS))
    rows = [_Row(int(r.cik), r.accession, r.primary_document, r.ticker, int(r.fiscal_year))
            for r in filings.itertuples()]
    per_worker_rps = config.SEC_MAX_RPS / workers
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(tqdm(ex.map(_extract_worker, [(r, per_worker_rps) for r in rows]), total=len(rows), desc="extract"))
    df = pd.DataFrame(out)
    df.to_parquet(interim(SECTIONS), index=False)
    log.info("extraction ok for %d/%d filings", int(df.ok.sum()), len(df))
    return df


def _blocks_from_json(s: str):
    import json
    from .html_blocks import Block

    return [Block(**d) for d in json.loads(s)]


QUANT = re.compile(r"\$\s?\d[\d,.]*\s*(million|billion|trillion)?|\b\d+(\.\d+)?\s?(%|percent)", re.I)
AI_TERMS = re.compile(
    r"artificial intelligence|generative ai|\bgen ?ai\b|machine learning|\bAI\b|large language model", re.I)


def stage_segment() -> pd.DataFrame:
    """Split each extracted section into risk factors (one row per risk factor)."""
    from .segment import segment

    filings = pd.read_parquet(interim(FILINGS))
    sections = pd.read_parquet(interim(SECTIONS))
    meta = filings.set_index("accession")
    rows = []
    for sec in sections[sections.ok].itertuples():
        rfs = segment(_blocks_from_json(sec.blocks_json))
        f = meta.loc[sec.accession]
        for i, rf in enumerate(rfs):
            rows.append({
                "rf_id": f"{sec.accession}-{i:03d}", "accession": sec.accession, "ticker": f.ticker,
                "company": f.company, "sector": f.sector, "fiscal_year": int(f.fiscal_year),
                "filing_date": f.filing_date, "rf_index": i, "group": rf.group, "heading": rf.heading,
                "body": rf.body, "style": rf.style, "n_chars": len(rf.text),
                "quantified": bool(QUANT.search(rf.text)),
                "ai_keyword": bool(AI_TERMS.search(rf.text)),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(interim(RISK_FACTORS), index=False)
    log.info("segmented %d risk factors from %d sections", len(df), int(sections.ok.sum()))
    return df

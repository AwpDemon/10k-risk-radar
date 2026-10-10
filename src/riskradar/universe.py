"""Load the company universe and resolve tickers to SEC CIKs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config
from .edgar import EdgarClient


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str
    sector: str


def load_universe(client: EdgarClient, path: Path | None = None) -> tuple[list[Company], set[int]]:
    cfg = yaml.safe_load(Path(path or config.CONFIGS / "universe.yaml").read_text(encoding="utf-8"))
    tickers = client.company_tickers()
    companies = []
    for row in cfg["companies"]:
        t = row["ticker"].upper()
        if t not in tickers:
            raise KeyError(f"Ticker {t} not found in SEC company_tickers.json")
        cik = tickers[t]
        name = client.submissions(cik).get("name", t)
        companies.append(Company(ticker=t, cik=cik, name=name, sector=row["sector"]))
    return companies, set(cfg["fiscal_years"])

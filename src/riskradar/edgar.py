"""Polite SEC EDGAR client: throttled, retried, and disk-cached.

SEC fair-access rules (https://www.sec.gov/os/accessing-edgar-data):
  * declare a User-Agent with contact info, otherwise EDGAR answers 403;
  * stay at or below 10 requests/second per IP.

We default to 5 req/s and cache every response on disk (gzip), so re-running the
pipeline never re-downloads a filing.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from . import config

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
FTS_URL = "https://efts.sec.gov/LATEST/search-index"


class RateLimiter:
    """Minimum-interval limiter, thread safe."""

    def __init__(self, max_per_second: float):
        self.min_interval = 1.0 / max_per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = self._next
            self._next = now + self.min_interval


class EdgarClient:
    def __init__(
        self,
        cache_dir: Path | None = None,
        user_agent: str = config.SEC_USER_AGENT,
        max_rps: float = config.SEC_MAX_RPS,
        session: requests.Session | None = None,
    ):
        if max_rps > 10:
            raise ValueError("SEC fair-access policy caps clients at 10 requests/second")
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent that includes contact information (an email)")
        self.cache_dir = Path(cache_dir or config.CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.limiter = RateLimiter(max_rps)
        self.network_requests = 0

    # ---------- low level ----------
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha1(url.encode()).hexdigest()
        tail = url.rstrip("/").rsplit("/", 1)[-1][:60].replace("?", "_").replace("&", "_").replace("=", "_")
        return self.cache_dir / h[:2] / f"{h[:16]}_{tail}.gz"

    def get_bytes(self, url: str, use_cache: bool = True) -> bytes:
        path = self._cache_path(url)
        if use_cache and path.exists():
            return gzip.decompress(path.read_bytes())
        delay = 1.0
        for _attempt in range(6):
            self.limiter.wait()
            self.network_requests += 1
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(gzip.compress(resp.content))
                return resp.content
            if resp.status_code == 403:
                raise PermissionError(
                    f"EDGAR returned 403 for {url}. Check the User-Agent header and request rate."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                log.warning("EDGAR %s for %s, backing off %.1fs", resp.status_code, url, delay)
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"EDGAR request failed after retries: {url}")

    def get_json(self, url: str, use_cache: bool = True) -> dict:
        return json.loads(self.get_bytes(url, use_cache=use_cache))

    # ---------- endpoints ----------
    def company_tickers(self) -> dict[str, int]:
        data = self.get_json(TICKERS_URL)
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}

    def submissions(self, cik: int, use_cache: bool = True) -> dict:
        return self.get_json(SUBMISSIONS_URL.format(cik=cik), use_cache=use_cache)

    def filing_document(self, cik: int, accession: str, document: str) -> bytes:
        url = ARCHIVE_URL.format(cik=cik, acc_nodash=accession.replace("-", ""), doc=document)
        return self.get_bytes(url)

    def filing_index(self, cik: int, accession: str) -> dict:
        url = ARCHIVE_URL.format(cik=cik, acc_nodash=accession.replace("-", ""), doc="index.json")
        return self.get_json(url)

    def filing_index_html(self, cik: int, accession: str) -> bytes:
        url = ARCHIVE_URL.format(cik=cik, acc_nodash=accession.replace("-", ""), doc=f"{accession}-index.htm")
        return self.get_bytes(url)

    def full_text_search(self, query: str, forms: str = "10-K", start: str = "", end: str = "",
                         ciks: list[int] | None = None) -> dict:
        """EDGAR full-text search (2001+). Returns the raw Elasticsearch-style JSON."""
        params = [f"q={requests.utils.quote(query)}", f"forms={forms}"]
        if start and end:
            params += ["dateRange=custom", f"startdt={start}", f"enddt={end}"]
        if ciks:
            params.append("ciks=" + ",".join(f"{c:010d}" for c in ciks))
        return self.get_json(FTS_URL + "?" + "&".join(params))


@dataclass(frozen=True)
class Filing:
    cik: int
    ticker: str
    company: str
    sector: str
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def fiscal_year(self) -> int:
        # reportDate is the period end. A fiscal year ending in Jan-Mar is labeled by the prior
        # calendar year (rare in this universe, but keeps e.g. FYE 2024-01-31 as FY2023).
        y, m = int(self.report_date[:4]), int(self.report_date[5:7])
        return y - 1 if m <= 3 else y

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc_nodash=self.accession.replace("-", ""),
                                  doc=self.primary_document)


def list_10k_filings(client: EdgarClient, cik: int, ticker: str, company: str, sector: str,
                     fiscal_years: set[int]) -> list[Filing]:
    """Original 10-K filings (not amendments) for the requested fiscal years.

    The submissions API puts the latest ~1000 filings in `filings.recent` and older ones in
    paginated side files (`filings.files`). Large banks file thousands of 424B2 prospectus
    supplements a year, so their older 10-Ks live in the side files -- we walk those too.
    """
    sub = client.submissions(cik)
    pages = [sub["filings"]["recent"]]
    side_files = [f["name"] for f in sub["filings"].get("files", [])]
    out: dict[int, Filing] = {}
    min_year = min(fiscal_years)
    while pages:
        page = pages.pop(0)
        for i, form in enumerate(page["form"]):
            if form != "10-K":
                continue
            f = Filing(
                cik=cik, ticker=ticker, company=company, sector=sector, form=form,
                accession=page["accessionNumber"][i], filing_date=page["filingDate"][i],
                report_date=page["reportDate"][i] or page["filingDate"][i],
                primary_document=page["primaryDocument"][i],
            )
            if f.fiscal_year in fiscal_years and f.fiscal_year not in out:
                out[f.fiscal_year] = f  # newest-first ordering: keep the latest original per FY
        oldest = min(page["filingDate"]) if page["filingDate"] else "0000"
        if len(out) < len(fiscal_years) and side_files and int(oldest[:4]) > min_year:
            pages.append(client.get_json(f"https://data.sec.gov/submissions/{side_files.pop(0)}"))
    return sorted(out.values(), key=lambda f: f.fiscal_year)

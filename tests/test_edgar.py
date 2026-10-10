import gzip
import time

import pytest

from riskradar.edgar import EdgarClient, RateLimiter, list_10k_filings


class FakeResp:
    def __init__(self, status, content=b"{}"):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def get(self, url, timeout):
        self.calls.append(url)
        return self.responses.pop(0)


def test_rate_limiter_spacing():
    rl = RateLimiter(20)  # 50 ms between requests
    t0 = time.monotonic()
    for _ in range(5):
        rl.wait()
    assert time.monotonic() - t0 >= 0.19


def test_refuses_unfair_settings(tmp_path):
    with pytest.raises(ValueError):
        EdgarClient(cache_dir=tmp_path, max_rps=11)
    with pytest.raises(ValueError):
        EdgarClient(cache_dir=tmp_path, user_agent="python-requests")


def test_user_agent_cache_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    sess = FakeSession([FakeResp(429), FakeResp(200, b'{"ok": 1}')])
    c = EdgarClient(cache_dir=tmp_path, session=sess, max_rps=10)
    assert "@" in sess.headers["User-Agent"]
    assert c.get_json("https://data.sec.gov/x.json") == {"ok": 1}
    assert len(sess.calls) == 2  # one retry after 429
    assert c.get_json("https://data.sec.gov/x.json") == {"ok": 1}
    assert len(sess.calls) == 2  # second read served from the disk cache
    cached = list(tmp_path.rglob("*.gz"))
    assert cached and gzip.decompress(cached[0].read_bytes()) == b'{"ok": 1}'


def test_403_raises_permission_error(tmp_path):
    c = EdgarClient(cache_dir=tmp_path, session=FakeSession([FakeResp(403)]))
    with pytest.raises(PermissionError):
        c.get_bytes("https://www.sec.gov/blocked")


class FakeClient:
    def __init__(self, sub, pages=None):
        self.sub, self.pages = sub, pages or {}

    def submissions(self, cik):
        return self.sub

    def get_json(self, url):
        return self.pages[url.rsplit("/", 1)[-1]]


def _page(rows):
    keys = ["form", "accessionNumber", "filingDate", "reportDate", "primaryDocument"]
    return {k: [r[i] for r in rows] for i, k in enumerate(keys)}


def test_list_10k_filters_amendments_years_and_walks_side_files():
    recent = _page([
        ("424B2", "a0", "2026-02-20", "", "x.htm"),
        ("10-K", "a1", "2026-02-15", "2025-12-31", "k25.htm"),
        ("10-K/A", "a2", "2025-04-01", "2024-12-31", "amend.htm"),
        ("10-K", "a3", "2025-02-14", "2024-12-31", "k24.htm"),
    ])
    older = _page([("10-K", "a4", "2024-02-14", "2023-12-31", "k23.htm"),
                   ("10-K", "a5", "2023-02-14", "2022-12-31", "k22.htm")])
    sub = {"filings": {"recent": recent, "files": [{"name": "CIK1-submissions-001.json"}]}}
    fc = FakeClient(sub, {"CIK1-submissions-001.json": older})
    fs = list_10k_filings(fc, 1, "TST", "Test Co", "Bank", {2023, 2024, 2025})
    assert [(f.fiscal_year, f.primary_document) for f in fs] == [
        (2023, "k23.htm"), (2024, "k24.htm"), (2025, "k25.htm")]
    assert fs[0].url.endswith("/1/a4/k23.htm")

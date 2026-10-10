"""Locate the Item 1A "Risk Factors" section inside a 10-K.

Filings differ a lot, so location is a cascade of strategies, and the strategy that
succeeded is recorded for every filing (that is how we report extraction success rate):

1. ``item1a``   -- the standard "Item 1A. Risk Factors" heading, ending at the next
                   Item heading (1B / 1C / 2 / 3). The table of contents also contains
                   these headings, so every start candidate is scored and the longest
                   span wins.
2. ``heading``  -- no usable Item 1A span (e.g. Citigroup's 10-K is laid out as an annual
                   report): a standalone "RISK FACTORS" heading, ending at the next
                   known top-level section heading.
3. ``exhibit13``-- Item 1A only cross-references the Annual Report to Shareholders filed
                   as Exhibit 13 (Wells Fargo). Strategy 2 is applied to the exhibit.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .html_blocks import Block

MIN_SECTION_CHARS = 5000

_ITEM1A = re.compile(r"^(part\s+i\W+)?item\s*1a\b", re.I)
_ITEM1A_BARE = re.compile(r"^1a\.?\s*(\||$|risk)", re.I)
_RISK_FACTORS_TXT = re.compile(r"risk\s*factors", re.I)
_ITEM_END = re.compile(r"^(part\s+i\W+)?item\s*(1b|1c|2|3|4)\b", re.I)
_ITEM_END_BARE = re.compile(r"^(1b|1c|2|3)\.?\s*(\||unresolved|cybersecurity|properties|legal)", re.I)
_RF_HEADING = re.compile(r"^(item\s*1a\W*)?risk\s+factors\W*$", re.I)
_NEXT_SECTION = re.compile(
    # whole-caption matches (so "Risk Management Strategies, Models..." inside Risk Factors
    # does not end the section) ...
    r"^((unresolved staff comments|properties|legal proceedings|cybersecurity|risk management|"
    r"managing global risk|human capital|controls and procedures|financial statements|"
    r"selected financial data|glossary|forward[- ]looking statements|executive summary|overview|"
    r"supervision and regulation|consolidated financial statements|financial review|"
    r"employees|available information)\W*$"
    # ... or captions that start a known next section
    r"|item\s*\d|management['’]s discussion|quantitative and qualitative|"
    r"market for (the )?registrant|report of independent)",
    re.I,
)
_PAGE_NO = re.compile(r"^(page\s*)?[ivxlc\d]{1,4}$|^-\s*\d+\s*-$", re.I)


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.replace("\xa0", " ")).strip()


@dataclass
class Section:
    method: str                      # item1a | heading | exhibit13 | failed
    blocks: list[Block] = field(default_factory=list)
    start: int = -1
    end: int = -1
    note: str = ""

    @property
    def n_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks if not b.is_table)

    @property
    def ok(self) -> bool:
        return self.method != "failed" and self.n_chars >= MIN_SECTION_CHARS


def boilerplate_texts(blocks: list[Block], min_repeats: int = 3) -> set[str]:
    """Short texts that repeat on many pages (running headers/footers)."""
    c = Counter(_norm(b.text) for b in blocks if len(b.text) < 120)
    return {t for t, n in c.items() if n >= min_repeats}


def is_noise(block: Block, boiler: set[str]) -> bool:
    t = _norm(block.text)
    return (
        not t
        or bool(_PAGE_NO.match(t))
        or t.lower() in ("table of contents", "index", "back to contents")
        or bool(re.search(r"\(continued\)$", t, re.I)) and len(t) < 80
        or t in boiler
    )


def _is_item1a_start(blocks: list[Block], k: int) -> bool:
    t = _norm(blocks[k].text)
    if len(t) > 150 or blocks[k].is_table and len(t) > 60:
        return False
    if _ITEM1A.match(t):
        # "Item 1A." alone is fine if the next block says Risk Factors
        if _RISK_FACTORS_TXT.search(t):
            return True
        nxt = _norm(blocks[k + 1].text) if k + 1 < len(blocks) else ""
        return bool(_RISK_FACTORS_TXT.match(nxt))
    return bool(_ITEM1A_BARE.match(t) and _RISK_FACTORS_TXT.search(t + " " + (blocks[k + 1].text if k + 1 < len(blocks) else "")))


def _is_item_end(t: str) -> bool:
    t = _norm(t)
    return len(t) <= 150 and bool(_ITEM_END.match(t) or _ITEM_END_BARE.match(t))


def _span_chars(blocks: list[Block], s: int, e: int) -> int:
    return sum(len(b.text) for b in blocks[s:e] if not b.is_table)


def locate_item1a(blocks: list[Block]) -> Section:
    starts = [k for k in range(len(blocks)) if _is_item1a_start(blocks, k)]
    best: tuple[int, int, int] | None = None
    for s in starts:
        e = next((j for j in range(s + 1, len(blocks)) if _is_item_end(blocks[j].text)), len(blocks))
        n = _span_chars(blocks, s + 1, e)
        if e == len(blocks):
            continue  # no closing Item heading: not trustworthy
        if best is None or n > best[2]:
            best = (s, e, n)
    if best and best[2] >= MIN_SECTION_CHARS:
        s, e, _ = best
        return Section("item1a", blocks[s + 1:e], s, e)
    note = f"item1a candidates={len(starts)} longest={best[2] if best else 0}"
    return Section("failed", note=note)


def locate_by_heading(blocks: list[Block], method: str = "heading") -> Section:
    starts = [
        k for k, b in enumerate(blocks)
        if (len(b.text) < 60 and _RF_HEADING.match(_norm(b.text)) and (b.emphasized or b.text.isupper()))
        or _RF_HEADING.match(_norm(b.lead_emph))  # run-in heading: "Risk Factors An investment..."
    ]
    best: tuple | None = None
    for s in starts:
        e = len(blocks)
        for j in range(s + 1, len(blocks)):
            t = _norm(blocks[j].text)
            if len(t) < 80 and (blocks[j].emphasized or t.isupper()) and _NEXT_SECTION.match(t):
                e = j
                break
        n = _span_chars(blocks, s + 1, e)
        # Prefer spans closed by a known next section; a span running to the end of the
        # document is accepted only if nothing better exists (U.S. Bancorp puts Risk
        # Factors last in its annual report).
        closed = e < len(blocks)
        score = (closed, n)
        if best is None or score > best[3]:
            best = (s, e, n, score)
    if best and best[2] >= MIN_SECTION_CHARS and (best[3][0] or best[2] < 400_000):
        s, e, _, sc = best
        return Section(method, blocks[s + 1:e], s, e, note="" if sc[0] else "ran to end of document")
    return Section("failed", note=f"heading candidates={len(starts)} longest={best[2] if best else 0}")


def locate_risk_factors(blocks: list[Block]) -> Section:
    sec = locate_item1a(blocks)
    if sec.ok:
        return sec
    sec2 = locate_by_heading(blocks)
    if sec2.ok:
        sec2.note = sec.note
        return sec2
    return Section("failed", note=f"{sec.note}; {sec2.note}")


def exhibit13_candidates(index_html: str | bytes) -> list[str]:
    """Annual-Report (EX-13) document names from a filing's ``-index.htm`` page.

    The index page has a table of (Seq, Description, Document, Type, Size); we keep the
    documents whose Type starts with EX-13.
    """
    from lxml import html as _html

    root = _html.fromstring(index_html)
    out = []
    for tr in root.iter("tr"):
        cells = [c.text_content().strip() for c in tr.findall("td")]
        if len(cells) >= 4 and cells[3].upper().startswith("EX-13"):
            name = cells[2].split()[0] if cells[2] else ""
            if name.lower().endswith((".htm", ".html")):
                out.append(name)
    return out

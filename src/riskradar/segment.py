"""Split an Item 1A section into individual risk factors.

SEC Regulation S-K Item 105 requires each risk factor to have an adequately descriptive
sub-caption (heading). In practice those headings are typeset as either

* a **standalone** emphasized paragraph (bold/italic) followed by plain body paragraphs,
* an emphasized **lead-in** sentence at the start of a plain paragraph (Travelers style), or
* a paragraph in a different **font colour/size** but not bold (The Hartford style).

Short lines without a verb ("Insurance-Related Risks") are *group* headers; we keep them as
context. A "Risk Factors Summary" (required since 2020 when the section is > 15 pages) is
skipped because its bullets duplicate the real risk factors.

Strategy: try emphasis-based segmentation first; if it finds fewer than ``MIN_EXPECTED``
risk factors, retry treating typographic outliers (blocks whose font signature differs from
the dominant body style) as emphasized, and keep whichever finds more.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

from .extract import boilerplate_texts, is_noise
from .html_blocks import Block

_TERMINAL = re.compile(r"[.!?:;][\"”’)]*$")
_BULLET = re.compile(r"^[•●▪◦·\-–—*]\s*|^\(?[a-z0-9]{1,2}[.)]\s")
_SUMMARY = re.compile(r"\bsummary\b|^overview$", re.I)
_VERB = re.compile(
    r"\b(may|could|might|can|will|would|should|must|is|are|was|were|be|been|has|have|had|"
    r"depends?|faces?|affects?|subjects?|requires?|relies|rely|exposes?|involves?|increases?|"
    r"reduces?|impacts?|limits?|harms?|results?)\b", re.I)
_INTRO = re.compile(
    r"(should carefully consider|following (discussion|risk factors|is a (summary|discussion))|"
    r"summary (above )?is (subject|qualified)|not the only risks|"
    r"in addition to the other information|discussion of risk factors|set forth below|"
    r"the risks described below|risk factors summary)", re.I,
)
MIN_RF_CHARS = 200
MIN_EXPECTED = 8


@dataclass
class RiskFactor:
    heading: str
    body: str = ""
    group: str = ""
    style: str = ""   # standalone | leadin | typographic

    @property
    def text(self) -> str:
        return (self.heading + "\n" + self.body).strip()


def _clean(blocks: list[Block]) -> list[Block]:
    boiler = boilerplate_texts(blocks)
    out: list[Block] = []
    for b in blocks:
        if b.is_table or is_noise(b, boiler):
            continue
        t = b.text.strip()
        prev = out[-1] if out else None
        # Paragraph split by a page break: continuation starts lower-case and the previous
        # (non-heading) block did not end a sentence.
        if (prev is not None and t and (t[0].islower() or t[0] in ",;)")
                and not prev.emphasized and not _TERMINAL.search(prev.text)):
            prev.text = prev.text + " " + t
            continue
        out.append(replace(b))
    return out


def _is_group(t: str) -> bool:
    """Short caption without a verb or final period: a category header, not a risk factor."""
    return len(t) <= 100 and not _TERMINAL.search(t) and not _VERB.search(t)


def _kind(b: Block, emphasized: bool) -> str:
    t = b.text.strip()
    if emphasized:
        if _BULLET.match(t) and len(t) < 300:
            return "body"
        if _is_group(t):
            return "group"
        if len(t) >= 40:
            return "heading"
        return "body"
    lead = b.lead_emph.strip()
    rest = t[len(lead):].lstrip() if lead and t.startswith(lead) else ""
    # Run-in heading: an emphasized lead sentence, then plain text. Most end with a period;
    # some generators (U.S. Bancorp) drop it, so also accept "<emphasized> <Capitalised...>".
    if (len(lead) >= 40 and len(rest) > 20 and not _BULLET.match(t)
            and (_TERMINAL.search(lead) or rest[:1].isupper())):
        return "leadin"
    return "body"


def _segment(clean: list[Block], outlier_sig: set[str] | None) -> list[RiskFactor]:
    rfs: list[RiskFactor] = []
    group = ""
    in_summary = False
    cur: RiskFactor | None = None
    for b in clean:
        typo = outlier_sig is not None and b.sig in outlier_sig and len(b.text) <= 600
        kind = _kind(b, b.emphasized or typo)
        t = b.text.strip()
        if kind == "group":
            # A summary often has its own group captions ("Market Risk") that repeat later,
            # so summary mode only ends at the first real risk-factor heading.
            if _SUMMARY.search(t):
                in_summary = True
                group = ""
            else:
                group = t
            cur = None
            continue
        if kind == "heading":
            if _INTRO.search(t):
                cur = None
                continue
            in_summary = False
            # Two consecutive emphasized blocks where the first doesn't end a sentence are
            # one heading split across lines.
            if cur is not None and not cur.body and not _TERMINAL.search(cur.heading):
                cur.heading += " " + t
                continue
            cur = RiskFactor(heading=t, group=group, style="typographic" if typo and not b.emphasized else "standalone")
            rfs.append(cur)
        elif kind == "leadin":
            in_summary = False
            lead = b.lead_emph.strip()
            cur = RiskFactor(heading=lead, body=t[len(lead):].strip(), group=group, style="leadin")
            rfs.append(cur)
        elif not in_summary and cur is not None:
            cur.body = (cur.body + "\n" + t).strip()
    return [rf for rf in rfs if len(rf.text) >= MIN_RF_CHARS and len(rf.body) >= 50]


def _outlier_signatures(clean: list[Block]) -> set[str]:
    weights: Counter[str] = Counter()
    for b in clean:
        weights[b.sig] += len(b.text)
    if not weights:
        return set()
    body_sig, _ = weights.most_common(1)[0]
    return {s for s in weights if s != body_sig and s}


def segment(blocks: list[Block]) -> list[RiskFactor]:
    clean = _clean(blocks)
    rfs = _segment(clean, None)
    if len(rfs) < MIN_EXPECTED:
        alt = _segment(clean, _outlier_signatures(clean))
        if len(alt) > len(rfs):
            return alt
    return rfs

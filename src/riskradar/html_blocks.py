"""Flatten 10-K HTML / inline-XBRL into an ordered list of text blocks with style info.

Why blocks instead of plain text? Risk-factor headings are almost always signalled by
*formatting* (bold/italic paragraphs, or a different font colour/size), not by wording.
Keeping per-block emphasis and a typographic signature lets the segmenter find headings
across very different filing generators (Workiva, Donnelley, Toppan Merrill, hand-rolled).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

from lxml import html

BLOCK_TAGS = {
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
    "blockquote", "ul", "ol", "center", "body", "dd", "dt", "pre",
}
SKIP_TAGS = {"script", "style", "head", "title", "ix:header", "header", "noscript"}
WS = re.compile(r"\s+")
BR = "\u2028"  # placeholder for <br>; whitespace, so it collapses to a space
_BOLD_ON = re.compile(r"font-weight\s*:\s*(bold|bolder|[6-9]00)", re.I)
_BOLD_OFF = re.compile(r"font-weight\s*:\s*(normal|lighter|[1-5]00)", re.I)
_ITALIC_ON = re.compile(r"font-style\s*:\s*(italic|oblique)", re.I)
_ITALIC_OFF = re.compile(r"font-style\s*:\s*normal", re.I)
_UNDER_RE = re.compile(r"text-decoration[^;]*underline", re.I)
_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
_SIZE_RE = re.compile(r"font-size\s*:\s*([\d.]+\s*(pt|px|em|%)?)", re.I)
_COLOR_RE = re.compile(r"(?<![-\w])color\s*:\s*([^;]+)", re.I)
_FAMILY_RE = re.compile(r"font-family\s*:\s*([^;,]+)", re.I)


class Style(NamedTuple):
    bold: bool = False
    italic: bool = False
    underline: bool = False
    size: str = ""
    color: str = ""
    family: str = ""

    @property
    def sig(self) -> str:
        """Typographic signature, used to spot headings set in a different font/colour."""
        return f"{self.size}|{self.color}|{self.family}|{'b' if self.bold else ''}{'i' if self.italic else ''}"


@dataclass
class Block:
    text: str
    bold: float       # fraction of non-space characters rendered bold
    italic: float
    underline: float
    is_table: bool = False
    lead_emph: str = ""  # leading run of emphasized (bold/italic/underlined) text, if any
    sig: str = ""        # dominant typographic signature of the block

    @property
    def emphasized(self) -> bool:
        return max(self.bold, self.italic, self.underline) >= 0.8


def _tag(el) -> str:
    t = el.tag
    if not isinstance(t, str):
        return ""
    t = t.lower()
    if t.startswith("{"):
        # namespaced XHTML/iXBRL, e.g. {http://www.xbrl.org/2013/inlineXBRL}header
        ns, local = t[1:].split("}", 1)
        return ("ix:" + local) if "inlinexbrl" in ns else local
    return t


def _nchars(t: str) -> int:
    return len(t) - t.count(" ") - t.count("\xa0") - t.count("\n") - t.count("\t")


class _Builder:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.runs: list[tuple[str, Style]] = []

    def add(self, text: str | None, st: Style) -> None:
        if text:
            self.runs.append((text, st))

    def last_is_br(self) -> bool:
        for t, _ in reversed(self.runs):
            if t == BR:
                return True
            if t.strip():
                return False
        return False

    def flush(self) -> None:
        if not self.runs:
            return
        text = WS.sub(" ", "".join(r[0] for r in self.runs)).strip()
        if text:
            total = bold = ital = und = 0
            sigs: Counter[str] = Counter()
            for t, st in self.runs:
                n = _nchars(t)
                total += n
                bold += n * st.bold
                ital += n * st.italic
                und += n * st.underline
                if n:
                    sigs[st.sig] += n
            total = max(total, 1)
            lead = []
            for t, st in self.runs:
                if not t.strip() or st.bold or st.italic or st.underline:
                    lead.append(t)
                else:
                    break
            lead_txt = WS.sub(" ", "".join(lead)).strip()
            sig = sigs.most_common(1)[0][0] if sigs else ""
            self.blocks.append(Block(text, bold / total, ital / total, und / total, False, lead_txt, sig))
        self.runs = []


def _style(el, st: Style) -> Style:
    tag = _tag(el)
    style = el.get("style") or ""
    b, i, u = st.bold, st.italic, st.underline
    if tag in ("b", "strong") or _BOLD_ON.search(style):
        b = True
    elif _BOLD_OFF.search(style):
        b = False
    if tag in ("i", "em") or _ITALIC_ON.search(style):
        i = True
    elif _ITALIC_OFF.search(style):
        i = False
    if tag == "u" or _UNDER_RE.search(style):
        u = True
    size, color, family = st.size, st.color, st.family
    m = _SIZE_RE.search(style)
    if m:
        size = m.group(1).replace(" ", "").lower()
    m = _COLOR_RE.search(style)
    if m:
        color = m.group(1).strip().lower()
    m = _FAMILY_RE.search(style)
    if m:
        family = m.group(1).strip().strip("'\"").lower()
    return Style(b, i, u, size, color, family)


def _walk(el, bld: _Builder, st: Style) -> None:
    tag = _tag(el)
    if tag in SKIP_TAGS or not isinstance(el.tag, str):
        return
    if _HIDDEN_RE.search(el.get("style") or ""):
        return
    if tag == "table":
        _table(el, bld, st)
        return
    st = _style(el, st)
    is_block = tag in BLOCK_TAGS
    if tag == "br":
        # "<br><br>" inside one <div> is a paragraph break (Workiva uses it between a body
        # paragraph and the next bold heading); a single <br> is just a line break.
        if bld.last_is_br():
            bld.flush()
        else:
            bld.add(BR, st)
    if is_block:
        bld.flush()
    bld.add(el.text, st)
    for child in el:
        _walk(child, bld, st)
        bld.add(child.tail, st)
    if is_block:
        bld.flush()


def _table(el, bld: _Builder, st: Style) -> None:
    bld.flush()
    rows = []
    for tr in el.iter():
        if _tag(tr) == "tr":
            cells = [WS.sub(" ", "".join(td.itertext())).strip() for td in tr if _tag(td) in ("td", "th")]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
    if not rows:
        return
    if len(rows) == 1:
        # A one-row table is usually a layout trick for a bullet or heading ("1A. | Risk Factors"):
        # keep it as a normal text block with its styling.
        sub = _Builder()
        for td in el.iter():
            if _tag(td) in ("td", "th"):
                _inline(td, sub, _style(td, st))
        sub.flush()
        if sub.blocks:
            blk = sub.blocks[0]
            blk.text = rows[0].replace(" | ", " ")
            bld.blocks.append(blk)
            return
    bld.blocks.append(Block("\n".join(rows), 0.0, 0.0, 0.0, is_table=True))


def _inline(el, bld: _Builder, st: Style) -> None:
    """Collect all text of a table cell as inline runs (no block flushing)."""
    bld.add(el.text, st)
    for child in el:
        if isinstance(child.tag, str) and _tag(child) not in SKIP_TAGS:
            _inline(child, bld, _style(child, st))
        bld.add(child.tail, st)
    bld.add(" ", st)


def decode(raw: bytes | str) -> str:
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def parse_blocks(raw: bytes | str) -> list[Block]:
    """Parse filing HTML (plain, XHTML, or inline XBRL) into text blocks."""
    text = decode(raw)
    # Drop the XML declaration (lxml refuses unicode strings that carry one).
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    parser = html.HTMLParser(recover=True, huge_tree=True)
    root = html.fromstring(text, parser=parser)
    bld = _Builder()
    body = root.find(".//body")
    _walk(body if body is not None else root, bld, Style())
    bld.flush()
    return bld.blocks


def debug_dump(blocks: list[Block], start: int = 0, n: int = 50) -> str:  # pragma: no cover
    out = []
    for k, blk in enumerate(blocks[start:start + n], start):
        flag = "B" if blk.bold >= .8 else ("I" if blk.italic >= .8 else ("U" if blk.underline >= .8 else " "))
        out.append(f"{k:5d} {flag}{'T' if blk.is_table else ' '} {blk.text[:140]}")
    return "\n".join(out)

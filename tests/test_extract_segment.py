from riskradar.extract import exhibit13_candidates, locate_by_heading, locate_risk_factors
from riskradar.html_blocks import Block, parse_blocks
from riskradar.segment import segment


def _section(mini_10k, monkeypatch=None, min_chars=None):
    import riskradar.extract as ex

    if monkeypatch is not None and min_chars is not None:
        monkeypatch.setattr(ex, "MIN_SECTION_CHARS", min_chars)
    return locate_risk_factors(parse_blocks(mini_10k))


def test_item1a_skips_table_of_contents(mini_10k, monkeypatch):
    sec = _section(mini_10k, monkeypatch, 100)
    assert sec.method == "item1a"
    texts = [b.text for b in sec.blocks]
    assert texts[0].startswith("You should carefully consider")
    assert not any("Unresolved" in t for t in texts)  # stops at Item 1B
    assert not any("headquarters" in t for t in texts)


def test_short_section_is_rejected(mini_10k):
    # the default minimum (5,000 chars) is far larger than the toy fixture
    assert not _section(mini_10k).ok


def test_segment_finds_three_risk_factors_and_skips_summary(mini_10k, monkeypatch):
    rfs = segment(_section(mini_10k, monkeypatch, 100).blocks)
    headings = [r.heading for r in rfs]
    assert len(rfs) == 3, headings
    assert headings[0].startswith("Catastrophe losses")
    assert rfs[0].group == "Insurance Risks"
    assert rfs[2].group == "Technology Risks"
    # page number "12" dropped and the page-split continuation merged into the body
    assert "12" not in rfs[0].body.split()
    assert "Our property business" in rfs[0].body and "reduce the capital" in rfs[0].body
    assert not any("Cyberattacks could hurt us" in r.text for r in rfs)  # summary bullet


def test_leadin_headings():
    blocks = [Block("Market Risks", 1, 0, 0)]
    for i in range(3):
        lead = f"Rising interest rates could reduce the value of our bond portfolio number {i}."
        body = "The body explains the exposure in detail. " * 8
        blocks.append(Block(lead + " " + body, 0.3, 0, 0, lead_emph=lead))
    rfs = segment(blocks)
    assert [r.style for r in rfs] == ["leadin"] * 3
    assert rfs[0].heading.endswith("number 0.") and rfs[0].body.startswith("The body")
    assert rfs[0].group == "Market Risks"


def test_typographic_fallback_when_no_bold():
    body_sig, head_sig = "10pt|#000|arial|", "11pt|#3b5978|montserrat|"
    blocks = []
    for i in range(10):
        blocks.append(Block(f"Unfavorable condition number {i} may adversely impact our business and results.",
                            0, 0, 0, sig=head_sig))
        blocks.append(Block("Body paragraph with plenty of explanatory detail about the risk. " * 5,
                            0, 0, 0, sig=body_sig))
    rfs = segment(blocks)
    assert len(rfs) == 10 and all(r.style == "typographic" for r in rfs)


def test_group_header_vs_heading():
    from riskradar.segment import _is_group

    assert _is_group("Risks Related to Our Industry")
    assert _is_group("MARKET-RELATED RISKS")
    assert not _is_group("We May Lose Business Due to a Downgrade in Our Ratings")
    assert not _is_group("Competition could reduce our margins.")


def test_heading_fallback_for_annual_report_layout(monkeypatch):
    import riskradar.extract as ex

    monkeypatch.setattr(ex, "MIN_SECTION_CHARS", 50)
    blocks = [Block("RISK FACTORS", 1, 0, 0), Block("Body text about many risks. " * 5, 0, 0, 0),
              Block("Risk Management Strategies, Models and Processes", 1, 0, 0),
              Block("Still inside risk factors. " * 3, 0, 0, 0),
              Block("Managing Global Risk", 1, 0, 0), Block("Not risk factors.", 0, 0, 0)]
    sec = locate_by_heading(blocks)
    assert sec.ok and sec.method == "heading"
    assert len(sec.blocks) == 3  # a caption that merely *starts* with "Risk Management" does not end it


def test_exhibit13_candidates_from_index_page():
    html = """<table><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>
    <tr><td>1</td><td>10-K</td><td>abc-10k.htm iXBRL</td><td>10-K</td></tr>
    <tr><td>2</td><td>EX-13</td><td>abc-ex13.htm iXBRL</td><td>EX-13</td></tr>
    <tr><td>3</td><td>EX-21</td><td>abc-ex21.htm</td><td>EX-21</td></tr></table>"""
    assert exhibit13_candidates(html) == ["abc-ex13.htm"]

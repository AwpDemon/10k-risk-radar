from riskradar.html_blocks import parse_blocks


def test_parses_ixbrl_and_skips_hidden_header(mini_10k):
    blocks = parse_blocks(mini_10k)
    text = " ".join(b.text for b in blocks)
    assert "HIDDEN XBRL FACT" not in text
    assert any(b.text == "Item 1A. Risk Factors" for b in blocks)


def test_bold_and_italic_fractions():
    b = parse_blocks('<html><body><p><b>Bold heading here.</b></p><p>plain <i>half</i></p></body></html>')
    assert b[0].bold == 1.0 and b[0].emphasized
    assert 0 < b[1].italic < 1 and not b[1].emphasized


def test_css_font_weight_and_lead_emphasis():
    html = ('<html><body><div><span style="font-weight:700">A lead-in sentence that is bold.</span>'
            '<span style="font-weight:400"> Then plain body text follows.</span></div></body></html>')
    (blk,) = parse_blocks(html)
    assert blk.lead_emph == "A lead-in sentence that is bold."
    assert not blk.emphasized


def test_double_br_is_paragraph_break():
    html = '<html><body><div>First paragraph.<br/><br/><b>Next heading.</b></div><div>a<br/>b</div></body></html>'
    texts = [b.text for b in parse_blocks(html)]
    assert texts == ["First paragraph.", "Next heading.", "a b"]


def test_one_row_table_becomes_text_block():
    html = '<html><body><table><tr><td><b>1A.</b></td><td><b>Risk Factors</b></td></tr></table></body></html>'
    (blk,) = parse_blocks(html)
    assert blk.text == "1A. Risk Factors" and blk.emphasized and not blk.is_table


def test_display_none_is_skipped():
    html = '<html><body><div style="display: none">secret</div><p>visible</p></body></html>'
    assert [b.text for b in parse_blocks(html)] == ["visible"]


def test_typographic_signature_captures_color_and_size():
    html = ('<html><body><div><span style="color:#3b5978;font-size:11pt">Heading in colour.</span></div>'
            '<div><span style="color:#000000;font-size:10pt">Body.</span></div></body></html>')
    a, b = parse_blocks(html)
    assert a.sig != b.sig and "#3b5978" in a.sig


def test_cp1252_bytes_decode():
    raw = "<html><body><p>Company\x92s risk</p></body></html>".encode("latin-1")
    assert parse_blocks(raw)[0].text == "Company’s risk"

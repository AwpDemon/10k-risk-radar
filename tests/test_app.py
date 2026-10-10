"""Smoke test: the dashboard renders from the committed explorer data without errors."""
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "10-K Risk Radar"
    at.text_input[0].input("artificial intelligence").run()
    assert not at.exception

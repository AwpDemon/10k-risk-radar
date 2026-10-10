from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_10k() -> bytes:
    return (FIXTURES / "mini_10k.htm").read_bytes()

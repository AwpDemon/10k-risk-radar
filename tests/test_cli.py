import pytest

from riskradar.cli import main
from riskradar.taxonomy import CATEGORIES, KEYS, taxonomy_markdown


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    assert "riskradar" in capsys.readouterr().out


def test_taxonomy_is_consistent():
    assert len(KEYS) == len(set(KEYS)) == 14
    assert all(c.keywords for c in CATEGORIES)
    assert taxonomy_markdown().count("\n") == len(KEYS) + 1


def test_committed_gold_set_is_valid():
    """The committed gold labels must use taxonomy keys and keep the dev/test split."""
    import pandas as pd

    from riskradar import config

    g = pd.read_csv(config.GOLD / "gold.csv", keep_default_na=False)
    assert len(g) >= 150 and g.rf_id.is_unique
    assert set(g.primary) <= set(KEYS)
    assert all(s in KEYS for row in g.secondary for s in row.split(";") if s)
    assert set(g.split) == {"dev", "test"}
    fs = pd.read_csv(config.GOLD / "few_shot.csv")
    assert not set(fs.rf_id) & set(g.rf_id)  # few-shot examples never leak gold items

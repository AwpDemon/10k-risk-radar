"""Turn classified + diffed risk factors into tables and charts.

All numbers come from data/interim/*.parquet produced by the pipeline; nothing is typed in.
Outputs: docs/*.png (embedded in the README) and data/results/insights_*.csv.
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .taxonomy import BY_KEY  # noqa: E402

log = logging.getLogger(__name__)

# Categorical slots in fixed order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SECTOR_ORDER = ["P&C insurer", "Life insurer", "Insurance broker", "Bank"]


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load() -> pd.DataFrame:
    rf = pd.read_parquet(config.INTERIM / "risk_factors.parquet")
    cls = pd.read_parquet(config.INTERIM / "classifications.parquet")
    df = rf.merge(cls, on="rf_id", how="left")
    sec = df.secondary.fillna("").str.split(";")
    df["is_ai"] = (df.primary == "ai_model") | sec.apply(lambda s: "ai_model" in s) | df.ai_related.fillna(False).astype(bool)
    df["is_cyber"] = (df.primary == "cyber_tech") | sec.apply(lambda s: "cyber_tech" in s)
    df["is_climate"] = (df.primary == "climate_catastrophe") | sec.apply(lambda s: "climate_catastrophe" in s)
    return df


def _line_by_sector(tab: pd.DataFrame, title: str, ylabel: str, fname: str, pct: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    years = list(tab.index)
    for i, sector in enumerate([s for s in SECTOR_ORDER if s in tab.columns]):
        y = tab[sector] * (100 if pct else 1)
        ax.plot(years, y, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=sector)
        ax.annotate(f"{sector}  {y.iloc[-1]:.0f}{'%' if pct else ''}", (years[-1], y.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=TEXT)
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.2, years[-1] + 1.1)
    if pct:
        ax.set_ylim(0, 105)
    _style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.DOCS / fname, dpi=150)
    plt.close(fig)


def build_all() -> dict:
    config.ensure_dirs()
    df = load()
    out: dict = {}

    # 1) share of filings with at least one AI-related risk factor, by sector and year
    per_filing = df.groupby(["accession", "sector", "fiscal_year"]).agg(
        ai=("is_ai", "any"), ai_kw=("ai_keyword", "any"), cyber_n=("is_cyber", "sum"),
        climate=("is_climate", "any"), n_rf=("rf_id", "size"),
        cyber_chars=("n_chars", lambda s: s[df.loc[s.index, "is_cyber"]].sum()),
        total_chars=("n_chars", "sum")).reset_index()
    ai = per_filing.pivot_table(index="fiscal_year", columns="sector", values="ai", aggfunc="mean")
    ai_kw = per_filing.pivot_table(index="fiscal_year", columns="sector", values="ai_kw", aggfunc="mean")
    ai.to_csv(config.RESULTS / "insights_ai_share_by_sector.csv")
    ai_kw.to_csv(config.RESULTS / "insights_ai_keyword_share_by_sector.csv")
    _line_by_sector(ai, "Filings with at least one AI-related risk factor (LLM-classified)",
                    "% of 10-K filings", "ai_risk_share_by_sector.png")
    out["ai_share_all_years"] = per_filing.groupby("fiscal_year").ai.mean().round(3).to_dict()
    out["ai_keyword_share_all_years"] = per_filing.groupby("fiscal_year").ai_kw.mean().round(3).to_dict()

    # 2) AI-related risk factors per year (count) - how many separate RFs mention AI
    ai_rf = df[df.is_ai].groupby("fiscal_year").size()
    out["ai_risk_factor_count"] = ai_rf.to_dict()
    ai_primary = df[df.primary == "ai_model"].groupby("fiscal_year").size()
    out["ai_primary_count"] = ai_primary.to_dict()

    # 3) cyber disclosure length: share of each filing's Item 1A text devoted to cyber RFs
    per_filing["cyber_share"] = per_filing.cyber_chars / per_filing.total_chars
    cy = per_filing.pivot_table(index="fiscal_year", columns="sector", values="cyber_share", aggfunc="mean")
    cy.to_csv(config.RESULTS / "insights_cyber_text_share.csv")
    _line_by_sector(cy, "Share of Item 1A text in cyber/technology risk factors", "% of risk-factor characters",
                    "cyber_text_share_by_sector.png")
    out["cyber_text_share_all_years"] = per_filing.groupby("fiscal_year").cyber_share.mean().round(4).to_dict()

    # 4) category mix (primary) by year - 100% stacked horizontal bars would hide change; use a heat table
    mix = pd.crosstab(df.primary, df.fiscal_year, normalize="columns")
    mix.to_csv(config.RESULTS / "insights_category_mix.csv")
    counts = pd.crosstab(df.primary, df.sector)
    counts.to_csv(config.RESULTS / "insights_category_by_sector_counts.csv")
    _heatmap(pd.crosstab(df.primary, df.sector, normalize="columns")[[s for s in SECTOR_ORDER if s in counts]],
             "Risk-factor mix by sector (share of each sector's risk factors, all years)", "category_mix_by_sector.png")

    # 5) year-over-year changes
    dpath = config.INTERIM / "diffs.parquet"
    if dpath.exists():
        d = pd.read_parquet(dpath)
        ch = d.merge(df[["rf_id", "primary", "sector", "heading", "is_ai"]], on="rf_id", how="left")
        ct = pd.crosstab(ch.fiscal_year, ch.change)
        ct.to_csv(config.RESULTS / "insights_change_counts.csv")
        out["change_counts"] = json.loads(ct.to_json(orient="index"))
        new = ch[ch.change == "new"]
        new_mix = pd.crosstab(new.primary, new.fiscal_year)
        new_mix.to_csv(config.RESULTS / "insights_new_by_category.csv")
        out["new_ai_related"] = new[new.is_ai].groupby("fiscal_year").size().to_dict()
        _new_bars(new_mix, "New risk factors by category (not matched to the prior year's 10-K)", "new_risk_factors_by_category.png")
        examples = (new[new.is_ai].sort_values(["fiscal_year", "ticker"])
                    [["ticker", "fiscal_year", "primary", "heading"]])
        examples.to_csv(config.RESULTS / "insights_new_ai_examples.csv", index=False)

    (config.RESULTS / "insights_summary.json").write_text(json.dumps(out, indent=2, default=str))
    log.info("insights written: %s", list(out))
    return out


def _heatmap(tab: pd.DataFrame, title: str, fname: str) -> None:
    import numpy as np

    tab = tab.loc[tab.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.imshow(tab.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(tab.shape[1]), tab.columns, fontsize=8, color=MUTED)
    ax.set_yticks(range(tab.shape[0]), [BY_KEY[k].name for k in tab.index], fontsize=8, color=MUTED)
    vmax = np.nanmax(tab.to_numpy())
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7.5,
                    color="white" if v > vmax * 0.6 else TEXT)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(config.DOCS / fname, dpi=150)
    plt.close(fig)


def _new_bars(tab: pd.DataFrame, title: str, fname: str) -> None:
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=True).index]
    years = list(tab.columns)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    h = 0.8 / len(years)
    for i, y in enumerate(years):
        pos = [k + i * h for k in range(len(tab))]
        ax.barh(pos, tab[y], height=h * 0.9, color=SERIES[i], label=str(y))
    ax.set_yticks([k + h * (len(years) - 1) / 2 for k in range(len(tab))], [BY_KEY[k].name for k in tab.index], fontsize=8)
    _style(ax, title, "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("number of new risk factors", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8, title="fiscal year", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(config.DOCS / fname, dpi=150)
    plt.close(fig)

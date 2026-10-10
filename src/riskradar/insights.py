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
    from .pipeline import AI_TERMS

    text = df.heading + " " + df.body
    df["ai_term"] = text.str.contains(AI_TERMS)          # explicit AI / ML / generative-AI wording
    df["ai_heading"] = df.heading.str.contains(AI_TERMS)  # a risk factor *titled* around AI
    # "AI-related" needs both the model's flag and explicit wording: the flag alone also fires on
    # generic model-risk text (see README, AI flag audit).
    df["is_ai"] = df.ai_related.fillna(False).astype(bool) & df.ai_term
    df["is_cyber"] = (df.primary == "cyber_tech") | sec.apply(lambda s: "cyber_tech" in s)
    df["is_climate"] = (df.primary == "climate_catastrophe") | sec.apply(lambda s: "climate_catastrophe" in s)
    return df


def _line_by_sector(tab: pd.DataFrame, title: str, ylabel: str, fname: str, pct: bool = True,
                    n_by_sector: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    years = list(tab.index)
    sectors = [s for s in SECTOR_ORDER if s in tab.columns]
    ends = {}
    for i, sector in enumerate(sectors):
        y = tab[sector] * (100 if pct else 1)
        label = f"{sector} (n={n_by_sector[sector]})" if n_by_sector else sector
        ax.plot(years, y, color=SERIES[i], linewidth=2, marker="o", markersize=5, label=label)
        ends[sector] = float(y.iloc[-1])
    top = max(float((tab * (100 if pct else 1)).max().max()), 1.0)
    ymax = 105 if pct and top > 60 else top * 1.35
    gap = ymax * 0.055  # dodge end labels so they never overlap
    placed: list[float] = []
    for sector, v in sorted(ends.items(), key=lambda kv: kv[1]):
        pos = v if not placed else max(v, placed[-1] + gap)
        placed.append(pos)
        ax.annotate(f"{sector}  {v:.0f}{'%' if pct else ''}", (years[-1], v), xytext=(years[-1] + 0.08, pos),
                    textcoords="data", va="center", fontsize=8, color=TEXT)
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.2, years[-1] + 1.1)
    ax.set_ylim(0, ymax)
    _style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(config.DOCS / fname, dpi=150)
    plt.close(fig)


def build_all() -> dict:
    config.ensure_dirs()
    df = load()
    out: dict = {}

    # 1) AI: share of filings with a risk factor whose HEADING names AI (deterministic), by sector
    per_filing = df.groupby(["accession", "sector", "fiscal_year"]).agg(
        ai_heading=("ai_heading", "any"), ai_any=("ai_keyword", "any"), ai_llm=("is_ai", "any"),
        n_rf=("rf_id", "size"),
        cyber_chars=("n_chars", lambda s: s[df.loc[s.index, "is_cyber"]].sum()),
        total_chars=("n_chars", "sum")).reset_index()
    ai = per_filing.pivot_table(index="fiscal_year", columns="sector", values="ai_heading", aggfunc="mean")
    ai.to_csv(config.RESULTS / "insights_ai_heading_share_by_sector.csv")
    n_sec = df.groupby("sector").ticker.nunique().to_dict()
    _line_by_sector(ai, "Filings with a dedicated AI risk factor (AI named in the heading)",
                    "% of companies' 10-K filings", "ai_risk_share_by_sector.png", n_by_sector=n_sec)
    by_year = per_filing.groupby("fiscal_year")
    out["filings_per_year"] = by_year.size().to_dict()
    out["filings_with_ai_heading"] = by_year.ai_heading.sum().astype(int).to_dict()
    out["filings_with_ai_anywhere_in_item1a"] = by_year.ai_any.sum().astype(int).to_dict()
    out["filings_with_llm_ai_related_rf"] = by_year.ai_llm.sum().astype(int).to_dict()

    # 2) AI risk-factor counts and the LLM flag vs explicit-wording agreement
    out["ai_heading_risk_factors"] = df[df.ai_heading].groupby("fiscal_year").size().to_dict()
    out["ai_heading_primary_category"] = df[df.ai_heading].primary.value_counts().to_dict()
    flag = df.ai_related.fillna(False).astype(bool)
    out["llm_ai_flag_vs_ai_wording"] = {
        "both": int((flag & df.ai_term).sum()), "flag_only": int((flag & ~df.ai_term).sum()),
        "wording_only": int((~flag & df.ai_term).sum()), "neither": int((~flag & ~df.ai_term).sum())}

    # 3) cyber disclosure length: share of each filing's Item 1A text devoted to cyber RFs
    per_filing["cyber_share"] = per_filing.cyber_chars / per_filing.total_chars
    cy = per_filing.pivot_table(index="fiscal_year", columns="sector", values="cyber_share", aggfunc="mean")
    cy.to_csv(config.RESULTS / "insights_cyber_text_share.csv")
    _line_by_sector(cy, "Share of Item 1A text in cyber/technology risk factors", "% of risk-factor characters",
                    "cyber_text_share_by_sector.png", n_by_sector=n_sec)
    out["cyber_text_share_all_years"] = per_filing.groupby("fiscal_year").cyber_share.mean().round(4).to_dict()

    # 4) category mix (primary) by year - 100% stacked horizontal bars would hide change; use a heat table
    mix = pd.crosstab(df.primary, df.fiscal_year, normalize="columns")
    mix.to_csv(config.RESULTS / "insights_category_mix.csv")
    counts = pd.crosstab(df.primary, df.sector)
    counts.to_csv(config.RESULTS / "insights_category_by_sector_counts.csv")
    _heatmap(pd.crosstab(df.primary, df.sector, normalize="columns")[[s for s in SECTOR_ORDER if s in counts]],
             "Risk-factor mix by sector (share of risk factors, FY2022-25)", "category_mix_by_sector.png")

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
        _new_bars(new_mix, "New risk factors by category (no match in prior year's 10-K)", "new_risk_factors_by_category.png")
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
        pos = [k + (len(years) - 1 - i) * h for k in range(len(tab))]  # earliest year on top
        ax.barh(pos, tab[y], height=h * 0.9, color=SERIES[i], label=str(y))
    ax.set_yticks([k + h * (len(years) - 1) / 2 for k in range(len(tab))], [BY_KEY[k].name for k in tab.index], fontsize=8)
    _style(ax, title, "")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("number of new risk factors", color=MUTED, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=8, title="fiscal year", title_fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.DOCS / fname, dpi=150)
    plt.close(fig)

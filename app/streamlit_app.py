"""10-K Risk Radar explorer (read-only).

    streamlit run app/streamlit_app.py

Reads the explorer table built by `python scripts/build_explorer_data.py`
(data/sample/explorer.parquet, committed) plus the evaluation results in data/results/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
DOCS = ROOT / "docs"

st.set_page_config(page_title="10-K Risk Radar", layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "sample" / "explorer.parquet")


df = load()
st.title("10-K Risk Radar")
st.caption(
    f"{len(df):,} risk factors from {df.accession.nunique()} Form 10-K filings "
    f"({df.ticker.nunique()} US insurers, brokers and banks, fiscal {df.fiscal_year.min()}-{df.fiscal_year.max()}), "
    "pulled from SEC EDGAR, segmented from Item 1A and classified by a local LLM "
    "(qwen2.5:14b-instruct, zero-shot, macro-F1 0.91 on a 180-item hand-labeled test set). "
    "Labels are model output: expect roughly 1 in 10 to be wrong."
)

with st.sidebar:
    st.header("Filters")
    sectors = st.multiselect("Sector", sorted(df.sector.unique()), default=sorted(df.sector.unique()))
    years = st.slider("Fiscal year", int(df.fiscal_year.min()), int(df.fiscal_year.max()),
                      (int(df.fiscal_year.min()), int(df.fiscal_year.max())))
    cats = st.multiselect("Primary category", sorted(df.primary.dropna().unique()))
    q = st.text_input("Search headings", placeholder="e.g. artificial intelligence")

f = df[df.sector.isin(sectors) & df.fiscal_year.between(*years)]
if cats:
    f = f[f.primary.isin(cats)]
if q:
    f = f[f.heading.str.contains(q, case=False, regex=False) | f.excerpt.str.contains(q, case=False, regex=False)]

tab1, tab2, tab3, tab4 = st.tabs(["Trends", "Explore risk factors", "Year-over-year changes", "Evaluation"])

with tab1:
    per = f.groupby(["accession", "sector", "fiscal_year"]).agg(ai=("is_ai", "any")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Filings with an AI-related risk factor")
        st.line_chart(per.pivot_table(index="fiscal_year", columns="sector", values="ai", aggfunc="mean") * 100,
                      y_label="% of filings")
    with c2:
        st.subheader("Risk factors per category")
        mix = f.groupby(["fiscal_year", "primary"]).size().unstack(fill_value=0)
        st.bar_chart(mix, y_label="risk factors")
    st.caption("Shares are per filing: a filing counts once if any of its risk factors is AI-related "
               "(primary/secondary category ai_model or the model's ai_related flag).")

with tab2:
    st.write(f"{len(f):,} risk factors match the filters.")
    show = f[["ticker", "fiscal_year", "primary", "secondary", "change", "heading"]].sort_values(["ticker", "fiscal_year"])
    st.dataframe(show, use_container_width=True, height=420, hide_index=True)
    pick = st.selectbox("Read one", show.index, format_func=lambda i: f"{df.at[i, 'ticker']} {df.at[i, 'fiscal_year']}: {df.at[i, 'heading'][:90]}")
    if pick is not None:
        r = df.loc[pick]
        st.markdown(f"**{r.heading}**")
        st.write(r.excerpt + " ...")
        st.caption(f"Group: {r.group or '-'} | category: {r.primary} | secondary: {r.secondary or '-'} | "
                   f"YoY: {r.change or '-'} | source: {r.url}")

with tab3:
    tick = st.selectbox("Company", sorted(f.ticker.unique()))
    g = df[(df.ticker == tick)]
    for y in sorted(g.fiscal_year.unique())[1:]:
        new = g[(g.fiscal_year == y) & (g.change == "new")]
        exp = g[(g.fiscal_year == y) & (g.change == "expanded")]
        with st.expander(f"FY{y}: {len(new)} new, {len(exp)} materially expanded"):
            for _, r in new.iterrows():
                st.markdown(f"- NEW [{r.primary}] {r.heading}")
            for _, r in exp.iterrows():
                st.markdown(f"- EXPANDED [{r.primary}] {r.heading}")
    st.caption("New = no risk factor in the prior year's 10-K with embedding cosine similarity >= 0.80 "
               "(one-to-one Hungarian matching). Expanded = matched, but >= 75 net words added and >= 25% longer.")

with tab4:
    p = RESULTS / "eval_table_test.md"
    if p.exists():
        st.markdown(p.read_text(encoding="utf-8"))
    img = DOCS / "confusion_matrix_test.png"
    if img.exists():
        st.image(str(img))
    c = RESULTS / "consistency.json"
    if c.exists():
        d = json.loads(c.read_text())
        st.write(f"Labeler self-consistency on a blind 50-item re-label: {d['exact_agreement']:.0%} "
                 f"(Cohen's kappa {d['cohen_kappa']:.2f}); same annotator, so an optimistic bound.")

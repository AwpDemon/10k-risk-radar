"""Gold-set evaluation of every classifier.

    riskradar eval --models qwen2.5:7b-instruct qwen2.5:14b-instruct

Approaches compared on the same items:

* ``keyword``          -- taxonomy keyword counts (no learning, no LLM)
* ``embed_lr``         -- nomic-embed-text embeddings + logistic regression, out-of-fold
                          predictions from 5-fold CV over all gold items
* ``zero_shot:<model>``-- LLM with taxonomy definitions only
* ``few_shot:<model>`` -- LLM with 14 labeled examples (one per class, none from the gold set)

Writes data/results/: eval_summary.json, eval_table.md, gold_predictions.csv,
per_class_f1.csv, confusion_<best>.csv, and docs/confusion_matrix.png.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from . import config
from .classify import baseline, embed_lr
from .classify.llm_classifier import PROMPT_VERSION, FewShotExample, classify_many
from .evaluate import bootstrap_ci, confusion, paired_bootstrap, per_class_report, summary
from .llm import EmbeddingClient, LLMClient
from .taxonomy import KEYS

log = logging.getLogger(__name__)
DEFAULT_MODELS = ["qwen2.5:7b-instruct", "qwen2.5:14b-instruct"]


def load_gold(split: str = "all") -> pd.DataFrame:
    g = pd.read_csv(config.GOLD / "gold.csv", keep_default_na=False)
    g = g[g.segment_ok == 1]
    rf = pd.read_parquet(config.INTERIM / "risk_factors.parquet", columns=["rf_id", "body"]) \
        if (config.INTERIM / "risk_factors.parquet").exists() else None
    if rf is not None:
        g = g.merge(rf, on="rf_id", how="left")
        g["body"] = g["body"].fillna(g["body_excerpt"])
    else:  # CI / fresh clone: fall back to the committed excerpt
        g["body"] = g["body_excerpt"]
    g["secondary_list"] = g.secondary.apply(lambda s: [x for x in str(s).split(";") if x])
    if split != "all":
        g = g[g.split == split]
    return g.reset_index(drop=True)


def load_few_shot() -> list[FewShotExample]:
    fs = pd.read_csv(config.GOLD / "few_shot.csv", keep_default_na=False)
    return [FewShotExample(r.heading, r.body_excerpt, r.primary, [s for s in r.secondary.split(";") if s],
                           int(r.severity), str(r.ai_related).lower() == "true", r.rationale)
            for r in fs.itertuples()]


def run(models: list[str] | None = None, skip_llm: bool = False, split: str = "test",
        workers: int = 4) -> dict:
    config.ensure_dirs()
    models = models or DEFAULT_MODELS
    all_gold = load_gold("all")
    preds = pd.DataFrame({"rf_id": all_gold.rf_id, "split": all_gold.split, "gold": all_gold.primary,
                          "gold_secondary": all_gold.secondary})
    timing: dict[str, dict] = {}

    # 1) keyword baseline
    t0 = time.perf_counter()
    preds["keyword"] = [baseline.predict(h, b) for h, b in zip(all_gold.heading, all_gold.body)]
    timing["keyword"] = {"seconds_per_item": (time.perf_counter() - t0) / len(all_gold)}

    # 2) embeddings + logistic regression (out-of-fold over ALL gold items)
    emb = EmbeddingClient()
    t0 = time.perf_counter()
    X = emb.embed([embed_lr.rf_text(h, b) for h, b in zip(all_gold.heading, all_gold.body)],
                  prefix=embed_lr.EMBED_PREFIX)
    preds["embed_lr"] = embed_lr.cross_val_predict(X, all_gold.primary.to_numpy())
    timing["embed_lr"] = {"seconds_per_item": (time.perf_counter() - t0) / len(all_gold),
                          "note": "embedding time (cached after first run) + CV fit"}

    # 3/4) LLMs
    llm_meta: dict[str, dict] = {}
    if not skip_llm:
        client = LLMClient()
        shots = load_few_shot()
        items = list(zip(all_gold.heading, all_gold.body))
        for model in models:
            for mode, sh in (("zero_shot", None), ("few_shot", shots)):
                name = f"{mode}:{model}"
                res = classify_many(client, items, shots=sh, model=model, workers=workers)
                preds[name] = [r.primary for r in res]
                preds[name + "|secondary"] = [";".join(r.secondary) for r in res]
                preds[name + "|severity"] = [r.severity for r in res]
                preds[name + "|ai"] = [r.ai_related for r in res]
                preds[name + "|valid"] = [r.valid for r in res]
                preds[name + "|latency_s"] = [r.latency_s for r in res]
                preds[name + "|prompt_tokens"] = [r.prompt_tokens for r in res]
                preds[name + "|completion_tokens"] = [r.completion_tokens for r in res]

    # ---- metrics on the requested split
    ev = preds if split == "all" else preds[preds.split == split]
    for name in [c for c in preds.columns if ":" in c and "|" not in c]:
        lat = ev[name + "|latency_s"].to_numpy(dtype=float)
        llm_meta[name] = {
            "invalid_output_rate": float(1 - ev[name + "|valid"].astype(bool).mean()),
            "mean_latency_s": float(lat.mean()), "p95_latency_s": float(np.percentile(lat, 95)),
            "mean_prompt_tokens": float(ev[name + "|prompt_tokens"].mean()),
            "mean_completion_tokens": float(ev[name + "|completion_tokens"].mean()),
            "prompt_version": PROMPT_VERSION,
        }
    sec = ev.gold_secondary.apply(lambda s: [x for x in str(s).split(";") if x]).tolist()
    approaches = [c for c in preds.columns if c in ("keyword", "embed_lr") or (":" in c and "|" not in c)]
    rows, per_class = [], []
    for a in approaches:
        s = summary(ev.gold, ev[a], secondary=sec)
        lo, hi = bootstrap_ci(ev.gold.to_numpy(), ev[a].to_numpy())
        s.update(approach=a, macro_f1_ci_low=lo, macro_f1_ci_high=hi, **llm_meta.get(a, {}))
        rows.append(s)
        pc = per_class_report(ev.gold, ev[a])
        pc.insert(0, "approach", a)
        per_class.append(pc)
    table = pd.DataFrame(rows).set_index("approach")
    best = table.macro_f1.idxmax()
    comparisons = {}
    for a in approaches:
        if a != best:
            comparisons[f"{best} vs {a}"] = paired_bootstrap(ev.gold, ev[best], ev[a])

    # severity agreement between model sizes (no gold severity: inter-model agreement only)
    sev_agree = {}
    for mode in ("zero_shot", "few_shot"):
        cols = [f"{mode}:{m}|severity" for m in models if f"{mode}:{m}|severity" in preds]
        if len(cols) == 2:
            a, b = ev[cols[0]], ev[cols[1]]
            ok = a.notna() & b.notna()
            from sklearn.metrics import cohen_kappa_score
            sev_agree[mode] = {"quadratic_kappa": float(cohen_kappa_score(a[ok].astype(int), b[ok].astype(int), weights="quadratic")),
                               "exact_agreement": float((a[ok] == b[ok]).mean()), "n": int(ok.sum())}

    out = {
        "split": split, "n_items": int(len(ev)), "best_by_macro_f1": best,
        "table": json.loads(table.to_json(orient="index")),
        "paired_bootstrap": comparisons, "timing": timing, "severity_inter_model": sev_agree,
        "label_distribution": ev.gold.value_counts().to_dict(),
    }
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / f"eval_summary_{split}.json").write_text(json.dumps(out, indent=2))
    preds.to_csv(config.RESULTS / "gold_predictions.csv", index=False)
    pd.concat(per_class).to_csv(config.RESULTS / f"per_class_f1_{split}.csv", index=False)
    confusion(ev.gold, ev[best]).to_csv(config.RESULTS / f"confusion_{split}.csv")
    _write_table_md(table, split)
    _plot_confusion(ev.gold, ev[best], best, split)
    log.info("best approach on %s: %s (macro-F1 %.3f)", split, best, table.loc[best, "macro_f1"])
    print(table[["accuracy", "macro_f1", "micro_f1", "lenient_accuracy"]].round(3).to_string())
    return out


def _write_table_md(table: pd.DataFrame, split: str) -> None:
    cols = ["n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "lenient_accuracy", "kappa"]
    extra = [c for c in ("invalid_output_rate", "mean_latency_s", "mean_prompt_tokens") if c in table]
    md = table[cols + extra].round(3).to_markdown()
    (config.RESULTS / f"eval_table_{split}.md").write_text(md + "\n", encoding="utf-8")


def _plot_confusion(y_true, y_pred, name: str, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion(y_true, y_pred).to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(KEYS)), KEYS, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(KEYS)), KEYS, fontsize=8)
    for i in range(len(KEYS)):
        for j in range(len(KEYS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold label")
    ax.set_title(f"Confusion matrix - {name} ({split} split, n={len(y_true)})", fontsize=10)
    fig.tight_layout()
    config.DOCS.mkdir(exist_ok=True)
    fig.savefig(config.DOCS / f"confusion_matrix_{split}.png", dpi=150)
    plt.close(fig)

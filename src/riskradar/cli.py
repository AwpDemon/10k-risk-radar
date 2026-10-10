"""Command-line interface.

    riskradar fetch                 # list + download 10-Ks for the universe (cached, SEC-polite)
    riskradar extract               # locate Item 1A in each filing
    riskradar segment               # split sections into individual risk factors
    riskradar classify              # LLM-classify every risk factor (OpenAI-compatible endpoint)
    riskradar diff                  # year-over-year new / removed / expanded risk factors
    riskradar insights              # charts + summary tables into docs/ and data/results/
    riskradar eval                  # gold-set evaluation of all classifiers
    riskradar run                   # fetch -> extract -> segment -> classify -> diff -> insights
"""
from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="riskradar", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download 10-K filings from EDGAR")
    p.add_argument("--tickers", nargs="*", help="limit to these tickers")
    sub.add_parser("extract", help="locate Item 1A in each filing")
    sub.add_parser("segment", help="split Item 1A into risk factors")
    p = sub.add_parser("classify", help="classify all risk factors with the configured LLM")
    p.add_argument("--model", help="override RISKRADAR_LLM_MODEL")
    p.add_argument("--few-shot", action="store_true", help="use the few-shot prompt")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="only the first N risk factors (smoke test)")
    sub.add_parser("diff", help="year-over-year change detection")
    sub.add_parser("insights", help="build charts and summary tables")
    p = sub.add_parser("eval", help="evaluate classifiers on the gold set")
    p.add_argument("--models", nargs="*", default=None, help="LLM models to evaluate")
    p.add_argument("--skip-llm", action="store_true", help="only baseline + embeddings")
    p.add_argument("--split", default="test", choices=["dev", "test", "all"])
    p = sub.add_parser("run", help="full pipeline")
    p.add_argument("--tickers", nargs="*")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from . import pipeline

    if args.cmd == "fetch":
        pipeline.stage_fetch(tickers=args.tickers)
    elif args.cmd == "extract":
        pipeline.stage_extract()
    elif args.cmd == "segment":
        pipeline.stage_segment()
    elif args.cmd == "classify":
        pipeline.stage_classify(model=args.model, few_shot=args.few_shot, workers=args.workers, limit=args.limit)
    elif args.cmd == "diff":
        pipeline.stage_diff()
    elif args.cmd == "insights":
        from . import insights
        insights.build_all()
    elif args.cmd == "eval":
        from . import eval_harness
        eval_harness.run(models=args.models, skip_llm=args.skip_llm, split=args.split)
    elif args.cmd == "run":
        pipeline.stage_fetch(tickers=args.tickers)
        pipeline.stage_extract()
        pipeline.stage_segment()
        pipeline.stage_classify()
        pipeline.stage_diff()
        from . import insights
        insights.build_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())

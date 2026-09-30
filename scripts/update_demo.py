"""Extend the demo corpus with the months that have passed since it was built.

Run monthly by CI, and by hand when needed:

    python scripts/update_demo.py --dry-run
    python scripts/update_demo.py

Exits with code 2 when there is nothing to add, which lets the workflow skip the
commit without treating it as a failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from topiclens.config import load_config
from topiclens.constants import PROJECT_ROOT
from topiclens.data.arxiv import ArxivClient, ArxivFetchError
from topiclens.data.base import corpus_summary, finalize_corpus
from topiclens.data.refresh import merge_corpora, missing_periods, slice_quota
from topiclens.pipeline import train
from topiclens.reporting import write_report
from topiclens.utils.logging_config import setup_logging

DEMO_CORPUS = PROJECT_ROOT / "data" / "demo" / "arxiv-demo.csv.gz"
DEMO_ARTIFACTS = PROJECT_ROOT / "artifacts" / "demo"

NOTHING_TO_DO = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--until", help="last month to fetch, YYYY-MM (default: last complete)")
    parser.add_argument("--max-months", type=int, default=6, help="cap on months per run")
    parser.add_argument("--dry-run", action="store_true", help="report what would be fetched")
    return parser.parse_args(argv)


def fetch_periods(
    client: ArxivClient, categories: list[str], periods: list[str], quota: int, include_title: bool
) -> pd.DataFrame:
    """Collect one quota per category for each of the given months."""
    records = []
    for period in periods:
        for category in categories:
            for paper in client.fetch_slice(category, period, limit=quota):
                text = f"{paper.title}. {paper.abstract}" if include_title else paper.abstract
                records.append(
                    {
                        "doc_id": paper.paper_id,
                        "text": text,
                        "title": paper.title,
                        "label": paper.primary_category,
                        "date": paper.published.isoformat(),
                    }
                )
    return pd.DataFrame.from_records(records)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config.logging.model_copy(update={"console_level": "INFO"}), force=True)

    if not DEMO_CORPUS.is_file():
        print(f"No demo corpus at {DEMO_CORPUS}; run scripts/make_demo.py first", file=sys.stderr)
        return 1

    existing = pd.read_csv(DEMO_CORPUS, parse_dates=["date"])
    periods = missing_periods(existing, until=args.until)
    if not periods:
        print("The demo corpus is already up to date.")
        return NOTHING_TO_DO

    periods = periods[: args.max_months]
    quota = slice_quota(existing)
    categories = sorted(existing["label"].dropna().unique())
    print(f"Fetching {len(periods)} month(s) — {', '.join(periods)} — at {quota} per category")

    if args.dry_run:
        return 0

    try:
        fresh = fetch_periods(
            ArxivClient(config.data.arxiv),
            list(categories),
            periods,
            quota,
            config.data.arxiv.include_title,
        )
    except ArxivFetchError as error:
        print(f"Fetch failed, the corpus is unchanged: {error}", file=sys.stderr)
        return 1

    if fresh.empty:
        print("The API returned nothing for those months; the corpus is unchanged.")
        return NOTHING_TO_DO

    merged = finalize_corpus(
        merge_corpora(existing, fresh),
        min_chars=config.data.min_abstract_chars,
        drop_duplicates=config.data.drop_duplicates,
    )
    added = len(merged) - len(existing)
    if added <= 0:
        print("Every fetched paper was already present; the corpus is unchanged.")
        return NOTHING_TO_DO

    merged.to_csv(DEMO_CORPUS, index=False, compression="gzip")
    summary = corpus_summary(merged)
    print(f"\nAdded {added} documents; the corpus now holds {summary['documents']}")
    print(f"Period: {summary.get('date_range', '')}")

    run = train(
        merged,
        config,
        corpus_fingerprint={"source": "demo", "size": len(merged), "updated_through": periods[-1]},
        show_progress=False,
    )
    print("\n" + run.metrics_frame().to_string(index=False))

    write_report(
        run,
        merged,
        DEMO_ARTIFACTS,
        top_words=config.evaluation.top_words,
        timeline_freq=config.viz.timeline_freq,
    )
    run.bundle.save(DEMO_ARTIFACTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

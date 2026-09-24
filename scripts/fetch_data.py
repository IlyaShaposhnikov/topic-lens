"""Build (or refresh) the corpus cache from the command line.

Examples:
    python scripts/fetch_data.py --limit-per-slice 20   # quick smoke run
    python scripts/fetch_data.py                        # full corpus
    python scripts/fetch_data.py --refresh              # ignore the cache
    python scripts/fetch_data.py --refresh --no-resume   # and the slice checkpoints
    python scripts/fetch_data.py --source csv --csv-path data/raw/my.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from topiclens.config import load_config
from topiclens.data.base import CorpusError, corpus_summary
from topiclens.data.corpus import build_source, cache_paths, load_corpus
from topiclens.utils.logging_config import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="path to a YAML config")
    parser.add_argument("--source", choices=("arxiv", "csv"), help="override data.source")
    parser.add_argument("--csv-path", type=Path, help="CSV file to read when --source csv")
    parser.add_argument(
        "--limit-per-slice",
        type=int,
        help="override data.arxiv.max_per_slice, useful for a quick trial run",
    )
    parser.add_argument("--refresh", action="store_true", help="refetch even if cached")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore checkpointed slices and fetch every one again",
    )
    parser.add_argument("--verbose", action="store_true", help="mirror the log to the console")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.source:
        config.data.source = args.source
    if args.csv_path:
        config.data.csv = config.data.csv.model_copy(update={"path": args.csv_path})
    if args.limit_per_slice:
        config.data.arxiv = config.data.arxiv.model_copy(
            update={"max_per_slice": args.limit_per_slice}
        )
    if args.verbose:
        config.logging = config.logging.model_copy(update={"console_level": "INFO"})

    setup_logging(config.logging, force=True)

    try:
        source_kwargs = {} if config.data.source == "csv" else {"resume": not args.no_resume}
        source = build_source(config, **source_kwargs)
        parquet_path, _ = cache_paths(config, source)
        frame = load_corpus(config, refresh=args.refresh, source=source)
    except CorpusError as error:
        print(f"Failed to build the corpus: {error}", file=sys.stderr)
        return 1

    summary = corpus_summary(frame)
    print(f"\nCorpus: {summary['documents']} documents, {summary['mean_chars']} chars on average")
    if "date_range" in summary:
        print(f"Period: {summary['date_range']}")
    if "by_label" in summary:
        print("\nDocuments per label:")
        for label, count in sorted(summary["by_label"].items()):
            print(f"  {label:<10} {count:>6}")
    if "by_year" in summary:
        print("\nDocuments per year:")
        for year, count in summary["by_year"].items():
            print(f"  {year:<10} {count:>6}")
    print(f"\nCached at: {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

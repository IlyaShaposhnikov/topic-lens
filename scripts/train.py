"""Train the models and write a report.

Examples:
    python scripts/train.py --sample 2000   # quick pass over a subset
    python scripts/train.py                 # full corpus
    python scripts/train.py --sweep         # also sweep the topic count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from topiclens.config import load_config
from topiclens.data.base import CorpusError
from topiclens.data.corpus import build_source, load_corpus
from topiclens.evaluation.coherence import CoherenceCalculator
from topiclens.evaluation.selection import best_topic_count, sweep_topic_counts
from topiclens.pipeline import train, vectorize
from topiclens.preprocessing import Preprocessor
from topiclens.reporting import write_report, write_sweep
from topiclens.utils.logging_config import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="path to a YAML config")
    parser.add_argument(
        "--sample",
        type=int,
        help="train on a random subset of this many documents (fast iteration)",
    )
    parser.add_argument("--n-topics", type=int, help="override models.n_topics for this run only")
    parser.add_argument("--artifacts", type=Path, help="where to write the bundle and report")
    parser.add_argument("--no-save", action="store_true", help="skip writing the model bundle")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="also fit every model at each evaluation.k_search value and write sweep.csv",
    )
    parser.add_argument("--quiet", action="store_true", help="keep the console free of progress")
    return parser.parse_args(argv)


def run_sweep(frame: pd.DataFrame, config, *, show_progress: bool) -> list:
    """Fit every model family across the configured topic counts."""
    processed = Preprocessor(config.preprocessing).transform_many(
        frame["text"].tolist(), show_progress=show_progress
    )
    vectorized = vectorize(processed, config)
    scorer = CoherenceCalculator(vectorized["count"].matrix, vectorized["count"].feature_names)

    rows = []
    for kind in ("lda", "nmf", "lsa"):
        from topiclens.models.factory import MODEL_CLASSES

        data = vectorized[MODEL_CLASSES[kind].vectorizer_kind]
        rows.extend(
            sweep_topic_counts(
                kind,
                config.models,
                data.matrix,
                data.feature_names,
                scorer,
                topic_counts=config.evaluation.k_search,
                top_words=config.evaluation.top_words,
                seed=config.project.seed,
            )
        )
        family = [row for row in rows if row.kind == kind]
        print(f"{kind}: best k = {best_topic_count(family, min_diversity=0.5)}")
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.n_topics:
        config.models = config.models.model_copy(update={"n_topics": args.n_topics})
    artifacts = args.artifacts or config.project.artifacts_path
    setup_logging(
        config.logging.model_copy(update={"console_level": "WARNING" if args.quiet else "INFO"}),
        force=True,
    )

    try:
        source = build_source(config)
        frame = load_corpus(config, source=source)
    except CorpusError as error:
        print(f"Could not load the corpus: {error}", file=sys.stderr)
        print("Run scripts/fetch_data.py first.", file=sys.stderr)
        return 1

    if args.sample and args.sample < len(frame):
        frame = frame.sample(args.sample, random_state=config.project.seed).reset_index(drop=True)
        print(f"Training on a random sample of {len(frame)} documents")

    run = train(
        frame, config, corpus_fingerprint=source.fingerprint(), show_progress=not args.quiet
    )

    print("\n" + run.metrics_frame().to_string(index=False))
    print("\nCross-model topic similarity:")
    for pair, value in sorted(run.similarity.items()):
        print(f"  {pair}: {value:.3f}")

    written = write_report(
        run,
        frame,
        artifacts,
        top_words=config.evaluation.top_words,
        timeline_freq=config.viz.timeline_freq,
    )
    if not args.no_save:
        written["bundle"] = run.bundle.save(artifacts)

    if args.sweep:
        written["sweep"] = write_sweep(
            run_sweep(frame, config, show_progress=not args.quiet), artifacts
        )

    print("\nWritten:")
    for name, path in sorted(written.items()):
        print(f"  {name:<12} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

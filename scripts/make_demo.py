"""Build the small corpus and artifacts that ship with the repository.

The hosted demo cannot carry a 15 MB corpus or a 5 MB bundle, and it should not
spend a minute training every time the container wakes up. This script writes a
stratified sample plus a bundle trained on it, both small enough to commit:

    python scripts/make_demo.py
    python scripts/make_demo.py --size 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from topiclens.config import load_config
from topiclens.constants import PROJECT_ROOT
from topiclens.data.base import CorpusError, corpus_summary
from topiclens.data.corpus import build_source, load_corpus
from topiclens.data.sample import stratified_sample
from topiclens.pipeline import train
from topiclens.reporting import write_report
from topiclens.utils.logging_config import setup_logging

DEMO_CORPUS = PROJECT_ROOT / "data" / "demo" / "arxiv-demo.csv.gz"
DEMO_ARTIFACTS = PROJECT_ROOT / "artifacts" / "demo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--size", type=int, default=2500, help="documents to keep")
    parser.add_argument("--corpus-only", action="store_true", help="skip training")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    setup_logging(config.logging.model_copy(update={"console_level": "INFO"}), force=True)

    try:
        frame = load_corpus(config, source=build_source(config))
    except CorpusError as error:
        print(f"Could not load the corpus: {error}", file=sys.stderr)
        return 1

    sample = stratified_sample(frame, args.size, seed=config.project.seed)
    DEMO_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(DEMO_CORPUS, index=False, compression="gzip")

    summary = corpus_summary(sample)
    print(f"\nDemo corpus: {summary['documents']} documents, {summary.get('date_range', '')}")
    if "by_label" in summary:
        print(
            "  "
            + "  ".join(f"{label}: {count}" for label, count in sorted(summary["by_label"].items()))
        )
    print(f"  {DEMO_CORPUS} ({DEMO_CORPUS.stat().st_size / 1e6:.1f} MB)")

    if args.corpus_only:
        return 0

    run = train(sample, config, corpus_fingerprint={"source": "demo", "size": len(sample)})
    print("\n" + run.metrics_frame().to_string(index=False))

    write_report(
        run,
        sample,
        DEMO_ARTIFACTS,
        top_words=config.evaluation.top_words,
        timeline_freq=config.viz.timeline_freq,
    )
    bundle_path = run.bundle.save(DEMO_ARTIFACTS)
    print(f"\n  {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB)")
    print("\nCommit data/demo/ and artifacts/demo/ so the hosted app starts without training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

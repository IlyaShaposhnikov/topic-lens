"""Render the README figures from a trained run.

Keeping this a script rather than a manual export means the pictures can be
regenerated after any retraining, so they never drift away from the numbers in
the text:

    pip install -e ".[dev,docs]"
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from topiclens.artifacts import ArtifactError, ModelBundle
from topiclens.config import load_config
from topiclens.constants import PROJECT_ROOT
from topiclens.ui.data import load_reports
from topiclens.viz.charts import (
    metrics_comparison_chart,
    similarity_heatmap,
    sweep_chart,
    timeline_chart,
    topic_words_chart,
)
from topiclens.viz.projection import topic_layout, topic_map_chart

IMAGES = PROJECT_ROOT / "docs" / "images"
WIDTH, HEIGHT, SCALE = 1200, 560, 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", type=Path, default=None, help="where the bundle lives")
    parser.add_argument(
        "--projection", choices=("pca", "mds"), default="mds", help="topic map layout"
    )
    return parser.parse_args(argv)


def corpus_shares(bundle: ModelBundle, texts: list[str]) -> dict[str, list[float]]:
    """Mean share of the corpus per topic, which sizes the dots on the map."""
    prepared = [bundle.preprocessor.transform(text) for text in texts]
    shares: dict[str, list[float]] = {}
    for kind, model in bundle.models.items():
        matrix = bundle.vectorizers[model.vectorizer_kind].transform(prepared)
        shares[kind] = model.document_topics(matrix).mean(axis=0).tolist()
    return shares


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    directory = args.artifacts or config.project.artifacts_path

    try:
        bundle = ModelBundle.load(directory)
    except ArtifactError as error:
        print(f"{error}", file=sys.stderr)
        return 1

    reports = load_reports(directory)
    IMAGES.mkdir(parents=True, exist_ok=True)

    figures = {}

    if reports.has_timeline:
        figures["timeline-nmf"] = timeline_chart(reports.timeline, "nmf")

    figures["metrics"] = metrics_comparison_chart(reports.metrics)
    if reports.sweep is not None and not reports.sweep.empty:
        rows = [
            type("Row", (), {"as_record": lambda self, r=record: r})()
            for record in reports.sweep.to_dict("records")
        ]
        figures["sweep"] = sweep_chart(rows)
    figures["similarity-lda-nmf"] = similarity_heatmap(bundle.models["lda"], bundle.models["nmf"])

    matches = sorted(glob.glob(str(PROJECT_ROOT / "data" / "cache" / "arxiv-*.parquet")))
    shares = None
    if matches:
        texts = pd.read_parquet(matches[0])["text"].tolist()
        shares = corpus_shares(bundle, texts)
        # The biggest topic of the best model makes the most convincing example.
        biggest = max(range(len(shares["nmf"])), key=lambda index: shares["nmf"][index])
        figures["topic-words-nmf"] = topic_words_chart(bundle.models["nmf"], biggest)
    else:
        print("No local corpus: the topic map will use equal dot sizes.")
        figures["topic-words-nmf"] = topic_words_chart(bundle.models["nmf"], 0)

    figures["topic-map"] = topic_map_chart(
        topic_layout(bundle.models, shares=shares, method=args.projection, seed=config.project.seed)
    )

    for name, figure in figures.items():
        path = IMAGES / f"{name}.png"
        figure.write_image(path, width=WIDTH, height=HEIGHT, scale=SCALE)
        print(f"  {path} ({path.stat().st_size / 1000:.0f} kB)")

    print(
        f"\nWrote {len(figures)} figures. Reference them from the README as docs/images/<name>.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

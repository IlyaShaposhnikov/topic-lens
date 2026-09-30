"""Writing a training run to disk.

Numbers that live only in a terminal cannot be put in a README, charted, or
compared against last week's run. Every artifact here is written in a shape
something downstream actually consumes: JSON for provenance, CSV for the eye and
for charts, long-format tables because that is what plotting libraries want.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from topiclens.constants import resolve_path
from topiclens.evaluation.matching import match_topics
from topiclens.pipeline import TrainingRun, topic_timeline
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

METRICS_JSON = "metrics.json"
METRICS_CSV = "metrics.csv"
TOPICS_CSV = "topics.csv"
TIMELINE_CSV = "timeline.csv"
MATCHES_CSV = "matches.csv"
SWEEP_CSV = "sweep.csv"


def timeline_frame(
    frame: pd.DataFrame,
    document_topics: dict[str, np.ndarray],
    models: dict[str, Any],
    *,
    freq: str = "Q",
) -> pd.DataFrame:
    """Topic shares over time for every model, in long format.

    One row per model, period and topic — the shape a stacked area chart or a
    grouped line chart expects, and the one that survives adding a model.
    """
    blocks: list[pd.DataFrame] = []
    for kind, shares in document_topics.items():
        wide = topic_timeline(frame, shares, freq=freq)
        labels = models[kind].topic_labels()
        long = wide.melt(id_vars="period", var_name="topic", value_name="share")
        long["topic"] = long["topic"].str.removeprefix("topic_").astype(int)
        long["label"] = [labels[index - 1] for index in long["topic"]]
        long.insert(0, "model", kind)
        blocks.append(long)
    return pd.concat(blocks, ignore_index=True)


def matches_frame(models: dict[str, Any], *, top_words: int = 10) -> pd.DataFrame:
    """Optimal topic pairings for every pair of models, as a table."""
    rows: list[dict[str, Any]] = []
    keys = sorted(models)
    for i, first in enumerate(keys):
        for second in keys[i + 1 :]:
            for match in match_topics(models[first], models[second], top_words=top_words):
                rows.append(
                    {
                        "pair": f"{first}-{second}",
                        "topic_a": match.topic_a + 1,
                        "label_a": models[first].topic_label(match.topic_a),
                        "topic_b": match.topic_b + 1,
                        "label_b": models[second].topic_label(match.topic_b),
                        "similarity": round(match.similarity, 4),
                        "shared_words": ", ".join(match.shared_words),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    run: TrainingRun,
    frame: pd.DataFrame,
    directory: Path | str,
    *,
    top_words: int = 10,
    timeline_freq: str = "Q",
) -> dict[str, Path]:
    """Write metrics, topics, timeline and matches; return the paths written.

    The timeline is skipped rather than fatal when the corpus has no dates: a
    CSV corpus without a date column is a legitimate use of this project.
    """
    target = resolve_path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "documents": int(len(frame)),
        "train_size": run.train_size,
        "holdout_size": run.holdout_size,
        "metrics": run.metrics,
        "cross_model_similarity": run.similarity,
        "corpus_fingerprint": run.bundle.corpus_fingerprint,
        "config": run.bundle.config.model_dump(mode="json"),
    }
    metrics_path = target / METRICS_JSON
    metrics_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    written["metrics_json"] = metrics_path

    metrics_csv = target / METRICS_CSV
    run.metrics_frame().to_csv(metrics_csv, index=False)
    written["metrics_csv"] = metrics_csv

    topics_csv = target / TOPICS_CSV
    pd.DataFrame(run.bundle.topic_table(top_words)).to_csv(topics_csv, index=False)
    written["topics"] = topics_csv

    matches_csv = target / MATCHES_CSV
    matches_frame(run.bundle.models, top_words=top_words).to_csv(matches_csv, index=False)
    written["matches"] = matches_csv

    if frame["date"].notna().any():
        timeline_csv = target / TIMELINE_CSV
        timeline_frame(frame, run.document_topics, run.bundle.models, freq=timeline_freq).to_csv(
            timeline_csv, index=False
        )
        written["timeline"] = timeline_csv
    else:
        logger.info("Corpus has no dates; skipping the timeline")

    logger.info("Wrote %d report files to %s", len(written), target)
    return written


def write_sweep(rows: list[Any], directory: Path | str) -> Path:
    """Persist a topic-count sweep so the selection curve is reproducible."""
    target = resolve_path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / SWEEP_CSV
    pd.DataFrame([row.as_record() for row in rows]).to_csv(path, index=False)
    logger.info("Wrote the sweep of %d fits to %s", len(rows), path)
    return path

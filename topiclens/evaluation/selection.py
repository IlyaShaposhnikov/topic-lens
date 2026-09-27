"""Choosing the number of topics.

The number of topics is the one hyperparameter of a topic model that cannot be
left at a default and defended. Sweeping it and plotting coherence turns the
choice into an argument: the curve typically rises, flattens, and then declines
as topics start splitting hairs.

Coherence alone is not enough to pick a point on that curve, because a model can
raise it by letting topics converge onto the same words. Diversity is measured
alongside for exactly that reason, and fit time is recorded because a marginally
better k that costs five times the compute is rarely the right answer.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from topiclens.config import ModelsConfig
from topiclens.evaluation.coherence import CoherenceCalculator
from topiclens.evaluation.diversity import topic_diversity
from topiclens.models.factory import build_model
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One (model, k) combination and how it scored."""

    kind: str
    n_topics: int
    coherence: float
    diversity: float
    seconds: float

    def as_record(self) -> dict[str, Any]:
        return {
            "model": self.kind,
            "n_topics": self.n_topics,
            "coherence": round(self.coherence, 4),
            "diversity": round(self.diversity, 4),
            "seconds": round(self.seconds, 1),
        }


def sweep_topic_counts(
    kind: str,
    config: ModelsConfig,
    matrix: Any,
    feature_names: Sequence[str],
    scorer: CoherenceCalculator,
    *,
    topic_counts: Sequence[int],
    top_words: int = 10,
    metric: str = "npmi",
    seed: int = 0,
) -> list[SweepRow]:
    """Fit one model family at several topic counts and score each fit.

    Args:
        kind: model key, as in the factory.
        config: models section, supplying the hyperparameters.
        matrix: the document-term matrix this family expects.
        feature_names: vocabulary of ``matrix``.
        scorer: coherence calculator; pass the same one for every model so the
            scores stay comparable — coherence is a property of the corpus, not
            of the weighting scheme used to fit.
        topic_counts: values of k to try.
        top_words: how many words per topic feed the metrics.
        metric: ``npmi`` or ``umass``.
        seed: random state shared by all fits, so only k varies.
    """
    rows: list[SweepRow] = []
    for k in topic_counts:
        started = time.perf_counter()
        model = build_model(kind, config, n_topics=k, seed=seed).fit(matrix, feature_names)
        elapsed = time.perf_counter() - started

        topics = [[entry.word for entry in topic] for topic in model.topics(top_words)]
        row = SweepRow(
            kind=kind,
            n_topics=k,
            coherence=scorer.mean_score(topics, metric),
            diversity=topic_diversity(topics),
            seconds=elapsed,
        )
        logger.info(
            "%s k=%d: coherence %.4f, diversity %.3f, %.1fs",
            kind,
            k,
            row.coherence,
            row.diversity,
            elapsed,
        )
        rows.append(row)
    return rows


def best_topic_count(rows: Sequence[SweepRow], *, min_diversity: float = 0.0) -> int:
    """Pick k by coherence, ignoring fits whose topics collapsed.

    Ties go to the smaller k: two models that explain the corpus equally well
    are not equally useful, and the simpler one is easier to read.

    Raises:
        ValueError: if no row clears ``min_diversity``.
    """
    eligible = [row for row in rows if row.diversity >= min_diversity]
    if not eligible:
        raise ValueError(f"no sweep result reaches diversity {min_diversity}")
    best = max(eligible, key=lambda row: (row.coherence, -row.n_topics))
    return best.n_topics

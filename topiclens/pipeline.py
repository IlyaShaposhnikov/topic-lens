"""End-to-end training: corpus in, evaluated bundle out.

The sequence is fixed and shared by the CLI, the tests and the app, so that a
number seen in the UI and a number in a report always came from the same steps:
preprocess once, vectorize twice (counts for LDA, TF-IDF for NMF and LSA), fit
every model on the training split, score them, and package the result.

Two choices worth stating. Models are fitted on a training split while LDA's
perplexity is measured on held-out documents — evaluating a likelihood on the
data it was fitted to says little. Document-topic shares, however, are computed
for the whole corpus, because the topic timeline needs every date.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from topiclens.artifacts import ModelBundle
from topiclens.config import AppConfig
from topiclens.constants import MODEL_KEYS
from topiclens.evaluation.alignment import alignment_scores
from topiclens.evaluation.coherence import CoherenceCalculator
from topiclens.evaluation.diversity import most_similar_pair, pairwise_overlap, topic_diversity
from topiclens.evaluation.matching import match_topics, mean_match_similarity
from topiclens.models.factory import build_model
from topiclens.preprocessing import Preprocessor, build_vectorizer
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Vectorized:
    """A fitted vectorizer with the training matrix it produced."""

    vectorizer: Any
    matrix: Any
    feature_names: np.ndarray


@dataclass
class TrainingRun:
    """The result of one training pass."""

    bundle: ModelBundle
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    similarity: dict[str, float] = field(default_factory=dict)
    document_topics: dict[str, np.ndarray] = field(default_factory=dict)
    train_size: int = 0
    holdout_size: int = 0

    def metrics_frame(self) -> pd.DataFrame:
        """Metrics as a table, one row per model."""
        return pd.DataFrame(self.metrics).T.rename_axis("model").reset_index()


def split_holdout(n_documents: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle document indices into a training and a held-out part."""
    indices = np.arange(n_documents)
    if fraction <= 0.0:
        return indices, np.empty(0, dtype=int)

    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(indices)
    cut = max(1, int(round(n_documents * fraction)))
    return np.sort(shuffled[cut:]), np.sort(shuffled[:cut])


def vectorize(texts: list[str], config: AppConfig) -> dict[str, Vectorized]:
    """Fit both vectorizers on the same preprocessed texts."""
    result: dict[str, Vectorized] = {}
    for kind in ("count", "tfidf"):
        vectorizer = build_vectorizer(kind, config.preprocessing)
        matrix = vectorizer.fit_transform(texts)
        names = vectorizer.get_feature_names_out()
        logger.info("%s matrix: %d x %d", kind, matrix.shape[0], matrix.shape[1])
        result[kind] = Vectorized(vectorizer, matrix, names)
    return result


def train(
    frame: pd.DataFrame,
    config: AppConfig,
    *,
    corpus_fingerprint: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> TrainingRun:
    """Preprocess, fit, evaluate and bundle all three models.

    Args:
        frame: corpus with the standard columns.
        config: validated application config.
        corpus_fingerprint: recorded in the bundle for provenance.
        show_progress: progress bar during preprocessing.

    Raises:
        ValueError: if the corpus is empty.
    """
    if frame.empty:
        raise ValueError("cannot train on an empty corpus")

    preprocessor = Preprocessor(config.preprocessing)
    processed = preprocessor.transform_many(frame["text"].tolist(), show_progress=show_progress)

    train_index, holdout_index = split_holdout(
        len(processed), config.evaluation.holdout_fraction, config.project.seed
    )
    train_texts = [processed[i] for i in train_index]
    holdout_texts = [processed[i] for i in holdout_index]
    logger.info("Training on %d documents, holding out %d", len(train_texts), len(holdout_texts))

    vectorized = vectorize(train_texts, config)
    scorer = CoherenceCalculator(vectorized["count"].matrix, vectorized["count"].feature_names)
    train_labels = frame["label"].to_numpy()[train_index]

    models: dict[str, Any] = {}
    metrics: dict[str, dict[str, float]] = {}
    document_topics: dict[str, np.ndarray] = {}

    for kind in MODEL_KEYS:
        model = build_model(kind, config.models, seed=config.project.seed)
        data = vectorized[model.vectorizer_kind]

        started = time.perf_counter()
        model.fit(data.matrix, data.feature_names)
        elapsed = time.perf_counter() - started

        topics = [
            [entry.word for entry in topic] for topic in model.topics(config.evaluation.top_words)
        ]
        scores: dict[str, float] = {
            "fit_seconds": round(elapsed, 1),
            "diversity": topic_diversity(topics),
            "pairwise_overlap": pairwise_overlap(topics),
            "most_similar_pair": most_similar_pair(topics)[2],
        }
        for metric in config.evaluation.coherence_metrics:
            scores[metric] = scorer.mean_score(topics, metric)
        scores.update(alignment_scores(model.dominant_topics(data.matrix), train_labels))
        scores.update(model.fit_info())

        if kind == "lda" and holdout_texts:
            # A likelihood measured on the data it was fitted to means little.
            holdout_matrix = data.vectorizer.transform(holdout_texts)
            scores["holdout_perplexity"] = model.perplexity(holdout_matrix)

        models[kind] = model
        metrics[kind] = scores
        # Shares for the entire corpus: the timeline needs every document.
        document_topics[kind] = model.document_topics(data.vectorizer.transform(processed))
        logger.info("%s: %s", kind, ", ".join(f"{key} {value}" for key, value in scores.items()))

    bundle = ModelBundle(
        config=config,
        preprocessor=preprocessor,
        vectorizers={kind: item.vectorizer for kind, item in vectorized.items()},
        models=models,
        corpus_fingerprint=corpus_fingerprint or {},
    )

    return TrainingRun(
        bundle=bundle,
        metrics=metrics,
        similarity=cross_model_similarity(models),
        document_topics=document_topics,
        train_size=len(train_texts),
        holdout_size=len(holdout_texts),
    )


def cross_model_similarity(models: dict[str, Any]) -> dict[str, float]:
    """Mean similarity of optimally matched topics for each pair of models.

    Answers the question three lists of topics immediately raise: did the models
    find the same structure or different structures?
    """
    keys = sorted(models)
    result: dict[str, float] = {}
    for i, first in enumerate(keys):
        for second in keys[i + 1 :]:
            matches = match_topics(models[first], models[second])
            result[f"{first}-{second}"] = round(mean_match_similarity(matches), 4)
    return result


#: pandas spells resampling and period frequencies differently — "QE" resamples,
#: "Q" is the period alias — and rejects each in the other's place.
_PERIOD_ALIASES = {"ME": "M", "QE": "Q", "YE": "Y", "WE": "W"}


def period_alias(freq: str) -> str:
    """Accept either spelling of a pandas frequency and return the period one."""
    return _PERIOD_ALIASES.get(freq.upper(), freq)


def topic_timeline(frame: pd.DataFrame, shares: np.ndarray, *, freq: str = "QE") -> pd.DataFrame:
    """Mean topic share per period — the data behind the evolution chart.

    Raises:
        ValueError: if the share matrix does not line up with the corpus.
    """
    if len(frame) != shares.shape[0]:
        raise ValueError(f"{shares.shape[0]} share rows for {len(frame)} documents")

    dated = frame["date"].notna().to_numpy()
    if not dated.any():
        raise ValueError("the corpus carries no dates; a timeline cannot be built")

    periods = pd.PeriodIndex(frame.loc[dated, "date"], freq=period_alias(freq))
    table = pd.DataFrame(
        shares[dated],
        columns=[f"topic_{index + 1}" for index in range(shares.shape[1])],
    )
    table["period"] = periods.to_timestamp().to_numpy()
    return table.groupby("period", as_index=False).mean()

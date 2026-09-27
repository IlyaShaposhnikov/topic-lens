"""Matching topics across models.

Looking at three lists of topics, the obvious question is whether the models
found the same structure. Cosine similarity between topic-term vectors answers
it for a single pair of topics; the Hungarian algorithm turns that into a global
one-to-one assignment, which is what makes the comparison honest — greedy
nearest-neighbour matching would happily map three NMF topics onto the same LDA
topic and report high similarity.

One catch drives the design: LDA is fitted on the count vocabulary while NMF and
LSA use the TF-IDF one, and those vocabularies differ. Topic vectors must be
restricted to the shared words before anything is compared, or the numbers come
out plausible and meaningless.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TopicMatch:
    """One pairing between a topic of model A and a topic of model B."""

    topic_a: int
    topic_b: int
    similarity: float
    shared_words: tuple[str, ...]


def align_vocabularies(
    components_a: np.ndarray,
    features_a: Sequence[str],
    components_b: np.ndarray,
    features_b: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Restrict two topic-term matrices to the words both models know.

    Raises:
        ValueError: if the vocabularies do not overlap at all.
    """
    index_a = {str(word): position for position, word in enumerate(features_a)}
    index_b = {str(word): position for position, word in enumerate(features_b)}

    shared = sorted(set(index_a) & set(index_b))
    if not shared:
        raise ValueError("the two models share no vocabulary; cannot compare their topics")

    columns_a = [index_a[word] for word in shared]
    columns_b = [index_b[word] for word in shared]
    logger.debug(
        "Comparing on %d shared words (of %d and %d)", len(shared), len(index_a), len(index_b)
    )
    return components_a[:, columns_a], components_b[:, columns_b], shared


def cosine_similarity_matrix(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between the rows of two matrices."""
    norms_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
    normalized_a = np.divide(matrix_a, norms_a, out=np.zeros_like(matrix_a), where=norms_a > 0)
    normalized_b = np.divide(matrix_b, norms_b, out=np.zeros_like(matrix_b), where=norms_b > 0)
    return normalized_a @ normalized_b.T


def topic_similarity(model_a: Any, model_b: Any) -> tuple[np.ndarray, list[str]]:
    """Similarity of every topic of A to every topic of B, on shared words."""
    aligned_a, aligned_b, shared = align_vocabularies(
        model_a.components, model_a.feature_names_, model_b.components, model_b.feature_names_
    )
    return cosine_similarity_matrix(aligned_a, aligned_b), shared


def match_topics(model_a: Any, model_b: Any, *, top_words: int = 10) -> list[TopicMatch]:
    """Optimal one-to-one pairing of the topics of two models.

    The assignment maximizes total similarity across all pairs at once. When the
    models have different topic counts, the surplus topics of the larger model
    stay unmatched — there is nothing to pair them with.
    """
    similarity, _ = topic_similarity(model_a, model_b)
    rows, columns = linear_sum_assignment(-similarity)

    matches: list[TopicMatch] = []
    for topic_a, topic_b in zip(rows, columns, strict=True):
        words_a = {entry.word for entry in model_a.top_words(int(topic_a), top_words)}
        words_b = {entry.word for entry in model_b.top_words(int(topic_b), top_words)}
        matches.append(
            TopicMatch(
                topic_a=int(topic_a),
                topic_b=int(topic_b),
                similarity=float(similarity[topic_a, topic_b]),
                shared_words=tuple(sorted(words_a & words_b)),
            )
        )
    return sorted(matches, key=lambda match: match.similarity, reverse=True)


def mean_match_similarity(matches: Sequence[TopicMatch]) -> float:
    """How close two models' topic sets are overall, in one number."""
    return float(np.mean([match.similarity for match in matches])) if matches else 0.0


def unmatched_topics(matches: Sequence[TopicMatch], n_topics: int, *, side: str = "a") -> list[int]:
    """Topics of one model left without a counterpart."""
    used = {match.topic_a if side == "a" else match.topic_b for match in matches}
    return [index for index in range(n_topics) if index not in used]

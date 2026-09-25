"""Topic coherence from document co-occurrence.

Coherence asks a simple question: do the words at the top of a topic actually
appear together in documents? Two standard formulations are implemented.

NPMI is normalized pointwise mutual information averaged over the word pairs of
a topic. It lands in [-1, 1]: near 1 when two words always co-occur, 0 when they
are independent, negative when they avoid each other. Being normalized, it is
comparable across corpora and across topic counts, which is what makes it the
metric to select k with.

UMass is an asymmetric, unnormalized alternative: for each pair it asks how
often the lower-ranked word appears given the higher-ranked one. Values are
negative and closer to zero is better. It is kept as a second opinion, because
two metrics that disagree are more informative than one that is trusted blindly.

Implementing these directly, rather than through gensim, keeps the dependency
tree light and makes the definition auditable — and the whole computation is a
co-occurrence matrix and two formulas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse

from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

#: Guards the logarithms when a pair never co-occurs.
EPSILON = 1e-12


class CoherenceCalculator:
    """Scores topics against the corpus they were trained on.

    The binary document-term matrix is kept once and co-occurrence counts are
    computed only for the words actually being scored: the full vocabulary would
    mean a dense matrix of tens of millions of cells, almost all of it unused.
    """

    def __init__(self, matrix: Any, feature_names: Sequence[str]) -> None:
        self.binary = (sparse.csc_matrix(matrix) > 0).astype(np.float64)
        self.n_documents = self.binary.shape[0]
        self.vocabulary = {str(word): index for index, word in enumerate(feature_names)}

    def _known_indices(self, words: Sequence[str]) -> list[int]:
        """Vocabulary positions of the given words, silently dropping strangers."""
        indices = [self.vocabulary[word] for word in words if word in self.vocabulary]
        if len(indices) < len(words):
            missing = [word for word in words if word not in self.vocabulary]
            logger.debug("Words outside the vocabulary are ignored: %s", ", ".join(missing))
        return indices

    def _statistics(self, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Document frequencies and pairwise co-occurrence counts for a few words."""
        columns = self.binary[:, list(indices)]
        document_frequency = np.asarray(columns.sum(axis=0)).ravel()
        co_occurrence = np.asarray((columns.T @ columns).todense())
        return document_frequency, co_occurrence

    def npmi(self, words: Sequence[str]) -> float:
        """Mean NPMI over the distinct word pairs of one topic."""
        indices = self._known_indices(words)
        if len(indices) < 2:
            return 0.0

        document_frequency, co_occurrence = self._statistics(indices)
        probability = document_frequency / self.n_documents
        joint = co_occurrence / self.n_documents

        scores: list[float] = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pair = joint[i, j]
                if pair <= 0.0:
                    # Words that never share a document are maximally incoherent.
                    scores.append(-1.0)
                    continue
                pointwise = np.log(pair / (probability[i] * probability[j] + EPSILON))
                scores.append(float(pointwise / -np.log(pair)))
        return float(np.mean(scores))

    def umass(self, words: Sequence[str]) -> float:
        """Mean UMass score; words must be ordered by descending topic weight."""
        indices = self._known_indices(words)
        if len(indices) < 2:
            return 0.0

        document_frequency, co_occurrence = self._statistics(indices)
        scores: list[float] = []
        for i in range(1, len(indices)):
            for j in range(i):
                # j precedes i in the topic, so it is the conditioning word.
                scores.append(
                    float(np.log((co_occurrence[i, j] + 1.0) / (document_frequency[j] + EPSILON)))
                )
        return float(np.mean(scores))

    def score(self, topics: Sequence[Sequence[str]], metric: str = "npmi") -> list[float]:
        """Score every topic with one metric.

        Raises:
            ValueError: on an unknown metric name.
        """
        if metric == "npmi":
            return [self.npmi(words) for words in topics]
        if metric == "umass":
            return [self.umass(words) for words in topics]
        raise ValueError(f"unknown coherence metric {metric!r}; expected 'npmi' or 'umass'")

    def mean_score(self, topics: Sequence[Sequence[str]], metric: str = "npmi") -> float:
        """Corpus-level coherence: the average over topics."""
        scores = self.score(topics, metric)
        return float(np.mean(scores)) if scores else 0.0

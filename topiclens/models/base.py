"""The contract every topic model in this project satisfies.

LDA, NMF and LSA differ in almost everything — a generative model of counts, a
constrained matrix factorization and a truncated SVD — yet the whole point of
this project is comparing them. A shared interface is what keeps the evaluation,
the charts and the UI free of per-model branching.

Two differences cannot be hidden and are handled explicitly:

* loadings are non-negative for LDA and NMF but not for SVD, so
  ``document_topics`` clips and renormalizes to make "share of a topic in a
  document" mean the same thing for all three;
* each algorithm has its own native fit diagnostics, which are reported
  separately through ``fit_info`` rather than pretended to be comparable.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

VectorizerKind = Literal["count", "tfidf"]


class ModelNotFittedError(RuntimeError):
    """Raised when a model is queried before ``fit``."""


@dataclass(frozen=True, slots=True)
class TopicWord:
    """A word and its weight within one topic."""

    word: str
    weight: float


class TopicModel(abc.ABC):
    """Base class for the topic models.

    Subclasses declare which matrix they consume and build their estimator; the
    shared behaviour — top words, topic labels, normalized document shares —
    lives here.
    """

    #: Short key used in configs, artifact names and report columns.
    kind: ClassVar[str]

    #: Which vectorizer this model expects; enforced by the training pipeline.
    vectorizer_kind: ClassVar[VectorizerKind]

    #: Whether loadings are guaranteed non-negative (false for SVD).
    non_negative: ClassVar[bool] = True

    #: Human-readable name for charts and tables.
    display_name: ClassVar[str]

    def __init__(self, n_topics: int, *, seed: int = 0) -> None:
        if n_topics < 2:
            raise ValueError(f"n_topics must be at least 2, got {n_topics}")
        self.n_topics = n_topics
        self.seed = seed
        self.estimator = self._build_estimator()
        self.feature_names_: np.ndarray | None = None

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}(n_topics={self.n_topics}, seed={self.seed}, {state})"

    # ------------------------------------------------------------- subclasses

    @abc.abstractmethod
    def _build_estimator(self) -> Any:
        """Create the underlying scikit-learn estimator."""

    def fit_info(self) -> dict[str, float]:
        """Native fit diagnostics, not comparable across model families."""
        return {}

    # ------------------------------------------------------------------ state

    @property
    def is_fitted(self) -> bool:
        return self.feature_names_ is not None

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelNotFittedError(f"{type(self).__name__} must be fitted first")

    # ------------------------------------------------------------- fit / use

    def fit(self, matrix: Any, feature_names: Any) -> TopicModel:
        """Fit on a document-term matrix.

        Args:
            matrix: documents x features, from the vectorizer named by
                ``vectorizer_kind``.
            feature_names: vocabulary in column order.

        Raises:
            ValueError: if the vocabulary does not match the matrix width.
        """
        names = np.asarray(feature_names)
        if names.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"vocabulary has {names.shape[0]} entries but the matrix has "
                f"{matrix.shape[1]} columns"
            )

        self.estimator.fit(matrix)
        self.feature_names_ = names
        logger.info(
            "Fitted %s: %d topics over %d documents and %d features",
            self.kind,
            self.n_topics,
            matrix.shape[0],
            matrix.shape[1],
        )
        return self

    @property
    def components(self) -> np.ndarray:
        """Topic-term matrix, topics x features."""
        self._require_fitted()
        return np.asarray(self.estimator.components_)

    def transform(self, matrix: Any) -> np.ndarray:
        """Raw document loadings, in whatever scale the algorithm produces."""
        self._require_fitted()
        return np.asarray(self.estimator.transform(matrix), dtype=float)

    def document_topics(self, matrix: Any) -> np.ndarray:
        """Topic shares per document: non-negative rows summing to one.

        For LDA and NMF this is a rescaling of the native output. For LSA the
        negative part of each loading is clipped, which is a deliberate
        approximation: SVD coordinates are not a mixture, but a common scale is
        what allows dominant-topic assignment and the topic timeline to be
        computed identically for every model.
        """
        weights = self.transform(matrix)
        if not self.non_negative:
            weights = np.clip(weights, 0.0, None)
        totals = weights.sum(axis=1, keepdims=True)
        return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)

    def dominant_topics(self, matrix: Any) -> np.ndarray:
        """Index of the strongest topic for each document."""
        return np.argmax(self.document_topics(matrix), axis=1)

    # ------------------------------------------------------------ inspection

    def top_words(self, topic_index: int, n_words: int = 10) -> list[TopicWord]:
        """The ``n_words`` highest-weighted words of one topic.

        Raises:
            IndexError: if the topic index is out of range.
        """
        self._require_fitted()
        if not 0 <= topic_index < self.n_topics:
            raise IndexError(f"topic {topic_index} is out of range for {self.n_topics} topics")

        weights = self.components[topic_index]
        order = np.argsort(weights)[::-1][:n_words]
        assert self.feature_names_ is not None
        return [TopicWord(str(self.feature_names_[i]), float(weights[i])) for i in order]

    def topics(self, n_words: int = 10) -> list[list[TopicWord]]:
        """Top words for every topic."""
        return [self.top_words(index, n_words) for index in range(self.n_topics)]

    def topic_label(self, topic_index: int, n_words: int = 3) -> str:
        """Compact label built from the leading words, for legends and tables."""
        words = [entry.word for entry in self.top_words(topic_index, n_words)]
        return f"{topic_index + 1}. " + " · ".join(words)

    def topic_labels(self, n_words: int = 3) -> list[str]:
        return [self.topic_label(index, n_words) for index in range(self.n_topics)]

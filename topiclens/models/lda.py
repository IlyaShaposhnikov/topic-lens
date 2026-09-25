"""Latent Dirichlet Allocation."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from sklearn.decomposition import LatentDirichletAllocation

from topiclens.models.base import TopicModel, VectorizerKind


class LdaModel(TopicModel):
    """LDA over raw counts.

    LDA models how often each word occurs, so it takes the count matrix: TF-IDF
    weights would break the generative assumption it is built on.
    """

    kind: ClassVar[str] = "lda"
    vectorizer_kind: ClassVar[VectorizerKind] = "count"
    display_name: ClassVar[str] = "LDA"

    def __init__(
        self,
        n_topics: int,
        *,
        seed: int = 0,
        max_iter: int = 20,
        learning_method: Literal["batch", "online"] = "online",
        learning_decay: float = 0.7,
        doc_topic_prior: float | None = None,
        topic_word_prior: float | None = None,
    ) -> None:
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.learning_decay = learning_decay
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        super().__init__(n_topics, seed=seed)

    def _build_estimator(self) -> Any:
        return LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            learning_decay=self.learning_decay,
            doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior,
            random_state=self.seed,
        )

    def perplexity(self, matrix: Any) -> float:
        """Perplexity on the given matrix — meaningful only on held-out data."""
        self._require_fitted()
        return float(self.estimator.perplexity(matrix))

    def fit_info(self) -> dict[str, float]:
        self._require_fitted()
        return {"n_iter": float(self.estimator.n_iter_)}

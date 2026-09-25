"""Latent Semantic Analysis: truncated SVD of the TF-IDF matrix."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from topiclens.models.base import TopicModel, VectorizerKind
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)


class LsaModel(TopicModel):
    """LSA/LSI over TF-IDF weights.

    Unlike LDA and NMF, SVD produces signed components, which has two
    consequences this class deals with.

    The sign of a singular vector is arbitrary: a component and its negation
    describe the same subspace. Left alone, roughly half the topics would show
    their least characteristic words at the top. ``fit`` therefore orients every
    component so its dominant side is positive; because ``transform`` multiplies
    by ``components_``, document scores follow the flip automatically.

    Document vectors are also L2-normalized by default, so that similarity
    between documents does not track abstract length.
    """

    kind: ClassVar[str] = "lsa"
    vectorizer_kind: ClassVar[VectorizerKind] = "tfidf"
    non_negative: ClassVar[bool] = False
    display_name: ClassVar[str] = "LSA"

    def __init__(
        self,
        n_topics: int,
        *,
        seed: int = 0,
        algorithm: Literal["randomized", "arpack"] = "randomized",
        n_iter: int = 10,
        normalize: bool = True,
    ) -> None:
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.normalize = normalize
        super().__init__(n_topics, seed=seed)

    def _build_estimator(self) -> Any:
        return TruncatedSVD(
            n_components=self.n_topics,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
            random_state=self.seed,
        )

    def fit(self, matrix: Any, feature_names: Any) -> LsaModel:
        super().fit(matrix, feature_names)
        self._orient_components()
        return self

    def _orient_components(self) -> None:
        """Flip components whose negative side carries the larger mass."""
        components = self.estimator.components_
        flipped = 0
        for index, row in enumerate(components):
            positive = float(row[row > 0].sum())
            negative = float(-row[row < 0].sum())
            if negative > positive:
                components[index] = -row
                flipped += 1
        if flipped:
            logger.info("Oriented %d of %d LSA components", flipped, self.n_topics)

    def transform(self, matrix: Any) -> np.ndarray:
        weights = super().transform(matrix)
        if self.normalize:
            weights = normalize(weights, norm="l2")
        return np.asarray(weights, dtype=float)

    def fit_info(self) -> dict[str, float]:
        self._require_fitted()
        return {
            "explained_variance_ratio": float(self.estimator.explained_variance_ratio_.sum()),
        }

"""Non-negative Matrix Factorization."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from sklearn.decomposition import NMF

from topiclens.models.base import TopicModel, VectorizerKind


class NmfModel(TopicModel):
    """NMF over TF-IDF weights.

    The non-negativity constraint is what makes the factors readable as topics:
    words add to a topic, never subtract from it.
    """

    kind: ClassVar[str] = "nmf"
    vectorizer_kind: ClassVar[VectorizerKind] = "tfidf"
    display_name: ClassVar[str] = "NMF"

    def __init__(
        self,
        n_topics: int,
        *,
        seed: int = 0,
        beta_loss: Literal["frobenius", "kullback-leibler"] = "kullback-leibler",
        solver: Literal["cd", "mu"] = "mu",
        max_iter: int = 400,
        alpha_W: float = 0.0,
        l1_ratio: float = 0.0,
    ) -> None:
        self.beta_loss = beta_loss
        self.solver = solver
        self.max_iter = max_iter
        self.alpha_W = alpha_W
        self.l1_ratio = l1_ratio
        super().__init__(n_topics, seed=seed)

    def _build_estimator(self) -> Any:
        # 'mu' cannot update zeros left by 'nndsvd', which silently cripples
        # part of the factorization; 'nndsvda' fills them with the matrix mean.
        init = "nndsvda" if self.solver == "mu" else "nndsvd"
        return NMF(
            n_components=self.n_topics,
            beta_loss=self.beta_loss,
            solver=self.solver,
            max_iter=self.max_iter,
            alpha_W=self.alpha_W,
            l1_ratio=self.l1_ratio,
            init=init,
            random_state=self.seed,
        )

    def fit_info(self) -> dict[str, float]:
        self._require_fitted()
        return {
            "reconstruction_error": float(self.estimator.reconstruction_err_),
            "n_iter": float(self.estimator.n_iter_),
        }

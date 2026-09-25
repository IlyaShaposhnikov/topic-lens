"""Building models from the configuration."""

from __future__ import annotations

from topiclens.config import ModelsConfig
from topiclens.constants import MODEL_KEYS
from topiclens.models.base import TopicModel
from topiclens.models.lda import LdaModel
from topiclens.models.lsa import LsaModel
from topiclens.models.nmf import NmfModel

MODEL_CLASSES: dict[str, type[TopicModel]] = {
    LdaModel.kind: LdaModel,
    NmfModel.kind: NmfModel,
    LsaModel.kind: LsaModel,
}


def build_model(
    kind: str,
    config: ModelsConfig,
    *,
    n_topics: int | None = None,
    seed: int = 0,
) -> TopicModel:
    """Instantiate one model with its configured hyperparameters.

    Args:
        kind: one of ``lda``, ``nmf``, ``lsa``.
        config: the ``models`` config section.
        n_topics: overrides ``config.n_topics``, used when sweeping k.
        seed: random state, taken from the project config by the caller.

    Raises:
        ValueError: on an unknown model kind.
    """
    if kind not in MODEL_CLASSES:
        raise ValueError(f"unknown model kind {kind!r}; expected one of {', '.join(MODEL_KEYS)}")

    model_class = MODEL_CLASSES[kind]
    settings = getattr(config, kind).model_dump()
    return model_class(n_topics or config.n_topics, seed=seed, **settings)


def build_all_models(
    config: ModelsConfig, *, n_topics: int | None = None, seed: int = 0
) -> dict[str, TopicModel]:
    """One instance per model family, in a stable order."""
    return {kind: build_model(kind, config, n_topics=n_topics, seed=seed) for kind in MODEL_KEYS}

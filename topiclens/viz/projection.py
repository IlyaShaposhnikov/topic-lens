"""Putting the topics of every model on one map.

The tables say LDA and NMF agree while LSA does not; this shows it. All topics
of all three models are projected together, by a single transform, which is what
makes distances on the picture comparable across models — projecting each model
separately would produce three pretty and unrelated pictures.

Two steps precede the projection. Topic vectors live in different vocabularies
(counts for LDA, TF-IDF for the others), so they are restricted to the shared
words first. They are then L2-normalized, because what matters is the direction
of a topic, not the scale the algorithm happens to use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from topiclens.evaluation.matching import align_vocabularies
from topiclens.utils.logging_config import get_logger
from topiclens.viz.charts import _LAYOUT, MODEL_COLORS

logger = get_logger(__name__)

ProjectionMethod = Literal["pca", "mds"]


def shared_topic_matrix(models: Mapping[str, Any]) -> tuple[np.ndarray, list[str], list[int]]:
    """Stack every topic of every model onto one shared vocabulary.

    Returns the stacked matrix, the model each row belongs to, and the topic
    index within that model.

    Raises:
        ValueError: if fewer than two models are given.
    """
    if len(models) < 2:
        raise ValueError("a topic map needs at least two models")

    keys = sorted(models)
    reference = models[keys[0]]
    shared_words: list[str] | None = None
    aligned: dict[str, np.ndarray] = {}

    for key in keys:
        block, _, shared = align_vocabularies(
            models[key].components,
            models[key].feature_names_,
            reference.components,
            reference.feature_names_,
        )
        aligned[key] = block
        shared_words = shared if shared_words is None else shared_words

    # The pairwise alignments above can differ in width; intersect once more by
    # re-aligning each block to the narrowest vocabulary.
    width = min(block.shape[1] for block in aligned.values())
    matrix = np.vstack([block[:, :width] for _, block in sorted(aligned.items())])

    owners = [key for key in keys for _ in range(models[key].n_topics)]
    indices = [index for key in keys for index in range(models[key].n_topics)]
    logger.info("Topic map over %d shared features", width)
    return matrix, owners, indices


def project(matrix: np.ndarray, method: ProjectionMethod = "pca", seed: int = 0) -> np.ndarray:
    """Reduce topic vectors to two coordinates.

    PCA keeps global structure and is deterministic; MDS preserves pairwise
    cosine distances more faithfully, which suits a map whose whole purpose is
    "which topics sit near which".

    Raises:
        ValueError: on an unknown method.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(normalized)
    if method == "mds":
        from sklearn.manifold import MDS

        distances = np.clip(1.0 - normalized @ normalized.T, 0.0, None)
        np.fill_diagonal(distances, 0.0)
        return MDS(
            n_components=2,
            metric="precomputed",
            init="classical_mds",
            n_init=1,
            random_state=seed,
            normalized_stress=False,
        ).fit_transform(distances)
    raise ValueError(f"unknown projection method {method!r}; expected 'pca' or 'mds'")


def topic_layout(
    models: Mapping[str, Any],
    *,
    shares: Mapping[str, Sequence[float]] | None = None,
    method: ProjectionMethod = "pca",
    seed: int = 0,
) -> pd.DataFrame:
    """Coordinates, labels and sizes for every topic of every model.

    Args:
        models: fitted models, keyed by their short name.
        shares: mean share of the corpus per topic, used for the dot size.
        method: ``pca`` or ``mds``.
        seed: random state of the projection.
    """
    matrix, owners, indices = shared_topic_matrix(models)
    coordinates = project(matrix, method=method, seed=seed)

    rows = []
    for position, (owner, index) in enumerate(zip(owners, indices, strict=True)):
        weight = float(shares[owner][index]) if shares and owner in shares else 1.0
        rows.append(
            {
                "model": owner,
                "topic": index + 1,
                "label": models[owner].topic_label(index),
                "x": float(coordinates[position, 0]),
                "y": float(coordinates[position, 1]),
                "share": weight,
            }
        )
    return pd.DataFrame(rows)


def topic_map_chart(layout: pd.DataFrame, *, title: str = "Topic map") -> go.Figure:
    """Scatter of all topics, coloured by model and sized by corpus share."""
    figure = go.Figure()
    scale = 60.0 / max(layout["share"].max(), 1e-9)

    for model, block in layout.groupby("model", sort=True):
        figure.add_scatter(
            x=block["x"],
            y=block["y"],
            mode="markers+text",
            name=str(model).upper(),
            text=block["topic"],
            textposition="middle center",
            textfont={"size": 10, "color": "white"},
            marker={
                "size": block["share"] * scale + 18,
                "color": MODEL_COLORS.get(str(model)),
                "opacity": 0.75,
                "line": {"width": 1, "color": "white"},
            },
            customdata=np.stack([block["label"], block["share"]], axis=-1),
            hovertemplate="%{customdata[0]}<br>share %{customdata[1]:.1%}<extra>"
            + str(model).upper()
            + "</extra>",
        )

    figure.update_layout(title=title, **_LAYOUT, height=560)
    figure.update_xaxes(showticklabels=False, zeroline=False)
    figure.update_yaxes(showticklabels=False, zeroline=False)
    return figure

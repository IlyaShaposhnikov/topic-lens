"""Tests for the cross-model topic map."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from topiclens.config import ModelsConfig
from topiclens.models.factory import build_model
from topiclens.viz.projection import project, shared_topic_matrix, topic_layout, topic_map_chart

CORPUS = [
    "language model translation corpus text",
    "language model translation corpus sentence",
    "robot motion planning control navigation",
    "robot motion planning control sensor",
] * 5


@pytest.fixture
def models():
    counts = CountVectorizer(min_df=1)
    count_matrix = counts.fit_transform(CORPUS)
    weights = TfidfVectorizer(min_df=1)
    tfidf_matrix = weights.fit_transform(CORPUS)
    config = ModelsConfig(n_topics=2)

    return {
        "lda": build_model("lda", config, seed=1).fit(count_matrix, counts.get_feature_names_out()),
        "nmf": build_model("nmf", config, seed=1).fit(
            tfidf_matrix, weights.get_feature_names_out()
        ),
        "lsa": build_model("lsa", config, seed=1).fit(
            tfidf_matrix, weights.get_feature_names_out()
        ),
    }


def test_shared_matrix_holds_every_topic(models):
    matrix, owners, indices = shared_topic_matrix(models)
    assert matrix.shape[0] == 6
    assert owners.count("lda") == 2
    assert indices == [0, 1, 0, 1, 0, 1]


def test_a_single_model_is_not_a_map(models):
    with pytest.raises(ValueError, match="at least two models"):
        shared_topic_matrix({"nmf": models["nmf"]})


def test_pca_projection_is_two_dimensional_and_deterministic(models):
    matrix, _, _ = shared_topic_matrix(models)
    first = project(matrix, method="pca", seed=1)
    second = project(matrix, method="pca", seed=1)

    assert first.shape == (6, 2)
    assert np.allclose(first, second)


def test_mds_projection_returns_coordinates(models):
    matrix, _, _ = shared_topic_matrix(models)
    assert project(matrix, method="mds", seed=1).shape == (6, 2)


def test_unknown_projection_is_rejected(models):
    matrix, _, _ = shared_topic_matrix(models)
    with pytest.raises(ValueError, match="unknown projection method"):
        project(matrix, method="umap")


def test_layout_has_a_row_per_topic_with_labels(models):
    layout = topic_layout(models, seed=1)
    assert len(layout) == 6
    assert set(layout["model"]) == {"lda", "nmf", "lsa"}
    assert layout["label"].str.contains("·").all()


def test_layout_uses_the_supplied_shares(models):
    shares = {"lda": [0.7, 0.3], "nmf": [0.5, 0.5], "lsa": [0.9, 0.1]}
    layout = topic_layout(models, shares=shares, seed=1)
    row = layout.query("model == 'lda' and topic == 1").iloc[0]
    assert row["share"] == pytest.approx(0.7)


def test_layout_defaults_to_equal_sizes(models):
    assert (topic_layout(models, seed=1)["share"] == 1.0).all()


def test_chart_draws_one_series_per_model(models):
    figure = topic_map_chart(topic_layout(models, seed=1))
    assert len(figure.data) == 3
    assert {trace.name for trace in figure.data} == {"LDA", "NMF", "LSA"}


def test_chart_hides_the_meaningless_axes(models):
    """Projection coordinates carry no units; showing ticks would imply they do."""
    figure = topic_map_chart(topic_layout(models, seed=1))
    assert figure.layout.xaxis.showticklabels is False
    assert figure.layout.yaxis.showticklabels is False

"""Tests for the model interface and the three implementations.

A tiny two-topic corpus is enough: the point is the contract — shapes, ordering,
determinism, normalization — not the quality of the topics.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from topiclens.config import ModelsConfig
from topiclens.models.base import ModelNotFittedError, TopicModel
from topiclens.models.factory import MODEL_CLASSES, build_all_models, build_model
from topiclens.models.lda import LdaModel
from topiclens.models.lsa import LsaModel
from topiclens.models.nmf import NmfModel

CORPUS = [
    "language model translation text corpus",
    "language text corpus translation model",
    "text corpus language model sentence",
    "robot motion planning control navigation",
    "motion planning robot navigation control",
    "robot control navigation motion sensor",
] * 4


@pytest.fixture
def counts():
    vectorizer = CountVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(CORPUS)
    return matrix, vectorizer.get_feature_names_out()


@pytest.fixture
def weights():
    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(CORPUS)
    return matrix, vectorizer.get_feature_names_out()


@pytest.fixture
def config() -> ModelsConfig:
    return ModelsConfig(n_topics=2)


def fitted(model: TopicModel, data) -> TopicModel:
    matrix, names = data
    return model.fit(matrix, names)


# ------------------------------------------------------------------- factory


def test_factory_builds_each_model_kind(config):
    assert isinstance(build_model("lda", config), LdaModel)
    assert isinstance(build_model("nmf", config), NmfModel)
    assert isinstance(build_model("lsa", config), LsaModel)


def test_factory_rejects_unknown_kind(config):
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("word2vec", config)


def test_factory_passes_configured_hyperparameters(config):
    config = config.model_copy(update={"lda": config.lda.model_copy(update={"max_iter": 3})})
    assert build_model("lda", config).estimator.max_iter == 3


def test_factory_can_override_the_topic_count(config):
    assert build_model("nmf", config, n_topics=5).n_topics == 5


def test_build_all_models_covers_every_kind(config):
    models = build_all_models(config)
    assert set(models) == set(MODEL_CLASSES)


def test_vectorizer_kinds_match_the_algorithms():
    assert LdaModel.vectorizer_kind == "count"
    assert NmfModel.vectorizer_kind == "tfidf"
    assert LsaModel.vectorizer_kind == "tfidf"


def test_too_few_topics_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        LdaModel(1)


# -------------------------------------------------------------- the contract


@pytest.mark.parametrize("kind", ["lda", "nmf", "lsa"])
def test_unfitted_model_refuses_to_answer(kind, config):
    model = build_model(kind, config)
    assert not model.is_fitted
    with pytest.raises(ModelNotFittedError):
        model.top_words(0)


def test_fit_rejects_a_mismatched_vocabulary(config, counts):
    matrix, names = counts
    with pytest.raises(ValueError, match="vocabulary"):
        build_model("lda", config).fit(matrix, names[:-1])


@pytest.mark.parametrize("kind", ["lda", "nmf", "lsa"])
def test_components_and_transform_have_the_right_shapes(kind, config, counts, weights):
    data = counts if MODEL_CLASSES[kind].vectorizer_kind == "count" else weights
    matrix, _ = data
    model = fitted(build_model(kind, config), data)

    assert model.components.shape == (2, matrix.shape[1])
    assert model.transform(matrix).shape == (matrix.shape[0], 2)


@pytest.mark.parametrize("kind", ["lda", "nmf", "lsa"])
def test_document_topics_are_shares(kind, config, counts, weights):
    data = counts if MODEL_CLASSES[kind].vectorizer_kind == "count" else weights
    matrix, _ = data
    shares = fitted(build_model(kind, config), data).document_topics(matrix)

    assert (shares >= 0).all()
    assert np.allclose(shares.sum(axis=1), 1.0)


@pytest.mark.parametrize("kind", ["lda", "nmf", "lsa"])
def test_top_words_are_ordered_and_sized(kind, config, counts, weights):
    data = counts if MODEL_CLASSES[kind].vectorizer_kind == "count" else weights
    model = fitted(build_model(kind, config), data)
    words = model.top_words(0, n_words=4)

    assert len(words) == 4
    assert [entry.weight for entry in words] == sorted(
        (entry.weight for entry in words), reverse=True
    )


@pytest.mark.parametrize("kind", ["lda", "nmf", "lsa"])
def test_topic_index_is_range_checked(kind, config, counts, weights):
    data = counts if MODEL_CLASSES[kind].vectorizer_kind == "count" else weights
    model = fitted(build_model(kind, config), data)
    with pytest.raises(IndexError):
        model.top_words(2)


@pytest.mark.parametrize("kind", ["lda", "nmf", "lsa"])
def test_same_seed_gives_the_same_topics(kind, config, counts, weights):
    data = counts if MODEL_CLASSES[kind].vectorizer_kind == "count" else weights
    first = fitted(build_model(kind, config, seed=7), data).components
    second = fitted(build_model(kind, config, seed=7), data).components
    assert np.allclose(first, second)


def test_topics_separate_the_two_themes(config, weights):
    """A sanity check that the wiring produces meaningful output at all."""
    model = fitted(build_model("nmf", config), weights)
    vocabularies = [{entry.word for entry in topic} for topic in model.topics(4)]

    language = {"language", "corpus", "translation", "text"}
    robotics = {"robot", "motion", "planning", "navigation"}
    assert any(len(words & language) >= 3 for words in vocabularies)
    assert any(len(words & robotics) >= 3 for words in vocabularies)


def test_topic_labels_are_numbered_and_readable(config, counts):
    labels = fitted(build_model("lda", config), counts).topic_labels(n_words=2)
    assert len(labels) == 2
    assert labels[0].startswith("1. ")
    assert "·" in labels[0]


def test_dominant_topics_index_within_range(config, weights):
    matrix, _ = weights
    assignments = fitted(build_model("nmf", config), weights).dominant_topics(matrix)
    assert assignments.shape == (matrix.shape[0],)
    assert set(np.unique(assignments)) <= {0, 1}


# ------------------------------------------------------------- per-model bits


def test_lda_reports_perplexity_and_iterations(config, counts):
    matrix, _ = counts
    model = fitted(build_model("lda", config), counts)
    assert model.perplexity(matrix) > 0
    assert model.fit_info()["n_iter"] >= 1


def test_nmf_reports_reconstruction_error(config, weights):
    info = fitted(build_model("nmf", config), weights).fit_info()
    assert info["reconstruction_error"] >= 0


def test_lsa_reports_explained_variance(config, weights):
    ratio = fitted(build_model("lsa", config), weights).fit_info()["explained_variance_ratio"]
    assert 0.0 < ratio <= 1.0


def test_lsa_orients_components_towards_their_dominant_side(config, weights):
    model = fitted(build_model("lsa", config), weights)
    for row in model.components:
        assert row[row > 0].sum() >= -row[row < 0].sum()


def test_lsa_normalizes_document_vectors(config, weights):
    matrix, _ = weights
    model = fitted(build_model("lsa", config), weights)
    norms = np.linalg.norm(model.transform(matrix), axis=1)
    assert np.allclose(norms, 1.0)


def test_lsa_normalization_can_be_disabled(config, weights):
    config = config.model_copy(update={"lsa": config.lsa.model_copy(update={"normalize": False})})
    matrix, _ = weights
    model = fitted(build_model("lsa", config), weights)
    norms = np.linalg.norm(model.transform(matrix), axis=1)
    assert not np.allclose(norms, 1.0)

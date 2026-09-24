"""Tests for cleaning, tokenization and vectorizer construction."""

from __future__ import annotations

import pytest

from topiclens.config import PreprocessingConfig
from topiclens.preprocessing import (
    TOKEN_PATTERN,
    Preprocessor,
    build_stopwords,
    build_vectorizer,
    lemmatize,
)


@pytest.fixture
def config() -> PreprocessingConfig:
    return PreprocessingConfig(extra_stopwords=["paper", "results"])


@pytest.fixture
def preprocessor(config) -> Preprocessor:
    return Preprocessor(config)


@pytest.fixture
def tiny_config(config) -> PreprocessingConfig:
    """Production min_df/max_df assume thousands of documents, not two."""
    vectorizer = config.vectorizer.model_copy(
        update={
            "count": config.vectorizer.count.model_copy(update={"min_df": 1, "max_df": 1.0}),
            "tfidf": config.vectorizer.tfidf.model_copy(update={"min_df": 1, "max_df": 1.0}),
        }
    )
    return config.model_copy(update={"vectorizer": vectorizer})


# ------------------------------------------------------------------ cleaning


def test_clean_removes_inline_math(preprocessor):
    assert "alpha" not in preprocessor.clean(r"We set $\alpha = 0.5$ for training")


def test_clean_removes_display_math(preprocessor):
    cleaned = preprocessor.clean(r"Objective $$\sum_{i=1}^n x_i$$ is convex")
    assert cleaned == "Objective is convex"


def test_clean_removes_latex_commands(preprocessor):
    cleaned = preprocessor.clean(r"the \textbf{robust} \mathbb{R} estimator")
    assert "textbf" not in cleaned
    assert "mathbb" not in cleaned


def test_clean_removes_urls(preprocessor):
    cleaned = preprocessor.clean("Code at https://github.com/user/repo and more")
    assert "github" not in cleaned


def test_clean_keeps_latex_when_disabled(config):
    preprocessor = Preprocessor(config.model_copy(update={"remove_latex": False}))
    assert "alpha" in preprocessor.clean(r"$\alpha$ matters")


def test_clean_collapses_whitespace(preprocessor):
    assert preprocessor.clean("two\n  spaced\tlines") == "two spaced lines"


# -------------------------------------------------------------- tokenization


def test_tokens_drop_stopwords_and_short_words(preprocessor):
    assert preprocessor.tokens("the model is in a robust state") == ["model", "robust", "state"]


def test_tokens_apply_domain_stopwords(preprocessor):
    assert "paper" not in preprocessor.tokens("this paper presents a model")


def test_tokens_apply_domain_stopwords_after_lemmatization(preprocessor):
    """'results' in the config must also filter the lemma 'result'."""
    assert "result" not in preprocessor.tokens("our results and our result")


def test_tokens_lemmatize_plurals_and_verbs(preprocessor):
    tokens = preprocessor.tokens("embeddings learning transformers")
    assert tokens == ["embedding", "learn", "transformer"]


def test_tokens_keep_hyphenated_terms(preprocessor):
    assert "self-supervised" in preprocessor.tokens("a self-supervised objective")


def test_tokens_drop_numbers_by_default(preprocessor):
    assert preprocessor.tokens("trained for 300 epochs on 4 gpus") == ["train", "epoch", "gpus"]


def test_tokens_keep_numbers_when_configured(config):
    preprocessor = Preprocessor(config.model_copy(update={"remove_numbers": False}))
    assert "300" in preprocessor.tokens("trained for 300 epochs")


def test_min_token_length_is_respected(config):
    preprocessor = Preprocessor(config.model_copy(update={"min_token_length": 6}))
    assert preprocessor.tokens("a robust language model") == ["robust", "language"]


def test_lemmatization_can_be_disabled(config):
    preprocessor = Preprocessor(config.model_copy(update={"lemmatize": False}))
    assert preprocessor.tokens("embeddings learning") == ["embeddings", "learning"]


def test_transform_joins_tokens(preprocessor):
    assert preprocessor.transform("Robust language models") == "robust language model"


def test_transform_many_handles_an_empty_document(preprocessor):
    processed = preprocessor.transform_many(["the a an", "robust language model"])
    assert processed[0] == ""
    assert processed[1].startswith("robust")


def test_lemmatize_is_idempotent():
    assert lemmatize(lemmatize("models")) == "model"


def test_lemmatize_folds_domain_plurals_unknown_to_wordnet():
    assert lemmatize("embeddings") == "embedding"
    assert lemmatize("autoencoders") == "autoencoder"


def test_lemmatize_leaves_words_wordnet_knows_alone():
    assert lemmatize("series") == "series"
    assert lemmatize("analysis") == "analysis"
    assert lemmatize("bias") == "bias"


def test_build_stopwords_includes_both_sources(config):
    words = build_stopwords(config)
    assert "the" in words
    assert "paper" in words
    assert "result" in words


# -------------------------------------------------------------- vectorizers


def test_count_vectorizer_uses_config_and_whitespace_tokens(tiny_config):
    vectorizer = build_vectorizer("count", tiny_config)
    matrix = vectorizer.fit_transform(["self-supervised model", "model model"])
    assert vectorizer.token_pattern == TOKEN_PATTERN
    assert "self-supervised" in vectorizer.get_feature_names_out()
    assert matrix[1, vectorizer.vocabulary_["model"]] == 2


def test_tfidf_vectorizer_emits_bigrams(tiny_config):
    vectorizer = build_vectorizer("tfidf", tiny_config)
    vectorizer.fit(["language model pretraining"] * 3)
    assert "language model" in set(vectorizer.get_feature_names_out())


def test_vectorizer_respects_min_df(tiny_config):
    settings = tiny_config.vectorizer.count.model_copy(update={"min_df": 2})
    tuned = tiny_config.model_copy(
        update={"vectorizer": tiny_config.vectorizer.model_copy(update={"count": settings})}
    )
    vectorizer = build_vectorizer("count", tuned)
    vectorizer.fit(["shared term", "shared other"])
    assert set(vectorizer.get_feature_names_out()) == {"shared"}


def test_vectorizer_respects_max_features(tiny_config):
    settings = tiny_config.vectorizer.count.model_copy(update={"max_features": 1})
    tuned = tiny_config.model_copy(
        update={"vectorizer": tiny_config.vectorizer.model_copy(update={"count": settings})}
    )
    vectorizer = build_vectorizer("count", tuned)
    vectorizer.fit(["frequent frequent rare"])
    assert list(vectorizer.get_feature_names_out()) == ["frequent"]

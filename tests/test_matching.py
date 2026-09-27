"""Tests for cross-model topic matching and topic-count selection."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from topiclens.config import ModelsConfig
from topiclens.evaluation.coherence import CoherenceCalculator
from topiclens.evaluation.matching import (
    align_vocabularies,
    cosine_similarity_matrix,
    match_topics,
    mean_match_similarity,
    topic_similarity,
    unmatched_topics,
)
from topiclens.evaluation.selection import SweepRow, best_topic_count, sweep_topic_counts
from topiclens.models.factory import build_model

CORPUS = [
    "language model translation corpus text",
    "language model translation corpus sentence",
    "language model text corpus translation",
    "robot motion planning control navigation",
    "robot motion planning control sensor",
    "robot motion control navigation planning",
] * 4


@pytest.fixture
def config() -> ModelsConfig:
    return ModelsConfig(n_topics=2)


@pytest.fixture
def counts():
    vectorizer = CountVectorizer(min_df=1)
    return vectorizer.fit_transform(CORPUS), vectorizer.get_feature_names_out()


@pytest.fixture
def weights():
    vectorizer = TfidfVectorizer(min_df=1)
    return vectorizer.fit_transform(CORPUS), vectorizer.get_feature_names_out()


@pytest.fixture
def lda(config, counts):
    matrix, names = counts
    return build_model("lda", config, seed=1).fit(matrix, names)


@pytest.fixture
def nmf(config, weights):
    matrix, names = weights
    return build_model("nmf", config, seed=1).fit(matrix, names)


# --------------------------------------------------------- vocabulary alignment


def test_alignment_keeps_only_shared_words():
    left = np.array([[1.0, 2.0, 3.0]])
    right = np.array([[4.0, 5.0]])
    aligned_left, aligned_right, shared = align_vocabularies(
        left, ["robot", "model", "query"], right, ["query", "robot"]
    )

    assert shared == ["query", "robot"]
    assert aligned_left.tolist() == [[3.0, 1.0]]
    assert aligned_right.tolist() == [[4.0, 5.0]]


def test_alignment_rejects_disjoint_vocabularies():
    with pytest.raises(ValueError, match="share no vocabulary"):
        align_vocabularies(np.ones((1, 2)), ["a", "b"], np.ones((1, 2)), ["c", "d"])


# -------------------------------------------------------------------- similarity


def test_cosine_of_identical_rows_is_one():
    matrix = np.array([[1.0, 2.0, 3.0]])
    assert cosine_similarity_matrix(matrix, matrix)[0, 0] == pytest.approx(1.0)


def test_cosine_is_scale_invariant():
    first = np.array([[1.0, 2.0]])
    assert cosine_similarity_matrix(first, first * 10)[0, 0] == pytest.approx(1.0)


def test_cosine_of_orthogonal_rows_is_zero():
    result = cosine_similarity_matrix(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert result[0, 0] == pytest.approx(0.0)


def test_zero_rows_do_not_produce_nan():
    result = cosine_similarity_matrix(np.zeros((1, 3)), np.ones((1, 3)))
    assert np.isfinite(result).all()


def test_similarity_matrix_has_one_entry_per_topic_pair(lda, nmf):
    similarity, shared = topic_similarity(lda, nmf)
    assert similarity.shape == (lda.n_topics, nmf.n_topics)
    assert shared


# ---------------------------------------------------------------------- matching


def test_matching_is_one_to_one(lda, nmf):
    matches = match_topics(lda, nmf)
    assert len(matches) == 2
    assert len({match.topic_a for match in matches}) == 2
    assert len({match.topic_b for match in matches}) == 2


def test_matching_is_sorted_by_similarity(lda, nmf):
    similarities = [match.similarity for match in match_topics(lda, nmf)]
    assert similarities == sorted(similarities, reverse=True)


def test_a_model_matched_to_itself_pairs_each_topic_with_itself(nmf):
    matches = match_topics(nmf, nmf)
    assert all(match.topic_a == match.topic_b for match in matches)
    assert mean_match_similarity(matches) == pytest.approx(1.0)


def test_matching_reports_the_words_two_topics_agree_on(lda, nmf):
    matches = match_topics(lda, nmf, top_words=5)
    assert any(match.shared_words for match in matches)


def test_hungarian_beats_greedy_on_a_crafted_case():
    """Greedy would grab the 0.9 pair and be forced into 0.1; optimal takes 0.8+0.7."""
    similarity = np.array([[0.9, 0.8], [0.7, 0.1]])
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-similarity)
    assert similarity[rows, columns].sum() == pytest.approx(1.5)


def test_surplus_topics_are_reported_as_unmatched(config, counts, weights):
    count_matrix, count_names = counts
    tfidf_matrix, tfidf_names = weights
    wide = build_model("lda", config, n_topics=4, seed=1).fit(count_matrix, count_names)
    narrow = build_model("nmf", config, n_topics=2, seed=1).fit(tfidf_matrix, tfidf_names)

    matches = match_topics(wide, narrow)
    assert len(matches) == 2
    assert len(unmatched_topics(matches, wide.n_topics, side="a")) == 2


def test_mean_similarity_of_no_matches_is_zero():
    assert mean_match_similarity([]) == 0.0


# --------------------------------------------------------------------- selection


def test_sweep_covers_every_requested_topic_count(config, counts):
    matrix, names = counts
    scorer = CoherenceCalculator(matrix, names)
    rows = sweep_topic_counts(
        "lda", config, matrix, names, scorer, topic_counts=[2, 3], top_words=5
    )

    assert [row.n_topics for row in rows] == [2, 3]
    assert all(row.kind == "lda" for row in rows)
    assert all(row.seconds > 0 for row in rows)
    assert all(0.0 <= row.diversity <= 1.0 for row in rows)


def test_sweep_rows_serialize_for_reports(config, counts):
    matrix, names = counts
    scorer = CoherenceCalculator(matrix, names)
    record = sweep_topic_counts(
        "nmf", config, matrix, names, scorer, topic_counts=[2], top_words=5
    )[0].as_record()

    assert record["model"] == "nmf"
    assert set(record) == {"model", "n_topics", "coherence", "diversity", "seconds"}


def test_best_topic_count_takes_the_highest_coherence():
    rows = [
        SweepRow("lda", 5, 0.10, 0.9, 1.0),
        SweepRow("lda", 10, 0.18, 0.9, 2.0),
        SweepRow("lda", 15, 0.14, 0.9, 3.0),
    ]
    assert best_topic_count(rows) == 10


def test_ties_prefer_the_simpler_model():
    rows = [SweepRow("lda", 5, 0.15, 0.9, 1.0), SweepRow("lda", 20, 0.15, 0.9, 4.0)]
    assert best_topic_count(rows) == 5


def test_collapsed_topics_are_excluded_by_the_diversity_floor():
    rows = [
        SweepRow("lda", 5, 0.30, 0.30, 1.0),  # high coherence, degenerate topics
        SweepRow("lda", 10, 0.20, 0.85, 2.0),
    ]
    assert best_topic_count(rows, min_diversity=0.5) == 10


def test_an_impossible_diversity_floor_is_an_error():
    with pytest.raises(ValueError, match="no sweep result"):
        best_topic_count([SweepRow("lda", 5, 0.3, 0.4, 1.0)], min_diversity=0.9)

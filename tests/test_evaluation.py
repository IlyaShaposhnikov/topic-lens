"""Tests for coherence, diversity and label-alignment metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from topiclens.evaluation.alignment import (
    alignment_scores,
    cluster_purity,
    topic_label_distribution,
)
from topiclens.evaluation.coherence import CoherenceCalculator
from topiclens.evaluation.diversity import most_similar_pair, pairwise_overlap, topic_diversity

# Two themes whose words co-occur strictly within their own documents.
CORPUS = [
    "language model translation corpus",
    "language model translation corpus",
    "language model translation sentence",
    "robot motion planning control",
    "robot motion planning control",
    "robot motion planning sensor",
]

COHERENT_TOPIC = ["language", "model", "translation", "corpus"]
MIXED_TOPIC = ["language", "robot", "corpus", "sensor"]


@pytest.fixture
def calculator() -> CoherenceCalculator:
    vectorizer = CountVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(CORPUS)
    return CoherenceCalculator(matrix, vectorizer.get_feature_names_out())


# ----------------------------------------------------------------- coherence


def test_npmi_is_within_its_theoretical_range(calculator):
    assert -1.0 <= calculator.npmi(COHERENT_TOPIC) <= 1.0


def test_npmi_prefers_words_that_travel_together(calculator):
    assert calculator.npmi(COHERENT_TOPIC) > calculator.npmi(MIXED_TOPIC)


def test_npmi_is_maximal_for_words_that_always_co_occur(calculator):
    assert calculator.npmi(["language", "model"]) == pytest.approx(1.0)


def test_npmi_is_minimal_for_words_that_never_co_occur(calculator):
    assert calculator.npmi(["corpus", "sensor"]) == pytest.approx(-1.0)


def test_umass_prefers_the_coherent_topic(calculator):
    assert calculator.umass(COHERENT_TOPIC) > calculator.umass(MIXED_TOPIC)


def test_coherence_ignores_words_outside_the_vocabulary(calculator):
    both_known = calculator.npmi(["language", "model"])
    with_stranger = calculator.npmi(["language", "model", "quantumcryptography"])
    assert with_stranger == pytest.approx(both_known)


def test_coherence_of_a_single_known_word_is_neutral(calculator):
    assert calculator.npmi(["language"]) == 0.0
    assert calculator.umass(["language"]) == 0.0


def test_score_returns_one_value_per_topic(calculator):
    scores = calculator.score([COHERENT_TOPIC, MIXED_TOPIC])
    assert len(scores) == 2
    assert calculator.mean_score([COHERENT_TOPIC, MIXED_TOPIC]) == pytest.approx(np.mean(scores))


def test_unknown_metric_is_rejected(calculator):
    with pytest.raises(ValueError, match="unknown coherence metric"):
        calculator.score([COHERENT_TOPIC], metric="cv")


def test_tfidf_matrix_is_accepted_as_well():
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(CORPUS)
    calculator = CoherenceCalculator(matrix, vectorizer.get_feature_names_out())
    assert calculator.npmi(["language", "model"]) == pytest.approx(1.0)


# ----------------------------------------------------------------- diversity


def test_diversity_is_one_for_disjoint_topics():
    assert topic_diversity([["a", "b"], ["c", "d"]]) == 1.0


def test_diversity_falls_when_topics_repeat_each_other():
    assert topic_diversity([["a", "b"], ["a", "b"]]) == 0.5


def test_diversity_respects_the_top_n_cutoff():
    topics = [["a", "b", "z"], ["c", "d", "z"]]
    assert topic_diversity(topics, top_n=2) == 1.0
    assert topic_diversity(topics) < 1.0


def test_diversity_of_no_topics_is_zero():
    assert topic_diversity([]) == 0.0


def test_pairwise_overlap_is_zero_for_disjoint_topics():
    assert pairwise_overlap([["a", "b"], ["c", "d"]]) == 0.0


def test_pairwise_overlap_is_one_for_identical_topics():
    assert pairwise_overlap([["a", "b"], ["a", "b"]]) == 1.0


def test_most_similar_pair_names_the_redundant_topics():
    topics = [["a", "b"], ["c", "d"], ["a", "b"]]
    first, second, score = most_similar_pair(topics)
    assert {first, second} == {0, 2}
    assert score == 1.0


# ----------------------------------------------------------------- alignment


def test_purity_is_one_when_topics_match_labels():
    assert cluster_purity([0, 0, 1, 1], ["cs.CL", "cs.CL", "cs.RO", "cs.RO"]) == 1.0


def test_purity_reflects_a_mixed_topic():
    assert cluster_purity([0, 0, 0, 0], ["cs.CL", "cs.CL", "cs.RO", "cs.RO"]) == 0.5


def test_alignment_scores_are_perfect_for_a_matching_partition():
    scores = alignment_scores([0, 0, 1, 1], ["cs.CL", "cs.CL", "cs.RO", "cs.RO"])
    assert scores["nmi"] == pytest.approx(1.0)
    assert scores["ari"] == pytest.approx(1.0)
    assert scores["purity"] == pytest.approx(1.0)


def test_alignment_scores_are_low_for_an_unrelated_partition():
    scores = alignment_scores([0, 1, 0, 1], ["cs.CL", "cs.CL", "cs.RO", "cs.RO"])
    assert scores["nmi"] < 0.1
    assert scores["ari"] < 0.1


def test_alignment_survives_more_topics_than_labels():
    """Ten topics over five categories is the normal case, not an error."""
    scores = alignment_scores([0, 1, 2, 3], ["cs.CL", "cs.CL", "cs.RO", "cs.RO"])
    assert 0.0 < scores["nmi"] <= 1.0
    assert scores["purity"] == 1.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="must match"):
        alignment_scores([0, 1], ["cs.CL"])


def test_label_distribution_covers_every_topic_including_empty_ones():
    distribution = topic_label_distribution([0, 0, 1], ["cs.CL", "cs.RO", "cs.CL"], n_topics=3)
    assert distribution[0] == {"cs.CL": 1, "cs.RO": 1}
    assert distribution[1] == {"cs.CL": 1}
    assert distribution[2] == {}

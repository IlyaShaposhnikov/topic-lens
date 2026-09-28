"""Tests for the chart builders.

Figures are inspected as data structures — traces, axes, titles — rather than
rendered: what matters here is that the right numbers reach the right places.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from topiclens.config import ModelsConfig
from topiclens.evaluation.selection import SweepRow
from topiclens.models.factory import build_model
from topiclens.viz.charts import (
    MODEL_COLORS,
    label_distribution_chart,
    metrics_comparison_chart,
    similarity_heatmap,
    sweep_chart,
    timeline_chart,
    topic_words_chart,
)

CORPUS = [
    "language model translation corpus text",
    "language model translation corpus sentence",
    "robot motion planning control navigation",
    "robot motion planning control sensor",
] * 5


@pytest.fixture
def lda():
    vectorizer = CountVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(CORPUS)
    config = ModelsConfig(n_topics=2)
    return build_model("lda", config, seed=1).fit(matrix, vectorizer.get_feature_names_out())


@pytest.fixture
def nmf():
    vectorizer = TfidfVectorizer(min_df=1)
    matrix = vectorizer.fit_transform(CORPUS)
    config = ModelsConfig(n_topics=2)
    return build_model("nmf", config, seed=1).fit(matrix, vectorizer.get_feature_names_out())


@pytest.fixture
def timeline() -> pd.DataFrame:
    rows = []
    for model in ("lda", "nmf"):
        for period in pd.date_range("2024-01-01", periods=3, freq="QS"):
            for topic in (1, 2):
                rows.append(
                    {
                        "model": model,
                        "period": period,
                        "topic": topic,
                        "share": 0.5,
                        "label": f"{topic}. word · other",
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- topic words


def test_topic_words_chart_has_one_bar_per_word(lda):
    figure = topic_words_chart(lda, 0, top_words=4)
    assert len(figure.data) == 1
    assert len(figure.data[0].y) == 4
    assert figure.data[0].orientation == "h"


def test_topic_words_chart_puts_the_heaviest_word_on_top(lda):
    """Plotly draws the first y entry at the bottom, so weights must ascend."""
    figure = topic_words_chart(lda, 0, top_words=5)
    assert list(figure.data[0].x) == sorted(figure.data[0].x)


def test_topic_words_chart_titles_carry_the_model_name(lda):
    assert "LDA" in figure_title(topic_words_chart(lda, 1))


def figure_title(figure) -> str:
    return figure.layout.title.text


# ------------------------------------------------------------------- metrics


def test_metrics_chart_has_one_series_per_model():
    metrics = {
        "lda": {"npmi": 0.11, "diversity": 0.81, "purity": 0.70},
        "nmf": {"npmi": 0.19, "diversity": 0.85, "purity": 0.81},
    }
    figure = metrics_comparison_chart(metrics)
    assert len(figure.data) == 2
    assert figure.layout.barmode == "group"
    assert figure.data[1].marker.color == MODEL_COLORS["nmf"]


def test_metrics_chart_tolerates_a_missing_metric():
    figure = metrics_comparison_chart({"lsa": {"npmi": 0.12}}, keys=("npmi", "purity"))
    assert list(figure.data[0].y) == [0.12, 0.0]


# ------------------------------------------------------------------ timeline


def test_timeline_chart_draws_one_area_per_topic(timeline):
    figure = timeline_chart(timeline, "lda")
    assert len(figure.data) == 2
    assert all(trace.stackgroup == "shares" for trace in figure.data)


def test_timeline_chart_can_skip_normalization(timeline):
    figure = timeline_chart(timeline, "nmf", normalize=False)
    assert all(trace.groupnorm is None for trace in figure.data)


def test_timeline_chart_uses_only_the_requested_model(timeline):
    figure = timeline_chart(timeline, "lda")
    assert len(figure.data[0].x) == 3


def test_timeline_chart_rejects_an_unknown_model(timeline):
    with pytest.raises(ValueError, match="no rows for model"):
        timeline_chart(timeline, "word2vec")


# ------------------------------------------------------------------- heatmap


def test_heatmap_is_shaped_by_the_two_topic_counts(lda, nmf):
    figure = similarity_heatmap(lda, nmf)
    assert len(figure.data[0].z) == lda.n_topics
    assert len(figure.data[0].z[0]) == nmf.n_topics
    assert len(figure.data[0].x) == nmf.n_topics


def test_heatmap_is_bounded_to_the_cosine_range(lda, nmf):
    figure = similarity_heatmap(lda, nmf)
    assert (figure.data[0].zmin, figure.data[0].zmax) == (0.0, 1.0)


# --------------------------------------------------------------------- sweep


def test_sweep_chart_draws_coherence_and_diversity_per_model():
    rows = [
        SweepRow("nmf", 5, 0.21, 0.98, 36.0),
        SweepRow("nmf", 8, 0.22, 0.89, 46.0),
        SweepRow("lda", 5, 0.11, 0.84, 325.0),
        SweepRow("lda", 8, 0.12, 0.88, 415.0),
    ]
    figure = sweep_chart(rows)
    assert len(figure.data) == 4
    assert any(trace.yaxis == "y2" for trace in figure.data)
    assert figure.layout.yaxis2.range == (0, 1)


# ------------------------------------------------------------- distributions


def test_label_distribution_stacks_one_series_per_category(lda):
    distribution = {0: {"cs.CL": 10, "cs.RO": 2}, 1: {"cs.RO": 9}}
    figure = label_distribution_chart(distribution, lda)

    assert {trace.name for trace in figure.data} == {"cs.CL", "cs.RO"}
    assert figure.layout.barmode == "stack"
    assert list(figure.data[0].y) == [10, 0]

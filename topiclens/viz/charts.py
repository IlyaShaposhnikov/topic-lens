"""Plotly figures for the reports, the README and the app.

Every chart is built here and used everywhere, so a figure in the README and the
same figure in the Streamlit app cannot drift apart. Functions return figures
rather than writing or displaying them: what to do with a figure is the caller's
business.

The colour of a model is fixed across all charts — a small thing that makes a
page of figures readable at a glance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from topiclens.evaluation.matching import topic_similarity

#: One colour per model, used consistently in every chart.
MODEL_COLORS = {
    "lda": "#4C78A8",
    "nmf": "#F58518",
    "lsa": "#54A24B",
}

#: Qualitative palette for topics within one model.
TOPIC_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#8C6D31",
    "#6B6ECF",
    "#B5CF6B",
    "#CE6DBD",
    "#AD494A",
]

_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Inter, Segoe UI, sans-serif", "size": 13},
    "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    "hovermode": "closest",
}


def _styled(figure: go.Figure, title: str, **layout: Any) -> go.Figure:
    figure.update_layout(title=title, **_LAYOUT, **layout)
    return figure


def topic_words_chart(model: Any, topic_index: int, top_words: int = 10) -> go.Figure:
    """Horizontal bars of the words that define one topic."""
    words = model.top_words(topic_index, top_words)
    figure = go.Figure(
        go.Bar(
            x=[entry.weight for entry in reversed(words)],
            y=[entry.word for entry in reversed(words)],
            orientation="h",
            marker_color=TOPIC_COLORS[topic_index % len(TOPIC_COLORS)],
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    return _styled(
        figure,
        f"{model.display_name} — {model.topic_label(topic_index)}",
        xaxis_title="weight",
        height=40 * len(words) + 120,
    )


def metrics_comparison_chart(
    metrics: dict[str, dict[str, float]],
    keys: Sequence[str] = ("npmi", "diversity", "purity"),
) -> go.Figure:
    """Grouped bars comparing models on metrics that share a scale.

    Only comparable metrics belong on one chart: perplexity in the thousands
    next to NPMI in the hundredths would say nothing.
    """
    figure = go.Figure()
    for model, scores in metrics.items():
        figure.add_bar(
            name=model.upper(),
            x=list(keys),
            y=[scores.get(key, 0.0) for key in keys],
            marker_color=MODEL_COLORS.get(model),
            hovertemplate="%{x}: %{y:.3f}<extra>" + model.upper() + "</extra>",
        )
    return _styled(figure, "Model comparison", barmode="group", yaxis_title="score")


def timeline_chart(timeline: pd.DataFrame, model: str, *, normalize: bool = True) -> go.Figure:
    """Stacked areas showing how topic shares move over time.

    Raises:
        ValueError: if the frame holds no rows for this model.
    """
    subset = timeline[timeline["model"] == model]
    if subset.empty:
        raise ValueError(f"the timeline contains no rows for model {model!r}")

    figure = go.Figure()
    for topic, block in subset.groupby("topic", sort=True):
        figure.add_scatter(
            x=block["period"],
            y=block["share"],
            name=str(block["label"].iloc[0]),
            mode="lines",
            stackgroup="shares" if normalize else None,
            groupnorm="fraction" if normalize else None,
            line={"width": 0.5, "color": TOPIC_COLORS[(topic - 1) % len(TOPIC_COLORS)]},
            hovertemplate="%{x|%Y-%m}: %{y:.1%}<extra>" + str(block["label"].iloc[0]) + "</extra>",
        )
    return _styled(
        figure,
        f"Topic shares over time — {model.upper()}",
        xaxis_title="period",
        yaxis_title="share of the corpus",
        yaxis_tickformat=".0%" if normalize else None,
        height=520,
    )


def similarity_heatmap(model_a: Any, model_b: Any) -> go.Figure:
    """Cosine similarity of every topic of one model to every topic of another.

    A bright diagonal after matching means the two models found the same
    structure; a diffuse field means they did not.
    """
    similarity, _ = topic_similarity(model_a, model_b)
    figure = go.Figure(
        go.Heatmap(
            z=similarity,
            x=model_b.topic_labels(2),
            y=model_a.topic_labels(2),
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar={"title": "cosine"},
        )
    )
    return _styled(
        figure,
        f"Topic similarity: {model_a.display_name} vs {model_b.display_name}",
        height=520,
        xaxis_title=model_b.display_name,
        yaxis_title=model_a.display_name,
    )


def sweep_chart(rows: Sequence[Any]) -> go.Figure:
    """Coherence and diversity against the number of topics.

    Both curves belong on one chart because they are read together: coherence
    that rises while diversity falls means the topics are collapsing, not
    improving.
    """
    figure = go.Figure()
    frame = pd.DataFrame([row.as_record() for row in rows])

    for model, block in frame.groupby("model", sort=True):
        color = MODEL_COLORS.get(str(model))
        figure.add_scatter(
            x=block["n_topics"],
            y=block["coherence"],
            name=f"{str(model).upper()} coherence",
            mode="lines+markers",
            line={"color": color},
        )
        figure.add_scatter(
            x=block["n_topics"],
            y=block["diversity"],
            name=f"{str(model).upper()} diversity",
            mode="lines",
            line={"color": color, "dash": "dot"},
            yaxis="y2",
        )

    return _styled(
        figure,
        "Choosing the number of topics",
        xaxis_title="topics",
        yaxis_title="NPMI coherence",
        yaxis2={"title": "diversity", "overlaying": "y", "side": "right", "range": [0, 1]},
        height=480,
    )


def label_distribution_chart(
    distribution: dict[int, dict[str, int]], model: Any, *, top_labels: int = 8
) -> go.Figure:
    """Stacked bars of the true categories inside each discovered topic.

    This is where alignment stops being a single number: it shows which topics
    are clean and which ones mix several categories.
    """
    labels = sorted({label for counts in distribution.values() for label in counts})[:top_labels]
    topics = sorted(distribution)

    figure = go.Figure()
    for position, label in enumerate(labels):
        figure.add_bar(
            name=label,
            x=[model.topic_label(topic, 2) for topic in topics],
            y=[distribution[topic].get(label, 0) for topic in topics],
            marker_color=TOPIC_COLORS[position % len(TOPIC_COLORS)],
        )
    return _styled(
        figure,
        f"Categories inside each topic — {model.display_name}",
        barmode="stack",
        yaxis_title="documents",
        height=520,
    )

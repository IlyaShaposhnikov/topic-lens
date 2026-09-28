"""Streamlit rendering. All computation lives in ``data.py``."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from topiclens.artifacts import ModelBundle
from topiclens.constants import ARXIV_CATEGORY_NAMES
from topiclens.ui.data import (
    ReportData,
    matches_for_pair,
    shares_table,
    similar_documents,
    topic_words_table,
)
from topiclens.viz.charts import (
    label_distribution_chart,
    metrics_comparison_chart,
    similarity_heatmap,
    sweep_chart,
    timeline_chart,
    topic_words_chart,
)

MODEL_NOTES = {
    "lda": "Generative model of word counts. Topics are probability distributions.",
    "nmf": "Non-negative factorization of TF-IDF weights. Parts add up, never subtract.",
    "lsa": "Truncated SVD. Finds directions of variance, which is not the same as topics.",
}


def render_topics_tab(bundle: ModelBundle, reports: ReportData) -> None:
    """Browse the topics of one model."""
    st.subheader("Topics")
    model_key = st.selectbox(
        "Model", sorted(bundle.models), format_func=str.upper, key="topics_model"
    )
    model = bundle.models[model_key]
    st.caption(MODEL_NOTES.get(model_key, ""))

    topic_index = (
        st.selectbox(
            "Topic",
            range(model.n_topics),
            format_func=lambda index: model.topic_label(index),
            key="topics_topic",
        )
        or 0
    )

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(topic_words_chart(model, topic_index), width="stretch")
    with right:
        st.dataframe(topic_words_table(model, topic_index), hide_index=True)

    with st.expander("All topics of this model"):
        st.dataframe(
            pd.DataFrame(bundle.topic_table()).query("model == @model_key"),
            hide_index=True,
            width="stretch",
        )
    if reports.metrics.get(model_key):
        scores = reports.metrics[model_key]
        columns = st.columns(4)
        for column, key in zip(
            columns, ("npmi", "diversity", "purity", "fit_seconds"), strict=False
        ):
            if key in scores:
                column.metric(key, f"{scores[key]:.3f}")


def render_comparison_tab(bundle: ModelBundle, reports: ReportData) -> None:
    """Metrics side by side, plus how the topic sets relate."""
    st.subheader("Model comparison")
    if not reports.metrics:
        st.info("No metrics report found. Run scripts/train.py to produce one.")
        return

    st.plotly_chart(metrics_comparison_chart(reports.metrics), width="stretch")
    st.dataframe(
        pd.DataFrame(reports.metrics).T.rename_axis("model").reset_index(),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Purity of about 0.20 is what random assignment gives on five balanced "
        "categories, which is the baseline these numbers should be read against."
    )

    st.markdown("#### Do the models agree?")
    keys = sorted(bundle.models)
    pair = st.selectbox(
        "Pair",
        [f"{first}-{second}" for i, first in enumerate(keys) for second in keys[i + 1 :]],
        key="comparison_pair",
    )
    first, second = str(pair).split("-")
    st.plotly_chart(
        similarity_heatmap(bundle.models[first], bundle.models[second]), width="stretch"
    )

    table = matches_for_pair(reports.matches, str(pair))
    if not table.empty:
        st.dataframe(
            table[["label_a", "label_b", "similarity", "shared_words"]],
            hide_index=True,
            width="stretch",
        )

    if reports.sweep is not None and not reports.sweep.empty:
        with st.expander("How the number of topics was chosen"):
            rows = [
                type("Row", (), {"as_record": lambda self, r=record: r})()
                for record in reports.sweep.to_dict("records")
            ]
            st.plotly_chart(sweep_chart(rows), width="stretch")


def render_timeline_tab(bundle: ModelBundle, reports: ReportData) -> None:
    """Topic shares over time."""
    st.subheader("Topics over time")
    if not reports.has_timeline:
        st.info("This corpus carries no dates, so there is no timeline to show.")
        return

    assert reports.timeline is not None
    model_key = st.selectbox(
        "Model", sorted(bundle.models), format_func=str.upper, key="timeline_model"
    )
    normalize = st.toggle("Normalize to 100%", value=True)
    st.plotly_chart(
        timeline_chart(reports.timeline, str(model_key), normalize=normalize),
        width="stretch",
    )
    st.caption(
        "Shares are relative: a band that narrows means the topic grew more slowly "
        "than the rest of the corpus, not that it shrank in absolute terms."
    )


def render_text_tab(
    bundle: ModelBundle,
    corpus: pd.DataFrame | None = None,
    corpus_vectors: Any | None = None,
) -> None:
    """Score a text the user provides, through the exact training pipeline."""
    st.subheader("Analyze a text")
    st.caption(
        "The text is cleaned, lemmatized and vectorized with the very objects the "
        "models were trained with, so this is what the models would have seen."
    )

    text = st.text_area(
        "Paste an abstract",
        height=160,
        placeholder="We propose a retrieval-augmented approach to question answering over...",
    )
    if not text.strip():
        return

    shares = bundle.transform_text(text)
    columns = st.columns(len(shares))
    for column, (model_key, values) in zip(columns, sorted(shares.items()), strict=True):
        with column:
            st.markdown(f"**{model_key.upper()}**")
            st.dataframe(
                shares_table(values, bundle.models[model_key], top_n=3),
                hide_index=True,
                column_config={
                    "share": st.column_config.ProgressColumn(
                        "share", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )

    if corpus is not None and corpus_vectors is not None:
        st.markdown("#### Closest papers in the corpus")
        st.caption("Nearest neighbours in the LSA space — what truncated SVD is actually good at.")
        table = similar_documents(bundle, corpus, corpus_vectors, text)
        table["category"] = table["label"].map(lambda key: ARXIV_CATEGORY_NAMES.get(key, key))
        st.dataframe(
            table[["title", "category", "date", "similarity"]],
            hide_index=True,
            width="stretch",
        )


def render_alignment_section(
    bundle: ModelBundle, distribution: dict[int, dict[str, int]], model_key: str
) -> None:
    """Which real categories ended up inside each discovered topic."""
    st.plotly_chart(
        label_distribution_chart(distribution, bundle.models[model_key]),
        width="stretch",
    )

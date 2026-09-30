"""The "bring your own data" tab.

The pipeline is not arXiv-specific, and this is where that stops being a claim
in the README: a user drops in a CSV, names the columns, and gets the same three
models with the same metrics.

Training happens in the browser session, so it is bounded. Anything beyond a few
thousand documents is sampled down, because LDA on a free hosting tier is slow
enough to look broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from topiclens.config import AppConfig
from topiclens.data.base import CorpusError
from topiclens.data.sample import stratified_sample
from topiclens.pipeline import TrainingRun, train
from topiclens.ui.data import corpus_from_csv, corpus_size_warning
from topiclens.viz.charts import metrics_comparison_chart, topic_words_chart

#: Above this, fitting in a web session takes long enough to look like a hang.
MAX_DOCUMENTS = 3000

NO_COLUMN = "— none —"


@dataclass
class PreparedUpload:
    """A user's file, turned into a corpus ready for training."""

    frame: pd.DataFrame
    original_size: int
    notes: list[str]

    @property
    def was_sampled(self) -> bool:
        return len(self.frame) < self.original_size


def prepare_upload(
    source: Any,
    *,
    text_column: str,
    label_column: str | None = None,
    date_column: str | None = None,
    id_column: str | None = None,
    max_documents: int = MAX_DOCUMENTS,
    min_chars: int = 100,
    seed: int = 0,
) -> PreparedUpload:
    """Read, validate and if necessary shrink an uploaded CSV.

    Raises:
        CorpusError: if a named column is missing or nothing survives filtering.
    """
    frame = corpus_from_csv(
        source,
        text_column=text_column,
        label_column=label_column,
        date_column=date_column,
        id_column=id_column,
        min_chars=min_chars,
    )
    original_size = len(frame)
    notes: list[str] = []

    if original_size > max_documents:
        frame = stratified_sample(frame, max_documents, seed=seed)
        notes.append(
            f"Sampled {len(frame)} of {original_size} documents so the models fit in a "
            "browser session; the category and date composition is preserved."
        )

    warning = corpus_size_warning(len(frame))
    if warning:
        notes.append(warning)
    return PreparedUpload(frame=frame, original_size=original_size, notes=notes)


def training_config(config: AppConfig, n_topics: int) -> AppConfig:
    """Config for a one-off fit: no held-out split, user's topic count."""
    tuned = config.model_copy(deep=True)
    tuned.models = tuned.models.model_copy(update={"n_topics": n_topics})
    # Perplexity on held-out data is a diagnostic for the arXiv run; here every
    # document is better spent on the fit itself.
    tuned.evaluation = tuned.evaluation.model_copy(update={"holdout_fraction": 0.0})
    return tuned


def _render_results(run: TrainingRun, frame: pd.DataFrame) -> None:
    st.success(f"Fitted three models on {len(frame)} documents.")
    st.plotly_chart(metrics_comparison_chart(run.metrics), width="stretch")

    if frame["label"].notna().any():
        st.caption(
            "Agreement metrics (NMI, ARI, purity) compare the discovered topics with the "
            "labels in your file."
        )

    model_key = st.selectbox(
        "Model", sorted(run.bundle.models), format_func=str.upper, key="upload_model"
    )
    model = run.bundle.models[model_key]
    topic_index = st.selectbox(
        "Topic",
        range(model.n_topics),
        format_func=lambda index: model.topic_label(index),
        key="upload_topic",
    )
    st.plotly_chart(topic_words_chart(model, topic_index or 0), width="stretch")

    with st.expander("All topics"):
        st.dataframe(pd.DataFrame(run.bundle.topic_table()), hide_index=True, width="stretch")


def render_upload_tab(config: AppConfig) -> None:
    """Fit the three models on a CSV the user provides."""
    st.subheader("Your own data")
    st.caption(
        "One row per document. Topic models learn from how words co-occur across "
        "documents, so a few hundred rows is the minimum and thousands work better."
    )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        if "upload_run" in st.session_state:
            _render_results(st.session_state["upload_run"], st.session_state["upload_frame"])
        return

    try:
        preview = pd.read_csv(uploaded, nrows=50)
    except Exception as error:  # noqa: BLE001 - anything malformed lands here
        st.error(f"Could not read that file: {error}")
        return
    uploaded.seek(0)

    columns = list(preview.columns)
    first, second, third = st.columns(3)
    text_column = first.selectbox("Text column", columns)
    label_column = second.selectbox("Label column (optional)", [NO_COLUMN, *columns])
    date_column = third.selectbox("Date column (optional)", [NO_COLUMN, *columns])
    n_topics = st.slider("Topics", min_value=4, max_value=12, value=8)

    st.caption(
        f"Up to {MAX_DOCUMENTS:,} documents are used; fitting all three models takes "
        "roughly half a minute, almost all of it LDA."
    )

    if not st.button("Fit three models", type="primary"):
        if "upload_run" in st.session_state:
            _render_results(st.session_state["upload_run"], st.session_state["upload_frame"])
        return

    try:
        prepared = prepare_upload(
            uploaded,
            text_column=str(text_column),
            label_column=None if label_column == NO_COLUMN else str(label_column),
            date_column=None if date_column == NO_COLUMN else str(date_column),
            seed=config.project.seed,
        )
    except CorpusError as error:
        st.error(str(error))
        return

    for note in prepared.notes:
        st.warning(note)

    with st.spinner("Fitting LDA, NMF and LSA…"):
        run = train(
            prepared.frame,
            training_config(config, n_topics),
            corpus_fingerprint={"source": "upload", "documents": len(prepared.frame)},
            show_progress=False,
        )

    st.session_state["upload_run"] = run
    st.session_state["upload_frame"] = prepared.frame
    _render_results(run, prepared.frame)

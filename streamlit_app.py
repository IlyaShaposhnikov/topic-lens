"""topic-lens — comparing LDA, NMF and LSA on one corpus.

Entry point for Streamlit, kept at the repository root because that is where
Streamlit Community Cloud looks for it. Caching is declared here rather than in
the modules, so the library code stays free of framework decorators:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import glob

import pandas as pd
import streamlit as st

from topiclens.artifacts import ArtifactError
from topiclens.config import load_config
from topiclens.constants import PROJECT_ROOT
from topiclens.ui.data import ReportData, document_vectors, load_bundle, load_reports
from topiclens.ui.views import (
    render_comparison_tab,
    render_text_tab,
    render_timeline_tab,
    render_topics_tab,
)

st.set_page_config(page_title="topic-lens", page_icon="🔭", layout="wide")


@st.cache_resource(show_spinner="Loading the trained models…")
def get_bundle(directory: str):
    return load_bundle(directory)


@st.cache_data(show_spinner="Reading the reports…")
def get_reports(directory: str) -> ReportData:
    return load_reports(directory)


@st.cache_data(show_spinner="Loading the corpus…")
def get_corpus(pattern: str) -> pd.DataFrame | None:
    matches = sorted(glob.glob(pattern))
    return pd.read_parquet(matches[0]) if matches else None


@st.cache_data(show_spinner="Embedding the corpus for similarity search…")
def get_corpus_vectors(directory: str, pattern: str):
    corpus = get_corpus(pattern)
    if corpus is None:
        return None
    return document_vectors(get_bundle(directory), corpus["text"].tolist())


def main() -> None:
    st.title("topic-lens")
    st.caption(
        "Three topic models — LDA, NMF and LSA — fitted on the same corpus of arXiv "
        "abstracts and compared on coherence, diversity and agreement with the real "
        "categories."
    )

    default_artifacts = str(load_config().project.artifacts_path)
    with st.sidebar:
        st.header("Artifacts")
        directory = st.text_input("Directory", value=default_artifacts)
        st.caption("Produced by `python scripts/train.py`.")

    try:
        bundle = get_bundle(directory)
    except ArtifactError as error:
        st.error(str(error))
        st.stop()

    reports = get_reports(directory)
    corpus_pattern = str(PROJECT_ROOT / "data" / "cache" / "arxiv-*.parquet")
    corpus = get_corpus(corpus_pattern)

    with st.sidebar:
        st.header("This run")
        st.metric("Documents", f"{reports.summary.get('documents', 0):,}")
        st.metric("Topics per model", next(iter(bundle.models.values())).n_topics)
        st.caption(f"Trained {bundle.created_at}")
        fingerprint = bundle.corpus_fingerprint
        if fingerprint.get("categories"):
            st.caption("Categories: " + ", ".join(fingerprint["categories"]))
        if corpus is None:
            st.caption("No local corpus found — similarity search is unavailable.")

    topics, comparison, timeline, text = st.tabs(
        ["Topics", "Comparison", "Over time", "Analyze a text"]
    )
    with topics:
        render_topics_tab(bundle, reports)
    with comparison:
        render_comparison_tab(bundle, reports)
    with timeline:
        render_timeline_tab(bundle, reports)
    with text:
        vectors = get_corpus_vectors(directory, corpus_pattern) if corpus is not None else None
        render_text_tab(bundle, corpus, vectors)


if __name__ == "__main__":
    main()

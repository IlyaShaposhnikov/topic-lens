"""topic-lens — comparing LDA, NMF and LSA on one corpus.

Entry point for Streamlit, kept at the repository root because that is where
Streamlit Community Cloud looks for it. Caching is declared here rather than in
the modules, so the library code stays free of framework decorators:

    streamlit run streamlit_app.py

With no locally trained artifacts — which is the case on a fresh clone and on
the hosted demo — the app falls back to the small bundle committed under
artifacts/demo, so it always has something to show.
"""

from __future__ import annotations

import glob

import pandas as pd
import streamlit as st

from topiclens.artifacts import BUNDLE_FILENAME, ArtifactError
from topiclens.config import load_config
from topiclens.constants import PROJECT_ROOT, resolve_path
from topiclens.ui.data import ReportData, document_vectors, load_bundle, load_reports
from topiclens.ui.views import (
    render_comparison_tab,
    render_text_tab,
    render_timeline_tab,
    render_topics_tab,
)

DEMO_ARTIFACTS = PROJECT_ROOT / "artifacts" / "demo"
DEMO_CORPUS = PROJECT_ROOT / "data" / "demo" / "arxiv-demo.csv.gz"
FULL_CORPUS_PATTERN = str(PROJECT_ROOT / "data" / "cache" / "arxiv-*.parquet")

st.set_page_config(page_title="topic-lens", page_icon="🔭", layout="wide")


def default_artifacts_directory() -> tuple[str, bool]:
    """Prefer locally trained artifacts, fall back to the committed demo."""
    configured = load_config().project.artifacts_path
    if (configured / BUNDLE_FILENAME).is_file():
        return str(configured), False
    return str(DEMO_ARTIFACTS), True


@st.cache_resource(show_spinner="Loading the trained models…")
def get_bundle(directory: str):
    return load_bundle(directory)


@st.cache_data(show_spinner="Reading the reports…")
def get_reports(directory: str) -> ReportData:
    return load_reports(directory)


@st.cache_data(show_spinner="Loading the corpus…")
def get_corpus(prefer_demo: bool) -> pd.DataFrame | None:
    """The corpus behind the loaded bundle, for similarity search."""
    if not prefer_demo:
        matches = sorted(glob.glob(FULL_CORPUS_PATTERN))
        if matches:
            return pd.read_parquet(matches[0])
    if DEMO_CORPUS.is_file():
        return pd.read_csv(DEMO_CORPUS, parse_dates=["date"])
    return None


@st.cache_data(show_spinner="Embedding the corpus for similarity search…")
def get_corpus_vectors(directory: str, prefer_demo: bool):
    corpus = get_corpus(prefer_demo)
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

    default_directory, is_demo = default_artifacts_directory()
    with st.sidebar:
        st.header("Artifacts")
        directory = st.text_input("Directory", value=default_directory)
        if is_demo and resolve_path(directory) == DEMO_ARTIFACTS:
            st.info(
                "Showing the demo bundle: a stratified sample of the full corpus, "
                "trained on fewer documents, so the metrics are lower than the ones "
                "in the README. Run `python scripts/train.py` for the full run."
            )
        else:
            st.caption("Produced by `python scripts/train.py`.")

    try:
        bundle = get_bundle(directory)
    except ArtifactError as error:
        st.error(str(error))
        st.stop()

    reports = get_reports(directory)
    prefer_demo = resolve_path(directory) == DEMO_ARTIFACTS
    corpus = get_corpus(prefer_demo)

    with st.sidebar:
        st.header("This run")
        st.metric("Documents", f"{reports.summary.get('documents', 0):,}")
        st.metric("Topics per model", next(iter(bundle.models.values())).n_topics)
        st.caption(f"Trained {bundle.created_at}")
        categories = bundle.corpus_fingerprint.get("categories")
        if categories:
            st.caption("Categories: " + ", ".join(categories))
        if corpus is None:
            st.caption("No corpus file found — similarity search is unavailable.")

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
        vectors = get_corpus_vectors(directory, prefer_demo) if corpus is not None else None
        render_text_tab(bundle, corpus, vectors)


if __name__ == "__main__":
    main()

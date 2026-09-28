"""Tests for the app's data layer. Streamlit is never imported here."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from topiclens.artifacts import ArtifactError
from topiclens.config import AppConfig
from topiclens.data.base import CorpusError
from topiclens.pipeline import train
from topiclens.reporting import write_report
from topiclens.ui.data import (
    MIN_CORPUS_DOCUMENTS,
    corpus_from_csv,
    corpus_size_warning,
    document_vectors,
    load_bundle,
    load_reports,
    matches_for_pair,
    nearest_documents,
    shares_table,
    similar_documents,
    topic_words_table,
)

LANGUAGE = "language model translation corpus text sentence word embedding"
ROBOTICS = "robot motion planning control navigation sensor trajectory autonomous"


def make_corpus(n_each: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(n_each):
        for prefix, text, label in (("cl", LANGUAGE, "cs.CL"), ("ro", ROBOTICS, "cs.RO")):
            rows.append(
                {
                    "doc_id": f"{prefix}-{index}",
                    "text": text,
                    "title": f"{prefix} paper {index}",
                    "label": label,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index * 10),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "project": {"seed": 3, "artifacts_dir": str(tmp_path / "artifacts")},
            "data": {
                "cache_dir": str(tmp_path / "cache"),
                "arxiv": {
                    "categories": ["cs.CL"],
                    "date_from": "2024-01",
                    "date_to": "2024-12",
                },
            },
            "preprocessing": {
                "lemmatize": False,
                "vectorizer": {
                    "count": {"min_df": 1, "max_df": 1.0, "ngram_range": [1, 1]},
                    "tfidf": {"min_df": 1, "max_df": 1.0, "ngram_range": [1, 1]},
                },
            },
            "models": {"n_topics": 2, "lda": {"max_iter": 5}, "nmf": {"max_iter": 50}},
            "evaluation": {"top_words": 5, "holdout_fraction": 0.2},
        }
    )


@pytest.fixture
def artifacts(config, tmp_path):
    """A trained bundle and its reports on disk."""
    frame = make_corpus()
    run = train(frame, config, show_progress=False)
    directory = tmp_path / "artifacts"
    write_report(run, frame, directory, top_words=5)
    run.bundle.save(directory)
    return directory, run


# ------------------------------------------------------------------- loading


def test_load_bundle_reads_a_trained_directory(artifacts):
    directory, run = artifacts
    bundle = load_bundle(directory)
    assert set(bundle.models) == set(run.bundle.models)


def test_load_bundle_explains_how_to_train(tmp_path):
    with pytest.raises(ArtifactError, match="scripts/train.py"):
        load_bundle(tmp_path / "empty")


def test_load_reports_reads_every_file(artifacts):
    directory, _ = artifacts
    reports = load_reports(directory)

    assert reports.metrics
    assert reports.topics is not None
    assert reports.has_timeline
    assert reports.matches is not None
    assert reports.sweep is None  # not produced unless --sweep was passed


def test_load_reports_tolerates_an_empty_directory(tmp_path):
    reports = load_reports(tmp_path)
    assert reports.summary == {}
    assert reports.metrics == {}
    assert not reports.has_timeline


def test_reports_expose_the_metrics_of_the_run(artifacts):
    directory, _ = artifacts
    summary = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert load_reports(directory).metrics == summary["metrics"]


# ---------------------------------------------------------- corpus guardrails


def test_small_corpora_are_flagged():
    message = corpus_size_warning(42)
    assert message is not None
    assert str(MIN_CORPUS_DOCUMENTS) in message


def test_large_enough_corpora_pass_silently():
    assert corpus_size_warning(MIN_CORPUS_DOCUMENTS) is None


def test_corpus_from_csv_maps_the_chosen_columns():
    buffer = io.StringIO("body,section,when\n" + f'"{LANGUAGE}",tech,2024-02-01\n')
    frame = corpus_from_csv(
        buffer, text_column="body", label_column="section", date_column="when", min_chars=10
    )
    assert frame.loc[0, "label"] == "tech"
    assert frame.loc[0, "date"] == pd.Timestamp("2024-02-01")
    assert frame.loc[0, "doc_id"] == "row-0"


def test_corpus_from_csv_rejects_a_missing_text_column():
    buffer = io.StringIO("body\nsome text here\n")
    with pytest.raises(CorpusError):
        corpus_from_csv(buffer, text_column="text")


# --------------------------------------------------------------------- tables


def test_shares_table_is_sorted_and_labelled(artifacts):
    _, run = artifacts
    table = shares_table(np.array([0.2, 0.8]), run.bundle.models["nmf"])

    assert table.loc[0, "topic"] == 2
    assert table.loc[0, "share"] == 0.8
    assert table.loc[0, "label"].startswith("2.")


def test_shares_table_can_be_truncated(artifacts):
    _, run = artifacts
    assert len(shares_table(np.array([0.5, 0.5]), run.bundle.models["lda"], top_n=1)) == 1


def test_topic_words_table_has_words_and_weights(artifacts):
    _, run = artifacts
    table = topic_words_table(run.bundle.models["nmf"], 0, top_words=4)
    assert list(table.columns) == ["word", "weight"]
    assert len(table) == 4


def test_matches_for_pair_filters_and_sorts(artifacts):
    directory, _ = artifacts
    reports = load_reports(directory)
    table = matches_for_pair(reports.matches, "lda-nmf")

    assert set(table["pair"]) == {"lda-nmf"}
    assert table["similarity"].is_monotonic_decreasing


def test_matches_for_pair_survives_a_missing_report():
    assert matches_for_pair(None, "lda-nmf").empty


# ----------------------------------------------------------------- similarity


def test_document_vectors_have_one_row_per_document(artifacts):
    _, run = artifacts
    vectors = document_vectors(run.bundle, [LANGUAGE, ROBOTICS])
    assert vectors.shape == (2, 2)


def test_nearest_documents_ranks_by_cosine():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    order, scores = nearest_documents(corpus, np.array([1.0, 0.0]), top_n=2)

    assert order[0] == 0
    assert scores[0] == pytest.approx(1.0)
    assert scores[0] >= scores[1]


def test_nearest_documents_handles_a_zero_query():
    order, scores = nearest_documents(np.eye(3), np.zeros(3), top_n=2)
    assert len(order) == 2
    assert np.isfinite(scores).all()


def test_similar_documents_finds_the_matching_theme(artifacts):
    _, run = artifacts
    frame = make_corpus()
    vectors = document_vectors(run.bundle, frame["text"].tolist())
    table = similar_documents(run.bundle, frame, vectors, "a robot planning its motion", top_n=3)

    assert len(table) == 3
    assert (table["label"] == "cs.RO").all()
    assert table["similarity"].is_monotonic_decreasing

"""Tests for the report writers."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from topiclens.config import AppConfig
from topiclens.evaluation.selection import SweepRow
from topiclens.pipeline import train
from topiclens.reporting import matches_frame, timeline_frame, write_report, write_sweep

LANGUAGE = "language model translation corpus text sentence word embedding"
ROBOTICS = "robot motion planning control navigation sensor trajectory autonomous"


def make_corpus(n_each: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(n_each):
        for prefix, text, label in (
            ("cl", LANGUAGE, "cs.CL"),
            ("ro", ROBOTICS, "cs.RO"),
        ):
            rows.append(
                {
                    "doc_id": f"{prefix}-{index}",
                    "text": text,
                    "title": f"{prefix} paper",
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
def run(config):
    return train(make_corpus(), config, show_progress=False)


# ------------------------------------------------------------------- timeline


def test_timeline_frame_is_long_and_labelled(run):
    table = timeline_frame(make_corpus(), run.document_topics, run.bundle.models, freq="Q")
    assert set(table.columns) == {"model", "period", "topic", "share", "label"}
    assert set(table["model"]) == {"lda", "nmf", "lsa"}
    assert table["topic"].min() == 1
    assert table["label"].str.startswith("1.").any()


def test_timeline_shares_sum_to_one_per_model_and_period(run):
    table = timeline_frame(make_corpus(), run.document_topics, run.bundle.models, freq="Q")
    totals = table.groupby(["model", "period"])["share"].sum()
    assert totals.between(0.999, 1.001).all()


# -------------------------------------------------------------------- matches


def test_matches_frame_covers_every_model_pair(run):
    table = matches_frame(run.bundle.models, top_words=5)
    assert set(table["pair"]) == {"lda-lsa", "lda-nmf", "lsa-nmf"}
    assert (table["similarity"] <= 1.0).all()
    assert "label_a" in table.columns


# --------------------------------------------------------------------- report


def test_report_writes_every_expected_file(run, tmp_path):
    written = write_report(run, make_corpus(), tmp_path / "out", top_words=5)
    assert set(written) == {"metrics_json", "metrics_csv", "topics", "matches", "timeline"}
    assert all(path.is_file() for path in written.values())


def test_metrics_json_carries_provenance(run, tmp_path):
    written = write_report(run, make_corpus(), tmp_path / "out", top_words=5)
    payload = json.loads(written["metrics_json"].read_text(encoding="utf-8"))

    assert payload["documents"] == 60
    assert payload["train_size"] + payload["holdout_size"] == 60
    assert "nmf" in payload["metrics"]
    assert payload["config"]["models"]["n_topics"] == 2
    assert "created_at" in payload


def test_topics_csv_lists_every_topic(run, tmp_path):
    written = write_report(run, make_corpus(), tmp_path / "out", top_words=5)
    table = pd.read_csv(written["topics"])
    assert len(table) == 6
    assert set(table["model"]) == {"lda", "nmf", "lsa"}


def test_timeline_is_skipped_for_an_undated_corpus(run, tmp_path):
    frame = make_corpus()
    frame["date"] = pd.NaT
    written = write_report(run, frame, tmp_path / "out", top_words=5)
    assert "timeline" not in written
    assert (tmp_path / "out" / "metrics.json").is_file()


def test_report_directory_is_created_if_absent(run, tmp_path):
    target = tmp_path / "deep" / "nested" / "out"
    write_report(run, make_corpus(), target, top_words=5)
    assert target.is_dir()


# ---------------------------------------------------------------------- sweep


def test_sweep_csv_has_a_row_per_fit(tmp_path):
    rows = [SweepRow("nmf", 5, 0.21, 0.98, 36.3), SweepRow("nmf", 8, 0.22, 0.89, 45.9)]
    path = write_sweep(rows, tmp_path / "out")
    table = pd.read_csv(path)

    assert len(table) == 2
    assert list(table.columns) == ["model", "n_topics", "coherence", "diversity", "seconds"]

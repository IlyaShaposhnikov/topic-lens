"""Tests for preparing an uploaded CSV before training."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from topiclens.config import AppConfig
from topiclens.data.base import CorpusError
from topiclens.ui.data import MIN_CORPUS_DOCUMENTS
from topiclens.ui.upload import prepare_upload, training_config

LONG_TEXT = (
    "a sufficiently long document about latent topics and their discovery, "
    "written so that it survives the minimum-length filter the uploader applies "
    "to anything a user drops into the app "
)


def make_csv(rows: int, *, with_label: bool = True) -> io.StringIO:
    frame = pd.DataFrame(
        {
            "body": [LONG_TEXT + f"number {index}" for index in range(rows)],
            "section": ["a" if index % 2 else "b" for index in range(rows)],
            "when": ["2024-01-01" if index % 2 else "2023-01-01" for index in range(rows)],
        }
    )
    if not with_label:
        frame = frame.drop(columns=["section"])
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def test_upload_is_read_with_the_chosen_columns():
    prepared = prepare_upload(
        make_csv(10), text_column="body", label_column="section", date_column="when"
    )
    assert len(prepared.frame) == 10
    assert set(prepared.frame["label"]) == {"a", "b"}
    assert prepared.frame["date"].notna().all()


def test_optional_columns_may_be_omitted():
    prepared = prepare_upload(make_csv(10, with_label=False), text_column="body")
    assert prepared.frame["label"].isna().all()


def test_a_missing_column_is_reported():
    with pytest.raises(CorpusError, match="not in"):
        prepare_upload(make_csv(5), text_column="text")


def test_large_uploads_are_sampled_down():
    prepared = prepare_upload(
        make_csv(500), text_column="body", label_column="section", max_documents=100
    )
    assert prepared.was_sampled
    assert len(prepared.frame) <= 100
    assert prepared.original_size == 500
    assert any("Sampled" in note for note in prepared.notes)


def test_small_uploads_are_kept_whole_but_flagged():
    prepared = prepare_upload(make_csv(20), text_column="body")
    assert not prepared.was_sampled
    assert any(str(MIN_CORPUS_DOCUMENTS) in note for note in prepared.notes)


def test_a_large_enough_upload_raises_no_warning():
    prepared = prepare_upload(make_csv(MIN_CORPUS_DOCUMENTS + 5), text_column="body")
    assert prepared.notes == []


def test_training_config_drops_the_holdout_and_takes_the_topic_count():
    config = AppConfig.model_validate(
        {
            "data": {
                "arxiv": {
                    "categories": ["cs.CL"],
                    "date_from": "2024-01",
                    "date_to": "2024-12",
                }
            },
            "evaluation": {"holdout_fraction": 0.2},
        }
    )
    tuned = training_config(config, 6)

    assert tuned.models.n_topics == 6
    assert tuned.evaluation.holdout_fraction == 0.0
    assert config.evaluation.holdout_fraction == 0.2  # the original is untouched

"""Tests for corpus cleaning, caching and the CSV source."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from topiclens.config import AppConfig
from topiclens.data.arxiv import ArxivPaper
from topiclens.data.base import CorpusError, corpus_summary, finalize_corpus
from topiclens.data.corpus import ArxivSource, cache_key, cache_paths, load_corpus
from topiclens.data.csv_source import CsvSource

LONG_TEXT = "a sufficiently long abstract about latent topics " * 4


def raw_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


@pytest.fixture
def config(tmp_path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "data": {
                "cache_dir": str(tmp_path / "cache"),
                "raw_dir": str(tmp_path / "raw"),
                "min_abstract_chars": 20,
                "arxiv": {
                    "categories": ["cs.CL"],
                    "date_from": "2024-01",
                    "date_to": "2024-12",
                },
            }
        }
    )


class StubSource:
    """Yields a fixed frame and counts how often it was asked for it."""

    name = "stub"

    def __init__(self, frame: pd.DataFrame, fingerprint: dict[str, Any] | None = None) -> None:
        self.frame = frame
        self._fingerprint = fingerprint or {"kind": "stub"}
        self.calls = 0

    def fingerprint(self) -> dict[str, Any]:
        return self._fingerprint

    def load(self) -> pd.DataFrame:
        self.calls += 1
        return self.frame.copy()


# ------------------------------------------------------------------ cleaning


def test_finalize_requires_text_column():
    with pytest.raises(CorpusError):
        finalize_corpus(raw_frame([{"doc_id": "1"}]))


def test_finalize_collapses_whitespace_and_orders_columns():
    frame = finalize_corpus(
        raw_frame([{"doc_id": "1", "text": "  spaced\n  out   text  ", "label": " cs.CL "}]),
    )
    assert frame.loc[0, "text"] == "spaced out text"
    assert frame.loc[0, "label"] == "cs.CL"
    assert list(frame.columns) == ["doc_id", "text", "title", "label", "date"]


def test_finalize_drops_short_documents():
    frame = finalize_corpus(
        raw_frame([{"doc_id": "1", "text": "too short"}, {"doc_id": "2", "text": LONG_TEXT}]),
        min_chars=50,
    )
    assert frame["doc_id"].tolist() == ["2"]


def test_finalize_drops_duplicate_ids():
    frame = finalize_corpus(
        raw_frame(
            [
                {"doc_id": "1", "text": LONG_TEXT},
                {"doc_id": "1", "text": LONG_TEXT + " variant"},
            ]
        )
    )
    assert len(frame) == 1


def test_finalize_drops_documents_with_identical_text():
    frame = finalize_corpus(
        raw_frame(
            [
                {"doc_id": "1", "text": LONG_TEXT},
                {"doc_id": "2", "text": LONG_TEXT.upper()},
            ]
        )
    )
    assert frame["doc_id"].tolist() == ["1"]


def test_finalize_keeps_identical_text_when_deduplication_is_off():
    frame = finalize_corpus(
        raw_frame([{"doc_id": "1", "text": LONG_TEXT}, {"doc_id": "2", "text": LONG_TEXT}]),
        drop_duplicates=False,
    )
    assert len(frame) == 2


def test_finalize_parses_dates_and_tolerates_garbage():
    frame = finalize_corpus(
        raw_frame(
            [
                {"doc_id": "1", "text": LONG_TEXT, "date": "2024-03-01"},
                {"doc_id": "2", "text": LONG_TEXT + " other", "date": "not a date"},
            ]
        )
    )
    assert frame.loc[0, "date"] == pd.Timestamp("2024-03-01")
    assert pd.isna(frame.loc[1, "date"])


def test_finalize_rejects_an_empty_result():
    with pytest.raises(CorpusError):
        finalize_corpus(raw_frame([{"doc_id": "1", "text": "short"}]), min_chars=500)


def test_corpus_summary_reports_labels_and_years():
    frame = finalize_corpus(
        raw_frame(
            [
                {"doc_id": "1", "text": LONG_TEXT, "label": "cs.CL", "date": "2023-05-01"},
                {"doc_id": "2", "text": LONG_TEXT + " two", "label": "cs.CV", "date": "2024-05-01"},
            ]
        )
    )
    summary = corpus_summary(frame)
    assert summary["documents"] == 2
    assert summary["by_label"] == {"cs.CL": 1, "cs.CV": 1}
    assert summary["by_year"] == {2023: 1, 2024: 1}
    assert summary["date_range"] == "2023-05 … 2024-05"


# --------------------------------------------------------------------- cache


def test_load_corpus_caches_and_reuses(config):
    source = StubSource(raw_frame([{"doc_id": "1", "text": LONG_TEXT, "date": "2024-01-01"}]))

    first = load_corpus(config, source=source)
    second = load_corpus(config, source=source)

    assert source.calls == 1
    assert len(first) == len(second) == 1

    parquet_path, metadata_path = cache_paths(config, source)
    assert parquet_path.is_file()
    assert metadata_path.is_file()
    assert "created_at" in metadata_path.read_text(encoding="utf-8")


def test_refresh_forces_a_second_fetch(config):
    source = StubSource(raw_frame([{"doc_id": "1", "text": LONG_TEXT}]))
    load_corpus(config, source=source)
    load_corpus(config, source=source, refresh=True)
    assert source.calls == 2


def test_disabling_the_cache_writes_nothing(config):
    source = StubSource(raw_frame([{"doc_id": "1", "text": LONG_TEXT}]))
    load_corpus(config, source=source, use_cache=False)
    parquet_path, _ = cache_paths(config, source)
    assert not parquet_path.exists()


def test_cache_key_changes_with_the_fingerprint(config):
    frame = raw_frame([{"doc_id": "1", "text": LONG_TEXT}])
    first = cache_key(config, StubSource(frame, {"categories": ["cs.CL"]}))
    second = cache_key(config, StubSource(frame, {"categories": ["cs.CV"]}))
    assert first != second


def test_cache_key_changes_with_filter_settings(config):
    source = StubSource(raw_frame([{"doc_id": "1", "text": LONG_TEXT}]))
    before = cache_key(config, source)
    config.data.min_abstract_chars = 999
    assert cache_key(config, source) != before


# ---------------------------------------------------------------- csv source


def write_csv(path, rows: list[dict[str, Any]]) -> str:
    pd.DataFrame.from_records(rows).to_csv(path, index=False)
    return str(path)


def test_csv_source_maps_configured_columns(config, tmp_path):
    path = write_csv(
        tmp_path / "docs.csv",
        [{"body": LONG_TEXT, "section": "tech", "ts": "2024-02-01", "key": "a1"}],
    )
    config.data.csv = config.data.csv.model_copy(
        update={
            "path": path,
            "text_column": "body",
            "label_column": "section",
            "date_column": "ts",
            "id_column": "key",
        }
    )
    frame = finalize_corpus(CsvSource(config).load(), min_chars=20)

    assert frame.loc[0, "doc_id"] == "a1"
    assert frame.loc[0, "label"] == "tech"
    assert frame.loc[0, "date"] == pd.Timestamp("2024-02-01")


def test_csv_source_generates_ids_when_none_configured(config, tmp_path):
    path = write_csv(tmp_path / "docs.csv", [{"text": LONG_TEXT}, {"text": LONG_TEXT + " two"}])
    config.data.csv = config.data.csv.model_copy(update={"path": path})
    frame = finalize_corpus(CsvSource(config).load(), min_chars=20)
    assert frame["doc_id"].tolist() == ["row-0", "row-1"]


def test_csv_source_reports_a_missing_column(config, tmp_path):
    path = write_csv(tmp_path / "docs.csv", [{"body": LONG_TEXT}])
    config.data.csv = config.data.csv.model_copy(update={"path": path, "text_column": "text"})
    with pytest.raises(CorpusError, match="not in"):
        CsvSource(config).load()


def test_csv_source_reports_a_missing_file(config, tmp_path):
    config.data.csv = config.data.csv.model_copy(update={"path": str(tmp_path / "nope.csv")})
    with pytest.raises(CorpusError, match="not found"):
        CsvSource(config).load()


def test_csv_source_without_a_path_is_rejected(config):
    with pytest.raises(CorpusError):
        CsvSource(config)


# --------------------------------------------------------------- checkpoints


class FakeClient:
    """Stands in for ArxivClient, recording which slices were asked for."""

    def __init__(self, categories: list[str], periods: list[str]) -> None:
        self._categories = categories
        self._periods = periods
        self.requested: list[tuple[str, str]] = []

    def slices(self):
        for category in self._categories:
            for period in self._periods:
                yield category, period

    def fetch_slice(self, category: str, period: str):
        self.requested.append((category, period))
        return [
            ArxivPaper(
                paper_id=f"{category}-{period}",
                title=f"Title for {category} {period}",
                abstract=LONG_TEXT,
                published=pd.Timestamp(f"{period}-01").date(),
                primary_category=category,
                categories=(category,),
            )
        ]


def arxiv_source(config: AppConfig, **kwargs) -> ArxivSource:
    client = FakeClient(["cs.CL"], ["2024-01", "2024-02"])
    return ArxivSource(config, client=client, show_progress=False, **kwargs)


def test_checkpoint_is_written_per_slice(config):
    source = arxiv_source(config)
    source.load()

    lines = source.checkpoint_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_second_run_reuses_checkpointed_slices(config):
    first = arxiv_source(config)
    first.load()

    second = arxiv_source(config)
    frame = second.load()

    assert second.client.requested == []
    assert len(frame) == 2


def test_resume_disabled_refetches_everything(config):
    arxiv_source(config).load()
    second = arxiv_source(config, resume=False)
    second.load()
    assert len(second.client.requested) == 2


def test_partial_slice_is_refetched_after_a_corrupt_line(config):
    source = arxiv_source(config)
    source.load()
    with source.checkpoint_path.open("a", encoding="utf-8") as stream:
        stream.write('{"category": "cs.CL", "period":')

    second = arxiv_source(config)
    frame = second.load()
    assert second.client.requested == []
    assert len(frame) == 2


def test_checkpoint_path_follows_the_fingerprint(config):
    first = arxiv_source(config).checkpoint_path
    config.data.arxiv = config.data.arxiv.model_copy(update={"max_per_slice": 7})
    assert arxiv_source(config).checkpoint_path != first

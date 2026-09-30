"""Tests for the scheduled-refresh arithmetic."""

from __future__ import annotations

import pandas as pd
import pytest

from topiclens.data.refresh import (
    MIN_QUOTA,
    last_complete_month,
    latest_period,
    merge_corpora,
    missing_periods,
    slice_quota,
)


def make_corpus(months: list[str], per_cell: int = 5) -> pd.DataFrame:
    rows = []
    for month in months:
        for label in ("cs.CL", "cs.RO"):
            for index in range(per_cell):
                rows.append(
                    {
                        "doc_id": f"{label}-{month}-{index}",
                        "text": "a document about latent topics",
                        "title": "paper",
                        "label": label,
                        "date": pd.Timestamp(f"{month}-05"),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- periods


def test_the_current_month_is_never_fetched():
    """A partial month would read as a dip on the timeline."""
    assert last_complete_month(pd.Timestamp("2026-09-15")) == "2026-08"


def test_last_complete_month_crosses_the_year_boundary():
    assert last_complete_month(pd.Timestamp("2026-01-03")) == "2025-12"


def test_latest_period_finds_the_newest_month():
    assert latest_period(make_corpus(["2026-04", "2026-06", "2026-05"])) == "2026-06"


def test_latest_period_of_an_undated_corpus_is_none():
    frame = make_corpus(["2026-04"])
    frame["date"] = pd.NaT
    assert latest_period(frame) is None


def test_missing_periods_lists_the_gap():
    frame = make_corpus(["2026-06"])
    assert missing_periods(frame, until="2026-09") == ["2026-07", "2026-08", "2026-09"]


def test_an_up_to_date_corpus_needs_nothing():
    assert missing_periods(make_corpus(["2026-09"]), until="2026-09") == []


def test_a_corpus_ahead_of_the_boundary_needs_nothing():
    assert missing_periods(make_corpus(["2026-12"]), until="2026-09") == []


def test_missing_periods_of_an_undated_corpus_is_empty():
    frame = make_corpus(["2026-04"])
    frame["date"] = pd.NaT
    assert missing_periods(frame, until="2026-09") == []


# --------------------------------------------------------------------- quota


def test_quota_matches_the_existing_density():
    assert slice_quota(make_corpus(["2026-05", "2026-06"], per_cell=7)) == 7


def test_quota_never_drops_below_the_floor():
    assert slice_quota(make_corpus(["2026-05"], per_cell=1)) == MIN_QUOTA


# --------------------------------------------------------------------- merge


def test_merge_appends_new_documents_in_date_order():
    existing = make_corpus(["2026-05"])
    fresh = make_corpus(["2026-06"])
    merged = merge_corpora(existing, fresh)

    assert len(merged) == len(existing) + len(fresh)
    assert merged["date"].is_monotonic_increasing


def test_merge_drops_papers_already_present():
    existing = make_corpus(["2026-05"])
    merged = merge_corpora(existing, existing.copy())
    assert len(merged) == len(existing)


def test_merge_keeps_the_existing_copy_of_a_duplicate():
    existing = make_corpus(["2026-05"])
    altered = existing.copy()
    altered["text"] = "rewritten text"
    merged = merge_corpora(existing, altered)

    assert len(merged) == len(existing)
    assert "rewritten" not in " ".join(merged["text"])


@pytest.mark.parametrize("months", [["2026-05"], ["2026-05", "2026-06"]])
def test_merge_preserves_the_corpus_schema(months):
    merged = merge_corpora(make_corpus(months), make_corpus(["2026-07"]))
    assert {"doc_id", "text", "title", "label", "date"} <= set(merged.columns)

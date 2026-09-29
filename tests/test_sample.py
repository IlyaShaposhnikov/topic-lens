"""Tests for the stratified sampler behind the demo corpus."""

from __future__ import annotations

import pandas as pd
import pytest

from topiclens.data.sample import stratified_sample


def make_corpus() -> pd.DataFrame:
    """600 documents: three labels over two years, deliberately unbalanced."""
    rows = []
    for label, count in (("cs.CL", 300), ("cs.RO", 200), ("cs.DB", 100)):
        for index in range(count):
            rows.append(
                {
                    "doc_id": f"{label}-{index}",
                    "text": f"document {index} about {label}",
                    "title": f"paper {index}",
                    "label": label,
                    "date": pd.Timestamp("2023-01-01" if index % 2 else "2024-01-01"),
                }
            )
    return pd.DataFrame(rows)


def test_sample_has_about_the_requested_size():
    sample = stratified_sample(make_corpus(), 120, seed=1)
    assert 100 <= len(sample) <= 120


def test_sample_preserves_the_label_proportions():
    sample = stratified_sample(make_corpus(), 120, seed=1)
    shares = sample["label"].value_counts(normalize=True)
    assert shares["cs.CL"] == pytest.approx(0.5, abs=0.05)
    assert shares["cs.DB"] == pytest.approx(1 / 6, abs=0.05)


def test_sample_keeps_every_period():
    sample = stratified_sample(make_corpus(), 60, seed=1)
    assert set(sample["date"].dt.year) == {2023, 2024}


def test_sample_keeps_every_label_even_when_tiny():
    sample = stratified_sample(make_corpus(), 12, seed=1)
    assert set(sample["label"]) == {"cs.CL", "cs.RO", "cs.DB"}


def test_sample_is_reproducible():
    first = stratified_sample(make_corpus(), 90, seed=5)
    second = stratified_sample(make_corpus(), 90, seed=5)
    assert first["doc_id"].tolist() == second["doc_id"].tolist()


def test_different_seeds_draw_different_documents():
    first = stratified_sample(make_corpus(), 90, seed=1)
    second = stratified_sample(make_corpus(), 90, seed=2)
    assert first["doc_id"].tolist() != second["doc_id"].tolist()


def test_requesting_more_than_available_returns_everything():
    frame = make_corpus()
    assert len(stratified_sample(frame, 10_000, seed=1)) == len(frame)


def test_undated_corpora_are_sampled_by_label_only():
    frame = make_corpus()
    frame["date"] = pd.NaT
    sample = stratified_sample(frame, 60, seed=1)
    assert set(sample["label"]) == {"cs.CL", "cs.RO", "cs.DB"}


def test_unlabelled_undated_corpora_still_sample():
    frame = make_corpus()
    frame["label"] = None
    frame["date"] = pd.NaT
    assert len(stratified_sample(frame, 50, seed=1)) == 50


def test_a_non_positive_size_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        stratified_sample(make_corpus(), 0)

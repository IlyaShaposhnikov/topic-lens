"""Tests for the training pipeline and the artifact bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topiclens.artifacts import ARTIFACT_VERSION, ArtifactError, ModelBundle
from topiclens.config import AppConfig
from topiclens.pipeline import cross_model_similarity, split_holdout, topic_timeline, train

pytestmark = pytest.mark.slow

LANGUAGE = "language model translation corpus text sentence word embedding"
ROBOTICS = "robot motion planning control navigation sensor trajectory autonomous"


def make_corpus(n_each: int = 30) -> pd.DataFrame:
    rows = []
    for index in range(n_each):
        rows.append(
            {
                "doc_id": f"cl-{index}",
                "text": LANGUAGE,
                "title": "language paper",
                "label": "cs.CL",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index * 10),
            }
        )
        rows.append(
            {
                "doc_id": f"ro-{index}",
                "text": ROBOTICS,
                "title": "robotics paper",
                "label": "cs.RO",
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=index * 10),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def config(tmp_path) -> AppConfig:
    """A config scaled down so the whole pipeline runs in seconds."""
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


# ------------------------------------------------------------------- splitting


def test_holdout_split_is_disjoint_and_complete():
    train_index, holdout_index = split_holdout(100, 0.2, seed=1)
    assert len(holdout_index) == 20
    assert len(train_index) == 80
    assert not set(train_index) & set(holdout_index)


def test_holdout_split_is_reproducible():
    first = split_holdout(50, 0.3, seed=7)[1]
    second = split_holdout(50, 0.3, seed=7)[1]
    assert np.array_equal(first, second)


def test_zero_fraction_keeps_everything_for_training():
    train_index, holdout_index = split_holdout(10, 0.0, seed=1)
    assert len(train_index) == 10
    assert holdout_index.size == 0


# -------------------------------------------------------------------- training


def test_training_fits_every_model(run):
    assert set(run.bundle.models) == {"lda", "nmf", "lsa"}
    assert all(model.is_fitted for model in run.bundle.models.values())


def test_training_reports_split_sizes(run):
    assert run.train_size == 48
    assert run.holdout_size == 12


def test_metrics_cover_coherence_diversity_and_alignment(run):
    scores = run.metrics["nmf"]
    for key in ("npmi", "diversity", "pairwise_overlap", "nmi", "ari", "purity", "fit_seconds"):
        assert key in scores


def test_lda_perplexity_is_measured_on_held_out_documents(run):
    assert run.metrics["lda"]["holdout_perplexity"] > 0


def test_perplexity_is_absent_without_a_holdout(config):
    config.evaluation = config.evaluation.model_copy(update={"holdout_fraction": 0.0})
    run = train(make_corpus(), config, show_progress=False)
    assert "holdout_perplexity" not in run.metrics["lda"]


def test_document_topics_cover_the_whole_corpus(run):
    frame = make_corpus()
    for shares in run.document_topics.values():
        assert shares.shape == (len(frame), 2)
        assert np.allclose(shares.sum(axis=1), 1.0)


def test_metrics_frame_has_one_row_per_model(run):
    table = run.metrics_frame()
    assert len(table) == 3
    assert "model" in table.columns


def test_empty_corpus_is_rejected(config):
    with pytest.raises(ValueError, match="empty corpus"):
        train(pd.DataFrame(columns=["doc_id", "text", "title", "label", "date"]), config)


def test_topics_recover_the_two_planted_themes(run):
    vocabularies = [{entry.word for entry in topic} for topic in run.bundle.models["nmf"].topics(5)]
    assert any("robot" in words for words in vocabularies)
    assert any("language" in words or "translation" in words for words in vocabularies)


# ------------------------------------------------------------------ similarity


def test_cross_model_similarity_covers_every_pair(run):
    assert set(run.similarity) == {"lda-lsa", "lda-nmf", "lsa-nmf"}
    assert all(-1.0 <= value <= 1.0 for value in run.similarity.values())


def test_a_model_is_identical_to_itself(run):
    models = run.bundle.models
    assert cross_model_similarity({"a": models["nmf"], "b": models["nmf"]})["a-b"] == 1.0


# -------------------------------------------------------------------- timeline


def test_timeline_aggregates_shares_by_period(run):
    frame = make_corpus()
    table = topic_timeline(frame, run.document_topics["nmf"], freq="QE")
    assert "period" in table.columns
    assert len(table) >= 2
    assert np.allclose(table.drop(columns="period").sum(axis=1), 1.0)


def test_timeline_rejects_a_mismatched_share_matrix(run):
    with pytest.raises(ValueError, match="share rows"):
        topic_timeline(make_corpus().head(5), run.document_topics["nmf"])


def test_timeline_requires_dates(run):
    frame = make_corpus()
    frame["date"] = pd.NaT
    with pytest.raises(ValueError, match="no dates"):
        topic_timeline(frame, run.document_topics["nmf"])


def test_timeline_accepts_both_pandas_frequency_spellings(run):
    frame = make_corpus()
    resample_style = topic_timeline(frame, run.document_topics["nmf"], freq="QE")
    period_style = topic_timeline(frame, run.document_topics["nmf"], freq="Q")
    assert resample_style.equals(period_style)


# -------------------------------------------------------------------- bundle


def test_bundle_round_trips_through_disk(run, tmp_path):
    run.bundle.save(tmp_path / "artifacts")
    restored = ModelBundle.load(tmp_path / "artifacts")

    assert restored.version == ARTIFACT_VERSION
    assert set(restored.models) == set(run.bundle.models)
    assert np.allclose(restored.models["nmf"].components, run.bundle.models["nmf"].components)


def test_saved_bundle_writes_a_readable_manifest(run, tmp_path):
    run.bundle.save(tmp_path / "artifacts")
    manifest = (tmp_path / "artifacts" / "manifest.json").read_text(encoding="utf-8")
    assert "vocabulary_size" in manifest
    assert "created_at" in manifest


def test_restored_bundle_scores_new_text_identically(run, tmp_path):
    run.bundle.save(tmp_path / "artifacts")
    restored = ModelBundle.load(tmp_path / "artifacts")

    before = run.bundle.transform_text("a robot planning its motion")
    after = restored.transform_text("a robot planning its motion")
    assert np.allclose(before["nmf"], after["nmf"])


def test_transform_text_applies_the_training_preprocessing(run):
    """Markup and case must not change the result: the bundle owns the cleaning."""
    plain = run.bundle.transform_text("robot motion planning control")
    noisy = run.bundle.transform_text("Robot \\emph{motion} planning $x_i$ control https://a.b")
    assert np.allclose(plain["nmf"], noisy["nmf"])


def test_transform_text_returns_shares_for_every_model(run):
    shares = run.bundle.transform_text("language model translation")
    assert set(shares) == {"lda", "nmf", "lsa"}
    for value in shares.values():
        assert value.shape == (2,)
        assert value.sum() == pytest.approx(1.0)


def test_topic_table_lists_every_topic_of_every_model(run):
    rows = run.bundle.topic_table(top_words=4)
    assert len(rows) == 6
    assert {row["model"] for row in rows} == {"lda", "nmf", "lsa"}
    assert all(row["top_words"] for row in rows)


def test_loading_a_missing_bundle_is_an_error(tmp_path):
    with pytest.raises(ArtifactError, match="no bundle"):
        ModelBundle.load(tmp_path / "nothing")


def test_loading_an_incompatible_version_is_an_error(run, tmp_path):
    import joblib

    run.bundle.save(tmp_path / "artifacts")
    path = tmp_path / "artifacts" / "bundle.joblib"
    payload = joblib.load(path)
    payload["version"] = ARTIFACT_VERSION + 1
    joblib.dump(payload, path)

    with pytest.raises(ArtifactError, match="cannot be read"):
        ModelBundle.load(tmp_path / "artifacts")


def test_unknown_vectorizer_kind_is_rejected(run):
    with pytest.raises(ArtifactError, match="no 'word2vec' vectorizer"):
        run.bundle.matrix_for("word2vec", ["some text"])

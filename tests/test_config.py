"""Tests for configuration loading, validation and environment overrides."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from topiclens.config import AppConfig, apply_env_overrides, load_config
from topiclens.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT

MINIMAL_CONFIG = {
    "data": {
        "arxiv": {
            "categories": ["cs.CL", "cs.CV"],
            "date_from": "2020-01",
            "date_to": "2024-12",
        }
    }
}


@pytest.fixture
def config_file(tmp_path):
    """Write a minimal valid config and return its path."""

    def _write(payload: dict) -> str:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return str(path)

    return _write


def test_shipped_config_is_valid():
    """The config committed to the repo must always load."""
    config = load_config(DEFAULT_CONFIG_PATH, use_env=False)
    assert config.models.n_topics >= 2
    assert config.data.arxiv.categories


def test_defaults_fill_optional_sections(config_file):
    config = load_config(config_file(MINIMAL_CONFIG), use_env=False)
    assert config.project.name == "topic-lens"
    assert config.models.lda.learning_method == "online"
    assert config.preprocessing.vectorizer.tfidf.ngram_range == (1, 2)


def test_relative_paths_resolve_against_project_root(config_file):
    config = load_config(config_file(MINIMAL_CONFIG), use_env=False)
    assert config.data.cache_path == PROJECT_ROOT / "data" / "cache"
    assert config.logging.file_path.is_absolute()


def test_unknown_key_is_rejected(config_file):
    payload = {"data": dict(MINIMAL_CONFIG["data"]), "models": {"n_topic": 12}}
    with pytest.raises(ValidationError):
        load_config(config_file(payload), use_env=False)


def test_reversed_date_range_is_rejected(config_file):
    payload = {"data": {"arxiv": {**MINIMAL_CONFIG["data"]["arxiv"], "date_from": "2025-01"}}}
    with pytest.raises(ValidationError):
        load_config(config_file(payload), use_env=False)


def test_malformed_month_is_rejected(config_file):
    payload = {"data": {"arxiv": {**MINIMAL_CONFIG["data"]["arxiv"], "date_to": "2024-13"}}}
    with pytest.raises(ValidationError):
        load_config(config_file(payload), use_env=False)


def test_kl_divergence_requires_multiplicative_solver(config_file):
    payload = {
        "data": dict(MINIMAL_CONFIG["data"]),
        "models": {"nmf": {"beta_loss": "kullback-leibler", "solver": "cd"}},
    }
    with pytest.raises(ValidationError):
        load_config(config_file(payload), use_env=False)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does-not-exist.yaml", use_env=False)


def test_k_search_is_deduplicated_and_sorted(config_file):
    payload = {"data": dict(MINIMAL_CONFIG["data"]), "evaluation": {"k_search": [20, 5, 5, 10]}}
    config = load_config(config_file(payload), use_env=False)
    assert config.evaluation.k_search == [5, 10, 20]


def test_env_override_replaces_scalar():
    raw = apply_env_overrides(MINIMAL_CONFIG, {"TOPICLENS_MODELS__N_TOPICS": "15"})
    assert AppConfig.model_validate(raw).models.n_topics == 15


def test_env_override_reaches_nested_keys():
    raw = apply_env_overrides(MINIMAL_CONFIG, {"TOPICLENS_DATA__ARXIV__MAX_PER_SLICE": "50"})
    assert AppConfig.model_validate(raw).data.arxiv.max_per_slice == 50


def test_env_override_parses_lists_and_booleans():
    raw = apply_env_overrides(
        MINIMAL_CONFIG,
        {
            "TOPICLENS_EVALUATION__K_SEARCH": "[4, 8]",
            "TOPICLENS_PREPROCESSING__LEMMATIZE": "false",
        },
    )
    config = AppConfig.model_validate(raw)
    assert config.evaluation.k_search == [4, 8]
    assert config.preprocessing.lemmatize is False


def test_unrelated_env_variables_are_ignored():
    raw = apply_env_overrides(MINIMAL_CONFIG, {"PATH": "/usr/bin", "N_TOPICS": "99"})
    assert raw == MINIMAL_CONFIG

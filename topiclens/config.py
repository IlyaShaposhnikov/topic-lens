"""Typed, validated configuration.

``configs/config.yaml`` is the single source of truth for the whole project.
This module turns it into pydantic models, which buys three things:

* typos are fatal instead of silent (``extra="forbid"``),
* value ranges are checked once, up front, rather than deep inside a fit call,
* every consumer gets autocompletion and static types instead of dict lookups.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from topiclens.constants import (
    DEFAULT_CONFIG_PATH,
    ENV_NESTING_SEPARATOR,
    ENV_PREFIX,
    resolve_path,
)

_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class _Section(BaseModel):
    """Base for all config sections: unknown keys are errors, not surprises."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Section):
    name: str = "topic-lens"
    seed: int = 12345
    artifacts_dir: Path = Path("artifacts")

    @property
    def artifacts_path(self) -> Path:
        return resolve_path(self.artifacts_dir)


class ArxivApiConfig(_Section):
    page_size: int = Field(default=200, ge=1, le=2000)
    delay_seconds: float = Field(default=3.0, ge=0.0)
    max_retries: int = Field(default=5, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    rate_limit_pause_seconds: float = Field(default=60.0, ge=0.0)


class ArxivConfig(_Section):
    categories: list[str] = Field(min_length=1)
    date_from: str
    date_to: str
    primary_category_only: bool = True
    include_title: bool = True
    slice_by: Literal["month", "year"] = "month"
    max_per_slice: int = Field(default=400, ge=1)
    api: ArxivApiConfig = Field(default_factory=ArxivApiConfig)

    @field_validator("date_from", "date_to")
    @classmethod
    def _validate_month(cls, value: str) -> str:
        if not _MONTH_PATTERN.match(value):
            raise ValueError(f"expected a YYYY-MM month, got {value!r}")
        return value

    @field_validator("categories")
    @classmethod
    def _validate_categories(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("categories must be unique")
        return value

    @model_validator(mode="after")
    def _validate_range(self) -> ArxivConfig:
        if self.date_from > self.date_to:
            raise ValueError(f"date_from ({self.date_from}) is after date_to ({self.date_to})")
        return self


class CsvConfig(_Section):
    path: Path | None = None
    text_column: str = "text"
    label_column: str | None = None
    date_column: str | None = None
    id_column: str | None = None


class DataConfig(_Section):
    source: Literal["arxiv", "csv"] = "arxiv"
    raw_dir: Path = Path("data/raw")
    cache_dir: Path = Path("data/cache")
    min_abstract_chars: int = Field(default=250, ge=0)
    drop_duplicates: bool = True
    arxiv: ArxivConfig
    csv: CsvConfig = Field(default_factory=CsvConfig)

    @property
    def raw_path(self) -> Path:
        return resolve_path(self.raw_dir)

    @property
    def cache_path(self) -> Path:
        return resolve_path(self.cache_dir)


class CountVectorizerConfig(_Section):
    max_features: int | None = Field(default=20000, ge=1)
    min_df: int | float = 5
    max_df: int | float = 0.5
    ngram_range: tuple[int, int] = (1, 1)

    @field_validator("ngram_range")
    @classmethod
    def _validate_ngrams(cls, value: tuple[int, int]) -> tuple[int, int]:
        low, high = value
        if low < 1 or high < low:
            raise ValueError(f"invalid ngram_range {value}")
        return value


class TfidfVectorizerConfig(CountVectorizerConfig):
    ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True


class VectorizerConfig(_Section):
    count: CountVectorizerConfig = Field(default_factory=CountVectorizerConfig)
    tfidf: TfidfVectorizerConfig = Field(default_factory=TfidfVectorizerConfig)


class PreprocessingConfig(_Section):
    lowercase: bool = True
    min_token_length: int = Field(default=3, ge=1)
    remove_latex: bool = True
    remove_numbers: bool = True
    lemmatize: bool = True
    extra_stopwords: list[str] = Field(default_factory=list)
    vectorizer: VectorizerConfig = Field(default_factory=VectorizerConfig)


class LdaConfig(_Section):
    max_iter: int = Field(default=20, ge=1)
    learning_method: Literal["batch", "online"] = "online"
    learning_decay: float = Field(default=0.7, gt=0.5, le=1.0)
    doc_topic_prior: float | None = Field(default=None, gt=0.0)
    topic_word_prior: float | None = Field(default=None, gt=0.0)
    evaluate_every: int = Field(default=-1, ge=-1)
    perplexity_tol: float = Field(default=0.1, gt=0.0)


class NmfConfig(_Section):
    beta_loss: Literal["frobenius", "kullback-leibler"] = "kullback-leibler"
    solver: Literal["cd", "mu"] = "mu"
    max_iter: int = Field(default=400, ge=1)
    alpha_W: float = Field(default=0.0, ge=0.0)
    l1_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_solver(self) -> NmfConfig:
        if self.beta_loss != "frobenius" and self.solver != "mu":
            raise ValueError(f"beta_loss={self.beta_loss!r} requires solver='mu'")
        return self


class LsaConfig(_Section):
    algorithm: Literal["randomized", "arpack"] = "randomized"
    n_iter: int = Field(default=10, ge=1)
    normalize: bool = True


class ModelsConfig(_Section):
    n_topics: int = Field(default=10, ge=2)
    lda: LdaConfig = Field(default_factory=LdaConfig)
    nmf: NmfConfig = Field(default_factory=NmfConfig)
    lsa: LsaConfig = Field(default_factory=LsaConfig)


class EvaluationConfig(_Section):
    top_words: int = Field(default=10, ge=1)
    coherence_metrics: list[Literal["npmi", "umass"]] = Field(default_factory=lambda: ["npmi"])
    topic_diversity_top_n: int = Field(default=25, ge=1)
    k_search: list[int] = Field(default_factory=lambda: [5, 10, 15, 20])
    holdout_fraction: float = Field(default=0.15, ge=0.0, lt=0.5)

    @field_validator("k_search")
    @classmethod
    def _validate_k_search(cls, value: list[int]) -> list[int]:
        if any(k < 2 for k in value):
            raise ValueError("k_search values must be >= 2")
        return sorted(set(value))


class VizConfig(_Section):
    top_words_chart: int = Field(default=10, ge=1)
    timeline_freq: str = "QE"
    projection: Literal["pca", "tsne"] = "pca"


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"
    file: Path = Path("logs/topiclens.log")

    @property
    def file_path(self) -> Path:
        return resolve_path(self.file)


class AppConfig(_Section):
    """Root configuration object."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _parse_env_value(raw: str) -> Any:
    """Interpret an environment value as YAML, falling back to the raw string."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _assign_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = target
    for key in keys[:-1]:
        nested = cursor.get(key)
        if not isinstance(nested, dict):
            nested = {}
            cursor[key] = nested
        cursor = nested
    cursor[keys[-1]] = value


def apply_env_overrides(
    raw: dict[str, Any], environ: dict[str, str] | None = None
) -> dict[str, Any]:
    """Overlay ``TOPICLENS_SECTION__KEY`` variables onto a raw config mapping."""
    environ = os.environ if environ is None else environ
    merged = dict(raw)
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = name[len(ENV_PREFIX) :].lower().split(ENV_NESTING_SEPARATOR)
        if not all(path):
            continue
        _assign_nested(merged, path, _parse_env_value(value))
    return merged


def load_config(
    path: Path | str | None = None,
    *,
    use_env: bool = True,
    environ: dict[str, str] | None = None,
) -> AppConfig:
    """Read, override and validate the project configuration.

    Args:
        path: YAML file to read; defaults to ``configs/config.yaml``.
        use_env: whether ``TOPICLENS_*`` variables may override file values.
        environ: explicit environment mapping, mainly for tests.

    Raises:
        FileNotFoundError: if the config file is missing.
        pydantic.ValidationError: if a value is absent, mistyped or unknown.
    """
    config_path = resolve_path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")

    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping at the top level")

    if use_env:
        raw = apply_env_overrides(raw, environ)
    return AppConfig.model_validate(raw)

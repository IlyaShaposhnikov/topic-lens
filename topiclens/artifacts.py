"""Saving and loading a trained set of models.

The bundle keeps the preprocessor, both vectorizers and all three models
together, for one reason: inference must repeat training exactly. Because
preprocessing is a separate step in this project rather than something buried
inside the vectorizer, a new text handed to a model directly would be tokenized
differently — or not at all. ``transform_text`` removes that possibility by
routing every new document through the very objects the models were fitted with.

The configuration is snapshotted alongside, so a chart found months later can be
traced back to the parameters that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from topiclens.config import AppConfig
from topiclens.constants import resolve_path
from topiclens.models.base import TopicModel
from topiclens.preprocessing import Preprocessor
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

#: Bumped whenever the bundle layout changes in a way old files cannot satisfy.
ARTIFACT_VERSION = 1

BUNDLE_FILENAME = "bundle.joblib"
MANIFEST_FILENAME = "manifest.json"


class ArtifactError(RuntimeError):
    """Raised when a bundle cannot be loaded or is of an incompatible version."""


@dataclass
class ModelBundle:
    """Everything needed to reproduce predictions, in one loadable object."""

    config: AppConfig
    preprocessor: Preprocessor
    vectorizers: dict[str, Any]
    models: dict[str, TopicModel]
    corpus_fingerprint: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    version: int = ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat(timespec="seconds")

    # ----------------------------------------------------------- persistence

    def save(self, directory: Path | str) -> Path:
        """Write the bundle plus a human-readable manifest; return the bundle path."""
        import joblib

        target = resolve_path(directory)
        target.mkdir(parents=True, exist_ok=True)
        bundle_path = target / BUNDLE_FILENAME

        joblib.dump(
            {
                "version": self.version,
                "created_at": self.created_at,
                "config": self.config.model_dump(mode="json"),
                "preprocessor": self.preprocessor,
                "vectorizers": self.vectorizers,
                "models": self.models,
                "corpus_fingerprint": self.corpus_fingerprint,
            },
            bundle_path,
            compress=3,
        )

        manifest = {
            "version": self.version,
            "created_at": self.created_at,
            "models": sorted(self.models),
            "n_topics": {kind: model.n_topics for kind, model in sorted(self.models.items())},
            "vocabulary_size": {
                kind: int(len(vectorizer.get_feature_names_out()))
                for kind, vectorizer in sorted(self.vectorizers.items())
            },
            "corpus_fingerprint": self.corpus_fingerprint,
        }
        (target / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )

        logger.info("Saved bundle to %s (%.1f MB)", bundle_path, bundle_path.stat().st_size / 1e6)
        return bundle_path

    @classmethod
    def load(cls, directory: Path | str) -> ModelBundle:
        """Read a bundle written by ``save``.

        Raises:
            ArtifactError: if the file is missing or was written by another version.
        """
        import joblib

        source = resolve_path(directory)
        bundle_path = source if source.is_file() else source / BUNDLE_FILENAME
        if not bundle_path.is_file():
            raise ArtifactError(f"no bundle at {bundle_path}")

        payload = joblib.load(bundle_path)
        version = int(payload.get("version", 0))
        if version != ARTIFACT_VERSION:
            raise ArtifactError(
                f"bundle version {version} cannot be read by version {ARTIFACT_VERSION}; retrain"
            )

        logger.info("Loaded bundle from %s (created %s)", bundle_path, payload["created_at"])
        return cls(
            config=AppConfig.model_validate(payload["config"]),
            preprocessor=payload["preprocessor"],
            vectorizers=payload["vectorizers"],
            models=payload["models"],
            corpus_fingerprint=payload.get("corpus_fingerprint", {}),
            created_at=payload["created_at"],
            version=version,
        )

    # -------------------------------------------------------------- inference

    def matrix_for(self, kind: str, texts: list[str], *, preprocess: bool = True) -> Any:
        """Vectorize documents for one model family, preprocessing them first.

        Raises:
            ArtifactError: on an unknown vectorizer kind.
        """
        if kind not in self.vectorizers:
            raise ArtifactError(f"bundle has no {kind!r} vectorizer")
        prepared = [self.preprocessor.transform(text) for text in texts] if preprocess else texts
        return self.vectorizers[kind].transform(prepared)

    def transform_text(self, text: str) -> dict[str, np.ndarray]:
        """Topic shares of one new document, per model.

        This is the only correct way to score a text against these models: it
        applies the same cleaning, the same lemmatization and the same
        vocabularies that were used during training.
        """
        prepared = self.preprocessor.transform(text)
        shares: dict[str, np.ndarray] = {}
        for kind, model in self.models.items():
            matrix = self.vectorizers[model.vectorizer_kind].transform([prepared])
            shares[kind] = model.document_topics(matrix)[0]
        return shares

    def topic_table(self, top_words: int = 10) -> list[dict[str, Any]]:
        """Flat listing of every topic of every model, for CSV export and tables."""
        rows: list[dict[str, Any]] = []
        for kind, model in self.models.items():
            for index, topic in enumerate(model.topics(top_words)):
                rows.append(
                    {
                        "model": kind,
                        "topic": index + 1,
                        "label": model.topic_label(index),
                        "top_words": ", ".join(entry.word for entry in topic),
                    }
                )
        return rows

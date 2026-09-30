"""Data access and preparation for the app, with no Streamlit in sight.

Keeping this module free of ``streamlit`` imports is deliberate: everything the
app computes — loading artifacts, shaping tables, ranking similar documents — is
plain Python that ordinary tests can exercise. ``views.py`` is then thin enough
that its correctness is a matter of looking at it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from topiclens.artifacts import ArtifactError, ModelBundle
from topiclens.constants import resolve_path
from topiclens.data.base import CorpusError, finalize_corpus
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

#: Below this, topic models describe noise rather than structure — the user is
#: warned instead of being handed meaningless topics.
MIN_CORPUS_DOCUMENTS = 200


@dataclass
class ReportData:
    """The CSV and JSON files written next to a bundle, as far as they exist."""

    summary: dict[str, Any]
    topics: pd.DataFrame | None = None
    timeline: pd.DataFrame | None = None
    matches: pd.DataFrame | None = None
    sweep: pd.DataFrame | None = None

    @property
    def metrics(self) -> dict[str, dict[str, float]]:
        return self.summary.get("metrics", {})

    @property
    def has_timeline(self) -> bool:
        return self.timeline is not None and not self.timeline.empty


def load_bundle(directory: Path | str) -> ModelBundle:
    """Load a trained bundle, explaining what to do when there is none.

    Raises:
        ArtifactError: with a message aimed at the person running the app.
    """
    try:
        return ModelBundle.load(directory)
    except ArtifactError as error:
        raise ArtifactError(
            f"{error}. Run 'python scripts/train.py' to build one, "
            "or point the app at another artifacts directory."
        ) from error


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path, **kwargs)


def load_reports(directory: Path | str) -> ReportData:
    """Read whatever report files are present; missing ones are not an error."""
    target = resolve_path(directory)
    summary_path = target / "metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return ReportData(
        summary=summary,
        topics=_read_csv(target / "topics.csv"),
        timeline=_read_csv(target / "timeline.csv", parse_dates=["period"]),
        matches=_read_csv(target / "matches.csv"),
        sweep=_read_csv(target / "sweep.csv"),
    )


def corpus_size_warning(n_documents: int) -> str | None:
    """Warn when a corpus is too small for topic modeling to mean anything."""
    if n_documents >= MIN_CORPUS_DOCUMENTS:
        return None
    return (
        f"Only {n_documents} documents. Topics are inferred from how words co-occur "
        f"across documents, so at least {MIN_CORPUS_DOCUMENTS} — ideally thousands — "
        "are needed before the result says anything about the corpus."
    )


def corpus_from_csv(
    source: Any,
    *,
    text_column: str,
    label_column: str | None = None,
    date_column: str | None = None,
    id_column: str | None = None,
    min_chars: int = 0,
) -> pd.DataFrame:
    """Turn an uploaded CSV into the standard corpus frame.

    Raises:
        CorpusError: if the chosen text column is absent or nothing survives.
    """
    raw = pd.read_csv(source)
    requested = {
        "text": text_column,
        "label": label_column,
        "date": date_column,
        "doc_id": id_column,
    }
    for target, column in requested.items():
        if column and column not in raw.columns:
            raise CorpusError(
                f"column {column!r} (mapped to {target!r}) is not in the file; "
                f"available columns: {', '.join(map(str, raw.columns))}"
            )
    frame = pd.DataFrame(index=raw.index)
    frame["text"] = raw[text_column]
    frame["title"] = None
    frame["label"] = raw[label_column] if label_column else None
    frame["date"] = raw[date_column] if date_column else None
    frame["doc_id"] = (
        raw[id_column].astype(str) if id_column else [f"row-{index}" for index in raw.index]
    )
    return finalize_corpus(frame, min_chars=min_chars, drop_duplicates=True)


def shares_table(shares: np.ndarray, model: Any, *, top_n: int | None = None) -> pd.DataFrame:
    """One document's topic shares as a sorted, labelled table."""
    table = pd.DataFrame(
        {
            "topic": range(1, len(shares) + 1),
            "label": model.topic_labels(),
            "share": shares,
        }
    ).sort_values("share", ascending=False, ignore_index=True)
    return table.head(top_n) if top_n else table


def topic_words_table(model: Any, topic_index: int, top_words: int = 10) -> pd.DataFrame:
    """Top words of one topic with their weights."""
    words = model.top_words(topic_index, top_words)
    return pd.DataFrame(
        {
            "word": [entry.word for entry in words],
            "weight": [round(entry.weight, 4) for entry in words],
        }
    )


def matches_for_pair(matches: pd.DataFrame | None, pair: str) -> pd.DataFrame:
    """Rows of the match report for one model pair, best first."""
    if matches is None or matches.empty:
        return pd.DataFrame()
    subset = matches[matches["pair"] == pair]
    return subset.sort_values("similarity", ascending=False, ignore_index=True)


def document_vectors(bundle: ModelBundle, texts: list[str], *, model: str = "lsa") -> np.ndarray:
    """Embed a corpus in one model's topic space, for similarity search.

    LSA is the default because this is what truncated SVD was built for: it lost
    the topic-quality comparison, but its vectors are a sound similarity space.
    """
    matrix = bundle.matrix_for(bundle.models[model].vectorizer_kind, texts)
    return bundle.models[model].transform(matrix)


def nearest_documents(
    corpus_vectors: np.ndarray, query_vector: np.ndarray, *, top_n: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Indices and cosine similarities of the closest documents to a query."""
    norms = np.linalg.norm(corpus_vectors, axis=1, keepdims=True)
    normalized = np.divide(
        corpus_vectors, norms, out=np.zeros_like(corpus_vectors), where=norms > 0
    )
    query_norm = np.linalg.norm(query_vector)
    query = query_vector / query_norm if query_norm > 0 else query_vector

    scores = normalized @ query
    order = np.argsort(scores)[::-1][:top_n]
    return order, scores[order]


def similar_documents(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    corpus_vectors: np.ndarray,
    query: str,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    """Documents of the corpus closest to a query text, with their scores."""
    query_vector = document_vectors(bundle, [query])[0]
    order, scores = nearest_documents(corpus_vectors, query_vector, top_n=top_n)

    result = frame.iloc[order][["doc_id", "title", "label", "date"]].copy()
    result["similarity"] = np.round(scores, 3)
    return result.reset_index(drop=True)

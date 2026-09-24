"""The corpus contract.

Everything downstream — preprocessing, models, evaluation, the UI — consumes a
dataframe with the columns defined here and nothing else. Sources are free to
fetch from anywhere as long as they produce this shape, which is what makes the
arXiv API and a user-supplied CSV interchangeable.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

#: Canonical column order of a corpus dataframe.
CORPUS_COLUMNS = ("doc_id", "text", "title", "label", "date")

#: Columns a source must provide; the rest are filled with nulls if absent.
REQUIRED_COLUMNS = ("doc_id", "text")


class CorpusError(ValueError):
    """Raised when a source produces something that is not a valid corpus."""


@runtime_checkable
class CorpusSource(Protocol):
    """A provider of raw documents.

    ``fingerprint`` must capture everything that affects the returned data: it
    is what the cache layer hashes to decide whether a stored corpus is still
    the one the current configuration asks for.
    """

    name: str

    def fingerprint(self) -> dict[str, Any]: ...

    def load(self) -> pd.DataFrame: ...


def finalize_corpus(
    frame: pd.DataFrame,
    *,
    min_chars: int = 0,
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Validate, clean and normalize a raw source frame.

    Args:
        frame: raw documents, with at least ``doc_id`` and ``text``.
        min_chars: documents with shorter text are dropped as uninformative.
        drop_duplicates: also deduplicate on normalized text, not just on id.

    Raises:
        CorpusError: if a required column is missing or nothing survives.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise CorpusError(f"source frame is missing required column(s): {', '.join(missing)}")

    result = frame.copy()
    initial = len(result)

    for column in CORPUS_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result["doc_id"] = result["doc_id"].astype("string").str.strip()
    result["text"] = (
        result["text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    )
    result["title"] = result["title"].astype("string").str.strip()
    result["label"] = result["label"].astype("string").str.strip()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")

    result = result.dropna(subset=["doc_id", "text"])
    result = result[result["doc_id"].str.len() > 0]
    empty_dropped = initial - len(result)

    if min_chars > 0:
        before = len(result)
        result = result[result["text"].str.len() >= min_chars]
        short_dropped = before - len(result)
    else:
        short_dropped = 0

    before = len(result)
    result = result.drop_duplicates(subset="doc_id", keep="first")
    if drop_duplicates:
        # Withdrawn-and-resubmitted papers appear under different identifiers
        # with a byte-identical abstract, which would skew every topic weight.
        result = result.loc[~result["text"].str.casefold().duplicated(keep="first")]
    duplicates_dropped = before - len(result)

    result = result[list(CORPUS_COLUMNS)].reset_index(drop=True)

    logger.info(
        "Corpus: %d documents kept from %d (dropped %d empty, %d short, %d duplicate)",
        len(result),
        initial,
        empty_dropped,
        short_dropped,
        duplicates_dropped,
    )
    if result.empty:
        raise CorpusError("no documents left after filtering; relax min_abstract_chars")
    return result


def corpus_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Compact description of a corpus, for CLI output and artifact metadata."""
    summary: dict[str, Any] = {
        "documents": int(len(frame)),
        "mean_chars": round(float(frame["text"].str.len().mean()), 1),
    }
    if frame["label"].notna().any():
        summary["by_label"] = frame["label"].value_counts().to_dict()
    if frame["date"].notna().any():
        summary["date_range"] = f"{frame['date'].min():%Y-%m} … {frame['date'].max():%Y-%m}"
        summary["by_year"] = frame["date"].dt.year.value_counts().sort_index().to_dict()
    return summary

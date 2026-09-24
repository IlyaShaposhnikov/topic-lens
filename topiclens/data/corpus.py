"""Corpus assembly: source selection, filtering and an on-disk cache.

``load_corpus`` is the single entry point used by the CLI, the tests and the
Streamlit app. Fetching thousands of abstracts takes minutes and is polite to do
only once, so the assembled corpus is cached as parquet, keyed by a hash of the
source fingerprint. Change the categories or the date range and you get a new
cache file; rerun with the same settings and nothing touches the network.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from topiclens.config import AppConfig
from topiclens.data.arxiv import ArxivClient
from topiclens.data.base import CorpusSource, corpus_summary, finalize_corpus
from topiclens.data.csv_source import CsvSource
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

_FINGERPRINT_LENGTH = 10


def fingerprint_hash(source: CorpusSource) -> str:
    """Short hash of a source fingerprint, used to key caches and checkpoints."""
    encoded = json.dumps(source.fingerprint(), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:_FINGERPRINT_LENGTH]


class ArxivSource:
    """Corpus source that pulls abstracts from the arXiv API.

    Fetching the full corpus takes tens of minutes and several hundred requests,
    any of which may fail. Each completed slice is therefore appended to a JSONL
    checkpoint keyed by the source fingerprint, so an interrupted run resumes
    where it stopped instead of starting over.
    """

    name = "arxiv"

    def __init__(
        self,
        config: AppConfig,
        *,
        client: ArxivClient | None = None,
        show_progress: bool = True,
        resume: bool = True,
    ) -> None:
        self.config = config
        self.settings = config.data.arxiv
        self.client = client or ArxivClient(self.settings)
        self.show_progress = show_progress
        self.resume = resume

    def fingerprint(self) -> dict[str, Any]:
        return {
            "categories": sorted(self.settings.categories),
            "date_from": self.settings.date_from,
            "date_to": self.settings.date_to,
            "primary_category_only": self.settings.primary_category_only,
            "max_per_slice": self.settings.max_per_slice,
            "include_title": self.settings.include_title,
            "slice_by": self.settings.slice_by,
        }

    @property
    def checkpoint_path(self) -> Path:
        return self.config.data.raw_path / "slices" / f"arxiv-{fingerprint_hash(self)}.jsonl"

    def _read_checkpoint(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Load the slices already fetched by an earlier run of this exact query."""
        path = self.checkpoint_path
        if not self.resume or not path.is_file():
            return {}

        done: dict[tuple[str, str], list[dict[str, Any]]] = {}
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    done[(entry["category"], entry["period"])] = entry["records"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A run killed mid-write leaves a partial last line; the
                    # slice is simply refetched.
                    logger.warning("Ignoring a corrupt checkpoint line in %s", path.name)
        if done:
            logger.info("Resuming from %d checkpointed slices", len(done))
        return done

    def _append_checkpoint(self, category: str, period: str, records: list[dict[str, Any]]) -> None:
        path = self.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"category": category, "period": period, "records": records}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, default=str) + "\n")

    def _slice_records(self, category: str, period: str) -> list[dict[str, Any]]:
        records = []
        for paper in self.client.fetch_slice(category, period):
            text = (
                f"{paper.title}. {paper.abstract}"
                if self.settings.include_title
                else paper.abstract
            )
            records.append(
                {
                    "doc_id": paper.paper_id,
                    "text": text,
                    "title": paper.title,
                    "label": paper.primary_category,
                    "date": paper.published.isoformat(),
                }
            )
        return records

    def load(self) -> pd.DataFrame:
        """Fetch every slice, reusing checkpoints and saving progress as it goes."""
        slices = list(self.client.slices())
        done = self._read_checkpoint()

        iterator: Any = slices
        if self.show_progress:
            from tqdm.auto import tqdm

            iterator = tqdm(slices, desc="arXiv slices", unit="slice")

        records: list[dict[str, Any]] = []
        for category, period in iterator:
            for paper in self.client.fetch_slice(category, period):
                if cached is not None:
                    records.extend(cached)
                    continue
                fetched = self._slice_records(category, period)
                self._append_checkpoint(category, period, fetched)
                records.extend(fetched)

        return pd.DataFrame.from_records(records)


def build_source(config: AppConfig, **kwargs: Any) -> CorpusSource:
    """Instantiate the source selected by ``data.source``."""
    if config.data.source == "csv":
        return CsvSource(config, **kwargs)
    return ArxivSource(config, **kwargs)


def cache_key(config: AppConfig, source: CorpusSource) -> str:
    """Stable short hash of everything that shapes the assembled corpus."""
    payload = {
        "source": source.name,
        "fingerprint": source.fingerprint(),
        "min_abstract_chars": config.data.min_abstract_chars,
        "drop_duplicates": config.data.drop_duplicates,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:_FINGERPRINT_LENGTH]


def cache_paths(config: AppConfig, source: CorpusSource) -> tuple[Path, Path]:
    """Return the parquet path and its metadata sidecar path."""
    stem = f"{source.name}-{cache_key(config, source)}"
    cache_dir = config.data.cache_path
    return cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.json"


def load_corpus(
    config: AppConfig,
    *,
    refresh: bool = False,
    source: CorpusSource | None = None,
    use_cache: bool = True,
    **source_kwargs: Any,
) -> pd.DataFrame:
    """Return the corpus, fetching and caching it if necessary.

    Args:
        config: validated application config.
        refresh: ignore any cached copy and fetch again.
        source: explicit source, mainly for tests; built from config otherwise.
        use_cache: set to False for one-off runs, e.g. an upload in the UI.
        **source_kwargs: forwarded to the source constructor.

    Raises:
        CorpusError: if the source yields no usable documents.
    """
    source = source or build_source(config, **source_kwargs)
    parquet_path, metadata_path = cache_paths(config, source)

    if use_cache and not refresh and parquet_path.is_file():
        frame = pd.read_parquet(parquet_path)
        logger.info("Loaded %d cached documents from %s", len(frame), parquet_path.name)
        return frame

    frame = finalize_corpus(
        source.load(),
        min_chars=config.data.min_abstract_chars,
        drop_duplicates=config.data.drop_duplicates,
    )

    if use_cache:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(parquet_path, index=False)
        metadata = {
            "source": source.name,
            "fingerprint": source.fingerprint(),
            "min_abstract_chars": config.data.min_abstract_chars,
            "drop_duplicates": config.data.drop_duplicates,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": corpus_summary(frame),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        logger.info("Cached corpus to %s", parquet_path.name)

    return frame

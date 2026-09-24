"""Corpus source backed by a local CSV file.

This is what lets the project work on the user's own data: the same pipeline,
the same metrics, the same charts, with column names taken from the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from topiclens.config import AppConfig
from topiclens.constants import resolve_path
from topiclens.data.base import CorpusError
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)


class CsvSource:
    """Reads documents from a delimited text file."""

    name = "csv"

    def __init__(self, config: AppConfig, *, path: Path | str | None = None) -> None:
        self.settings = config.data.csv
        raw_path = path if path is not None else self.settings.path
        if raw_path is None:
            raise CorpusError("data.csv.path is not set; nothing to read")
        self.path = resolve_path(raw_path)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of the file plus the column mapping applied to it."""
        stat = self.path.stat() if self.path.is_file() else None
        return {
            "path": str(self.path),
            "size": stat.st_size if stat else None,
            "mtime": int(stat.st_mtime) if stat else None,
            "text_column": self.settings.text_column,
            "label_column": self.settings.label_column,
            "date_column": self.settings.date_column,
            "id_column": self.settings.id_column,
        }

    def load(self) -> pd.DataFrame:
        """Read the file and rename its columns to the corpus schema.

        Raises:
            CorpusError: if the file is missing or a configured column is not there.
        """
        if not self.path.is_file():
            raise CorpusError(f"CSV file not found: {self.path}")

        frame = pd.read_csv(self.path)
        logger.info("Read %d rows from %s", len(frame), self.path.name)

        mapping = {
            "text": self.settings.text_column,
            "label": self.settings.label_column,
            "date": self.settings.date_column,
            "doc_id": self.settings.id_column,
        }
        for target, column in mapping.items():
            if column and column not in frame.columns:
                raise CorpusError(
                    f"column {column!r} (mapped to {target!r}) is not in {self.path.name}; "
                    f"available columns: {', '.join(map(str, frame.columns))}"
                )

        result = pd.DataFrame(index=frame.index)
        result["text"] = frame[mapping["text"]]
        result["title"] = None
        result["label"] = frame[mapping["label"]] if mapping["label"] else None
        result["date"] = frame[mapping["date"]] if mapping["date"] else None
        result["doc_id"] = (
            frame[mapping["doc_id"]].astype(str)
            if mapping["doc_id"]
            else pd.Series(frame.index, index=frame.index).map(lambda i: f"row-{i}")
        )
        return result

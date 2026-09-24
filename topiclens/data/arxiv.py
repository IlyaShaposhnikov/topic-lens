"""Fetching paper metadata from the public arXiv API.

The API is a paginated Atom feed. Three of its traits shape this module:

* a single query returns a limited window of results, so the requested date
  range is split into ``(category, year)`` slices;
* results are only stable if an explicit sort order is requested, which matters
  for reproducible corpora;
* the service occasionally answers with an empty feed even though matches exist,
  so an empty page is a reason to retry rather than to stop.

Nothing here writes to disk: assembling and caching a corpus is a separate
concern, handled one layer up.
"""

from __future__ import annotations

import calendar
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from topiclens.config import ArxivConfig
from topiclens.utils.logging_config import get_logger

logger = get_logger(__name__)

ARXIV_API_URL = "https://export.arxiv.org/api/query"

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

#: Safety valve: a slice never issues more than this many requests, however
#: aggressively the primary-category filter thins out the pages.
_MAX_PAGES_PER_SLICE = 30


class ArxivFetchError(RuntimeError):
    """Raised when a page could not be retrieved after exhausting retries."""


@dataclass(frozen=True, slots=True)
class ArxivPaper:
    """One paper, reduced to the fields the pipeline actually uses."""

    paper_id: str
    title: str
    abstract: str
    published: date
    primary_category: str
    categories: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        """Flatten into a row for a dataframe."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "published": self.published,
            "primary_category": self.primary_category,
            "categories": ",".join(self.categories),
        }


def normalize_whitespace(text: str) -> str:
    """Collapse the line wrapping that Atom applies to titles and abstracts."""
    return " ".join(text.split())


def strip_version(entry_id: str) -> str:
    """Turn ``http://arxiv.org/abs/2401.01234v2`` into ``2401.01234``."""
    tail = entry_id.rstrip("/").rsplit("/", 1)[-1]
    head, separator, version = tail.rpartition("v")
    return head if separator and version.isdigit() else tail


def _parse_timestamp(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _parse_entry(entry: ET.Element) -> ArxivPaper | None:
    """Convert one ``<entry>`` element, or return ``None`` if it is unusable."""
    try:
        entry_id = entry.findtext("atom:id", namespaces=_NS)
        title = entry.findtext("atom:title", namespaces=_NS)
        abstract = entry.findtext("atom:summary", namespaces=_NS)
        published = entry.findtext("atom:published", namespaces=_NS)
        primary = entry.find("arxiv:primary_category", namespaces=_NS)
        if not (entry_id and title and abstract and published and primary is not None):
            raise ValueError("entry is missing a required field")

        categories = tuple(
            term
            for element in entry.findall("atom:category", namespaces=_NS)
            if (term := element.get("term"))
        )
        primary_term = primary.get("term")
        if not primary_term:
            raise ValueError("primary_category has no term")

        return ArxivPaper(
            paper_id=strip_version(entry_id),
            title=normalize_whitespace(title),
            abstract=normalize_whitespace(abstract),
            published=_parse_timestamp(published),
            primary_category=primary_term,
            categories=categories or (primary_term,),
        )
    except (ValueError, TypeError) as error:
        logger.warning("Skipping malformed arXiv entry: %s", error)
        return None


def parse_feed(xml_text: str) -> tuple[list[ArxivPaper], int]:
    """Parse an Atom feed into papers plus the reported total match count.

    Raises:
        ArxivFetchError: if the payload is not a parsable Atom feed.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ArxivFetchError(f"response is not valid XML: {error}") from error

    if root.tag != f"{{{_NS['atom']}}}feed":
        raise ArxivFetchError(f"unexpected root element {root.tag!r}, expected an Atom feed")

    papers = [
        paper
        for entry in root.findall("atom:entry", namespaces=_NS)
        if (paper := _parse_entry(entry)) is not None
    ]
    total_text = root.findtext("opensearch:totalResults", namespaces=_NS)
    total = int(total_text) if total_text and total_text.isdigit() else len(papers)
    return papers, total


def build_query(category: str, start: str, end: str) -> str:
    """Compose the ``search_query`` value for one slice."""
    return f"cat:{category} AND submittedDate:[{start} TO {end}]"


class ArxivClient:
    """Paginating, retrying, rate-limited reader of the arXiv API.

    The HTTP session and the sleep function are injected so the whole retry and
    pagination logic can be exercised in tests without network or waiting.
    """

    def __init__(
        self,
        config: ArxivConfig,
        *,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._sleep = sleep
        self._session = session
        self._last_request_at: float | None = None

    @property
    def session(self) -> Any:
        if self._session is None:
            import requests  # imported lazily so parsing helpers stay dependency-free

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "topic-lens/0.1 (research project)"})
        return self._session

    # ---------------------------------------------------------------- slicing

    def years(self) -> range:
        first = int(self.config.date_from[:4])
        last = int(self.config.date_to[:4])
        return range(first, last + 1)

    def slice_bounds(self, year: int) -> tuple[str, str]:
        """Return API timestamps for one year, clipped to the configured range."""
        first_year, first_month = int(self.config.date_from[:4]), int(self.config.date_from[5:7])
        last_year, last_month = int(self.config.date_to[:4]), int(self.config.date_to[5:7])

        start_month = first_month if year == first_year else 1
        end_month = last_month if year == last_year else 12
        end_day = calendar.monthrange(year, end_month)[1]

        return (
            f"{year}{start_month:02d}010000",
            f"{year}{end_month:02d}{end_day:02d}2359",
        )

    def slices(self) -> Iterator[tuple[str, int]]:
        """Yield every ``(category, year)`` pair to be requested."""
        for category in self.config.categories:
            for year in self.years():
                yield category, year

    # ---------------------------------------------------------------- fetching

    def _throttle(self) -> None:
        """Keep at least ``delay_seconds`` between consecutive requests."""
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.config.api.delay_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)

    def _fetch_page(self, params: dict[str, Any]) -> tuple[list[ArxivPaper], int]:
        """Request one page, retrying transient failures and empty feeds."""
        attempts = self.config.api.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            if attempt:
                backoff = self.config.api.delay_seconds * 2**attempt
                logger.info("Retrying arXiv request in %.1fs (attempt %d)", backoff, attempt + 1)
                self._sleep(backoff)

            self._throttle()
            try:
                response = self.session.get(
                    ARXIV_API_URL, params=params, timeout=self.config.api.timeout_seconds
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                papers, total = parse_feed(response.text)
            except Exception as error:  # noqa: BLE001 - any transport error is retryable
                self._last_request_at = time.monotonic()
                last_error = error
                logger.warning("arXiv request failed: %s", error)
                continue

            # A feed can come back empty even when matches exist; only trust
            # emptiness once the service itself reports no results.
            if not papers and total > params["start"]:
                last_error = ArxivFetchError("empty feed despite reported matches")
                logger.warning("arXiv returned an empty page for %s", params["search_query"])
                continue

            return papers, total

        raise ArxivFetchError(
            f"could not fetch {params['search_query']} after {attempts} attempts"
        ) from last_error

    def fetch_slice(
        self, category: str, year: int, *, limit: int | None = None
    ) -> list[ArxivPaper]:
        """Collect papers for one ``(category, year)`` slice.

        Pages are requested until ``limit`` papers pass the primary-category
        filter, or the service runs out of matches.
        """
        target = limit if limit is not None else self.config.max_per_slice
        start_bound, end_bound = self.slice_bounds(year)
        query = build_query(category, start_bound, end_bound)
        page_size = min(self.config.api.page_size, target if target < 2000 else 2000)

        collected: list[ArxivPaper] = []
        offset = 0

        for _ in range(_MAX_PAGES_PER_SLICE):
            papers, total = self._fetch_page(
                {
                    "search_query": query,
                    "start": offset,
                    "max_results": page_size,
                    "sortBy": "submittedDate",
                    "sortOrder": "ascending",
                }
            )
            if self.config.primary_category_only:
                papers = [paper for paper in papers if paper.primary_category == category]
            collected.extend(papers)
            offset += page_size

            if len(collected) >= target or offset >= total:
                break

        logger.info("%s %d: collected %d papers", category, year, len(collected[:target]))
        return collected[:target]

    def fetch_all(self) -> list[ArxivPaper]:
        """Collect every configured slice, dropping duplicates across slices."""
        unique: dict[str, ArxivPaper] = {}
        for category, year in self.slices():
            for paper in self.fetch_slice(category, year):
                unique.setdefault(paper.paper_id, paper)
        logger.info("Fetched %d unique papers", len(unique))
        return list(unique.values())

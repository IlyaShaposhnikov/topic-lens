"""Fetching paper metadata from the public arXiv API.

The API is a paginated Atom feed. Three of its traits shape this module:

* a single query returns a limited window of results, so the requested date
  range is split into ``(category, period)`` slices;
* results are only stable if an explicit sort order is requested, which matters
  for reproducible corpora — but combined with coarse slices that sorting also
  biases the sample, which is why the period granularity is configurable and
  defaults to months;
* the service occasionally answers with an empty feed even though matches exist,
  so an empty page is a reason to retry rather than to stop.

Nothing here writes to disk: assembling and caching a corpus is a separate
concern, handled one layer up.
"""

from __future__ import annotations

import calendar
import re
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

_PERIOD_PATTERN = re.compile(r"^(\d{4})(?:-(0[1-9]|1[0-2]))?$")

#: Rate limiting is answered with a long pause, not the ordinary backoff: the
#: limit is enforced per IP address, so a shared or mobile connection can hit it
#: through no fault of ours.
_RATE_LIMIT_STATUS = 429


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

    @property
    def _first_month(self) -> tuple[int, int]:
        return int(self.config.date_from[:4]), int(self.config.date_from[5:7])

    @property
    def _last_month(self) -> tuple[int, int]:
        return int(self.config.date_to[:4]), int(self.config.date_to[5:7])

    def periods(self) -> list[str]:
        """List the periods covering the configured range.

        Monthly slices (``YYYY-MM``) keep the sample evenly spread over time;
        yearly ones (``YYYY``) are cheaper but, together with date sorting,
        collapse each year onto its first days.
        """
        first, last = self._first_month, self._last_month
        if self.config.slice_by == "year":
            return [str(year) for year in range(first[0], last[0] + 1)]

        periods: list[str] = []
        year, month = first
        while (year, month) <= last:
            periods.append(f"{year}-{month:02d}")
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return periods

    def period_bounds(self, period: str) -> tuple[str, str]:
        """Return API timestamps for a period, clipped to the configured range.

        Raises:
            ValueError: if the period is neither ``YYYY`` nor ``YYYY-MM``.
        """
        match = _PERIOD_PATTERN.match(period)
        if not match:
            raise ValueError(f"expected YYYY or YYYY-MM, got {period!r}")

        year = int(match.group(1))
        if month := match.group(2):
            start_month = end_month = int(month)
        else:
            first_year, first_month = self._first_month
            last_year, last_month = self._last_month
            start_month = first_month if year == first_year else 1
            end_month = last_month if year == last_year else 12

        end_day = calendar.monthrange(year, end_month)[1]
        return (
            f"{year}{start_month:02d}010000",
            f"{year}{end_month:02d}{end_day:02d}2359",
        )

    def slices(self) -> Iterator[tuple[str, str]]:
        """Yield every ``(category, period)`` pair to be requested."""
        periods = self.periods()
        for category in self.config.categories:
            for period in periods:
                yield category, period

    # ---------------------------------------------------------------- fetching

    def _throttle(self) -> None:
        """Keep at least ``delay_seconds`` between consecutive requests."""
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.config.api.delay_seconds - elapsed
        if remaining > 0:
            self._sleep(remaining)

    @staticmethod
    def _retry_after(response: Any, fallback: float) -> float:
        """Honour a ``Retry-After`` header, never waiting less than the fallback."""
        headers = getattr(response, "headers", None) or {}
        value = headers.get("Retry-After")
        if value is None:
            return fallback
        try:
            return max(float(value), fallback)
        except (TypeError, ValueError):
            return fallback

    def _fetch_page(self, params: dict[str, Any]) -> tuple[list[ArxivPaper], int]:
        """Request one page, retrying transient failures, rate limits and empty feeds."""
        attempts = self.config.api.max_retries + 1
        last_error: Exception | None = None
        pause = 0.0

        for attempt in range(attempts):
            if attempt:
                logger.info("Retrying arXiv request in %.0fs (attempt %d)", pause, attempt + 1)
                self._sleep(pause)

            self._throttle()
            try:
                response = self.session.get(
                    ARXIV_API_URL, params=params, timeout=self.config.api.timeout_seconds
                )
                self._last_request_at = time.monotonic()

                if getattr(response, "status_code", 200) == _RATE_LIMIT_STATUS:
                    pause = self._retry_after(response, self.config.api.rate_limit_pause_seconds)
                    last_error = ArxivFetchError("rate limited by arXiv (HTTP 429)")
                    logger.warning("Rate limited by arXiv; waiting %.0fs", pause)
                    continue

                response.raise_for_status()
                papers, total = parse_feed(response.text)
            except Exception as error:  # noqa: BLE001 - any transport error is retryable
                self._last_request_at = time.monotonic()
                last_error = error
                pause = self.config.api.delay_seconds * 2 ** (attempt + 1)
                logger.warning("arXiv request failed: %s", error)
                continue

            # A feed can come back empty even when matches exist; only trust
            # emptiness once the service itself reports no results.
            if not papers and total > params["start"]:
                last_error = ArxivFetchError("empty feed despite reported matches")
                pause = self.config.api.delay_seconds * 2 ** (attempt + 1)
                logger.warning("arXiv returned an empty page for %s", params["search_query"])
                continue

            return papers, total

        raise ArxivFetchError(
            f"could not fetch {params['search_query']} after {attempts} attempts"
        ) from last_error

    def fetch_slice(
        self, category: str, period: str, *, limit: int | None = None
    ) -> list[ArxivPaper]:
        """Collect papers for one ``(category, period)`` slice.

        Pages are requested until ``limit`` papers pass the primary-category
        filter, or the service runs out of matches.
        """
        target = limit if limit is not None else self.config.max_per_slice
        start_bound, end_bound = self.period_bounds(period)
        query = build_query(category, start_bound, end_bound)
        page_size = min(self.config.api.page_size, max(target, 1))

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

        logger.info("%s %s: collected %d papers", category, period, len(collected[:target]))
        return collected[:target]

    def fetch_all(self) -> list[ArxivPaper]:
        """Collect every configured slice, dropping duplicates across slices."""
        unique: dict[str, ArxivPaper] = {}
        for category, period in self.slices():
            for paper in self.fetch_slice(category, period):
                unique.setdefault(paper.paper_id, paper)
        logger.info("Fetched %d unique papers", len(unique))
        return list(unique.values())

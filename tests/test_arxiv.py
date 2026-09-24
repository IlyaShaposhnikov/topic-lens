"""Tests for the arXiv client. No network: a fake session serves canned feeds."""

from __future__ import annotations

from datetime import date

import pytest

from topiclens.config import ArxivConfig
from topiclens.data.arxiv import (
    ArxivClient,
    ArxivFetchError,
    build_query,
    parse_feed,
    strip_version,
)

FEED_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{total}</opensearch:totalResults>
{entries}
</feed>
"""

ENTRY_TEMPLATE = """  <entry>
    <id>http://arxiv.org/abs/{paper_id}v1</id>
    <title>{title}</title>
    <summary>An abstract
      wrapped over lines.</summary>
    <published>{published}T12:00:00Z</published>
    <arxiv:primary_category term="{primary}" scheme="http://arxiv.org/schemas/atom"/>
    <category term="{primary}"/>
    <category term="cs.LG"/>
  </entry>"""


def make_entry(paper_id: str, primary: str = "cs.CL", published: str = "2024-03-01") -> str:
    return ENTRY_TEMPLATE.format(
        paper_id=paper_id, title=f"Paper {paper_id}", published=published, primary=primary
    )


def make_feed(entries: list[str], total: int | None = None) -> str:
    return FEED_TEMPLATE.format(
        total=total if total is not None else len(entries), entries="\n".join(entries)
    )


class FakeResponse:
    def __init__(
        self, text: str, status_code: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.text = text
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves a queued list of responses and records the requests it received."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        self.calls.append(dict(params))
        if not self.responses:
            raise AssertionError("unexpected extra request")
        return self.responses.pop(0)


@pytest.fixture
def config() -> ArxivConfig:
    return ArxivConfig(
        categories=["cs.CL", "cs.CV"],
        date_from="2023-11",
        date_to="2024-02",
        max_per_slice=4,
        api={"page_size": 2, "delay_seconds": 0.0, "max_retries": 2},
    )


def build_client(
    config: ArxivConfig,
    responses: list[FakeResponse],
    sleeps: list[float] | None = None,
) -> ArxivClient:
    recorder = sleeps.append if sleeps is not None else (lambda _: None)
    return ArxivClient(config, session=FakeSession(responses), sleep=recorder)


# ------------------------------------------------------------------- parsing


def test_strip_version_handles_ids_with_and_without_version():
    assert strip_version("http://arxiv.org/abs/2401.01234v2") == "2401.01234"
    assert strip_version("http://arxiv.org/abs/2401.01234") == "2401.01234"


def test_parse_feed_normalizes_whitespace_and_fields():
    papers, total = parse_feed(make_feed([make_entry("2403.00001")], total=7))
    assert total == 7
    paper = papers[0]
    assert paper.paper_id == "2403.00001"
    assert paper.abstract == "An abstract wrapped over lines."
    assert paper.published == date(2024, 3, 1)
    assert paper.primary_category == "cs.CL"
    assert "cs.LG" in paper.categories


def test_parse_feed_skips_entry_without_abstract():
    broken = """  <entry>
    <id>http://arxiv.org/abs/2403.99999v1</id>
    <title>No summary here</title>
    <published>2024-03-01T12:00:00Z</published>
    <arxiv:primary_category term="cs.CL"/>
  </entry>"""
    papers, _ = parse_feed(make_feed([broken, make_entry("2403.00002")]))
    assert [paper.paper_id for paper in papers] == ["2403.00002"]


def test_parse_feed_rejects_html_error_page():
    with pytest.raises(ArxivFetchError):
        parse_feed("<html>gateway timeout</html>")


def test_parse_feed_rejects_truncated_xml():
    with pytest.raises(ArxivFetchError):
        parse_feed("<feed><entry>")


def test_build_query_shape():
    assert build_query("cs.CL", "202401010000", "202412312359") == (
        "cat:cs.CL AND submittedDate:[202401010000 TO 202412312359]"
    )


# ------------------------------------------------------------------- slicing


def test_monthly_periods_span_the_range(config):
    client = build_client(config, [])
    assert client.periods() == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_yearly_periods_span_the_range(config):
    config = config.model_copy(update={"slice_by": "year"})
    client = build_client(config, [])
    assert client.periods() == ["2023", "2024"]


def test_month_bounds_cover_exactly_that_month(config):
    client = build_client(config, [])
    assert client.period_bounds("2023-11") == ("202311010000", "202311302359")


def test_month_bounds_use_february_length(config):
    client = build_client(config, [])
    assert client.period_bounds("2024-02") == ("202402010000", "202402292359")


def test_year_bounds_are_clipped_to_the_configured_range(config):
    client = build_client(config, [])
    assert client.period_bounds("2023") == ("202311010000", "202312312359")
    assert client.period_bounds("2024") == ("202401010000", "202402292359")


def test_malformed_period_is_rejected(config):
    client = build_client(config, [])
    with pytest.raises(ValueError, match="YYYY-MM"):
        client.period_bounds("2024-13")


def test_slices_cover_categories_and_periods(config):
    config = config.model_copy(update={"date_to": "2023-12"})
    client = build_client(config, [])
    assert list(client.slices()) == [
        ("cs.CL", "2023-11"),
        ("cs.CL", "2023-12"),
        ("cs.CV", "2023-11"),
        ("cs.CV", "2023-12"),
    ]


# ------------------------------------------------------------------ fetching


def test_fetch_slice_paginates_until_target_is_reached(config):
    pages = [
        FakeResponse(make_feed([make_entry("1"), make_entry("2")], total=10)),
        FakeResponse(make_feed([make_entry("3"), make_entry("4")], total=10)),
    ]
    client = build_client(config, pages)
    papers = client.fetch_slice("cs.CL", "2024-01")

    assert [paper.paper_id for paper in papers] == ["1", "2", "3", "4"]
    assert [call["start"] for call in client.session.calls] == [0, 2]
    assert client.session.calls[0]["sortBy"] == "submittedDate"
    assert "202401010000" in client.session.calls[0]["search_query"]


def test_fetch_slice_stops_when_results_are_exhausted(config):
    client = build_client(config, [FakeResponse(make_feed([make_entry("1")], total=1))])
    assert len(client.fetch_slice("cs.CL", "2024-01")) == 1


def test_fetch_slice_filters_cross_listed_papers(config):
    pages = [
        FakeResponse(make_feed([make_entry("1"), make_entry("2", primary="cs.CV")], total=2)),
    ]
    client = build_client(config, pages)
    papers = client.fetch_slice("cs.CL", "2024-01")
    assert [paper.paper_id for paper in papers] == ["1"]


def test_fetch_slice_keeps_cross_listed_papers_when_configured(config):
    config = config.model_copy(update={"primary_category_only": False})
    pages = [
        FakeResponse(make_feed([make_entry("1"), make_entry("2", primary="cs.CV")], total=2)),
    ]
    client = build_client(config, pages)
    assert len(client.fetch_slice("cs.CL", "2024-01")) == 2


def test_fetch_slice_respects_explicit_limit(config):
    client = build_client(config, [FakeResponse(make_feed([make_entry("1")], total=5))])
    papers = client.fetch_slice("cs.CL", "2024-01", limit=1)
    assert len(papers) == 1
    assert client.session.calls[0]["max_results"] == 1


def test_fetch_retries_after_server_error(config):
    pages = [
        FakeResponse("", status_code=503),
        FakeResponse(make_feed([make_entry("1")], total=1)),
    ]
    client = build_client(config, pages)
    assert len(client.fetch_slice("cs.CL", "2024-01")) == 1
    assert len(client.session.calls) == 2


def test_fetch_retries_on_empty_feed_with_pending_matches(config):
    pages = [
        FakeResponse(make_feed([], total=5)),
        FakeResponse(make_feed([make_entry("1")], total=5)),
    ]
    client = build_client(config, pages)
    assert len(client.fetch_slice("cs.CL", "2024-01", limit=1)) == 1


def test_rate_limited_request_waits_the_configured_pause(config):
    config = config.model_copy(
        update={"api": config.api.model_copy(update={"rate_limit_pause_seconds": 45.0})}
    )
    pages = [
        FakeResponse("", status_code=429),
        FakeResponse(make_feed([make_entry("1")], total=1)),
    ]
    sleeps: list[float] = []
    client = build_client(config, pages, sleeps)

    assert len(client.fetch_slice("cs.CL", "2024-01")) == 1
    assert 45.0 in sleeps


def test_rate_limit_respects_a_longer_retry_after_header(config):
    pages = [
        FakeResponse("", status_code=429, headers={"Retry-After": "120"}),
        FakeResponse(make_feed([make_entry("1")], total=1)),
    ]
    sleeps: list[float] = []
    client = build_client(config, pages, sleeps)

    client.fetch_slice("cs.CL", "2024-01")
    assert 120.0 in sleeps


def test_fetch_gives_up_after_max_retries(config):
    pages = [FakeResponse("", status_code=500) for _ in range(3)]
    client = build_client(config, pages)
    with pytest.raises(ArxivFetchError):
        client.fetch_slice("cs.CL", "2024-01")


def test_fetch_all_deduplicates_across_slices(config):
    config = config.model_copy(update={"categories": ["cs.CL"], "date_to": "2023-11"})
    pages = [FakeResponse(make_feed([make_entry("1"), make_entry("1")], total=2))]
    client = build_client(config, pages)
    assert len(client.fetch_all()) == 1

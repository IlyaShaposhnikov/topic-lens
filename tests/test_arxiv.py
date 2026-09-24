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
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

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
        date_from="2023-03",
        date_to="2024-08",
        max_per_slice=4,
        api={"page_size": 2, "delay_seconds": 0.0, "max_retries": 2},
    )


def build_client(config: ArxivConfig, responses: list[FakeResponse]) -> ArxivClient:
    return ArxivClient(config, session=FakeSession(responses), sleep=lambda _: None)


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


def test_slice_bounds_clip_first_and_last_year(config):
    client = build_client(config, [])
    assert client.slice_bounds(2023) == ("202303010000", "202312312359")
    assert client.slice_bounds(2024) == ("202401010000", "202408312359")


def test_slice_bounds_use_february_length(config):
    config = config.model_copy(update={"date_to": "2024-02"})
    client = build_client(config, [])
    assert client.slice_bounds(2024)[1] == "202402292359"


def test_slices_cover_categories_and_years(config):
    client = build_client(config, [])
    assert list(client.slices()) == [
        ("cs.CL", 2023),
        ("cs.CL", 2024),
        ("cs.CV", 2023),
        ("cs.CV", 2024),
    ]


# ------------------------------------------------------------------ fetching


def test_fetch_slice_paginates_until_target_is_reached(config):
    pages = [
        FakeResponse(make_feed([make_entry("1"), make_entry("2")], total=10)),
        FakeResponse(make_feed([make_entry("3"), make_entry("4")], total=10)),
    ]
    client = build_client(config, pages)
    papers = client.fetch_slice("cs.CL", 2024)

    assert [paper.paper_id for paper in papers] == ["1", "2", "3", "4"]
    assert [call["start"] for call in client.session.calls] == [0, 2]
    assert client.session.calls[0]["sortBy"] == "submittedDate"


def test_fetch_slice_stops_when_results_are_exhausted(config):
    client = build_client(config, [FakeResponse(make_feed([make_entry("1")], total=1))])
    assert len(client.fetch_slice("cs.CL", 2024)) == 1


def test_fetch_slice_filters_cross_listed_papers(config):
    pages = [
        FakeResponse(make_feed([make_entry("1"), make_entry("2", primary="cs.CV")], total=2)),
    ]
    client = build_client(config, pages)
    papers = client.fetch_slice("cs.CL", 2024)
    assert [paper.paper_id for paper in papers] == ["1"]


def test_fetch_slice_keeps_cross_listed_papers_when_configured(config):
    config = config.model_copy(update={"primary_category_only": False})
    pages = [
        FakeResponse(make_feed([make_entry("1"), make_entry("2", primary="cs.CV")], total=2)),
    ]
    client = build_client(config, pages)
    assert len(client.fetch_slice("cs.CL", 2024)) == 2


def test_fetch_slice_respects_explicit_limit(config):
    client = build_client(config, [FakeResponse(make_feed([make_entry("1")], total=5))])
    papers = client.fetch_slice("cs.CL", 2024, limit=1)
    assert len(papers) == 1
    assert client.session.calls[0]["max_results"] == 1


def test_fetch_retries_after_server_error(config):
    pages = [
        FakeResponse("", status_code=503),
        FakeResponse(make_feed([make_entry("1")], total=1)),
    ]
    client = build_client(config, pages)
    assert len(client.fetch_slice("cs.CL", 2024)) == 1
    assert len(client.session.calls) == 2


def test_fetch_retries_on_empty_feed_with_pending_matches(config):
    pages = [
        FakeResponse(make_feed([], total=5)),
        FakeResponse(make_feed([make_entry("1")], total=5)),
    ]
    client = build_client(config, pages)
    assert len(client.fetch_slice("cs.CL", 2024, limit=1)) == 1


def test_fetch_gives_up_after_max_retries(config):
    pages = [FakeResponse("", status_code=500) for _ in range(3)]
    client = build_client(config, pages)
    with pytest.raises(ArxivFetchError):
        client.fetch_slice("cs.CL", 2024)


def test_fetch_all_deduplicates_across_slices(config):
    config = config.model_copy(update={"categories": ["cs.CL"], "date_to": "2023-12"})
    pages = [FakeResponse(make_feed([make_entry("1"), make_entry("1")], total=2))]
    client = build_client(config, pages)
    assert len(client.fetch_all()) == 1

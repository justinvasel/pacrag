from __future__ import annotations

from pac_crawler.listing_parser import (
    _extract_agency_name,
    parse_opinion_links,
    parse_year_links,
)
from pac_crawler.models import OpinionKind

ADVISORY_INDEX_HTML = """
<html><body>
<section id="content_container_356097">
<ul>
<li><a href="/pac/advisory/2025-advisory-opinions">Advisory Opinions 2025</a></li>
<li><a href="/pac/advisory/advisory-opinions-2024">Advisory Opinions 2024</a></li>
<li><a href="/pac/advisory/advisory-opinions-2023">Advisory Opinions 2023</a></li>
<li><a href="/pac/advisory/advisory-opinions-1999">Advisory Opinions 1999</a></li>
</ul>
</section>
<p><a href="/pac/about/">About us</a></p>
</body></html>
"""

INFORMAL_INDEX_HTML = """
<html><body>
<section id="content_container_343661">
<ul>
<li><a href="/pac/informal/informal-opinions-2024">Informal Opinions 2024</a></li>
<li><a href="/pac/informal/informal-opinions-2023">Informal Opinions 2023</a></li>
</ul>
</section>
</body></html>
"""

YEAR_PAGE_HTML = (
    "<html><body>"
    '<section id="content_container_995532">'
    "<ul>"
    '<li><a href="/pac/files/advisory/25-FC-00B-AO.pdf" target="_blank">'
    "25-FC-00B: Alleged Violation of APRA by the New Carlisle Town Council"
    "</a></li>"
    '<li><a href="/pac/files/advisory/25-FC-002A-Kimbrough.pdf" target="_blank">'
    "25-FC-002A: Alleged Violation of APRA by Some Agency"
    "</a></li>"
    '<li><a href="/pac/files/advisory/25-FC-138-LaFontaine.pdf" target="_blank">'
    "25-FC-138: Alleged Violation of APRA by City of Bloomington"
    "</a></li>"
    '<li><a href="/pac/files/advisory/25-FC-100-REVISED.pdf" target="_blank">'
    "25-FC-100 REVISED: Alleged Violation of ODL by Some Board"
    "</a></li>"
    "</ul>"
    "</section>"
    '<footer><a href="/pac/about/">About</a></footer>'
    "</body></html>"
)

INFORMAL_YEAR_PAGE_HTML = """
<html><body>
<section id="content_container_773301">
<ul>
<li><a href="/pac/files/informal/24-INF-01.pdf">24-INF-01: Question about records</a></li>
<li><a href="/pac/files/informal/24-INF-05.pdf">24-INF-05: Another question</a></li>
</ul>
</section>
</body></html>
"""

EMPTY_PAGE_HTML = """
<html><body><p>No opinions here.</p></body></html>
"""


class TestParseYearLinks:
    def test_advisory_index(self) -> None:
        links = parse_year_links(ADVISORY_INDEX_HTML, "https://www.in.gov/pac/advisory/")
        assert len(links) == 4
        urls = [url for url, _ in links]
        years = [y for _, y in links]
        assert 2025 in years
        assert 2024 in years
        assert 1999 in years
        assert all(url.startswith("https://") for url in urls)

    def test_informal_index(self) -> None:
        links = parse_year_links(INFORMAL_INDEX_HTML, "https://www.in.gov/pac/informal/")
        assert len(links) == 2
        years = [y for _, y in links]
        assert 2024 in years
        assert 2023 in years

    def test_sorted_descending(self) -> None:
        links = parse_year_links(ADVISORY_INDEX_HTML, "https://www.in.gov/pac/advisory/")
        years = [y for _, y in links]
        assert years == sorted(years, reverse=True)

    def test_empty_page_returns_empty(self) -> None:
        links = parse_year_links(EMPTY_PAGE_HTML, "https://www.in.gov/pac/advisory/")
        assert links == []

    def test_filters_non_year_links(self) -> None:
        links = parse_year_links(ADVISORY_INDEX_HTML, "https://www.in.gov/pac/advisory/")
        urls = [url for url, _ in links]
        assert not any("about" in url for url in urls)

    def test_handles_both_url_patterns(self) -> None:
        links = parse_year_links(ADVISORY_INDEX_HTML, "https://www.in.gov/pac/advisory/")
        urls = [url for url, _ in links]
        assert any("2025-advisory-opinions" in url for url in urls)
        assert any("advisory-opinions-2024" in url for url in urls)


class TestParseOpinionLinks:
    def test_parses_entries(self) -> None:
        entries = parse_opinion_links(
            YEAR_PAGE_HTML,
            "https://www.in.gov/pac/advisory/2025-advisory-opinions/",
            2025,
            OpinionKind.FC,
        )
        assert len(entries) == 4

    def test_opinion_id_from_text_not_filename(self) -> None:
        entries = parse_opinion_links(
            YEAR_PAGE_HTML,
            "https://www.in.gov/pac/advisory/2025-advisory-opinions/",
            2025,
            OpinionKind.FC,
        )
        ids = {e.opinion_id for e in entries}
        assert "25-FC-138" in ids

    def test_sub_letter_parsed(self) -> None:
        entries = parse_opinion_links(
            YEAR_PAGE_HTML,
            "https://www.in.gov/pac/advisory/2025-advisory-opinions/",
            2025,
            OpinionKind.FC,
        )
        entry_a = next(e for e in entries if e.opinion_id == "25-FC-002A")
        assert entry_a.sub_letter == "A"

    def test_revised_detected(self) -> None:
        entries = parse_opinion_links(
            YEAR_PAGE_HTML,
            "https://www.in.gov/pac/advisory/2025-advisory-opinions/",
            2025,
            OpinionKind.FC,
        )
        revised = next(e for e in entries if e.opinion_id == "25-FC-100")
        assert revised.is_revised is True

    def test_pdf_url_is_absolute(self) -> None:
        entries = parse_opinion_links(
            YEAR_PAGE_HTML,
            "https://www.in.gov/pac/advisory/2025-advisory-opinions/",
            2025,
            OpinionKind.FC,
        )
        for entry in entries:
            assert entry.pdf_url.startswith("https://")

    def test_informal_entries(self) -> None:
        entries = parse_opinion_links(
            INFORMAL_YEAR_PAGE_HTML,
            "https://www.in.gov/pac/informal/informal-opinions-2024/",
            2024,
            OpinionKind.INF,
        )
        assert len(entries) == 2
        assert entries[0].opinion_id == "24-INF-01"
        assert entries[0].kind == OpinionKind.INF

    def test_empty_page_returns_empty(self) -> None:
        entries = parse_opinion_links(
            EMPTY_PAGE_HTML,
            "https://example.com/2025/",
            2025,
            OpinionKind.FC,
        )
        assert entries == []

    def test_agency_name_extracted(self) -> None:
        entries = parse_opinion_links(
            YEAR_PAGE_HTML,
            "https://www.in.gov/pac/advisory/2025-advisory-opinions/",
            2025,
            OpinionKind.FC,
        )
        entry = next(e for e in entries if e.opinion_id == "25-FC-00B")
        assert "New Carlisle Town Council" in entry.agency_name

    def test_listing_url_set(self) -> None:
        entries = parse_opinion_links(
            YEAR_PAGE_HTML,
            "https://www.in.gov/pac/advisory/2025-advisory-opinions/",
            2025,
            OpinionKind.FC,
        )
        for entry in entries:
            assert entry.listing_url == "https://www.in.gov/pac/advisory/2025-advisory-opinions/"


class TestExtractAgencyName:
    def test_simple(self) -> None:
        result = _extract_agency_name(
            "25-FC-118: Alleged Violation of APRA by City of Bloomington",
            "25-FC-118",
        )
        assert result == "Alleged Violation of APRA by City of Bloomington"

    def test_no_colon(self) -> None:
        result = _extract_agency_name("25-FC-118 Some text without colon", "25-FC-118")
        assert result == "25-FC-118 Some text without colon"

    def test_multiple_colons(self) -> None:
        result = _extract_agency_name(
            "25-FC-118: Part one: Part two",
            "25-FC-118",
        )
        assert result == "Part one: Part two"

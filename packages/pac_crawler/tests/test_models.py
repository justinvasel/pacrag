from __future__ import annotations

from datetime import UTC

from pac_crawler.models import (
    DownloadStatus,
    ListingEntry,
    OpinionKind,
    OpinionRow,
    parse_opinion_id,
    two_digit_year_to_four,
)


class TestParseOpinionId:
    def test_simple_fc(self) -> None:
        result = parse_opinion_id("25-FC-118")
        assert result is not None
        oid, year, kind, sub, revised = result
        assert oid == "25-FC-118"
        assert year == 2025
        assert kind == OpinionKind.FC
        assert sub is None
        assert revised is False

    def test_simple_inf(self) -> None:
        result = parse_opinion_id("24-INF-01")
        assert result is not None
        oid, year, kind, sub, revised = result
        assert oid == "24-INF-01"
        assert year == 2024
        assert kind == OpinionKind.INF
        assert sub is None
        assert revised is False

    def test_sub_letter_a(self) -> None:
        result = parse_opinion_id("24-FC-77A")
        assert result is not None
        oid, year, kind, sub, revised = result
        assert oid == "24-FC-77A"
        assert year == 2024
        assert kind == OpinionKind.FC
        assert sub == "A"
        assert revised is False

    def test_sub_letter_b(self) -> None:
        result = parse_opinion_id("25-FC-00B")
        assert result is not None
        oid, year, _kind, sub, _revised = result
        assert oid == "25-FC-00B"
        assert year == 2025
        assert sub == "B"

    def test_revised(self) -> None:
        result = parse_opinion_id("23-FC-100 REVISED")
        assert result is not None
        oid, year, _kind, sub, revised = result
        assert oid == "23-FC-100"
        assert year == 2023
        assert sub is None
        assert revised is True

    def test_revised_with_sub_letter(self) -> None:
        result = parse_opinion_id("23-FC-100B REVISED")
        assert result is not None
        oid, _year, _kind, sub, revised = result
        assert oid == "23-FC-100B"
        assert sub == "B"
        assert revised is True

    def test_old_year(self) -> None:
        result = parse_opinion_id("99-FC-001")
        assert result is not None
        _, year, _, _, _ = result
        assert year == 1999

    def test_year_2000(self) -> None:
        result = parse_opinion_id("00-FC-001")
        assert result is not None
        _, year, _, _, _ = result
        assert year == 2000

    def test_no_match_returns_none(self) -> None:
        assert parse_opinion_id("not-an-opinion") is None
        assert parse_opinion_id("") is None
        assert parse_opinion_id("25-XX-100") is None

    def test_embedded_in_title(self) -> None:
        result = parse_opinion_id("25-FC-138: Alleged Violation of APRA by City of Bloomington")
        assert result is not None
        oid, year, _kind, _sub, _revised = result
        assert oid == "25-FC-138"
        assert year == 2025

    def test_leading_whitespace(self) -> None:
        result = parse_opinion_id("  25-FC-001")
        assert result is not None
        assert result[0] == "25-FC-001"

    def test_zero_padded_number(self) -> None:
        result = parse_opinion_id("25-FC-007")
        assert result is not None
        assert result[0] == "25-FC-007"


class TestTwoDigitYearToFour:
    def test_recent_years(self) -> None:
        assert two_digit_year_to_four(25) == 2025
        assert two_digit_year_to_four(24) == 2024
        assert two_digit_year_to_four(10) == 2010
        assert two_digit_year_to_four(0) == 2000

    def test_old_years(self) -> None:
        assert two_digit_year_to_four(99) == 1999
        assert two_digit_year_to_four(50) == 1950

    def test_boundary(self) -> None:
        assert two_digit_year_to_four(49) == 2049
        assert two_digit_year_to_four(50) == 1950


class TestOpinionKind:
    def test_values(self) -> None:
        assert OpinionKind.FC == "FC"
        assert OpinionKind.INF == "INF"

    def test_from_string(self) -> None:
        assert OpinionKind("FC") == OpinionKind.FC
        assert OpinionKind("INF") == OpinionKind.INF


class TestDownloadStatus:
    def test_values(self) -> None:
        assert DownloadStatus.PENDING == "pending"
        assert DownloadStatus.OK == "ok"
        assert DownloadStatus.HTTP_ERROR == "http_error"
        assert DownloadStatus.MISSING == "missing"


class TestListingEntry:
    def test_valid_entry(self) -> None:
        entry = ListingEntry(
            opinion_id="25-FC-118",
            year=2025,
            kind=OpinionKind.FC,
            sub_letter=None,
            is_revised=False,
            raw_title="25-FC-118: Alleged Violation of APRA by City of Bloomington",
            agency_name="Alleged Violation of APRA by City of Bloomington",
            pdf_url="https://www.in.gov/pac/files/advisory/25-FC-118.pdf",
            listing_url="https://www.in.gov/pac/advisory/2025-advisory-opinions/",
        )
        assert entry.opinion_id == "25-FC-118"
        assert entry.year == 2025
        assert entry.kind == OpinionKind.FC

    def test_with_sub_letter(self) -> None:
        entry = ListingEntry(
            opinion_id="24-FC-77A",
            year=2024,
            kind=OpinionKind.FC,
            sub_letter="A",
            is_revised=False,
            raw_title="24-FC-77A: Some complaint",
            agency_name="Some complaint",
            pdf_url="https://example.com/24-FC-77A.pdf",
            listing_url="https://example.com/2024/",
        )
        assert entry.sub_letter == "A"


class TestOpinionRow:
    def test_minimal_valid(self) -> None:
        from datetime import datetime

        now = datetime.now(UTC)
        row = OpinionRow(
            opinion_id="25-FC-118",
            year=2025,
            kind="FC",
            sub_letter=None,
            revision=0,
            parent_opinion_id=None,
            raw_title="25-FC-118: Some title",
            agency_name="Some title",
            listing_url="https://example.com/2025/",
            pdf_url="https://example.com/25-FC-118.pdf",
            pdf_path=None,
            pdf_sha256=None,
            first_seen_at=now,
            last_checked_at=now,
            download_status="pending",
        )
        assert row.opinion_id == "25-FC-118"
        assert row.download_status == "pending"

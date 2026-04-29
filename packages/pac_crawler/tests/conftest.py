from __future__ import annotations

from pathlib import Path

import pytest
from pac_crawler.models import ListingEntry, OpinionKind


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.duckdb"


@pytest.fixture
def sample_entry() -> ListingEntry:
    return ListingEntry(
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


@pytest.fixture
def sample_entry_b() -> ListingEntry:
    return ListingEntry(
        opinion_id="25-FC-002A",
        year=2025,
        kind=OpinionKind.FC,
        sub_letter="A",
        is_revised=False,
        raw_title="25-FC-002A: Alleged Violation of APRA by Some Agency",
        agency_name="Alleged Violation of APRA by Some Agency",
        pdf_url="https://www.in.gov/pac/files/advisory/25-FC-002A.pdf",
        listing_url="https://www.in.gov/pac/advisory/2025-advisory-opinions/",
    )


@pytest.fixture
def sample_inf_entry() -> ListingEntry:
    return ListingEntry(
        opinion_id="24-INF-01",
        year=2024,
        kind=OpinionKind.INF,
        sub_letter=None,
        is_revised=False,
        raw_title="24-INF-01: Some Informal Inquiry",
        agency_name="Some Informal Inquiry",
        pdf_url="https://www.in.gov/pac/files/informal/24-INF-01.pdf",
        listing_url="https://www.in.gov/pac/informal/informal-opinions-2024/",
    )

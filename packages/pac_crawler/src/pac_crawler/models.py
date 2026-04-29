from __future__ import annotations

import enum
import re
from datetime import datetime

from pydantic import BaseModel


class OpinionKind(enum.StrEnum):
    FC = "FC"
    INF = "INF"


class DownloadStatus(enum.StrEnum):
    PENDING = "pending"
    OK = "ok"
    HTTP_ERROR = "http_error"
    MISSING = "missing"


OPINION_ID_PATTERN = re.compile(r"(\d{2})-(FC|INF)-(\d+)(A|B)?(\s*REVISED)?")


def two_digit_year_to_four(yy: int) -> int:
    return 2000 + yy if yy < 50 else 1900 + yy


def parse_opinion_id(
    text: str,
) -> tuple[str, int, OpinionKind, str | None, bool] | None:
    """Parse opinion ID from text. Returns (id, year, kind, sub_letter, is_revised) or None."""
    m = OPINION_ID_PATTERN.search(text)
    if m is None:
        return None

    yy_str, kind_str, num_str, sub_letter, revised_str = m.groups()
    year = two_digit_year_to_four(int(yy_str))
    kind = OpinionKind(kind_str)
    is_revised = revised_str is not None

    base_id = f"{yy_str}-{kind_str}-{num_str}"
    if sub_letter:
        base_id += sub_letter

    return base_id, year, kind, sub_letter, is_revised


class ListingEntry(BaseModel, frozen=True):
    """A single opinion entry parsed from a year listing page."""

    opinion_id: str
    year: int
    kind: OpinionKind
    sub_letter: str | None
    is_revised: bool
    raw_title: str
    agency_name: str
    pdf_url: str
    listing_url: str


class OpinionRow(BaseModel, frozen=True):
    """Full manifest row as stored in DuckDB."""

    opinion_id: str
    year: int
    kind: str
    sub_letter: str | None
    revision: int
    parent_opinion_id: str | None
    raw_title: str
    agency_name: str
    listing_url: str
    pdf_url: str
    pdf_path: str | None
    pdf_sha256: str | None
    first_seen_at: datetime
    last_checked_at: datetime
    download_status: str

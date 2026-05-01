from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pytest
from pac_api.db import OpinionDB

_CREATE_OPINIONS = """
CREATE TABLE IF NOT EXISTS opinions (
    opinion_id TEXT PRIMARY KEY,
    year INTEGER NOT NULL,
    kind TEXT NOT NULL,
    sub_letter TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    parent_opinion_id TEXT,
    raw_title TEXT NOT NULL,
    agency_name TEXT NOT NULL,
    listing_url TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    pdf_path TEXT,
    pdf_sha256 TEXT,
    first_seen_at TIMESTAMP NOT NULL,
    last_checked_at TIMESTAMP NOT NULL,
    download_status TEXT NOT NULL DEFAULT 'pending',
    parse_status TEXT DEFAULT 'pending',
    parse_path TEXT,
    parsed_at TIMESTAMP,
    parser_version TEXT,
    extract_status TEXT DEFAULT 'pending',
    embed_status TEXT DEFAULT 'pending'
)
"""

_CREATE_METADATA = """
CREATE TABLE IF NOT EXISTS opinion_metadata (
    opinion_id TEXT PRIMARY KEY,
    issue_date DATE,
    counselor TEXT,
    complainant TEXT,
    public_agency TEXT NOT NULL,
    agency_type TEXT NOT NULL,
    statutes_invoked TEXT[] NOT NULL,
    statute_areas TEXT[] NOT NULL,
    issue_tags TEXT[] NOT NULL,
    tags_need_review BOOLEAN NOT NULL DEFAULT false,
    finding TEXT NOT NULL,
    summary TEXT NOT NULL,
    extracted_at TIMESTAMP NOT NULL,
    extractor_version TEXT NOT NULL
)
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    opinion_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    year INTEGER NOT NULL,
    agency_type TEXT NOT NULL,
    statute_areas TEXT[] NOT NULL,
    finding TEXT NOT NULL,
    issue_tags TEXT[] NOT NULL,
    chunker_version TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
)
"""


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    now = datetime.now(UTC)
    conn = duckdb.connect(str(db_path))
    conn.execute(_CREATE_OPINIONS)
    conn.execute(_CREATE_METADATA)
    conn.execute(_CREATE_CHUNKS)

    test_opinions = [
        (
            "25-FC-118",
            2025,
            "FC",
            "City of Bloomington",
            "violation",
            ["APRA"],
            "https://example.com/25-FC-118.pdf",
        ),
        (
            "25-FC-002A",
            2025,
            "FC",
            "Some School Board",
            "no_violation",
            ["ODL"],
            "https://example.com/25-FC-002A.pdf",
        ),
        (
            "24-INF-01",
            2024,
            "INF",
            "State Agency",
            "moot",
            ["APRA", "ODL"],
            "https://example.com/24-INF-01.pdf",
        ),
    ]
    for oid, year, kind, agency, finding, areas, pdf_url in test_opinions:
        conn.execute(
            """INSERT INTO opinions (opinion_id, year, kind, raw_title, agency_name,
               listing_url, pdf_url, first_seen_at, last_checked_at, download_status,
               parse_status, extract_status, embed_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', 'ok', 'ok', 'ok')""",
            [oid, year, kind, f"{oid}: Test", agency, "https://example.com/", pdf_url, now, now],
        )
        conn.execute(
            """INSERT INTO opinion_metadata (opinion_id, issue_date, counselor, complainant,
               public_agency, agency_type, statutes_invoked, statute_areas, issue_tags,
               tags_need_review, finding, summary, extracted_at, extractor_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, false, ?, ?, ?, '0.4.0')""",
            [
                oid,
                date(2025, 1, 15),
                "Luke H. Britt",
                None,
                agency,
                "city"
                if "City" in agency
                else "school_corporation"
                if "School" in agency
                else "state_agency",
                ["IC 5-14-3-4(b)(8)"],
                areas,
                ["denial-of-access"],
                finding,
                f"Summary for {oid}.",
                now,
            ],
        )

    for i in range(5):
        conn.execute(
            """INSERT INTO chunks (chunk_id, opinion_id, chunk_index, section, page_num,
               text, token_count, year, agency_type, statute_areas, finding, issue_tags,
               chunker_version, created_at)
            VALUES (?, '25-FC-118', ?, 'analysis', 1, 'chunk text', 100, 2025,
               'city', ['APRA'], 'violation', ['denial-of-access'], '0.1.0', ?)""",
            [f"25-FC-118#c{i}", i, now],
        )

    conn.close()
    return db_path


class TestOpinionDB:
    def test_get_opinion_found(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            row = db.get_opinion("25-FC-118")
            assert row is not None
            assert row["opinion_id"] == "25-FC-118"
            assert row["year"] == 2025
            assert row["agency_name"] == "City of Bloomington"
            assert row["pdf_url"] == "https://example.com/25-FC-118.pdf"
        finally:
            db.close()

    def test_get_opinion_not_found(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            row = db.get_opinion("99-FC-999")
            assert row is None
        finally:
            db.close()

    def test_get_opinion_metadata_found(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            meta = db.get_opinion_metadata("25-FC-118")
            assert meta is not None
            assert meta.opinion_id == "25-FC-118"
            assert meta.public_agency == "City of Bloomington"
            assert meta.agency_type == "city"
            assert meta.finding == "violation"
            assert meta.counselor == "Luke H. Britt"
        finally:
            db.close()

    def test_get_opinion_metadata_not_found(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            meta = db.get_opinion_metadata("99-FC-999")
            assert meta is None
        finally:
            db.close()

    def test_list_opinions_no_filters(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            items, total = db.list_opinions()
            assert total == 3
            assert len(items) == 3
        finally:
            db.close()

    def test_list_opinions_filter_by_year(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            items, total = db.list_opinions(year_min=2025)
            assert total == 2
            assert all(i["year"] == 2025 for i in items)
        finally:
            db.close()

    def test_list_opinions_filter_by_agency_type(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            items, total = db.list_opinions(agency_type="city")
            assert total == 1
            assert items[0]["opinion_id"] == "25-FC-118"
        finally:
            db.close()

    def test_list_opinions_filter_by_finding(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            items, total = db.list_opinions(finding="violation")
            assert total == 1
            assert items[0]["finding"] == "violation"
        finally:
            db.close()

    def test_list_opinions_pagination(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            items, total = db.list_opinions(page=1, page_size=2)
            assert total == 3
            assert len(items) == 2

            items2, total2 = db.list_opinions(page=2, page_size=2)
            assert total2 == 3
            assert len(items2) == 1
        finally:
            db.close()

    def test_list_opinions_filter_by_statute_areas(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            _items, total = db.list_opinions(statute_areas=["ODL"])
            assert total == 2
        finally:
            db.close()

    def test_get_chunk_count(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            assert db.get_chunk_count() == 5
        finally:
            db.close()

    def test_get_opinion_count(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            assert db.get_opinion_count() == 3
        finally:
            db.close()

    def test_get_parsed_text_found(self, seeded_db: Path, tmp_path: Path) -> None:
        parsed_dir = tmp_path / "parsed" / "2025"
        parsed_dir.mkdir(parents=True)
        parsed_json = {
            "opinion_id": "25-FC-118",
            "full_text": "The full opinion text here.",
            "pages": [{"page_num": 1, "text": "Page 1 text", "char_count": 11, "ocr_used": False}],
            "sections": {"BACKGROUND": [0, 100], "ANALYSIS": [100, 500]},
            "parser_version": "0.1.0",
        }
        parsed_path = parsed_dir / "25-FC-118.json"
        parsed_path.write_text(json.dumps(parsed_json))

        conn = duckdb.connect(str(seeded_db))
        conn.execute(
            "UPDATE opinions SET parse_path = ? WHERE opinion_id = '25-FC-118'",
            [str(parsed_path)],
        )
        conn.close()

        db = OpinionDB(seeded_db)
        try:
            result = db.get_parsed_text("25-FC-118")
            assert result is not None
            assert result["full_text"] == "The full opinion text here."
            assert "BACKGROUND" in result["sections"]
        finally:
            db.close()

    def test_get_parsed_text_no_parse_path(self, seeded_db: Path) -> None:
        db = OpinionDB(seeded_db)
        try:
            result = db.get_parsed_text("25-FC-118")
            assert result is None
        finally:
            db.close()

    def test_context_manager(self, seeded_db: Path) -> None:
        with OpinionDB(seeded_db) as db:
            assert db.get_opinion_count() == 3

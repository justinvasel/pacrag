from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

CREATE_OPINIONS_TABLE = """
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
    download_status TEXT NOT NULL DEFAULT 'pending'
)
"""

INSERT_OPINION = """
INSERT INTO opinions (
    opinion_id, year, kind, sub_letter, revision, parent_opinion_id,
    raw_title, agency_name, listing_url, pdf_url,
    pdf_path, pdf_sha256, first_seen_at, last_checked_at, download_status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(CREATE_OPINIONS_TABLE)
    conn.close()
    return db_path


@pytest.fixture
def tmp_db_with_opinions(tmp_db: Path) -> Path:
    now = datetime.now(UTC)
    conn = duckdb.connect(str(tmp_db))
    conn.execute(
        INSERT_OPINION,
        [
            "25-FC-118",
            2025,
            "FC",
            None,
            0,
            None,
            "25-FC-118: Alleged Violation of APRA by City of Bloomington",
            "City of Bloomington",
            "https://example.com/2025/",
            "https://example.com/25-FC-118.pdf",
            "data/pdfs/2025/25-FC-118.pdf",
            "abc123",
            now,
            now,
            "ok",
        ],
    )
    conn.execute(
        INSERT_OPINION,
        [
            "25-FC-002A",
            2025,
            "FC",
            "A",
            0,
            "25-FC-002",
            "25-FC-002A: Some complaint",
            "Some complaint",
            "https://example.com/2025/",
            "https://example.com/25-FC-002A.pdf",
            "data/pdfs/2025/25-FC-002A.pdf",
            "def456",
            now,
            now,
            "ok",
        ],
    )
    conn.execute(
        INSERT_OPINION,
        [
            "24-INF-01",
            2024,
            "INF",
            None,
            0,
            None,
            "24-INF-01: Informal inquiry",
            "Informal inquiry",
            "https://example.com/2024/",
            "https://example.com/24-INF-01.pdf",
            None,
            None,
            now,
            now,
            "pending",
        ],
    )
    conn.close()
    return tmp_db


@pytest.fixture
def tmp_db_with_parsed_opinions(tmp_db_with_opinions: Path) -> Path:
    now = datetime.now(UTC)
    conn = duckdb.connect(str(tmp_db_with_opinions))
    conn.execute(
        "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parse_status TEXT DEFAULT 'pending'"
    )
    conn.execute("ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parse_path TEXT")
    conn.execute("ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMP")
    conn.execute("ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parser_version TEXT")
    conn.execute(
        """
        UPDATE opinions
        SET parse_status = 'ok',
            parse_path = 'data/parsed/' || year || '/' || opinion_id || '.json',
            parsed_at = ?,
            parser_version = '0.1.0'
        WHERE download_status = 'ok'
        """,
        [now],
    )
    conn.close()
    return tmp_db_with_opinions

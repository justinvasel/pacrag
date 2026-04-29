from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import duckdb
import structlog

from pac_crawler.models import ListingEntry, OpinionRow

log = structlog.get_logger()

CREATE_TABLE_SQL = """
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

UPSERT_SQL = """
INSERT INTO opinions (
    opinion_id, year, kind, sub_letter, revision, parent_opinion_id,
    raw_title, agency_name, listing_url, pdf_url,
    first_seen_at, last_checked_at, download_status
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
ON CONFLICT (opinion_id) DO UPDATE SET
    last_checked_at = excluded.last_checked_at,
    raw_title = excluded.raw_title,
    pdf_url = excluded.pdf_url,
    revision = CASE WHEN excluded.revision > opinions.revision
                    THEN excluded.revision ELSE opinions.revision END
"""

_SUB_LETTER_RE = re.compile(r"^(.+?)([AB])$")


def _compute_revision_and_parent(
    opinion_id: str,
    sub_letter: str | None,
    is_revised: bool,
) -> tuple[int, str | None]:
    revision = 1 if is_revised else 0

    parent: str | None = None
    if sub_letter:
        m = _SUB_LETTER_RE.match(opinion_id)
        if m:
            parent = m.group(1)

    return revision, parent


class ManifestDB:
    """Manages the DuckDB opinions manifest. Single-writer, not thread-safe."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        self.ensure_table()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def ensure_table(self) -> None:
        self._conn.execute(CREATE_TABLE_SQL)

    def upsert_listing(self, entry: ListingEntry) -> None:
        now = datetime.now(UTC)
        revision, parent = _compute_revision_and_parent(
            entry.opinion_id, entry.sub_letter, entry.is_revised
        )
        self._conn.execute(
            UPSERT_SQL,
            [
                entry.opinion_id,
                entry.year,
                entry.kind.value,
                entry.sub_letter,
                revision,
                parent,
                entry.raw_title,
                entry.agency_name,
                entry.listing_url,
                entry.pdf_url,
                now,
                now,
            ],
        )

    def upsert_listings(self, entries: list[ListingEntry]) -> None:
        for entry in entries:
            self.upsert_listing(entry)

    def update_download(
        self,
        opinion_id: str,
        pdf_path: str,
        pdf_sha256: str,
        status: str,
    ) -> None:
        self._conn.execute(
            """
            UPDATE opinions
            SET pdf_path = ?, pdf_sha256 = ?, download_status = ?
            WHERE opinion_id = ?
            """,
            [pdf_path, pdf_sha256, status, opinion_id],
        )

    def mark_missing(self, seen_ids: set[str], kind: str) -> int:
        if not seen_ids:
            result = self._conn.execute(
                """
                UPDATE opinions
                SET download_status = 'missing'
                WHERE kind = ? AND download_status != 'missing'
                RETURNING opinion_id
                """,
                [kind],
            ).fetchall()
        else:
            placeholders = ", ".join("?" for _ in seen_ids)
            result = self._conn.execute(
                f"""
                UPDATE opinions
                SET download_status = 'missing'
                WHERE kind = ? AND download_status != 'missing'
                  AND opinion_id NOT IN ({placeholders})
                RETURNING opinion_id
                """,
                [kind, *seen_ids],
            ).fetchall()

        count = len(result)
        if count > 0:
            log.info("marked_missing", kind=kind, count=count)
        return count

    def get_existing_hashes(self) -> dict[str, str | None]:
        rows = self._conn.execute("SELECT opinion_id, pdf_sha256 FROM opinions").fetchall()
        return {row[0]: row[1] for row in rows}

    def get_pending_downloads(self) -> list[OpinionRow]:
        rows = self._conn.execute(
            "SELECT * FROM opinions WHERE download_status = 'pending' ORDER BY opinion_id"
        ).fetchall()
        columns = [
            "opinion_id",
            "year",
            "kind",
            "sub_letter",
            "revision",
            "parent_opinion_id",
            "raw_title",
            "agency_name",
            "listing_url",
            "pdf_url",
            "pdf_path",
            "pdf_sha256",
            "first_seen_at",
            "last_checked_at",
            "download_status",
        ]
        return [OpinionRow(**dict(zip(columns, row, strict=True))) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT download_status, count(*) FROM opinions GROUP BY download_status"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

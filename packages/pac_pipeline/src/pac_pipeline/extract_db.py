from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import duckdb
import structlog

from pac_pipeline.extract_models import OpinionMetadata

log = structlog.get_logger()

_CREATE_METADATA_TABLE = """
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

_CREATE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS opinion_metadata_history (
    id INTEGER PRIMARY KEY DEFAULT nextval('opinion_metadata_history_seq'),
    opinion_id TEXT NOT NULL,
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

_ALTER_SQLS: list[str] = [
    "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS extract_status TEXT DEFAULT 'pending'",
    "ALTER TABLE opinion_metadata ADD COLUMN IF NOT EXISTS tags_need_review BOOLEAN DEFAULT false",
    (
        "ALTER TABLE opinion_metadata_history"
        " ADD COLUMN IF NOT EXISTS tags_need_review BOOLEAN DEFAULT false"
    ),
]

_UPSERT_METADATA = """
INSERT INTO opinion_metadata (
    opinion_id, issue_date, counselor, complainant, public_agency,
    agency_type, statutes_invoked, statute_areas, issue_tags,
    tags_need_review, finding, summary, extracted_at, extractor_version
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (opinion_id) DO UPDATE SET
    issue_date = excluded.issue_date,
    counselor = excluded.counselor,
    complainant = excluded.complainant,
    public_agency = excluded.public_agency,
    agency_type = excluded.agency_type,
    statutes_invoked = excluded.statutes_invoked,
    statute_areas = excluded.statute_areas,
    issue_tags = excluded.issue_tags,
    tags_need_review = excluded.tags_need_review,
    finding = excluded.finding,
    summary = excluded.summary,
    extracted_at = excluded.extracted_at,
    extractor_version = excluded.extractor_version
"""

_COPY_TO_HISTORY = """
INSERT INTO opinion_metadata_history (
    opinion_id, issue_date, counselor, complainant, public_agency,
    agency_type, statutes_invoked, statute_areas, issue_tags,
    tags_need_review, finding, summary, extracted_at, extractor_version
)
SELECT
    opinion_id, issue_date, counselor, complainant, public_agency,
    agency_type, statutes_invoked, statute_areas, issue_tags,
    tags_need_review, finding, summary, extracted_at, extractor_version
FROM opinion_metadata
WHERE opinion_id = ?
"""


class ExtractDB:
    def __init__(self, db_path: Path) -> None:
        self._conn = duckdb.connect(str(db_path))
        self._ensure_tables()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_tables(self) -> None:
        self._conn.execute("CREATE SEQUENCE IF NOT EXISTS opinion_metadata_history_seq START 1")
        self._conn.execute(_CREATE_METADATA_TABLE)
        self._conn.execute(_CREATE_HISTORY_TABLE)
        for sql in _ALTER_SQLS:
            self._conn.execute(sql)

    def get_extractable_opinions(
        self,
        *,
        opinion_id: str | None = None,
        reextract: bool = False,
        limit: int | None = None,
        extractor_version: str | None = None,
    ) -> list[tuple[str, int, str]]:
        conditions = ["parse_status = 'ok'", "parse_path IS NOT NULL"]
        params: list[object] = []

        if not reextract:
            if extractor_version:
                conditions.append(
                    "(extract_status IS NULL OR extract_status = 'pending'"
                    " OR (extract_status = 'ok'"
                    " AND opinion_id IN ("
                    "   SELECT om.opinion_id FROM opinion_metadata om"
                    "   WHERE om.extractor_version != ?"
                    " )))"
                )
                params.append(extractor_version)
            else:
                conditions.append("(extract_status IS NULL OR extract_status = 'pending')")

        if opinion_id is not None:
            conditions.append("opinion_id = ?")
            params.append(opinion_id)

        where = " AND ".join(conditions)
        query = (
            f"SELECT opinion_id, year, parse_path FROM opinions"
            f" WHERE {where} ORDER BY year DESC, opinion_id"
        )

        if limit is not None:
            query += f" LIMIT {limit}"

        rows = self._conn.execute(query, params).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def upsert_metadata(
        self,
        metadata: OpinionMetadata,
        *,
        extractor_version: str,
    ) -> None:
        now = datetime.now(UTC)

        existing = self._conn.execute(
            "SELECT 1 FROM opinion_metadata WHERE opinion_id = ?",
            [metadata.opinion_id],
        ).fetchone()
        if existing:
            self._conn.execute(_COPY_TO_HISTORY, [metadata.opinion_id])

        self._conn.execute(
            _UPSERT_METADATA,
            [
                metadata.opinion_id,
                metadata.issue_date,
                metadata.counselor,
                metadata.complainant,
                metadata.public_agency,
                metadata.agency_type,
                list(metadata.statutes_invoked),
                list(metadata.statute_areas),
                list(metadata.issue_tags),
                metadata.tags_need_review,
                metadata.finding,
                metadata.summary,
                now,
                extractor_version,
            ],
        )

    def update_extract_status(self, opinion_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE opinions SET extract_status = ? WHERE opinion_id = ?",
            [status, opinion_id],
        )

    def count_by_extract_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT extract_status, count(*) FROM opinions GROUP BY extract_status"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

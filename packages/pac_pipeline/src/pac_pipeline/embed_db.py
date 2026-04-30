from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import duckdb
import structlog

from pac_pipeline.chunk_models import Chunk
from pac_pipeline.extract_models import OpinionMetadata

log = structlog.get_logger()

_CREATE_CHUNKS_TABLE = """
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

_ALTER_SQLS: list[str] = [
    "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS embed_status TEXT DEFAULT 'pending'",
]

_INSERT_CHUNK = """
INSERT INTO chunks (
    chunk_id, opinion_id, chunk_index, section, page_num,
    text, token_count, year, agency_type, statute_areas,
    finding, issue_tags, chunker_version, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class EmbedDB:
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
        self._conn.execute(_CREATE_CHUNKS_TABLE)
        for sql in _ALTER_SQLS:
            self._conn.execute(sql)

    def get_embeddable_opinions(
        self,
        *,
        opinion_id: str | None = None,
        reembed: bool = False,
        limit: int | None = None,
        chunker_version: str | None = None,
    ) -> list[tuple[str, int, str]]:
        conditions = [
            "parse_status = 'ok'",
            "parse_path IS NOT NULL",
            "extract_status = 'ok'",
        ]
        params: list[object] = []

        if not reembed:
            if chunker_version:
                conditions.append(
                    "(embed_status IS NULL OR embed_status = 'pending'"
                    " OR (embed_status = 'ok'"
                    " AND opinion_id IN ("
                    "   SELECT c.opinion_id FROM chunks c"
                    "   WHERE c.chunker_version != ?"
                    " )))"
                )
                params.append(chunker_version)
            else:
                conditions.append("(embed_status IS NULL OR embed_status = 'pending')")

        if opinion_id is not None:
            conditions.append("opinion_id = ?")
            params.append(opinion_id)

        where = " AND ".join(conditions)
        query = (
            f"SELECT opinion_id, year, parse_path FROM opinions"
            f" WHERE {where} ORDER BY year DESC, opinion_id"
        )

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        *,
        chunker_version: str,
    ) -> None:
        if not chunks:
            return
        opinion_id = chunks[0].opinion_id
        self.delete_chunks_for_opinion(opinion_id)

        now = datetime.now(UTC)
        for chunk in chunks:
            self._conn.execute(
                _INSERT_CHUNK,
                [
                    chunk.chunk_id,
                    chunk.opinion_id,
                    chunk.chunk_index,
                    chunk.section,
                    chunk.page_num,
                    chunk.text,
                    chunk.token_count,
                    chunk.year,
                    chunk.agency_type,
                    list(chunk.statute_areas),
                    chunk.finding,
                    list(chunk.issue_tags),
                    chunker_version,
                    now,
                ],
            )

    def delete_chunks_for_opinion(self, opinion_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE opinion_id = ?", [opinion_id])

    def update_embed_status(self, opinion_id: str, status: str) -> None:
        self._conn.execute(
            "UPDATE opinions SET embed_status = ? WHERE opinion_id = ?",
            [status, opinion_id],
        )

    def count_by_embed_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT embed_status, count(*) FROM opinions GROUP BY embed_status"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_chunk_count(self) -> int:
        rows = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchall()
        return rows[0][0]

    def get_metadata_for_opinion(self, opinion_id: str) -> OpinionMetadata | None:
        row = self._conn.execute(
            """
            SELECT opinion_id, issue_date, counselor, complainant, public_agency,
                   agency_type, statutes_invoked, statute_areas, issue_tags,
                   tags_need_review, finding, summary
            FROM opinion_metadata WHERE opinion_id = ?
            """,
            [opinion_id],
        ).fetchone()
        if row is None:
            return None
        return OpinionMetadata(
            opinion_id=row[0],
            issue_date=row[1],
            counselor=row[2],
            complainant=row[3],
            public_agency=row[4],
            agency_type=row[5],
            statutes_invoked=list(row[6]),
            statute_areas=list(row[7]),
            issue_tags=list(row[8]),
            tags_need_review=row[9],
            finding=row[10],
            summary=row[11],
        )

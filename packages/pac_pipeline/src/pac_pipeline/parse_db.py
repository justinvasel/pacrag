from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import duckdb
import structlog

log = structlog.get_logger()

_ALTER_SQLS: list[str] = [
    "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parse_status TEXT DEFAULT 'pending'",
    "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parse_path TEXT",
    "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMP",
    "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parser_version TEXT",
]


class ParseDB:
    """Manages parse-related columns on the opinions table."""

    def __init__(self, db_path: Path) -> None:
        self._conn = duckdb.connect(str(db_path))
        self._ensure_columns()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_columns(self) -> None:
        for sql in _ALTER_SQLS:
            self._conn.execute(sql)

    def get_parseable_opinions(
        self,
        *,
        year: int | None = None,
        opinion_id: str | None = None,
        reparse: bool = False,
        parser_version: str | None = None,
    ) -> list[tuple[str, int, str]]:
        conditions = ["download_status = 'ok'", "pdf_path IS NOT NULL"]
        params: list[object] = []

        if not reparse:
            if parser_version:
                conditions.append(
                    "(parse_status IS NULL OR parse_status = 'pending'"
                    " OR (parse_status = 'ok' AND parser_version != ?))"
                )
                params.append(parser_version)
            else:
                conditions.append("(parse_status IS NULL OR parse_status = 'pending')")

        if year is not None:
            conditions.append("year = ?")
            params.append(year)

        if opinion_id is not None:
            conditions.append("opinion_id = ?")
            params.append(opinion_id)

        where = " AND ".join(conditions)
        query = (
            f"SELECT opinion_id, year, pdf_path FROM opinions"
            f" WHERE {where} ORDER BY year DESC, opinion_id"
        )

        rows = self._conn.execute(query, params).fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    def update_parse_status(
        self,
        opinion_id: str,
        *,
        parse_status: str,
        parse_path: str | None = None,
        parser_version: str | None = None,
    ) -> None:
        now = datetime.now(UTC)
        self._conn.execute(
            """
            UPDATE opinions
            SET parse_status = ?, parse_path = ?, parsed_at = ?, parser_version = ?
            WHERE opinion_id = ?
            """,
            [parse_status, parse_path, now, parser_version, opinion_id],
        )

    def count_by_parse_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT parse_status, count(*) FROM opinions GROUP BY parse_status"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

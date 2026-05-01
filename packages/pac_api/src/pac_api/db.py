from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Self

import duckdb
import structlog
from pac_pipeline.extract_models import OpinionMetadata

log = structlog.get_logger()


class OpinionDB:
    def __init__(self, db_path: Path) -> None:
        self._conn = duckdb.connect(str(db_path), read_only=True)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_opinion(self, opinion_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM opinions WHERE opinion_id = ?", [opinion_id]
        ).fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in self._conn.description]
        return dict(zip(columns, row, strict=True))

    def get_opinion_metadata(self, opinion_id: str) -> OpinionMetadata | None:
        row = self._conn.execute(
            """SELECT opinion_id, issue_date, counselor, complainant, public_agency,
                      agency_type, statutes_invoked, statute_areas, issue_tags,
                      tags_need_review, finding, summary
               FROM opinion_metadata WHERE opinion_id = ?""",
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

    def list_opinions(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        year_min: int | None = None,
        year_max: int | None = None,
        agency_type: str | None = None,
        finding: str | None = None,
        statute_areas: list[str] | None = None,
        issue_tags: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        conditions: list[str] = []
        params: list[object] = []

        if year_min is not None:
            conditions.append("o.year >= ?")
            params.append(year_min)
        if year_max is not None:
            conditions.append("o.year <= ?")
            params.append(year_max)
        if agency_type is not None:
            conditions.append("m.agency_type = ?")
            params.append(agency_type)
        if finding is not None:
            conditions.append("m.finding = ?")
            params.append(finding)
        if statute_areas:
            area_conditions = " OR ".join(
                "list_contains(m.statute_areas, ?)" for _ in statute_areas
            )
            conditions.append(f"({area_conditions})")
            params.extend(statute_areas)
        if issue_tags:
            tag_conditions = " OR ".join("list_contains(m.issue_tags, ?)" for _ in issue_tags)
            conditions.append(f"({tag_conditions})")
            params.extend(issue_tags)

        where = " AND ".join(conditions)
        where_clause = f"WHERE {where}" if where else ""

        count_query = f"""
            SELECT COUNT(*) FROM opinions o
            JOIN opinion_metadata m ON o.opinion_id = m.opinion_id
            {where_clause}
        """
        count_row = self._conn.execute(count_query, params).fetchone()
        total: int = count_row[0] if count_row else 0

        offset = (page - 1) * page_size
        data_query = f"""
            SELECT o.opinion_id, o.year, o.agency_name,
                   m.agency_type, m.finding, m.statute_areas,
                   m.summary, m.counselor, m.issue_date
            FROM opinions o
            JOIN opinion_metadata m ON o.opinion_id = m.opinion_id
            {where_clause}
            ORDER BY o.year DESC, o.opinion_id
            LIMIT ? OFFSET ?
        """
        rows = self._conn.execute(data_query, [*params, page_size, offset]).fetchall()

        items = [
            {
                "opinion_id": r[0],
                "year": r[1],
                "agency_name": r[2],
                "agency_type": r[3],
                "finding": r[4],
                "statute_areas": list(r[5]),
                "summary": r[6],
                "counselor": r[7],
                "issue_date": r[8],
            }
            for r in rows
        ]
        return items, total

    def get_chunk_count(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return result[0] if result else 0

    def get_opinion_count(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM opinions").fetchone()
        return result[0] if result else 0

    def get_parsed_text(self, opinion_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT parse_path FROM opinions WHERE opinion_id = ?", [opinion_id]
        ).fetchone()
        if row is None or row[0] is None:
            return None

        parse_path = Path(row[0])
        if not parse_path.exists():
            log.warning("parsed_file_missing", opinion_id=opinion_id, path=str(parse_path))
            return None

        data: dict[str, Any] = json.loads(parse_path.read_text())
        sections_raw = data.get("sections", {})
        sections = {
            name: data.get("full_text", "")[start:end]
            for name, (start, end) in sections_raw.items()
        }
        return {
            "full_text": data.get("full_text", ""),
            "sections": sections,
        }

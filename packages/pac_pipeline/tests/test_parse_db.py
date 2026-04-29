from __future__ import annotations

from pathlib import Path

import duckdb
from pac_pipeline.parse_db import ParseDB


class TestEnsureColumns:
    def test_adds_columns(self, tmp_db: Path) -> None:
        with ParseDB(tmp_db):
            conn = duckdb.connect(str(tmp_db))
            cols = conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'opinions'"
            ).fetchall()
            col_names = {c[0] for c in cols}
            conn.close()

        assert "parse_status" in col_names
        assert "parse_path" in col_names
        assert "parsed_at" in col_names
        assert "parser_version" in col_names

    def test_idempotent(self, tmp_db: Path) -> None:
        with ParseDB(tmp_db):
            pass
        with ParseDB(tmp_db):
            pass

    def test_default_parse_status(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions):
            conn = duckdb.connect(str(tmp_db_with_opinions))
            rows = conn.execute(
                "SELECT opinion_id, parse_status FROM opinions ORDER BY opinion_id"
            ).fetchall()
            conn.close()
        for _oid, status in rows:
            assert status == "pending"


class TestGetParseableOpinions:
    def test_returns_downloaded_pending_parse(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            results = db.get_parseable_opinions()
        opinion_ids = [r[0] for r in results]
        assert "25-FC-118" in opinion_ids
        assert "25-FC-002A" in opinion_ids

    def test_excludes_not_downloaded(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            results = db.get_parseable_opinions()
        opinion_ids = [r[0] for r in results]
        assert "24-INF-01" not in opinion_ids

    def test_excludes_already_parsed(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            db.update_parse_status(
                "25-FC-118",
                parse_status="ok",
                parse_path="data/parsed/2025/25-FC-118.json",
                parser_version="0.1.0",
            )
            results = db.get_parseable_opinions()
        opinion_ids = [r[0] for r in results]
        assert "25-FC-118" not in opinion_ids

    def test_includes_already_parsed_when_reparse(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            db.update_parse_status(
                "25-FC-118",
                parse_status="ok",
                parse_path="data/parsed/2025/25-FC-118.json",
                parser_version="0.1.0",
            )
            results = db.get_parseable_opinions(reparse=True)
        opinion_ids = [r[0] for r in results]
        assert "25-FC-118" in opinion_ids

    def test_filters_by_year(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            results = db.get_parseable_opinions(year=2025)
        opinion_ids = [r[0] for r in results]
        assert "25-FC-118" in opinion_ids
        assert all(r[1] == 2025 for r in results)

    def test_filters_by_opinion_id(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            results = db.get_parseable_opinions(opinion_id="25-FC-118")
        assert len(results) == 1
        assert results[0][0] == "25-FC-118"

    def test_old_parser_version_included(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            db.update_parse_status(
                "25-FC-118",
                parse_status="ok",
                parse_path="data/parsed/2025/25-FC-118.json",
                parser_version="0.0.1",
            )
            results = db.get_parseable_opinions(parser_version="0.1.0")
        opinion_ids = [r[0] for r in results]
        assert "25-FC-118" in opinion_ids

    def test_returns_year_and_pdf_path(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            results = db.get_parseable_opinions(opinion_id="25-FC-118")
        assert results[0] == ("25-FC-118", 2025, "data/pdfs/2025/25-FC-118.pdf")


class TestUpdateParseStatus:
    def test_sets_ok(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            db.update_parse_status(
                "25-FC-118",
                parse_status="ok",
                parse_path="data/parsed/2025/25-FC-118.json",
                parser_version="0.1.0",
            )

        conn = duckdb.connect(str(tmp_db_with_opinions))
        row = conn.execute(
            "SELECT parse_status, parse_path, parser_version FROM opinions WHERE opinion_id = ?",
            ["25-FC-118"],
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "ok"
        assert row[1] == "data/parsed/2025/25-FC-118.json"
        assert row[2] == "0.1.0"

    def test_sets_error(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            db.update_parse_status("25-FC-118", parse_status="error")

        conn = duckdb.connect(str(tmp_db_with_opinions))
        row = conn.execute(
            "SELECT parse_status FROM opinions WHERE opinion_id = ?",
            ["25-FC-118"],
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "error"

    def test_updates_parsed_at(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            db.update_parse_status(
                "25-FC-118",
                parse_status="ok",
                parse_path="path",
                parser_version="0.1.0",
            )

        conn = duckdb.connect(str(tmp_db_with_opinions))
        row = conn.execute(
            "SELECT parsed_at FROM opinions WHERE opinion_id = ?",
            ["25-FC-118"],
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] is not None


class TestCountByParseStatus:
    def test_counts(self, tmp_db_with_opinions: Path) -> None:
        with ParseDB(tmp_db_with_opinions) as db:
            db.update_parse_status(
                "25-FC-118",
                parse_status="ok",
                parse_path="p",
                parser_version="v",
            )
            counts = db.count_by_parse_status()
        assert counts.get("ok", 0) == 1
        assert counts.get("pending", 0) >= 1

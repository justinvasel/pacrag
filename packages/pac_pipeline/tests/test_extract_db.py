from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pac_pipeline.extract_db import ExtractDB
from pac_pipeline.extract_models import OpinionMetadata


@pytest.fixture
def extract_db(tmp_db_with_parsed_opinions: Path) -> ExtractDB:
    with ExtractDB(tmp_db_with_parsed_opinions) as db:
        yield db


class TestTableCreation:
    def test_creates_opinion_metadata_table(self, extract_db: ExtractDB) -> None:
        rows = extract_db._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'opinion_metadata'"
        ).fetchall()
        assert len(rows) == 1

    def test_creates_history_table(self, extract_db: ExtractDB) -> None:
        rows = extract_db._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'opinion_metadata_history'"
        ).fetchall()
        assert len(rows) == 1

    def test_adds_extract_status_column(self, extract_db: ExtractDB) -> None:
        rows = extract_db._conn.execute("SELECT extract_status FROM opinions LIMIT 1").fetchall()
        assert rows[0][0] == "pending"

    def test_idempotent(self, tmp_db_with_parsed_opinions: Path) -> None:
        with ExtractDB(tmp_db_with_parsed_opinions):
            pass
        with ExtractDB(tmp_db_with_parsed_opinions):
            pass


class TestGetExtractableOpinions:
    def test_returns_parsed_opinions(self, extract_db: ExtractDB) -> None:
        opinions = extract_db.get_extractable_opinions()
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" in opinion_ids
        assert "25-FC-002A" in opinion_ids

    def test_excludes_unparsed(self, extract_db: ExtractDB) -> None:
        opinions = extract_db.get_extractable_opinions()
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "24-INF-01" not in opinion_ids

    def test_excludes_already_extracted(self, extract_db: ExtractDB) -> None:
        extract_db._conn.execute(
            "UPDATE opinions SET extract_status = 'ok' WHERE opinion_id = '25-FC-118'"
        )
        opinions = extract_db.get_extractable_opinions()
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" not in opinion_ids
        assert "25-FC-002A" in opinion_ids

    def test_reextract_includes_already_extracted(self, extract_db: ExtractDB) -> None:
        extract_db._conn.execute(
            "UPDATE opinions SET extract_status = 'ok' WHERE opinion_id = '25-FC-118'"
        )
        opinions = extract_db.get_extractable_opinions(reextract=True)
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" in opinion_ids

    def test_filters_by_opinion_id(self, extract_db: ExtractDB) -> None:
        opinions = extract_db.get_extractable_opinions(opinion_id="25-FC-118")
        assert len(opinions) == 1
        assert opinions[0][0] == "25-FC-118"

    def test_respects_limit(self, extract_db: ExtractDB) -> None:
        opinions = extract_db.get_extractable_opinions(limit=1)
        assert len(opinions) == 1

    def test_returns_opinion_id_year_parse_path(self, extract_db: ExtractDB) -> None:
        opinions = extract_db.get_extractable_opinions(opinion_id="25-FC-118")
        oid, year, parse_path = opinions[0]
        assert oid == "25-FC-118"
        assert year == 2025
        assert "25-FC-118.json" in parse_path

    def test_version_based_reextract(self, extract_db: ExtractDB) -> None:
        extract_db._conn.execute(
            "UPDATE opinions SET extract_status = 'ok' WHERE opinion_id = '25-FC-118'"
        )
        extract_db.upsert_metadata(
            _make_metadata("25-FC-118"),
            extractor_version="0.0.1",
        )
        opinions = extract_db.get_extractable_opinions(extractor_version="0.1.0")
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" in opinion_ids


class TestUpsertMetadata:
    def test_inserts_new_metadata(self, extract_db: ExtractDB) -> None:
        meta = _make_metadata("25-FC-118")
        extract_db.upsert_metadata(meta, extractor_version="0.1.0")

        rows = extract_db._conn.execute(
            "SELECT opinion_id, public_agency, finding FROM opinion_metadata "
            "WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "City of Bloomington"
        assert rows[0][2] == "violation"

    def test_stores_arrays(self, extract_db: ExtractDB) -> None:
        meta = _make_metadata("25-FC-118")
        extract_db.upsert_metadata(meta, extractor_version="0.1.0")

        rows = extract_db._conn.execute(
            "SELECT statutes_invoked, statute_areas, issue_tags FROM opinion_metadata "
            "WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] == ["IC 5-14-3-4(b)(20)"]
        assert rows[0][1] == ["APRA"]
        assert rows[0][2] == ["denial-of-access"]

    def test_stores_tags_need_review(self, extract_db: ExtractDB) -> None:
        meta = OpinionMetadata(
            opinion_id="25-FC-118",
            issue_date=date(2026, 2, 2),
            counselor="Jennifer G. Ruby",
            complainant="David Askins",
            public_agency="City of Bloomington",
            agency_type="city",
            statutes_invoked=[],
            statute_areas=["APRA"],
            issue_tags=[],
            tags_need_review=True,
            finding="violation",
            summary="Summary.",
        )
        extract_db.upsert_metadata(meta, extractor_version="0.1.0")

        rows = extract_db._conn.execute(
            "SELECT tags_need_review FROM opinion_metadata WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] is True

    def test_overwrites_existing(self, extract_db: ExtractDB) -> None:
        meta1 = _make_metadata("25-FC-118", finding="violation")
        extract_db.upsert_metadata(meta1, extractor_version="0.1.0")

        meta2 = _make_metadata("25-FC-118", finding="no_violation")
        extract_db.upsert_metadata(meta2, extractor_version="0.2.0")

        rows = extract_db._conn.execute(
            "SELECT finding, extractor_version FROM opinion_metadata "
            "WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "no_violation"
        assert rows[0][1] == "0.2.0"

    def test_copies_old_row_to_history(self, extract_db: ExtractDB) -> None:
        meta1 = _make_metadata("25-FC-118", finding="violation")
        extract_db.upsert_metadata(meta1, extractor_version="0.1.0")

        meta2 = _make_metadata("25-FC-118", finding="no_violation")
        extract_db.upsert_metadata(meta2, extractor_version="0.2.0")

        history = extract_db._conn.execute(
            "SELECT opinion_id, finding, extractor_version FROM opinion_metadata_history "
            "WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert len(history) == 1
        assert history[0][1] == "violation"
        assert history[0][2] == "0.1.0"


class TestUpdateExtractStatus:
    def test_updates_status(self, extract_db: ExtractDB) -> None:
        extract_db.update_extract_status("25-FC-118", "ok")
        rows = extract_db._conn.execute(
            "SELECT extract_status FROM opinions WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] == "ok"

    def test_updates_error_status(self, extract_db: ExtractDB) -> None:
        extract_db.update_extract_status("25-FC-118", "error")
        rows = extract_db._conn.execute(
            "SELECT extract_status FROM opinions WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] == "error"


class TestCountByExtractStatus:
    def test_counts(self, extract_db: ExtractDB) -> None:
        counts = extract_db.count_by_extract_status()
        assert counts.get("pending", 0) >= 2

    def test_reflects_updates(self, extract_db: ExtractDB) -> None:
        extract_db.update_extract_status("25-FC-118", "ok")
        counts = extract_db.count_by_extract_status()
        assert counts.get("ok", 0) == 1


def _make_metadata(
    opinion_id: str,
    *,
    finding: str = "violation",
) -> OpinionMetadata:
    return OpinionMetadata(
        opinion_id=opinion_id,
        issue_date=date(2026, 2, 2),
        counselor="Jennifer G. Ruby",
        complainant="David Askins",
        public_agency="City of Bloomington",
        agency_type="city",
        statutes_invoked=["IC 5-14-3-4(b)(20)"],
        statute_areas=["APRA"],
        issue_tags=["denial-of-access"],
        tags_need_review=False,
        finding=finding,  # type: ignore[arg-type]
        summary="The PAC found a violation of APRA.",
    )

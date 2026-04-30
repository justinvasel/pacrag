from __future__ import annotations

from pathlib import Path

import pytest
from pac_pipeline.chunk_models import Chunk
from pac_pipeline.embed_db import EmbedDB


@pytest.fixture
def embed_db(tmp_db_with_extracted_opinions: Path) -> EmbedDB:
    with EmbedDB(tmp_db_with_extracted_opinions) as db:
        yield db


def _make_chunks(opinion_id: str = "25-FC-118", count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{opinion_id}#c{i}",
            opinion_id=opinion_id,
            chunk_index=i,
            section="card" if i == 0 else "analysis",
            page_num=1,
            text=f"Chunk {i} text for {opinion_id}.",
            token_count=6,
            year=2025,
            agency_type="city",
            statute_areas=["APRA"],
            finding="violation",
            issue_tags=["denial-of-access"],
        )
        for i in range(count)
    ]


class TestTableCreation:
    def test_creates_chunks_table(self, embed_db: EmbedDB) -> None:
        rows = embed_db._conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'chunks'"
        ).fetchall()
        assert len(rows) == 1

    def test_adds_embed_status_column(self, embed_db: EmbedDB) -> None:
        rows = embed_db._conn.execute("SELECT embed_status FROM opinions LIMIT 1").fetchall()
        assert rows[0][0] == "pending"

    def test_idempotent(self, tmp_db_with_extracted_opinions: Path) -> None:
        with EmbedDB(tmp_db_with_extracted_opinions):
            pass
        with EmbedDB(tmp_db_with_extracted_opinions):
            pass


class TestGetEmbeddableOpinions:
    def test_returns_extracted_opinions(self, embed_db: EmbedDB) -> None:
        opinions = embed_db.get_embeddable_opinions()
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" in opinion_ids
        assert "25-FC-002A" in opinion_ids

    def test_excludes_unextracted(self, embed_db: EmbedDB) -> None:
        opinions = embed_db.get_embeddable_opinions()
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "24-INF-01" not in opinion_ids

    def test_excludes_already_embedded(self, embed_db: EmbedDB) -> None:
        embed_db._conn.execute(
            "UPDATE opinions SET embed_status = 'ok' WHERE opinion_id = '25-FC-118'"
        )
        opinions = embed_db.get_embeddable_opinions()
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" not in opinion_ids
        assert "25-FC-002A" in opinion_ids

    def test_reembed_includes_already_embedded(self, embed_db: EmbedDB) -> None:
        embed_db._conn.execute(
            "UPDATE opinions SET embed_status = 'ok' WHERE opinion_id = '25-FC-118'"
        )
        opinions = embed_db.get_embeddable_opinions(reembed=True)
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" in opinion_ids

    def test_filters_by_opinion_id(self, embed_db: EmbedDB) -> None:
        opinions = embed_db.get_embeddable_opinions(opinion_id="25-FC-118")
        assert len(opinions) == 1
        assert opinions[0][0] == "25-FC-118"

    def test_respects_limit(self, embed_db: EmbedDB) -> None:
        opinions = embed_db.get_embeddable_opinions(limit=1)
        assert len(opinions) == 1

    def test_returns_opinion_id_year_parse_path(self, embed_db: EmbedDB) -> None:
        opinions = embed_db.get_embeddable_opinions(opinion_id="25-FC-118")
        oid, year, parse_path = opinions[0]
        assert oid == "25-FC-118"
        assert year == 2025
        assert "25-FC-118.json" in parse_path

    def test_version_based_reembed(self, embed_db: EmbedDB) -> None:
        embed_db._conn.execute(
            "UPDATE opinions SET embed_status = 'ok' WHERE opinion_id = '25-FC-118'"
        )
        embed_db.upsert_chunks(_make_chunks(), chunker_version="0.0.1")
        opinions = embed_db.get_embeddable_opinions(chunker_version="0.1.0")
        opinion_ids = [oid for oid, _, _ in opinions]
        assert "25-FC-118" in opinion_ids


class TestUpsertChunks:
    def test_inserts_chunks(self, embed_db: EmbedDB) -> None:
        chunks = _make_chunks()
        embed_db.upsert_chunks(chunks, chunker_version="0.1.0")
        rows = embed_db._conn.execute(
            "SELECT chunk_id, opinion_id, section, token_count FROM chunks ORDER BY chunk_index"
        ).fetchall()
        assert len(rows) == 3
        assert rows[0][0] == "25-FC-118#c0"
        assert rows[0][2] == "card"

    def test_stores_arrays(self, embed_db: EmbedDB) -> None:
        chunks = _make_chunks(count=1)
        embed_db.upsert_chunks(chunks, chunker_version="0.1.0")
        rows = embed_db._conn.execute("SELECT statute_areas, issue_tags FROM chunks").fetchall()
        assert rows[0][0] == ["APRA"]
        assert rows[0][1] == ["denial-of-access"]

    def test_overwrites_on_reupsert(self, embed_db: EmbedDB) -> None:
        chunks1 = _make_chunks(count=3)
        embed_db.upsert_chunks(chunks1, chunker_version="0.1.0")

        chunks2 = _make_chunks(count=2)
        embed_db.upsert_chunks(chunks2, chunker_version="0.2.0")

        rows = embed_db._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] == 2


class TestDeleteChunksForOpinion:
    def test_deletes_all_chunks(self, embed_db: EmbedDB) -> None:
        embed_db.upsert_chunks(_make_chunks(), chunker_version="0.1.0")
        embed_db.delete_chunks_for_opinion("25-FC-118")
        rows = embed_db._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] == 0

    def test_does_not_affect_other_opinions(self, embed_db: EmbedDB) -> None:
        embed_db.upsert_chunks(_make_chunks("25-FC-118"), chunker_version="0.1.0")
        embed_db.upsert_chunks(_make_chunks("25-FC-002A"), chunker_version="0.1.0")
        embed_db.delete_chunks_for_opinion("25-FC-118")
        rows = embed_db._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE opinion_id = '25-FC-002A'"
        ).fetchall()
        assert rows[0][0] == 3


class TestUpdateEmbedStatus:
    def test_updates_status(self, embed_db: EmbedDB) -> None:
        embed_db.update_embed_status("25-FC-118", "ok")
        rows = embed_db._conn.execute(
            "SELECT embed_status FROM opinions WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] == "ok"

    def test_updates_error_status(self, embed_db: EmbedDB) -> None:
        embed_db.update_embed_status("25-FC-118", "error")
        rows = embed_db._conn.execute(
            "SELECT embed_status FROM opinions WHERE opinion_id = '25-FC-118'"
        ).fetchall()
        assert rows[0][0] == "error"


class TestCountByEmbedStatus:
    def test_counts(self, embed_db: EmbedDB) -> None:
        counts = embed_db.count_by_embed_status()
        assert counts.get("pending", 0) >= 2

    def test_reflects_updates(self, embed_db: EmbedDB) -> None:
        embed_db.update_embed_status("25-FC-118", "ok")
        counts = embed_db.count_by_embed_status()
        assert counts.get("ok", 0) == 1


class TestGetChunkCount:
    def test_zero_initially(self, embed_db: EmbedDB) -> None:
        assert embed_db.get_chunk_count() == 0

    def test_reflects_inserts(self, embed_db: EmbedDB) -> None:
        embed_db.upsert_chunks(_make_chunks(), chunker_version="0.1.0")
        assert embed_db.get_chunk_count() == 3


class TestGetMetadataForOpinion:
    def test_returns_metadata(self, embed_db: EmbedDB) -> None:
        meta = embed_db.get_metadata_for_opinion("25-FC-118")
        assert meta is not None
        assert meta.opinion_id == "25-FC-118"
        assert meta.finding == "violation"

    def test_returns_none_for_missing(self, embed_db: EmbedDB) -> None:
        meta = embed_db.get_metadata_for_opinion("99-FC-999")
        assert meta is None

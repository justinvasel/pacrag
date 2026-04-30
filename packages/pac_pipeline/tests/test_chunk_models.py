from __future__ import annotations

import pytest
from pac_pipeline.chunk_models import (
    CHUNKER_VERSION,
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    Chunk,
)
from pydantic import ValidationError


class TestChunk:
    def test_valid_construction(self) -> None:
        chunk = Chunk(
            chunk_id="25-FC-118#c0",
            opinion_id="25-FC-118",
            chunk_index=0,
            section="card",
            page_num=1,
            text="Summary of the opinion.",
            token_count=5,
            year=2025,
            agency_type="city",
            statute_areas=["APRA"],
            finding="violation",
            issue_tags=["denial-of-access"],
        )
        assert chunk.chunk_id == "25-FC-118#c0"
        assert chunk.opinion_id == "25-FC-118"
        assert chunk.chunk_index == 0
        assert chunk.section == "card"
        assert chunk.page_num == 1
        assert chunk.token_count == 5
        assert chunk.year == 2025
        assert chunk.agency_type == "city"
        assert chunk.statute_areas == ("APRA",)
        assert chunk.finding == "violation"
        assert chunk.issue_tags == ("denial-of-access",)

    def test_chunk_id_format(self) -> None:
        chunk = Chunk(
            chunk_id="24-INF-01#c3",
            opinion_id="24-INF-01",
            chunk_index=3,
            section="analysis",
            page_num=2,
            text="Analysis text.",
            token_count=3,
            year=2024,
            agency_type="state_agency",
            statute_areas=["ODL"],
            finding="no_violation",
            issue_tags=[],
        )
        assert chunk.chunk_id == "24-INF-01#c3"
        assert chunk.chunk_index == 3

    def test_frozen(self) -> None:
        chunk = Chunk(
            chunk_id="25-FC-118#c0",
            opinion_id="25-FC-118",
            chunk_index=0,
            section="card",
            page_num=1,
            text="Text.",
            token_count=1,
            year=2025,
            agency_type="city",
            statute_areas=["APRA"],
            finding="violation",
            issue_tags=[],
        )
        with pytest.raises(ValidationError, match="frozen_instance"):
            chunk.text = "changed"  # type: ignore[misc]

    def test_multiple_statute_areas(self) -> None:
        chunk = Chunk(
            chunk_id="25-FC-118#c1",
            opinion_id="25-FC-118",
            chunk_index=1,
            section="analysis",
            page_num=1,
            text="Both APRA and ODL discussed.",
            token_count=6,
            year=2025,
            agency_type="city",
            statute_areas=["APRA", "ODL"],
            finding="mixed",
            issue_tags=["serial-meetings", "denial-of-access"],
        )
        assert chunk.statute_areas == ("APRA", "ODL")
        assert len(chunk.issue_tags) == 2

    def test_section_values(self) -> None:
        for section in ("card", "background", "analysis", "conclusion", "discussion"):
            chunk = Chunk(
                chunk_id="25-FC-118#c0",
                opinion_id="25-FC-118",
                chunk_index=0,
                section=section,
                page_num=1,
                text="Text.",
                token_count=1,
                year=2025,
                agency_type="city",
                statute_areas=["APRA"],
                finding="violation",
                issue_tags=[],
            )
            assert chunk.section == section


class TestConstants:
    def test_chunker_version_format(self) -> None:
        parts = CHUNKER_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_qdrant_collection_name(self) -> None:
        assert QDRANT_COLLECTION == "pac_opinions_v1"

    def test_embedding_dim(self) -> None:
        assert EMBEDDING_DIM == 1024

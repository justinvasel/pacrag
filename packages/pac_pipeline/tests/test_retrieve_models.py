from __future__ import annotations

from pac_pipeline.chunk_models import Chunk
from pac_pipeline.retrieve_models import (
    MIN_CANDIDATES_FOR_FILTERS,
    RRF_K,
    TOP_K_FUSED,
    TOP_K_PER_SOURCE,
    TOP_K_RERANKED,
    FusedChunk,
    InferredFilters,
    RankedChunkRef,
    RetrievalResult,
    RetrievedChunk,
)


def _make_chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "chunk_id": "25-FC-118#c0",
        "opinion_id": "25-FC-118",
        "chunk_index": 0,
        "section": "analysis",
        "page_num": 1,
        "text": "Test chunk text.",
        "token_count": 5,
        "year": 2025,
        "agency_type": "city",
        "statute_areas": ("APRA",),
        "finding": "violation",
        "issue_tags": ("denial-of-access",),
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


class TestInferredFilters:
    def test_defaults(self) -> None:
        f = InferredFilters()
        assert f.year_min is None
        assert f.year_max is None
        assert f.agency_type is None
        assert f.statute_areas == []
        assert f.finding is None
        assert f.issue_tags == []

    def test_with_values(self) -> None:
        f = InferredFilters(
            year_min=2020,
            year_max=2025,
            agency_type="school_corporation",
            statute_areas=["ODL"],
            finding="violation",
            issue_tags=["executive-session-purpose"],
        )
        assert f.year_min == 2020
        assert f.agency_type == "school_corporation"
        assert f.statute_areas == ["ODL"]

    def test_frozen(self) -> None:
        f = InferredFilters()
        try:
            f.year_min = 2020  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except Exception:
            pass

    def test_has_any_filters_false(self) -> None:
        f = InferredFilters()
        assert not f.has_any()

    def test_has_any_filters_true(self) -> None:
        f = InferredFilters(agency_type="city")
        assert f.has_any()

    def test_to_dict(self) -> None:
        f = InferredFilters(year_min=2020, agency_type="city")
        d = f.to_applied_dict()
        assert d == {"year_min": 2020, "agency_type": "city"}

    def test_to_dict_empty(self) -> None:
        f = InferredFilters()
        d = f.to_applied_dict()
        assert d == {}


class TestRankedChunkRef:
    def test_creation(self) -> None:
        chunk = _make_chunk()
        ref = RankedChunkRef(chunk_id="25-FC-118#c0", rank=0, score=0.95, chunk=chunk)
        assert ref.rank == 0
        assert ref.score == 0.95
        assert ref.chunk.opinion_id == "25-FC-118"


class TestFusedChunk:
    def test_creation(self) -> None:
        chunk = _make_chunk()
        fc = FusedChunk(chunk=chunk, dense_rank=1, lexical_rank=3, fused_score=0.03)
        assert fc.dense_rank == 1
        assert fc.lexical_rank == 3
        assert fc.fused_score == 0.03

    def test_none_ranks(self) -> None:
        chunk = _make_chunk()
        fc = FusedChunk(chunk=chunk, dense_rank=None, lexical_rank=5, fused_score=0.015)
        assert fc.dense_rank is None


class TestRetrievedChunk:
    def test_creation(self) -> None:
        chunk = _make_chunk()
        rc = RetrievedChunk(
            chunk=chunk,
            dense_rank=2,
            lexical_rank=None,
            fused_score=0.025,
            rerank_score=0.88,
        )
        assert rc.rerank_score == 0.88
        assert rc.lexical_rank is None

    def test_frozen(self) -> None:
        chunk = _make_chunk()
        rc = RetrievedChunk(
            chunk=chunk,
            dense_rank=2,
            lexical_rank=None,
            fused_score=0.025,
            rerank_score=0.88,
        )
        try:
            rc.rerank_score = 0.5  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except Exception:
            pass


class TestRetrievalResult:
    def test_creation(self) -> None:
        chunk = _make_chunk()
        rc = RetrievedChunk(
            chunk=chunk,
            dense_rank=1,
            lexical_rank=2,
            fused_score=0.03,
            rerank_score=0.9,
        )
        result = RetrievalResult(
            question="test question",
            applied_filters={"agency_type": "city"},
            relaxed=False,
            chunks=[rc],
        )
        assert result.question == "test question"
        assert len(result.chunks) == 1
        assert result.relaxed is False

    def test_empty_result(self) -> None:
        result = RetrievalResult(
            question="no results",
            applied_filters={},
            relaxed=True,
            chunks=[],
        )
        assert result.relaxed is True
        assert result.chunks == []

    def test_serialization_roundtrip(self) -> None:
        chunk = _make_chunk()
        rc = RetrievedChunk(
            chunk=chunk,
            dense_rank=1,
            lexical_rank=2,
            fused_score=0.03,
            rerank_score=0.9,
        )
        result = RetrievalResult(
            question="test",
            applied_filters={"year_min": 2020},
            relaxed=False,
            chunks=[rc],
        )
        json_str = result.model_dump_json()
        restored = RetrievalResult.model_validate_json(json_str)
        assert restored.question == result.question
        assert len(restored.chunks) == 1
        assert restored.chunks[0].rerank_score == 0.9


class TestConstants:
    def test_constants_values(self) -> None:
        assert TOP_K_PER_SOURCE == 30
        assert RRF_K == 60
        assert TOP_K_FUSED == 30
        assert TOP_K_RERANKED == 10
        assert MIN_CANDIDATES_FOR_FILTERS == 8

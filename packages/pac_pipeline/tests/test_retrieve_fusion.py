from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from pac_pipeline.chunk_models import Chunk
from pac_pipeline.retrieve_fusion import fuse_rrf, rerank
from pac_pipeline.retrieve_models import (
    RRF_K,
    FusedChunk,
    RankedChunkRef,
)


def _make_chunk(chunk_id: str = "25-FC-118#c0", **overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "chunk_id": chunk_id,
        "opinion_id": "25-FC-118",
        "chunk_index": 0,
        "section": "analysis",
        "page_num": 1,
        "text": f"Text for {chunk_id}.",
        "token_count": 5,
        "year": 2025,
        "agency_type": "city",
        "statute_areas": ("APRA",),
        "finding": "violation",
        "issue_tags": ("denial-of-access",),
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def _make_ranked(chunk_id: str, rank: int, score: float = 0.9) -> RankedChunkRef:
    return RankedChunkRef(
        chunk_id=chunk_id,
        rank=rank,
        score=score,
        chunk=_make_chunk(chunk_id),
    )


class TestFuseRRF:
    def test_single_source_dense_only(self) -> None:
        dense = [_make_ranked("a#c0", 0), _make_ranked("a#c1", 1)]
        result = fuse_rrf(dense, [])
        assert len(result) == 2
        assert result[0].chunk.chunk_id == "a#c0"
        assert result[0].dense_rank == 0
        assert result[0].lexical_rank is None
        expected_score = 1.0 / (RRF_K + 0)
        assert abs(result[0].fused_score - expected_score) < 1e-9

    def test_single_source_lexical_only(self) -> None:
        lexical = [_make_ranked("a#c0", 0)]
        result = fuse_rrf([], lexical)
        assert len(result) == 1
        assert result[0].dense_rank is None
        assert result[0].lexical_rank == 0

    def test_overlapping_results_boost_score(self) -> None:
        dense = [_make_ranked("a#c0", 0), _make_ranked("a#c1", 1)]
        lexical = [_make_ranked("a#c0", 0), _make_ranked("a#c2", 1)]

        result = fuse_rrf(dense, lexical)

        scores_by_id = {fc.chunk.chunk_id: fc for fc in result}

        overlap = scores_by_id["a#c0"]
        assert overlap.dense_rank == 0
        assert overlap.lexical_rank == 0
        expected = 1.0 / (RRF_K + 0) + 1.0 / (RRF_K + 0)
        assert abs(overlap.fused_score - expected) < 1e-9

        dense_only = scores_by_id["a#c1"]
        assert dense_only.dense_rank == 1
        assert dense_only.lexical_rank is None
        single_score = 1.0 / (RRF_K + 1)
        assert abs(dense_only.fused_score - single_score) < 1e-9

        assert overlap.fused_score > dense_only.fused_score

    def test_disjoint_lists(self) -> None:
        dense = [_make_ranked("a#c0", 0)]
        lexical = [_make_ranked("b#c0", 0)]
        result = fuse_rrf(dense, lexical)
        assert len(result) == 2
        assert result[0].fused_score == result[1].fused_score

    def test_sorted_descending_by_score(self) -> None:
        dense = [_make_ranked("a#c0", 0), _make_ranked("a#c1", 1), _make_ranked("a#c2", 2)]
        lexical = [_make_ranked("a#c2", 0), _make_ranked("a#c1", 1)]

        result = fuse_rrf(dense, lexical)
        scores = [fc.fused_score for fc in result]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_output(self) -> None:
        dense = [_make_ranked(f"d#c{i}", i) for i in range(10)]
        lexical = [_make_ranked(f"l#c{i}", i) for i in range(10)]
        result = fuse_rrf(dense, lexical, top_k=5)
        assert len(result) == 5

    def test_empty_inputs(self) -> None:
        result = fuse_rrf([], [])
        assert result == []


class TestRerank:
    def test_rerank_sorts_by_score(self) -> None:
        candidates = [
            FusedChunk(chunk=_make_chunk("a#c0"), dense_rank=0, lexical_rank=0, fused_score=0.03),
            FusedChunk(chunk=_make_chunk("a#c1"), dense_rank=1, lexical_rank=1, fused_score=0.02),
            FusedChunk(chunk=_make_chunk("a#c2"), dense_rank=2, lexical_rank=2, fused_score=0.01),
        ]

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.3, 0.9, 0.6])

        result = rerank("test question", candidates, mock_reranker)

        assert len(result) == 3
        assert result[0].chunk.chunk_id == "a#c1"
        assert result[0].rerank_score == 0.9
        assert result[1].chunk.chunk_id == "a#c2"
        assert result[1].rerank_score == 0.6
        assert result[2].chunk.chunk_id == "a#c0"
        assert result[2].rerank_score == 0.3

    def test_rerank_top_k(self) -> None:
        candidates = [
            FusedChunk(chunk=_make_chunk(f"a#c{i}"), dense_rank=i, lexical_rank=i, fused_score=0.01)
            for i in range(5)
        ]

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.1, 0.5, 0.9, 0.3, 0.7])

        result = rerank("test", candidates, mock_reranker, top_k=3)
        assert len(result) == 3
        assert result[0].rerank_score == 0.9
        assert result[1].rerank_score == 0.7
        assert result[2].rerank_score == 0.5

    def test_rerank_preserves_ranks(self) -> None:
        candidates = [
            FusedChunk(
                chunk=_make_chunk("a#c0"), dense_rank=5, lexical_rank=None, fused_score=0.01
            ),
        ]

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.8])

        result = rerank("test", candidates, mock_reranker)
        assert result[0].dense_rank == 5
        assert result[0].lexical_rank is None
        assert result[0].fused_score == 0.01

    def test_rerank_calls_predict_with_pairs(self) -> None:
        candidates = [
            FusedChunk(chunk=_make_chunk("a#c0"), dense_rank=0, lexical_rank=0, fused_score=0.03),
        ]

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([0.5])

        rerank("my question", candidates, mock_reranker)

        mock_reranker.predict.assert_called_once()
        pairs = mock_reranker.predict.call_args[0][0]
        assert pairs == [("my question", candidates[0].chunk.text)]

    def test_rerank_empty_candidates(self) -> None:
        mock_reranker = MagicMock()
        result = rerank("test", [], mock_reranker)
        assert result == []
        mock_reranker.predict.assert_not_called()

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pac_pipeline.chunk_models import Chunk
from pac_pipeline.retrieve import _embedding_cache, retrieve
from pac_pipeline.retrieve_models import InferredFilters, RankedChunkRef


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


def _make_ranked_refs(count: int = 5) -> list[RankedChunkRef]:
    return [
        RankedChunkRef(
            chunk_id=f"op#c{i}",
            rank=i,
            score=1.0 - i * 0.1,
            chunk=_make_chunk(f"op#c{i}"),
        )
        for i in range(count)
    ]


@contextmanager
def _patch_retrieval_infra(
    mock_models: MagicMock,
) -> Generator[dict[str, MagicMock | AsyncMock], None, None]:
    qdrant_instance = AsyncMock()
    with (
        patch("pac_pipeline.retrieve.get_retrieval_models", return_value=mock_models),
        patch("pac_pipeline.retrieve.AsyncQdrantClient", return_value=qdrant_instance),
        patch("pac_pipeline.retrieve._load_tantivy_index"),
        patch("pac_pipeline.retrieve.search_dense", new_callable=AsyncMock) as mock_dense,
        patch("pac_pipeline.retrieve.search_lexical") as mock_lexical,
    ):
        yield {"dense": mock_dense, "lexical": mock_lexical, "qdrant": qdrant_instance}


@pytest.fixture(autouse=True)
def _clear_embedding_cache() -> Generator[None, None, None]:
    _embedding_cache.clear()
    yield
    _embedding_cache.clear()


@pytest.fixture
def mock_models() -> MagicMock:
    models = MagicMock()
    models.embedder.encode.return_value = np.zeros(1024)
    models.reranker.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    return models


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_basic_retrieval(self, mock_models: MagicMock) -> None:
        with _patch_retrieval_infra(mock_models) as mocks:
            mocks["dense"].return_value = _make_ranked_refs(5)
            mocks["lexical"].return_value = _make_ranked_refs(5)

            result = await retrieve("test question", skip_filter_inference=True)

        assert result.question == "test question"
        assert len(result.chunks) > 0
        assert result.relaxed is False

    @pytest.mark.asyncio
    async def test_skip_filter_inference(self, mock_models: MagicMock) -> None:
        with (
            _patch_retrieval_infra(mock_models) as mocks,
            patch("pac_pipeline.retrieve.infer_filters") as mock_infer,
        ):
            mocks["dense"].return_value = _make_ranked_refs(5)
            mocks["lexical"].return_value = _make_ranked_refs(5)

            await retrieve("test", skip_filter_inference=True)
            mock_infer.assert_not_called()

    @pytest.mark.asyncio
    async def test_filter_inference_called(self, mock_models: MagicMock) -> None:
        with (
            _patch_retrieval_infra(mock_models) as mocks,
            patch("pac_pipeline.retrieve.infer_filters", new_callable=AsyncMock) as mock_infer,
            patch("pac_pipeline.retrieve.AsyncAnthropic"),
        ):
            mock_infer.return_value = InferredFilters(agency_type="city")
            mocks["dense"].return_value = _make_ranked_refs(10)
            mocks["lexical"].return_value = _make_ranked_refs(10)
            mock_models.reranker.predict.return_value = np.arange(10, 0, -1, dtype=float)

            result = await retrieve("test about city records")
            mock_infer.assert_called_once()
            assert result.applied_filters.get("agency_type") == "city"

    @pytest.mark.asyncio
    async def test_filter_relaxation(self, mock_models: MagicMock) -> None:
        sparse_results = _make_ranked_refs(2)

        with (
            _patch_retrieval_infra(mock_models) as mocks,
            patch("pac_pipeline.retrieve.infer_filters", new_callable=AsyncMock) as mock_infer,
            patch("pac_pipeline.retrieve.AsyncAnthropic"),
        ):
            mock_infer.return_value = InferredFilters(agency_type="library")
            full_results = _make_ranked_refs(10)
            mocks["dense"].side_effect = [sparse_results, full_results]
            mocks["lexical"].side_effect = [[], full_results]
            mock_models.reranker.predict.return_value = np.arange(10, 0, -1, dtype=float)

            result = await retrieve("test about library records")
            assert result.relaxed is True
            assert mocks["dense"].call_count == 2

    @pytest.mark.asyncio
    async def test_embedding_cached(self, mock_models: MagicMock) -> None:
        with _patch_retrieval_infra(mock_models) as mocks:
            mocks["dense"].return_value = _make_ranked_refs(5)
            mocks["lexical"].return_value = _make_ranked_refs(5)

            await retrieve("same question", skip_filter_inference=True)
            await retrieve("same question", skip_filter_inference=True)

            assert mock_models.embedder.encode.call_count == 1

    @pytest.mark.asyncio
    async def test_different_questions_not_cached(self, mock_models: MagicMock) -> None:
        with _patch_retrieval_infra(mock_models) as mocks:
            mocks["dense"].return_value = _make_ranked_refs(5)
            mocks["lexical"].return_value = _make_ranked_refs(5)

            await retrieve("question one", skip_filter_inference=True)
            await retrieve("question two", skip_filter_inference=True)

            assert mock_models.embedder.encode.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_models: MagicMock) -> None:
        mock_models.reranker.predict.return_value = np.array([])

        with _patch_retrieval_infra(mock_models) as mocks:
            mocks["dense"].return_value = []
            mocks["lexical"].return_value = []

            result = await retrieve("obscure question", skip_filter_inference=True)
            assert result.chunks == []

    @pytest.mark.asyncio
    async def test_explicit_filters_passed_through(self, mock_models: MagicMock) -> None:
        mock_models.reranker.predict.return_value = np.arange(10, 0, -1, dtype=float)

        with (
            _patch_retrieval_infra(mock_models) as mocks,
            patch("pac_pipeline.retrieve.infer_filters") as mock_infer,
        ):
            mocks["dense"].return_value = _make_ranked_refs(10)
            mocks["lexical"].return_value = _make_ranked_refs(10)

            result = await retrieve(
                "test",
                filters={"agency_type": "school_corporation", "statute_areas": ["ODL"]},
            )
            mock_infer.assert_not_called()
            assert result.applied_filters["agency_type"] == "school_corporation"

    @pytest.mark.asyncio
    async def test_custom_top_k(self, mock_models: MagicMock) -> None:
        mock_models.reranker.predict.return_value = np.arange(10, 0, -1, dtype=float)

        with _patch_retrieval_infra(mock_models) as mocks:
            mocks["dense"].return_value = _make_ranked_refs(10)
            mocks["lexical"].return_value = _make_ranked_refs(10)

            result = await retrieve("test", skip_filter_inference=True, top_k=3)
            assert len(result.chunks) == 3

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from pac_pipeline.retrieve_models import RetrievalResult


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(
        self,
        client: AsyncClient,
        sample_retrieval_result: RetrievalResult,
    ) -> None:
        with patch("pac_api.routes.search.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = sample_retrieval_result
            response = await client.get("/search", params={"question": "fee charging"})

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "Can an agency charge fees for public records?"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["opinion_id"] == "25-FC-118"

    @pytest.mark.asyncio
    async def test_search_with_filters(
        self,
        client: AsyncClient,
        sample_retrieval_result: RetrievalResult,
    ) -> None:
        with patch("pac_api.routes.search.retrieve", new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = sample_retrieval_result
            response = await client.get(
                "/search",
                params={
                    "question": "open meetings",
                    "filters": '{"statute_areas": ["ODL"]}',
                },
            )

        assert response.status_code == 200
        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args
        assert call_kwargs.kwargs.get("filters") == {"statute_areas": ["ODL"]}

    @pytest.mark.asyncio
    async def test_search_invalid_filters_json(self, client: AsyncClient) -> None:
        response = await client.get("/search", params={"question": "test", "filters": "not-json"})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_search_empty_question(self, client: AsyncClient) -> None:
        response = await client.get("/search", params={"question": ""})
        assert response.status_code == 422

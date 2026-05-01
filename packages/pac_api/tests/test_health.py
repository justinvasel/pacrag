from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient


class TestHealthz:
    @pytest.mark.asyncio
    async def test_healthz_returns_counts(self, client: AsyncClient) -> None:
        mock_collection = MagicMock()
        mock_collection.points_count = 3

        with (
            patch("pac_api.routes.health._qdrant_count", new_callable=AsyncMock, return_value=3),
            patch("pac_api.routes.health._tantivy_count", return_value=3),
        ):
            response = await client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["qdrant_chunks"] == 3
        assert data["tantivy_chunks"] == 3
        assert data["duckdb_chunks"] == 3
        assert data["duckdb_opinions"] == 3
        assert data["counts_match"] is True

    @pytest.mark.asyncio
    async def test_healthz_counts_mismatch(self, client: AsyncClient) -> None:
        with (
            patch("pac_api.routes.health._qdrant_count", new_callable=AsyncMock, return_value=10),
            patch("pac_api.routes.health._tantivy_count", return_value=5),
        ):
            response = await client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["counts_match"] is False

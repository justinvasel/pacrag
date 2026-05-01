from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestListOpinions:
    @pytest.mark.asyncio
    async def test_list_all(self, client: AsyncClient) -> None:
        response = await client.get("/opinions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        assert data["page"] == 1
        assert data["total_pages"] == 1

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, client: AsyncClient) -> None:
        response = await client.get("/opinions", params={"page_size": 2})
        data = response.json()
        assert data["total"] == 3
        assert len(data["items"]) == 2
        assert data["total_pages"] == 2

    @pytest.mark.asyncio
    async def test_list_filter_by_year_min(self, client: AsyncClient) -> None:
        response = await client.get("/opinions", params={"year_min": 2025})
        data = response.json()
        assert data["total"] == 2
        assert all(item["year"] == 2025 for item in data["items"])

    @pytest.mark.asyncio
    async def test_list_filter_by_finding(self, client: AsyncClient) -> None:
        response = await client.get("/opinions", params={"finding": "violation"})
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["finding"] == "violation"

    @pytest.mark.asyncio
    async def test_list_filter_by_statute_areas(self, client: AsyncClient) -> None:
        response = await client.get("/opinions", params={"statute_areas": "APRA"})
        data = response.json()
        assert data["total"] >= 1
        assert all("APRA" in item["statute_areas"] for item in data["items"])


class TestGetOpinion:
    @pytest.mark.asyncio
    async def test_get_existing(self, client: AsyncClient) -> None:
        response = await client.get("/opinions/25-FC-118")
        assert response.status_code == 200
        data = response.json()
        assert data["opinion_id"] == "25-FC-118"
        assert data["year"] == 2025
        assert data["agency_name"] == "City of Bloomington"
        assert data["metadata"] is not None
        assert data["metadata"]["finding"] == "violation"

    @pytest.mark.asyncio
    async def test_get_not_found(self, client: AsyncClient) -> None:
        response = await client.get("/opinions/99-FC-999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_has_metadata_fields(self, client: AsyncClient) -> None:
        response = await client.get("/opinions/25-FC-118")
        data = response.json()
        meta = data["metadata"]
        assert "counselor" in meta
        assert "statute_areas" in meta
        assert "issue_tags" in meta
        assert "summary" in meta

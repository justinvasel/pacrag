from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from pac_pipeline.retrieve_models import RetrievalResult


async def _mock_stream_answer(question, chunks, client):
    yield ("token", "The agency violated APRA ")
    yield ("token", "(25-FC-118, p. 1).")
    yield ("done", "The agency violated APRA (25-FC-118, p. 1).")


async def _mock_stream_invalid_citation(question, chunks, client):
    yield ("token", "According to (99-FC-999, p. 1).")
    yield ("done", "According to (99-FC-999, p. 1).")


async def _mock_stream_no_citations(question, chunks, client):
    yield ("token", "The retrieved opinions don't directly address this.")
    yield ("done", "The retrieved opinions don't directly address this.")


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        event: dict[str, str] = {}
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = line[6:]
        if event:
            events.append(event)
    return events


class TestAsk:
    @pytest.mark.asyncio
    async def test_ask_streams_tokens(
        self,
        client: AsyncClient,
        sample_retrieval_result: RetrievalResult,
    ) -> None:
        with (
            patch(
                "pac_api.routes.ask.do_retrieve",
                new_callable=AsyncMock,
                return_value=sample_retrieval_result,
            ),
            patch("pac_api.routes.ask.stream_answer", side_effect=_mock_stream_answer),
        ):
            response = await client.post("/ask", json={"question": "Did the agency violate APRA?"})

        assert response.status_code == 200
        events = _parse_sse(response.text)

        token_events = [e for e in events if e.get("event") == "token"]
        assert len(token_events) >= 1

        metadata_events = [e for e in events if e.get("event") == "metadata"]
        assert len(metadata_events) == 1
        meta = json.loads(metadata_events[0]["data"])
        assert len(meta["citations"]) == 1
        assert meta["citations"][0]["opinion_id"] == "25-FC-118"
        assert meta["citations"][0]["valid"] is True

    @pytest.mark.asyncio
    async def test_ask_invalid_citation_yields_error(
        self,
        client: AsyncClient,
        sample_retrieval_result: RetrievalResult,
    ) -> None:
        with (
            patch(
                "pac_api.routes.ask.do_retrieve",
                new_callable=AsyncMock,
                return_value=sample_retrieval_result,
            ),
            patch(
                "pac_api.routes.ask.stream_answer",
                side_effect=_mock_stream_invalid_citation,
            ),
        ):
            response = await client.post("/ask", json={"question": "test question"})

        assert response.status_code == 200
        events = _parse_sse(response.text)
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 1
        error_data = json.loads(error_events[0]["data"])
        assert "Citation validation failed" in error_data["error"]

    @pytest.mark.asyncio
    async def test_ask_no_results(self, client: AsyncClient) -> None:
        empty_result = RetrievalResult(
            question="unknown topic",
            applied_filters={},
            relaxed=False,
            chunks=[],
        )
        with patch(
            "pac_api.routes.ask.do_retrieve",
            new_callable=AsyncMock,
            return_value=empty_result,
        ):
            response = await client.post("/ask", json={"question": "something irrelevant"})

        assert response.status_code == 200
        events = _parse_sse(response.text)
        token_events = [e for e in events if e.get("event") == "token"]
        assert len(token_events) >= 1
        token_text = json.loads(token_events[0]["data"])["token"]
        assert "No relevant opinions" in token_text

    @pytest.mark.asyncio
    async def test_ask_no_citations_passes(
        self,
        client: AsyncClient,
        sample_retrieval_result: RetrievalResult,
    ) -> None:
        with (
            patch(
                "pac_api.routes.ask.do_retrieve",
                new_callable=AsyncMock,
                return_value=sample_retrieval_result,
            ),
            patch(
                "pac_api.routes.ask.stream_answer",
                side_effect=_mock_stream_no_citations,
            ),
        ):
            response = await client.post("/ask", json={"question": "off topic question"})

        events = _parse_sse(response.text)
        error_events = [e for e in events if e.get("event") == "error"]
        assert len(error_events) == 0

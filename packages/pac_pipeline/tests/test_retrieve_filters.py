from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pac_pipeline.retrieve_filters import (
    _TOOL_NAME,
    infer_filters,
    to_qdrant_filter,
)
from pac_pipeline.retrieve_models import InferredFilters
from qdrant_client.models import (
    FieldCondition,
    MatchAny,
    MatchValue,
    Range,
)


def _mock_anthropic_response(tool_input: dict[str, object]) -> MagicMock:
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input

    response = MagicMock()
    response.content = [tool_block]
    return response


class TestInferFilters:
    @pytest.mark.asyncio
    async def test_school_board_question(self) -> None:
        client = AsyncMock()
        client.messages.create.return_value = _mock_anthropic_response(
            {
                "agency_type": "school_corporation",
                "statute_areas": ["ODL"],
                "issue_tags": ["executive-session-purpose"],
            }
        )

        result = await infer_filters("Can a school board hold executive sessions?", client)
        assert result.agency_type == "school_corporation"
        assert result.statute_areas == ["ODL"]
        assert "executive-session-purpose" in result.issue_tags

    @pytest.mark.asyncio
    async def test_public_records_question(self) -> None:
        client = AsyncMock()
        client.messages.create.return_value = _mock_anthropic_response(
            {
                "statute_areas": ["APRA"],
                "issue_tags": ["denial-of-access"],
            }
        )

        result = await infer_filters("Can an agency deny a public records request?", client)
        assert result.statute_areas == ["APRA"]

    @pytest.mark.asyncio
    async def test_open_meetings_question(self) -> None:
        client = AsyncMock()
        client.messages.create.return_value = _mock_anthropic_response(
            {
                "statute_areas": ["ODL"],
            }
        )

        result = await infer_filters("What are the notice requirements for open meetings?", client)
        assert result.statute_areas == ["ODL"]

    @pytest.mark.asyncio
    async def test_empty_filters(self) -> None:
        client = AsyncMock()
        client.messages.create.return_value = _mock_anthropic_response({})

        result = await infer_filters("Tell me about PAC opinions", client)
        assert not result.has_any()

    @pytest.mark.asyncio
    async def test_year_filters(self) -> None:
        client = AsyncMock()
        client.messages.create.return_value = _mock_anthropic_response(
            {"year_min": 2022, "year_max": 2025}
        )

        result = await infer_filters("Recent opinions about fee charging", client)
        assert result.year_min == 2022
        assert result.year_max == 2025

    @pytest.mark.asyncio
    async def test_no_tool_use_returns_empty(self) -> None:
        text_block = MagicMock()
        text_block.type = "text"

        response = MagicMock()
        response.content = [text_block]

        client = AsyncMock()
        client.messages.create.return_value = response

        result = await infer_filters("test question", client)
        assert not result.has_any()

    @pytest.mark.asyncio
    async def test_uses_tool_choice(self) -> None:
        client = AsyncMock()
        client.messages.create.return_value = _mock_anthropic_response({})

        await infer_filters("test", client)

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["tool_choice"]["name"] == _TOOL_NAME

    @pytest.mark.asyncio
    async def test_invalid_tags_stripped(self) -> None:
        client = AsyncMock()
        client.messages.create.return_value = _mock_anthropic_response(
            {
                "issue_tags": ["denial-of-access", "made-up-tag"],
            }
        )

        result = await infer_filters("test", client)
        assert result.issue_tags == ["denial-of-access"]


class TestToQdrantFilter:
    def test_empty_filters_returns_none(self) -> None:
        assert to_qdrant_filter(InferredFilters()) is None

    def test_agency_type_filter(self) -> None:
        f = to_qdrant_filter(InferredFilters(agency_type="school_corporation"))
        assert f is not None
        assert len(f.must) == 1  # type: ignore[arg-type]
        cond = f.must[0]  # type: ignore[index]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "agency_type"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "school_corporation"

    def test_year_range_filter(self) -> None:
        f = to_qdrant_filter(InferredFilters(year_min=2020, year_max=2024))
        assert f is not None
        assert len(f.must) == 1  # type: ignore[arg-type]
        cond = f.must[0]  # type: ignore[index]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "year"
        assert isinstance(cond.range, Range)
        assert cond.range.gte == 2020
        assert cond.range.lte == 2024

    def test_statute_areas_filter(self) -> None:
        f = to_qdrant_filter(InferredFilters(statute_areas=["APRA"]))
        assert f is not None
        cond = f.must[0]  # type: ignore[index]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "statute_areas"
        assert isinstance(cond.match, MatchAny)
        assert cond.match.any == ["APRA"]

    def test_finding_filter(self) -> None:
        f = to_qdrant_filter(InferredFilters(finding="violation"))
        assert f is not None
        cond = f.must[0]  # type: ignore[index]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "finding"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "violation"

    def test_issue_tags_filter(self) -> None:
        f = to_qdrant_filter(InferredFilters(issue_tags=["fee-charging", "denial-of-access"]))
        assert f is not None
        cond = f.must[0]  # type: ignore[index]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "issue_tags"
        assert isinstance(cond.match, MatchAny)
        assert cond.match.any == ["fee-charging", "denial-of-access"]

    def test_combined_filters(self) -> None:
        f = to_qdrant_filter(
            InferredFilters(
                year_min=2020,
                agency_type="city",
                statute_areas=["APRA"],
            )
        )
        assert f is not None
        assert len(f.must) == 3  # type: ignore[arg-type]

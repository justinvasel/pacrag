from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pac_eval.metrics_judge import (
    _format_numbered_facts,
    _parse_judge_response,
    judge_fact_coverage,
)


class TestFormatNumberedFacts:
    def test_formats_correctly(self) -> None:
        result = _format_numbered_facts(["Fact A", "Fact B", "Fact C"])
        assert result == "1. Fact A\n2. Fact B\n3. Fact C"

    def test_single_fact(self) -> None:
        result = _format_numbered_facts(["Only fact"])
        assert result == "1. Only fact"


class TestParseJudgeResponse:
    def test_valid_json(self) -> None:
        text = '{"facts": [{"fact_index": 1, "covered": true, "reasoning": "present"}]}'
        result = _parse_judge_response(text, 1)
        assert len(result.facts) == 1
        assert result.facts[0].covered is True

    def test_strips_code_fences(self) -> None:
        text = (
            '```json\n{"facts": [{"fact_index": 1, "covered": false, "reasoning": "missing"}]}\n```'
        )
        result = _parse_judge_response(text, 1)
        assert result.facts[0].covered is False

    def test_wrong_count_raises(self) -> None:
        text = '{"facts": [{"fact_index": 1, "covered": true, "reasoning": "ok"}]}'
        with pytest.raises(ValueError, match="expected 2"):
            _parse_judge_response(text, 2)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_judge_response("not json", 1)


class TestJudgeFactCoverage:
    @pytest.fixture()
    def mock_client(self) -> AsyncMock:
        client = AsyncMock()
        return client

    def _make_response(self, text: str) -> MagicMock:
        block = MagicMock()
        block.text = text
        msg = MagicMock()
        msg.content = [block]
        return msg

    async def test_all_facts_covered(self, mock_client: AsyncMock) -> None:
        judge_json = (
            '{"facts": ['
            '{"fact_index": 1, "covered": true, "reasoning": "stated directly"},'
            '{"fact_index": 2, "covered": true, "reasoning": "paraphrased"}'
            "]}"
        )
        mock_client.messages.create.return_value = self._make_response(judge_json)

        coverage, found, missing = await judge_fact_coverage(
            question="Q?",
            ideal_answer="Ideal.",
            key_facts=["Fact A", "Fact B"],
            generated_answer="Answer with both facts.",
            client=mock_client,
        )
        assert coverage == 1.0
        assert found == ["Fact A", "Fact B"]
        assert missing == []

    async def test_no_facts_covered(self, mock_client: AsyncMock) -> None:
        judge_json = (
            '{"facts": ['
            '{"fact_index": 1, "covered": false, "reasoning": "absent"},'
            '{"fact_index": 2, "covered": false, "reasoning": "absent"}'
            "]}"
        )
        mock_client.messages.create.return_value = self._make_response(judge_json)

        coverage, found, missing = await judge_fact_coverage(
            question="Q?",
            ideal_answer="Ideal.",
            key_facts=["Fact A", "Fact B"],
            generated_answer="Irrelevant answer.",
            client=mock_client,
        )
        assert coverage == 0.0
        assert found == []
        assert missing == ["Fact A", "Fact B"]

    async def test_partial_coverage(self, mock_client: AsyncMock) -> None:
        judge_json = (
            '{"facts": ['
            '{"fact_index": 1, "covered": true, "reasoning": "present"},'
            '{"fact_index": 2, "covered": false, "reasoning": "absent"},'
            '{"fact_index": 3, "covered": true, "reasoning": "present"}'
            "]}"
        )
        mock_client.messages.create.return_value = self._make_response(judge_json)

        coverage, found, missing = await judge_fact_coverage(
            question="Q?",
            ideal_answer="Ideal.",
            key_facts=["Fact A", "Fact B", "Fact C"],
            generated_answer="Answer.",
            client=mock_client,
        )
        assert coverage == pytest.approx(2 / 3)
        assert found == ["Fact A", "Fact C"]
        assert missing == ["Fact B"]

    async def test_retry_on_parse_failure(self, mock_client: AsyncMock) -> None:
        good_json = '{"facts": [{"fact_index": 1, "covered": true, "reasoning": "ok"}]}'
        mock_client.messages.create.side_effect = [
            self._make_response("not json"),
            self._make_response(good_json),
        ]

        coverage, _found, _missing = await judge_fact_coverage(
            question="Q?",
            ideal_answer="Ideal.",
            key_facts=["Fact A"],
            generated_answer="Answer.",
            client=mock_client,
        )
        assert coverage == 1.0
        assert mock_client.messages.create.call_count == 2

    async def test_fallback_after_two_failures(self, mock_client: AsyncMock) -> None:
        mock_client.messages.create.side_effect = [
            self._make_response("bad1"),
            self._make_response("bad2"),
        ]

        coverage, found, missing = await judge_fact_coverage(
            question="Q?",
            ideal_answer="Ideal.",
            key_facts=["Fact A", "Fact B"],
            generated_answer="Answer.",
            client=mock_client,
        )
        assert coverage == 0.0
        assert found == []
        assert missing == ["Fact A", "Fact B"]

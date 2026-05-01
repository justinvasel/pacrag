from __future__ import annotations

import pytest
from pac_pipeline.retrieve_dense import _payload_to_chunk


class TestPayloadToChunk:
    def test_valid_payload(self) -> None:
        payload: dict[str, object] = {
            "chunk_id": "25-FC-118#c0",
            "opinion_id": "25-FC-118",
            "chunk_index": 0,
            "section": "analysis",
            "page_num": 2,
            "text": "Some text here.",
            "token_count": 10,
            "year": 2025,
            "agency_type": "city",
            "statute_areas": ["APRA"],
            "finding": "violation",
            "issue_tags": ["denial-of-access"],
        }

        chunk = _payload_to_chunk(payload)
        assert chunk.chunk_id == "25-FC-118#c0"
        assert chunk.opinion_id == "25-FC-118"
        assert chunk.year == 2025
        assert chunk.statute_areas == ("APRA",)
        assert chunk.issue_tags == ("denial-of-access",)

    def test_missing_optional_fields_use_defaults(self) -> None:
        payload: dict[str, object] = {
            "chunk_id": "25-FC-118#c0",
            "opinion_id": "25-FC-118",
            "year": 2025,
        }

        chunk = _payload_to_chunk(payload)
        assert chunk.chunk_index == 0
        assert chunk.section == ""
        assert chunk.page_num == 0
        assert chunk.token_count == 0
        assert chunk.agency_type == "other"
        assert chunk.statute_areas == ()
        assert chunk.issue_tags == ()

    def test_missing_required_key_raises(self) -> None:
        payload: dict[str, object] = {"opinion_id": "25-FC-118", "year": 2025}
        with pytest.raises(KeyError):
            _payload_to_chunk(payload)

    def test_empty_lists(self) -> None:
        payload: dict[str, object] = {
            "chunk_id": "25-FC-118#c0",
            "opinion_id": "25-FC-118",
            "year": 2025,
            "statute_areas": [],
            "issue_tags": [],
        }

        chunk = _payload_to_chunk(payload)
        assert chunk.statute_areas == ()
        assert chunk.issue_tags == ()

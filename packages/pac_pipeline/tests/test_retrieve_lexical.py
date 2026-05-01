from __future__ import annotations

import tantivy
from pac_pipeline.retrieve_lexical import _doc_to_chunk


def _make_tantivy_doc(**fields: object) -> tantivy.Document:
    return tantivy.Document(**fields)  # type: ignore[arg-type]


class TestDocToChunk:
    def test_valid_document(self) -> None:
        doc = _make_tantivy_doc(
            chunk_id="25-FC-118#c0",
            opinion_id="25-FC-118",
            text="Some analysis text.",
            year=2025,
            agency_type="city",
            statute_areas="APRA ODL",
            finding="violation",
            issue_tags="denial-of-access fee-charging",
        )

        chunk = _doc_to_chunk(doc)
        assert chunk.chunk_id == "25-FC-118#c0"
        assert chunk.opinion_id == "25-FC-118"
        assert chunk.text == "Some analysis text."
        assert chunk.year == 2025
        assert chunk.agency_type == "city"
        assert chunk.statute_areas == ("APRA", "ODL")
        assert chunk.finding == "violation"
        assert chunk.issue_tags == ("denial-of-access", "fee-charging")

    def test_empty_statute_areas_and_tags(self) -> None:
        doc = _make_tantivy_doc(
            chunk_id="25-FC-118#c0",
            opinion_id="25-FC-118",
            text="Text.",
            year=2025,
            agency_type="other",
            statute_areas="",
            finding="no_violation",
            issue_tags="",
        )

        chunk = _doc_to_chunk(doc)
        assert chunk.statute_areas == ()
        assert chunk.issue_tags == ()

    def test_zeroed_metadata_fields(self) -> None:
        doc = _make_tantivy_doc(
            chunk_id="25-FC-118#c0",
            opinion_id="25-FC-118",
            text="Text.",
            year=2025,
            agency_type="city",
            statute_areas="APRA",
            finding="violation",
            issue_tags="",
        )

        chunk = _doc_to_chunk(doc)
        assert chunk.chunk_index == 0
        assert chunk.section == ""
        assert chunk.page_num == 0
        assert chunk.token_count == 0

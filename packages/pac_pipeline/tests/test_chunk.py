from __future__ import annotations

from datetime import date

import pytest
from pac_pipeline.chunk import (
    chunk_opinion,
    count_tokens,
    split_paragraphs_into_chunks,
)
from pac_pipeline.extract_models import OpinionMetadata
from pac_pipeline.models import ParsedOpinion, ParsedPage


class _FakeTokenizer:
    """Splits on whitespace for deterministic token counts in tests."""

    def __init__(self) -> None:
        self._last_words: list[str] = []

    def encode(self, text: str) -> list[int]:
        self._last_words = text.split()
        return list(range(len(self._last_words)))

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        return " ".join(self._last_words[i] for i in token_ids if i < len(self._last_words))


@pytest.fixture
def tokenizer() -> _FakeTokenizer:
    return _FakeTokenizer()


def _make_metadata(
    opinion_id: str = "25-FC-118",
    *,
    finding: str = "violation",
) -> OpinionMetadata:
    return OpinionMetadata(
        opinion_id=opinion_id,
        issue_date=date(2026, 2, 2),
        counselor="Jennifer G. Ruby",
        complainant="David Askins",
        public_agency="City of Bloomington",
        agency_type="city",
        statutes_invoked=["IC 5-14-3-4(b)(20)"],
        statute_areas=["APRA"],
        issue_tags=["denial-of-access"],
        tags_need_review=False,
        finding=finding,  # type: ignore[arg-type]
        summary="The PAC found that the City of Bloomington violated APRA.",
    )


def _make_parsed(
    opinion_id: str = "25-FC-118",
    *,
    with_sections: bool = True,
) -> ParsedOpinion:
    if with_sections:
        full_text = (
            "STATE OF INDIANA  PUBLIC ACCESS COUNSELOR\n"
            "JENNIFER G. RUBY\n\n"
            "February 2, 2026\n\n"
            "Re: Complaint 25-FC-118\n"
            "David Askins (Complainant) v.\n"
            "City of Bloomington (Respondent)\n\n"
            "This advisory opinion is in response to the complaint.\n\n"
            "BACKGROUND\n\n"
            "The complainant alleges that Respondent violated APRA by denying "
            "access to public records. The complainant requested records from "
            "the City of Bloomington.\n\n"
            "ANALYSIS\n\n"
            "APRA provides persons with the right to inspect and copy public records. "
            "IC 5-14-3-3(a). The agency cited IC 5-14-3-4(b)(20).\n\n"
            "The PAC must determine whether the exemption applies to the "
            "records at issue. After reviewing the submissions, the PAC finds "
            "that the City improperly applied the exemption.\n\n"
            "CONCLUSION\n\n"
            "Based on the foregoing, the PAC finds a violation of APRA.\n\n"
            "Jennifer G. Ruby\n"
            "Public Access Counselor"
        )
        bg_start = full_text.index("BACKGROUND\n\n") + len("BACKGROUND\n\n")
        bg_end = full_text.index("\n\nANALYSIS")
        an_start = full_text.index("ANALYSIS\n\n") + len("ANALYSIS\n\n")
        an_end = full_text.index("\n\nCONCLUSION")
        co_start = full_text.index("CONCLUSION\n\n") + len("CONCLUSION\n\n")
        co_end = full_text.index("\n\nJennifer G. Ruby\nPublic Access Counselor")
        sections = {
            "BACKGROUND": (bg_start, bg_end),
            "ANALYSIS": (an_start, an_end),
            "CONCLUSION": (co_start, co_end),
        }
    else:
        full_text = (
            "STATE OF INDIANA  PUBLIC ACCESS COUNSELOR\n"
            "LUKE H. BRITT\n\n"
            "Some opinion text without standard sections.\n"
            "The agency is City of Indianapolis. IC 5-14-3-3(a) applies.\n"
        )
        sections = {}

    return ParsedOpinion(
        opinion_id=opinion_id,
        full_text=full_text,
        pages=[ParsedPage(page_num=1, text=full_text, char_count=len(full_text), ocr_used=False)],
        sections=sections,
        parser_version="0.1.0",
    )


class TestCountTokens:
    def test_counts_words(self, tokenizer: _FakeTokenizer) -> None:
        assert count_tokens("hello world foo", tokenizer) == 3

    def test_empty_string(self, tokenizer: _FakeTokenizer) -> None:
        assert count_tokens("", tokenizer) == 0


class TestSplitParagraphs:
    def test_short_paragraphs_pack_together(self, tokenizer: _FakeTokenizer) -> None:
        paragraphs = ["word " * 50, "word " * 50, "word " * 50]
        chunks = split_paragraphs_into_chunks(
            paragraphs, tokenizer, target_min=100, target_max=200, overlap_tokens=25
        )
        assert len(chunks) == 1
        assert "word" in chunks[0]

    def test_respects_max_tokens(self, tokenizer: _FakeTokenizer) -> None:
        paragraphs = ["word " * 100, "word " * 100, "word " * 100]
        chunks = split_paragraphs_into_chunks(
            paragraphs, tokenizer, target_min=50, target_max=150, overlap_tokens=25
        )
        assert len(chunks) >= 2

    def test_single_long_paragraph_becomes_own_chunk(self, tokenizer: _FakeTokenizer) -> None:
        paragraphs = ["word " * 800]
        chunks = split_paragraphs_into_chunks(
            paragraphs, tokenizer, target_min=300, target_max=600, overlap_tokens=75
        )
        assert len(chunks) == 1
        assert "word" in chunks[0]

    def test_overlap_between_chunks(self, tokenizer: _FakeTokenizer) -> None:
        paragraphs = [f"para{i} " + "word " * 99 for i in range(5)]
        chunks = split_paragraphs_into_chunks(
            paragraphs, tokenizer, target_min=50, target_max=150, overlap_tokens=25
        )
        assert len(chunks) >= 2
        for i in range(1, len(chunks)):
            assert chunks[i].startswith("para")

    def test_empty_paragraphs(self, tokenizer: _FakeTokenizer) -> None:
        chunks = split_paragraphs_into_chunks(
            [], tokenizer, target_min=100, target_max=200, overlap_tokens=25
        )
        assert chunks == []

    def test_single_short_paragraph(self, tokenizer: _FakeTokenizer) -> None:
        paragraphs = ["just a few words"]
        chunks = split_paragraphs_into_chunks(
            paragraphs, tokenizer, target_min=100, target_max=200, overlap_tokens=25
        )
        assert len(chunks) == 1
        assert chunks[0] == "just a few words"


class TestChunkOpinion:
    def test_produces_card_chunk(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed()
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        card_chunks = [c for c in chunks if c.section == "card"]
        assert len(card_chunks) == 1
        card = card_chunks[0]
        assert card.chunk_index == 0
        assert card.chunk_id == "25-FC-118#c0"
        assert "violation" in card.text.lower()
        assert "City of Bloomington" in card.text

    def test_produces_section_chunks(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed()
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        sections = {c.section for c in chunks}
        assert "card" in sections
        assert "background" in sections
        assert "analysis" in sections
        assert "conclusion" in sections

    def test_denormalized_metadata(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed()
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        for chunk in chunks:
            assert chunk.opinion_id == "25-FC-118"
            assert chunk.year == 2025
            assert chunk.agency_type == "city"
            assert chunk.statute_areas == ("APRA",)
            assert chunk.finding == "violation"
            assert chunk.issue_tags == ("denial-of-access",)

    def test_chunk_ids_sequential(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed()
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.chunk_id == f"25-FC-118#c{i}"

    def test_token_counts_positive(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed()
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_page_num_assigned(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed()
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        for chunk in chunks:
            assert chunk.page_num >= 1

    def test_no_sections_falls_back(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed(with_sections=False)
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        assert len(chunks) >= 2
        card_chunks = [c for c in chunks if c.section == "card"]
        assert len(card_chunks) == 1
        body_chunks = [c for c in chunks if c.section == "full_text"]
        assert len(body_chunks) >= 1

    def test_card_chunk_includes_summary(self, tokenizer: _FakeTokenizer) -> None:
        parsed = _make_parsed()
        metadata = _make_metadata()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        card = next(c for c in chunks if c.section == "card")
        assert metadata.summary in card.text

    def test_multi_page_opinion(self, tokenizer: _FakeTokenizer) -> None:
        page1_text = "Page 1 content. " * 50 + "\n\nBACKGROUND\n\nThe facts."
        page2_text = "\n\nANALYSIS\n\nThe analysis here."
        page3_text = "\n\nCONCLUSION\n\nThe conclusion."
        full_text = page1_text + page2_text + page3_text

        bg_start = full_text.index("BACKGROUND\n\n") + len("BACKGROUND\n\n")
        bg_end = full_text.index("\n\nANALYSIS")
        an_start = full_text.index("ANALYSIS\n\n") + len("ANALYSIS\n\n")
        an_end = full_text.index("\n\nCONCLUSION")
        co_start = full_text.index("CONCLUSION\n\n") + len("CONCLUSION\n\n")
        co_end = len(full_text)

        parsed = ParsedOpinion(
            opinion_id="25-FC-200",
            full_text=full_text,
            pages=[
                ParsedPage(page_num=1, text=page1_text, char_count=len(page1_text), ocr_used=False),
                ParsedPage(page_num=2, text=page2_text, char_count=len(page2_text), ocr_used=False),
                ParsedPage(page_num=3, text=page3_text, char_count=len(page3_text), ocr_used=False),
            ],
            sections={
                "BACKGROUND": (bg_start, bg_end),
                "ANALYSIS": (an_start, an_end),
                "CONCLUSION": (co_start, co_end),
            },
            parser_version="0.1.0",
        )
        metadata = _make_metadata("25-FC-200")
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)

        analysis_chunk = next(c for c in chunks if c.section == "analysis")
        assert analysis_chunk.page_num == 2

        conclusion_chunk = next(c for c in chunks if c.section == "conclusion")
        assert conclusion_chunk.page_num == 3

    def test_card_chunk_truncated_to_limit(self, tokenizer: _FakeTokenizer) -> None:
        long_summary = "word " * 500
        metadata = OpinionMetadata(
            opinion_id="25-FC-118",
            issue_date=date(2026, 2, 2),
            counselor="Jennifer G. Ruby",
            complainant="David Askins",
            public_agency="City of Bloomington",
            agency_type="city",
            statutes_invoked=["IC 5-14-3-4(b)(20)"],
            statute_areas=["APRA"],
            issue_tags=["denial-of-access"],
            tags_need_review=False,
            finding="violation",
            summary=long_summary.strip(),
        )
        parsed = _make_parsed()
        chunks = chunk_opinion(parsed, metadata, year=2025, tokenizer=tokenizer)
        card = next(c for c in chunks if c.section == "card")
        assert card.token_count <= 480

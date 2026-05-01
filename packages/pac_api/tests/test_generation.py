from __future__ import annotations

from pac_api.generation import SYSTEM_PROMPT, format_context
from pac_pipeline.chunk_models import Chunk
from pac_pipeline.retrieve_models import RetrievedChunk


def _make_chunk(
    opinion_id: str = "25-FC-118",
    page_num: int = 1,
    section: str = "analysis",
    text: str = "The agency violated APRA by failing to respond.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{opinion_id}#c0",
            opinion_id=opinion_id,
            chunk_index=0,
            section=section,
            page_num=page_num,
            text=text,
            token_count=50,
            year=2025,
            agency_type="city",
            statute_areas=("APRA",),
            finding="violation",
            issue_tags=("denial-of-access",),
        ),
        dense_rank=0,
        lexical_rank=1,
        fused_score=0.85,
        rerank_score=0.92,
    )


class TestSystemPrompt:
    def test_mentions_pac_opinions(self) -> None:
        assert "Public Access Counselor" in SYSTEM_PROMPT

    def test_requires_citation_format(self) -> None:
        assert "(opinion_id, p. N)" in SYSTEM_PROMPT

    def test_requires_dont_address(self) -> None:
        assert "don't directly address" in SYSTEM_PROMPT.lower()

    def test_requires_disclaimer(self) -> None:
        assert "advisory" in SYSTEM_PROMPT.lower()
        assert "not binding" in SYSTEM_PROMPT.lower()

    def test_forbids_fabrication(self) -> None:
        assert "fabricat" in SYSTEM_PROMPT.lower()


class TestFormatContext:
    def test_single_chunk(self) -> None:
        chunk = _make_chunk()
        result = format_context([chunk])
        assert "[Opinion 25-FC-118, p. 1, analysis]" in result
        assert "The agency violated APRA" in result

    def test_multiple_chunks_separated(self) -> None:
        chunk1 = _make_chunk(opinion_id="25-FC-118", page_num=1)
        chunk2 = _make_chunk(opinion_id="24-INF-01", page_num=3, section="background")
        result = format_context([chunk1, chunk2])
        assert "---" in result
        assert "[Opinion 25-FC-118, p. 1, analysis]" in result
        assert "[Opinion 24-INF-01, p. 3, background]" in result

    def test_empty_chunks(self) -> None:
        result = format_context([])
        assert result == ""

    def test_preserves_chunk_text(self) -> None:
        text = "IC 5-14-3-4(b)(8) requires agencies to respond within 24 hours."
        chunk = _make_chunk(text=text)
        result = format_context([chunk])
        assert text in result

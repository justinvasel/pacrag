from __future__ import annotations

from pac_api.citations import check_response, parse_citations, validate_citations
from pac_api.schemas import CitationRef


class TestParseCitations:
    def test_single_citation(self) -> None:
        text = "The agency violated APRA (25-FC-118, p. 3)."
        result = parse_citations(text)
        assert result == [CitationRef(opinion_id="25-FC-118", page_num=3)]

    def test_multiple_citations(self) -> None:
        text = (
            "The PAC found a violation (25-FC-118, p. 3) "
            "and cited a prior opinion (24-FC-077, p. 12)."
        )
        result = parse_citations(text)
        assert len(result) == 2
        assert result[0] == CitationRef(opinion_id="25-FC-118", page_num=3)
        assert result[1] == CitationRef(opinion_id="24-FC-077", page_num=12)

    def test_sub_letter_citation(self) -> None:
        text = "See companion opinion (24-FC-077A, p. 5)."
        result = parse_citations(text)
        assert result == [CitationRef(opinion_id="24-FC-077A", page_num=5)]

    def test_sub_letter_b(self) -> None:
        text = "Also (24-FC-077B, p. 1)."
        result = parse_citations(text)
        assert result == [CitationRef(opinion_id="24-FC-077B", page_num=1)]

    def test_informal_opinion_citation(self) -> None:
        text = "An informal inquiry (24-INF-01, p. 2) addressed this."
        result = parse_citations(text)
        assert result == [CitationRef(opinion_id="24-INF-01", page_num=2)]

    def test_no_citations(self) -> None:
        text = "This text has no citations at all."
        result = parse_citations(text)
        assert result == []

    def test_empty_text(self) -> None:
        result = parse_citations("")
        assert result == []

    def test_malformed_not_matched(self) -> None:
        text = "Something (not-a-citation, p. 3) here."
        result = parse_citations(text)
        assert result == []

    def test_duplicate_citations(self) -> None:
        text = "First (25-FC-118, p. 3) and again (25-FC-118, p. 3)."
        result = parse_citations(text)
        assert len(result) == 2

    def test_varied_spacing(self) -> None:
        text = "Citation (25-FC-118,  p.  3) with extra spaces."
        result = parse_citations(text)
        assert result == [CitationRef(opinion_id="25-FC-118", page_num=3)]

    def test_multiline_text(self) -> None:
        text = "Line one (25-FC-118, p. 1).\nLine two (24-INF-05, p. 7)."
        result = parse_citations(text)
        assert len(result) == 2


class TestValidateCitations:
    def test_all_valid(self) -> None:
        citations = [
            CitationRef(opinion_id="25-FC-118", page_num=3),
            CitationRef(opinion_id="24-INF-01", page_num=2),
        ]
        retrieved_ids = frozenset({"25-FC-118", "24-INF-01"})
        valid, invalid = validate_citations(citations, retrieved_ids)
        assert len(valid) == 2
        assert len(invalid) == 0
        assert all(c.valid for c in valid)

    def test_one_invalid(self) -> None:
        citations = [
            CitationRef(opinion_id="25-FC-118", page_num=3),
            CitationRef(opinion_id="99-FC-999", page_num=1),
        ]
        retrieved_ids = frozenset({"25-FC-118"})
        valid, invalid = validate_citations(citations, retrieved_ids)
        assert len(valid) == 1
        assert len(invalid) == 1
        assert invalid[0].opinion_id == "99-FC-999"
        assert not invalid[0].valid

    def test_all_invalid(self) -> None:
        citations = [
            CitationRef(opinion_id="99-FC-999", page_num=1),
        ]
        retrieved_ids = frozenset({"25-FC-118"})
        valid, invalid = validate_citations(citations, retrieved_ids)
        assert len(valid) == 0
        assert len(invalid) == 1

    def test_empty_citations(self) -> None:
        valid, invalid = validate_citations([], frozenset({"25-FC-118"}))
        assert valid == []
        assert invalid == []

    def test_empty_retrieved_set(self) -> None:
        citations = [CitationRef(opinion_id="25-FC-118", page_num=3)]
        valid, invalid = validate_citations(citations, frozenset())
        assert len(valid) == 0
        assert len(invalid) == 1


class TestCheckResponse:
    def test_valid_response(self) -> None:
        text = "The agency violated APRA (25-FC-118, p. 3)."
        retrieved_ids = frozenset({"25-FC-118"})
        citations, all_valid = check_response(text, retrieved_ids)
        assert all_valid
        assert len(citations) == 1
        assert citations[0].valid

    def test_corrupted_response_rejected(self) -> None:
        text = (
            "According to (25-FC-118, p. 3), the agency violated APRA. "
            "This is also supported by (99-FC-999, p. 1) which found a violation."
        )
        retrieved_ids = frozenset({"25-FC-118"})
        citations, all_valid = check_response(text, retrieved_ids)
        assert not all_valid
        invalid = [c for c in citations if not c.valid]
        assert len(invalid) == 1
        assert invalid[0].opinion_id == "99-FC-999"

    def test_no_citations_passes(self) -> None:
        text = "The retrieved opinions don't directly address this."
        citations, all_valid = check_response(text, frozenset({"25-FC-118"}))
        assert all_valid
        assert citations == []

    def test_multiple_fabricated_citations(self) -> None:
        text = "See (99-FC-001, p. 1) and (99-INF-002, p. 5) for more details."
        retrieved_ids = frozenset({"25-FC-118"})
        citations, all_valid = check_response(text, retrieved_ids)
        assert not all_valid
        assert len([c for c in citations if not c.valid]) == 2

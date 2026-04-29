from __future__ import annotations

from pac_pipeline.section_detector import _find_section_headers, detect_sections

OPINION_WITH_ALL_SECTIONS = """\
STATE OF INDIANA   PUBLIC ACCESS COUNSELOR

Re: Complaint 25-FC-001
     John Doe v. City of Springfield

This advisory opinion is in response to the above complaint.

BACKGROUND

The complainant alleges that the City of Springfield failed to respond
to a public records request within the required timeframe. The request
was made on January 15, 2025.

ANALYSIS

APRA requires that a public agency respond to a records request within
a reasonable time. IC 5-14-3-9(a) states that a denial must be in writing.

The City argues that the request was overly broad. However, a public
agency must still provide a timely response indicating this.

CONCLUSION

Based on the foregoing, this office concludes that the City of Springfield
violated APRA by failing to respond to the records request in a timely manner.

Luke H. Britt
Public Access Counselor
"""

OPINION_NO_BACKGROUND = """\
STATE OF INDIANA   PUBLIC ACCESS COUNSELOR

Re: Complaint 25-FC-118
     David Askins v. City of Bloomington

This advisory opinion is in response to the above complaint filed June 6, 2025.

ANALYSIS

The public policy of APRA states that providing persons with information is
an essential function of representative government. IC 5-14-3-1. Respondent
is a public agency for purposes of APRA.

CONCLUSION

This office finds that the Respondent did not violate APRA.

Jennifer G. Ruby
Public Access Counselor
"""

INFORMAL_OPINION = """\
STATE OF INDIANA   PUBLIC ACCESS COUNSELOR

Re: Informal Inquiry 24-INF-05

Thank you for contacting this office. You have asked about the
applicability of the Open Door Law to virtual meetings.

The Open Door Law generally requires that meetings of governing
bodies be open to the public. IC 5-14-1.5-3.

We hope this information is helpful.

Joe Hoage
Public Access Counselor
"""

OPINION_WITH_DISCUSSION = """\
STATE OF INDIANA   PUBLIC ACCESS COUNSELOR

Re: Complaint 10-FC-42

BACKGROUND

The complainant filed a public records request.

DISCUSSION

This office has reviewed the matter and finds that the agency
complied with APRA requirements.

CONCLUSION

No violation found.

Heather Willis Neal
Public Access Counselor
"""

OPINION_WITH_SIGNATURE_VARIATIONS = """\
STATE OF INDIANA   PUBLIC ACCESS COUNSELOR

Re: Complaint 23-FC-200

ANALYSIS

Some analysis text here that discusses the matter at length.

CONCLUSION

Based on the analysis, this office finds no violation.


    Luke H. Britt
    Public Access Counselor



    Page 3 of 3
"""


class TestDetectSections:
    def test_all_three_sections(self) -> None:
        sections = detect_sections(OPINION_WITH_ALL_SECTIONS, "25-FC-001")
        assert "BACKGROUND" in sections
        assert "ANALYSIS" in sections
        assert "CONCLUSION" in sections

    def test_section_content_is_correct(self) -> None:
        sections = detect_sections(OPINION_WITH_ALL_SECTIONS, "25-FC-001")
        bg_start, bg_end = sections["BACKGROUND"]
        bg_text = OPINION_WITH_ALL_SECTIONS[bg_start:bg_end]
        assert "complainant alleges" in bg_text
        assert "January 15, 2025" in bg_text

    def test_analysis_content_is_correct(self) -> None:
        sections = detect_sections(OPINION_WITH_ALL_SECTIONS, "25-FC-001")
        a_start, a_end = sections["ANALYSIS"]
        a_text = OPINION_WITH_ALL_SECTIONS[a_start:a_end]
        assert "APRA requires" in a_text
        assert "timely response" in a_text

    def test_conclusion_content_is_correct(self) -> None:
        sections = detect_sections(OPINION_WITH_ALL_SECTIONS, "25-FC-001")
        c_start, c_end = sections["CONCLUSION"]
        c_text = OPINION_WITH_ALL_SECTIONS[c_start:c_end]
        assert "violated APRA" in c_text

    def test_no_background(self) -> None:
        sections = detect_sections(OPINION_NO_BACKGROUND, "25-FC-118")
        assert "BACKGROUND" not in sections
        assert "ANALYSIS" in sections
        assert "CONCLUSION" in sections

    def test_no_sections_found(self) -> None:
        sections = detect_sections(INFORMAL_OPINION, "24-INF-05")
        assert sections == {}

    def test_discussion_section(self) -> None:
        sections = detect_sections(OPINION_WITH_DISCUSSION, "10-FC-42")
        assert "DISCUSSION" in sections
        assert "BACKGROUND" in sections
        assert "CONCLUSION" in sections

    def test_sections_dont_include_header_text(self) -> None:
        sections = detect_sections(OPINION_WITH_ALL_SECTIONS, "25-FC-001")
        bg_start, _bg_end = sections["BACKGROUND"]
        bg_text = OPINION_WITH_ALL_SECTIONS[bg_start : bg_start + 20]
        assert "BACKGROUND" not in bg_text

    def test_sections_dont_overlap(self) -> None:
        sections = detect_sections(OPINION_WITH_ALL_SECTIONS, "25-FC-001")
        sorted_sections = sorted(sections.values(), key=lambda x: x[0])
        for i in range(len(sorted_sections) - 1):
            _start_a, end_a = sorted_sections[i]
            start_b, _end_b = sorted_sections[i + 1]
            assert end_a <= start_b

    def test_conclusion_excludes_signature(self) -> None:
        sections = detect_sections(OPINION_WITH_ALL_SECTIONS, "25-FC-001")
        c_start, c_end = sections["CONCLUSION"]
        c_text = OPINION_WITH_ALL_SECTIONS[c_start:c_end]
        assert "Public Access Counselor" not in c_text

    def test_conclusion_excludes_page_markers(self) -> None:
        sections = detect_sections(OPINION_WITH_SIGNATURE_VARIATIONS, "23-FC-200")
        c_start, c_end = sections["CONCLUSION"]
        c_text = OPINION_WITH_SIGNATURE_VARIATIONS[c_start:c_end]
        assert "Page 3 of 3" not in c_text

    def test_handles_leading_whitespace_on_headers(self) -> None:
        text = "Intro text.\n  ANALYSIS  \nSome analysis.\n  CONCLUSION  \nDone."
        sections = detect_sections(text)
        assert "ANALYSIS" in sections
        assert "CONCLUSION" in sections

    def test_case_sensitive(self) -> None:
        text = "The analysis shows that APRA was violated.\nConclusion: no issue."
        sections = detect_sections(text)
        assert sections == {}

    def test_empty_text(self) -> None:
        sections = detect_sections("")
        assert sections == {}

    def test_header_only_no_content(self) -> None:
        text = "ANALYSIS\nCONCLUSION"
        sections = detect_sections(text)
        assert "ANALYSIS" in sections


class TestFindSectionHeaders:
    def test_finds_all_standard_headers(self) -> None:
        text = "intro\nBACKGROUND\nbg text\nANALYSIS\nanalysis text\nCONCLUSION\nconc"
        headers = _find_section_headers(text)
        names = [name for name, _, _ in headers]
        assert "BACKGROUND" in names
        assert "ANALYSIS" in names
        assert "CONCLUSION" in names

    def test_sorted_by_position(self) -> None:
        text = "intro\nCONCLUSION\nconc\nBACKGROUND\nbg\nANALYSIS\nanal"
        headers = _find_section_headers(text)
        positions = [start for _, start, _ in headers]
        assert positions == sorted(positions)

    def test_returns_header_end_after_newline(self) -> None:
        text = "ANALYSIS\nSome text"
        headers = _find_section_headers(text)
        assert len(headers) == 1
        _name, _start, end = headers[0]
        assert text[end:].startswith("Some text")

    def test_no_headers(self) -> None:
        text = "Just some ordinary text with no section markers."
        headers = _find_section_headers(text)
        assert headers == []

    def test_discussion_header(self) -> None:
        text = "DISCUSSION\nSome discussion."
        headers = _find_section_headers(text)
        assert len(headers) == 1
        assert headers[0][0] == "DISCUSSION"

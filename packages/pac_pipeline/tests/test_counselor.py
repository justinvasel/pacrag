from __future__ import annotations

from pac_pipeline.counselor import extract_counselor


class TestExtractCounselor:
    def test_extracts_luke_britt(self) -> None:
        text = (
            "Some opinion text here.\n\n"
            "CONCLUSION\n\n"
            "Based on the foregoing, it is the opinion of the Public Access Counselor "
            "that the agency violated APRA.\n\n"
            "    Luke H. Britt\n"
            "    Public Access Counselor"
        )
        assert extract_counselor(text) == "Luke H. Britt"

    def test_extracts_jennifer_ruby(self) -> None:
        text = (
            "Opinion text.\n\n"
            "CONCLUSION\n\n"
            "The complaint is sustained.\n\n"
            "Jennifer G. Ruby\n"
            "Public Access Counselor\n"
        )
        assert extract_counselor(text) == "Jennifer G. Ruby"

    def test_extracts_joseph_hoage(self) -> None:
        text = "Analysis text.\n\n  Joseph B. Hoage\n  Public Access Counselor\n"
        assert extract_counselor(text) == "Joseph B. Hoage"

    def test_extracts_andrew_kossack(self) -> None:
        text = "Some text.\n\nAndrew J. Kossack\nPublic Access Counselor\n"
        assert extract_counselor(text) == "Andrew J. Kossack"

    def test_extracts_heather_willis_neal(self) -> None:
        text = "Conclusion text.\n\nHeather Willis Neal\nPublic Access Counselor"
        assert extract_counselor(text) == "Heather Willis Neal"

    def test_extracts_karen_davis(self) -> None:
        text = "Conclusion.\n\n    Karen Davis\n    Public Access Counselor\n"
        assert extract_counselor(text) == "Karen Davis"

    def test_returns_none_when_no_signature(self) -> None:
        text = "This is just some opinion text without a signature block."
        assert extract_counselor(text) is None

    def test_returns_none_for_empty_text(self) -> None:
        assert extract_counselor("") is None

    def test_does_not_match_complainant_name(self) -> None:
        text = (
            "Re: Formal Complaint 25-FC-118\n"
            "    David Askins (Complainant) v.\n"
            "    City of Bloomington (Respondent)\n\n"
            "This is the opinion text without a proper signature.\n"
        )
        assert extract_counselor(text) is None

    def test_handles_extra_whitespace(self) -> None:
        text = "Text.\n\n     Luke H. Britt   \n     Public Access Counselor  \n"
        assert extract_counselor(text) == "Luke H. Britt"

    def test_extracts_name_without_middle_initial(self) -> None:
        text = "Conclusion.\n\nKaren Davis\nPublic Access Counselor\n"
        assert extract_counselor(text) == "Karen Davis"

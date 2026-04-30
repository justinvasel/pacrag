from __future__ import annotations

from datetime import date

import pytest
from pac_pipeline.extract_models import (
    EXTRACTOR_VERSION,
    ISSUE_TAGS,
    LLMExtractionResponse,
    OpinionMetadata,
)
from pydantic import ValidationError


class TestOpinionMetadata:
    def test_valid_construction(self) -> None:
        meta = OpinionMetadata(
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
            summary="The PAC found that the City of Bloomington violated APRA.",
        )
        assert meta.opinion_id == "25-FC-118"
        assert meta.issue_date == date(2026, 2, 2)
        assert meta.agency_type == "city"
        assert meta.finding == "violation"

    def test_null_optional_fields(self) -> None:
        meta = OpinionMetadata(
            opinion_id="25-FC-100",
            issue_date=None,
            counselor=None,
            complainant=None,
            public_agency="Some Agency",
            agency_type="state_agency",
            statutes_invoked=[],
            statute_areas=["APRA"],
            issue_tags=[],
            tags_need_review=False,
            finding="no_violation",
            summary="No violation found.",
        )
        assert meta.issue_date is None
        assert meta.counselor is None
        assert meta.complainant is None

    def test_invalid_issue_tags_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid issue tags"):
            OpinionMetadata(
                opinion_id="25-FC-100",
                issue_date=None,
                counselor=None,
                complainant=None,
                public_agency="Some Agency",
                agency_type="city",
                statutes_invoked=[],
                statute_areas=["APRA"],
                issue_tags=["not-a-real-tag"],
                tags_need_review=False,
                finding="no_violation",
                summary="Summary.",
            )

    def test_invalid_agency_type_rejected(self) -> None:
        with pytest.raises(ValueError):
            OpinionMetadata(
                opinion_id="25-FC-100",
                issue_date=None,
                counselor=None,
                complainant=None,
                public_agency="Some Agency",
                agency_type="invalid_type",  # type: ignore[arg-type]
                statutes_invoked=[],
                statute_areas=["APRA"],
                issue_tags=[],
                tags_need_review=False,
                finding="no_violation",
                summary="Summary.",
            )

    def test_invalid_finding_rejected(self) -> None:
        with pytest.raises(ValueError):
            OpinionMetadata(
                opinion_id="25-FC-100",
                issue_date=None,
                counselor=None,
                complainant=None,
                public_agency="Some Agency",
                agency_type="city",
                statutes_invoked=[],
                statute_areas=["APRA"],
                issue_tags=[],
                tags_need_review=False,
                finding="guilty",  # type: ignore[arg-type]
                summary="Summary.",
            )

    def test_frozen(self) -> None:
        meta = OpinionMetadata(
            opinion_id="25-FC-100",
            issue_date=None,
            counselor=None,
            complainant=None,
            public_agency="Some Agency",
            agency_type="city",
            statutes_invoked=[],
            statute_areas=["APRA"],
            issue_tags=[],
            tags_need_review=False,
            finding="no_violation",
            summary="Summary.",
        )
        with pytest.raises(ValidationError, match="frozen_instance"):
            meta.opinion_id = "changed"  # type: ignore[misc]

    def test_multiple_statute_areas(self) -> None:
        meta = OpinionMetadata(
            opinion_id="24-INF-01",
            issue_date=date(2024, 2, 1),
            counselor="Luke H. Britt",
            complainant=None,
            public_agency="Indiana Office of Court Services",
            agency_type="state_agency",
            statutes_invoked=["IC 5-14-1.5-2"],
            statute_areas=["APRA", "ODL"],
            issue_tags=["electronic-meetings"],
            tags_need_review=False,
            finding="no_violation",
            summary="The PAC addressed Open Door Law applicability.",
        )
        assert meta.statute_areas == ["APRA", "ODL"]

    def test_multiple_issue_tags(self) -> None:
        meta = OpinionMetadata(
            opinion_id="25-FC-100",
            issue_date=None,
            counselor=None,
            complainant=None,
            public_agency="Some Agency",
            agency_type="city",
            statutes_invoked=[],
            statute_areas=["APRA"],
            issue_tags=["denial-of-access", "response-timeline", "fee-charging"],
            tags_need_review=False,
            finding="mixed",
            summary="Mixed findings.",
        )
        assert len(meta.issue_tags) == 3

    def test_tags_need_review(self) -> None:
        meta = OpinionMetadata(
            opinion_id="25-FC-100",
            issue_date=None,
            counselor=None,
            complainant=None,
            public_agency="Some Agency",
            agency_type="city",
            statutes_invoked=[],
            statute_areas=["APRA"],
            issue_tags=[],
            tags_need_review=True,
            finding="no_violation",
            summary="Summary.",
        )
        assert meta.tags_need_review is True


class TestLLMExtractionResponse:
    def test_valid_construction(self) -> None:
        resp = LLMExtractionResponse(
            issue_date="2026-02-02",
            complainant="David Askins",
            public_agency="City of Bloomington",
            agency_type="city",
            statutes_invoked=["IC 5-14-3-4(b)(20)"],
            statute_areas=["APRA"],
            issue_tags=["denial-of-access"],
            tags_need_review=False,
            finding="violation",
            summary="The PAC found a violation.",
        )
        assert resp.public_agency == "City of Bloomington"
        assert resp.issue_date == "2026-02-02"

    def test_null_complainant(self) -> None:
        resp = LLMExtractionResponse(
            issue_date=None,
            complainant=None,
            public_agency="Some Agency",
            agency_type="state_agency",
            statutes_invoked=[],
            statute_areas=["ODL"],
            issue_tags=[],
            tags_need_review=True,
            finding="no_violation",
            summary="No violation.",
        )
        assert resp.complainant is None
        assert resp.tags_need_review is True

    def test_json_schema_generation(self) -> None:
        schema = LLMExtractionResponse.model_json_schema()
        assert "properties" in schema
        assert "public_agency" in schema["properties"]
        assert "finding" in schema["properties"]
        assert "issue_tags" in schema["properties"]
        assert "tags_need_review" in schema["properties"]


class TestIssueTagsVocabulary:
    def test_has_minimum_tags(self) -> None:
        assert len(ISSUE_TAGS) >= 25

    def test_no_duplicates(self) -> None:
        assert len(ISSUE_TAGS) == len(set(ISSUE_TAGS))

    def test_expected_tags_present(self) -> None:
        expected = [
            "deliberative-materials",
            "executive-session-purpose",
            "fee-charging",
            "response-timeline",
            "attorney-client-privilege",
            "serial-meetings",
        ]
        for tag in expected:
            assert tag in ISSUE_TAGS


class TestExtractorVersion:
    def test_version_format(self) -> None:
        parts = EXTRACTOR_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

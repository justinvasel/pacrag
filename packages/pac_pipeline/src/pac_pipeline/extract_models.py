from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from pydantic import BaseModel, field_validator

EXTRACTOR_VERSION: str = "0.3.0"

ISSUE_TAGS: list[str] = [
    "deliberative-materials",
    "executive-session-purpose",
    "executive-session-notice",
    "executive-session-minutes",
    "fee-charging",
    "response-timeline",
    "reasonable-particularity",
    "personnel-files",
    "investigatory-records",
    "attorney-client-privilege",
    "commercial-trade-secrets",
    "serial-meetings",
    "electronic-meetings",
    "public-meeting-notice",
    "meeting-memoranda",
    "recording-meetings",
    "denial-of-access",
    "failure-to-respond",
    "redaction",
    "copying-charges",
    "format-of-records",
    "law-enforcement-records",
    "medical-records",
    "student-records",
    "adoption-records",
    "juvenile-records",
    "utility-customer-records",
    "draft-documents",
    "email-as-public-record",
    "social-media-records",
    "body-camera-footage",
    "audio-recordings",
    "public-agency-definition",
    "standing-to-complain",
]

ISSUE_TAGS_SET: frozenset[str] = frozenset(ISSUE_TAGS)

AgencyType = Literal[
    "county",
    "city",
    "town",
    "school_corporation",
    "state_agency",
    "law_enforcement",
    "court",
    "library",
    "utility",
    "other",
]

Finding = Literal[
    "violation",
    "no_violation",
    "mixed",
    "moot",
    "no_jurisdiction",
    "withdrawn",
]

StatuteArea = Literal["APRA", "ODL"]


class OpinionMetadata(BaseModel, frozen=True):
    opinion_id: str
    issue_date: date | None
    counselor: str | None
    complainant: str | None
    public_agency: str
    agency_type: AgencyType
    statutes_invoked: list[str]
    statute_areas: list[StatuteArea]
    issue_tags: list[str]
    tags_need_review: bool
    finding: Finding
    summary: str

    @field_validator("issue_tags")
    @classmethod
    def validate_issue_tags(cls, v: list[str]) -> list[str]:
        invalid = [t for t in v if t not in ISSUE_TAGS_SET]
        if invalid:
            raise ValueError(f"Invalid issue tags: {invalid}. Must be from controlled vocabulary.")
        return v


class LLMExtractionResponse(BaseModel, frozen=True):
    issue_date: str | None
    complainant: str | None
    public_agency: str
    agency_type: AgencyType
    statutes_invoked: list[str]
    statute_areas: list[StatuteArea]
    issue_tags: list[str]
    tags_need_review: bool
    finding: Finding
    summary: str


@dataclass(frozen=True)
class RejectedTagSuggestion:
    opinion_id: str
    suggested_tag: str
    attempt: int
    all_suggested_tags: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionResult:
    metadata: OpinionMetadata
    rejected_tags: tuple[RejectedTagSuggestion, ...]

from __future__ import annotations

from datetime import date

from pac_pipeline.extract_models import AgencyType, Finding, StatuteArea
from pac_pipeline.retrieve_models import RetrievedChunk
from pydantic import BaseModel, Field


class AskRequest(BaseModel, frozen=True):
    question: str = Field(min_length=1, max_length=2000)
    filters: dict[str, object] | None = None


class ChunkResponse(BaseModel, frozen=True):
    chunk_id: str
    opinion_id: str
    section: str
    page_num: int
    text: str
    score: float


class CitationRef(BaseModel, frozen=True):
    opinion_id: str
    page_num: int | None
    valid: bool = True


class AskResponseMeta(BaseModel, frozen=True):
    citations: list[CitationRef]
    applied_filters: dict[str, object]
    relaxed: bool
    chunks: list[ChunkResponse]


class SearchResponse(BaseModel, frozen=True):
    question: str
    applied_filters: dict[str, object]
    relaxed: bool
    chunks: list[ChunkResponse]


class OpinionSummary(BaseModel, frozen=True):
    opinion_id: str
    year: int
    agency_name: str
    agency_type: str
    finding: str
    statute_areas: list[str]
    summary: str
    counselor: str | None = None
    issue_date: date | None = None


class OpinionListResponse(BaseModel, frozen=True):
    items: list[OpinionSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class OpinionDetail(BaseModel, frozen=True):
    opinion_id: str
    year: int
    agency_name: str
    pdf_url: str | None
    metadata: OpinionMetadataResponse | None
    full_text: str | None
    sections: dict[str, str] | None


class OpinionMetadataResponse(BaseModel, frozen=True):
    issue_date: date | None
    counselor: str | None
    complainant: str | None
    public_agency: str
    agency_type: AgencyType
    statutes_invoked: list[str]
    statute_areas: list[StatuteArea]
    issue_tags: list[str]
    finding: Finding
    summary: str


class HealthResponse(BaseModel, frozen=True):
    status: str
    qdrant_chunks: int
    tantivy_chunks: int
    duckdb_chunks: int
    duckdb_opinions: int
    counts_match: bool


# Rebuild OpinionDetail since it forward-references OpinionMetadataResponse
OpinionDetail.model_rebuild()


def chunk_to_response(rc: RetrievedChunk) -> ChunkResponse:
    return ChunkResponse(
        chunk_id=rc.chunk.chunk_id,
        opinion_id=rc.chunk.opinion_id,
        section=rc.chunk.section,
        page_num=rc.chunk.page_num,
        text=rc.chunk.text,
        score=rc.rerank_score,
    )

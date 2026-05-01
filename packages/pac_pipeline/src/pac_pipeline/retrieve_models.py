from __future__ import annotations

from pydantic import BaseModel

from pac_pipeline.chunk_models import Chunk
from pac_pipeline.extract_models import AgencyType, Finding, StatuteArea

TOP_K_PER_SOURCE: int = 30
RRF_K: int = 60
TOP_K_FUSED: int = 30
TOP_K_RERANKED: int = 10
MIN_CANDIDATES_FOR_FILTERS: int = 8


class InferredFilters(BaseModel, frozen=True):
    year_min: int | None = None
    year_max: int | None = None
    agency_type: AgencyType | None = None
    statute_areas: list[StatuteArea] = []
    finding: Finding | None = None
    issue_tags: list[str] = []

    def has_any(self) -> bool:
        return (
            self.year_min is not None
            or self.year_max is not None
            or self.agency_type is not None
            or len(self.statute_areas) > 0
            or self.finding is not None
            or len(self.issue_tags) > 0
        )

    def to_applied_dict(self) -> dict[str, object]:
        d: dict[str, object] = {}
        if self.year_min is not None:
            d["year_min"] = self.year_min
        if self.year_max is not None:
            d["year_max"] = self.year_max
        if self.agency_type is not None:
            d["agency_type"] = self.agency_type
        if self.statute_areas:
            d["statute_areas"] = self.statute_areas
        if self.finding is not None:
            d["finding"] = self.finding
        if self.issue_tags:
            d["issue_tags"] = self.issue_tags
        return d


class RankedChunkRef(BaseModel, frozen=True):
    chunk_id: str
    rank: int
    score: float
    chunk: Chunk


class FusedChunk(BaseModel, frozen=True):
    chunk: Chunk
    dense_rank: int | None
    lexical_rank: int | None
    fused_score: float


class RetrievedChunk(BaseModel, frozen=True):
    chunk: Chunk
    dense_rank: int | None
    lexical_rank: int | None
    fused_score: float
    rerank_score: float


class RetrievalResult(BaseModel, frozen=True):
    question: str
    applied_filters: dict[str, object]
    relaxed: bool
    chunks: list[RetrievedChunk]

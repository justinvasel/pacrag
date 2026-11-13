from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GoldEntry(BaseModel, frozen=True):
    id: str
    question: str
    statute_area: Literal["APRA", "ODL", "both"] | None = None
    expected_opinions: list[str] = Field(min_length=1)
    ideal_answer: str
    key_facts: list[str] = Field(min_length=1)


class GoldSet(BaseModel, frozen=True):
    version: str
    entries: list[GoldEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldSet:
        ids = [e.id for e in self.entries]
        dupes = [i for i in ids if ids.count(i) > 1]
        if dupes:
            raise ValueError(f"duplicate entry ids: {sorted(set(dupes))}")
        return self


class RetrievalMetrics(BaseModel, frozen=True):
    recall_at_8: float
    recall_at_16: float
    mrr: float
    first_relevant_rank: int | None
    retrieved_opinion_ids: tuple[str, ...]


class AnswerMetrics(BaseModel, frozen=True):
    citation_precision: float
    citation_count: int
    valid_citation_count: int
    fact_coverage: float
    facts_found: tuple[str, ...]
    facts_missing: tuple[str, ...]
    hedging_present: bool
    overclaim_detected: bool
    hedging_score: float


class QuestionResult(BaseModel, frozen=True):
    entry_id: str
    question: str
    statute_area: str | None
    retrieval: RetrievalMetrics
    answer: AnswerMetrics | None = None
    generated_answer: str | None = None
    error: str | None = None
    duration_seconds: float


class ScorecardSummary(BaseModel, frozen=True):
    mean_recall_at_8: float
    mean_recall_at_16: float
    mean_mrr: float
    mean_citation_precision: float
    mean_fact_coverage: float
    mean_hedging_score: float
    hedging_rate: float
    questions_evaluated: int
    questions_with_errors: int


class Scorecard(BaseModel, frozen=True):
    run_id: str
    timestamp: datetime
    gold_set_version: str
    results: list[QuestionResult]
    summary: ScorecardSummary

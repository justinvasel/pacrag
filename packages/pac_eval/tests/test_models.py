from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pac_eval.models import (
    AnswerMetrics,
    GoldEntry,
    GoldSet,
    QuestionResult,
    RetrievalMetrics,
    Scorecard,
    ScorecardSummary,
)
from pydantic import ValidationError


class TestGoldEntry:
    def test_valid_entry(self) -> None:
        entry = GoldEntry(
            id="q001",
            question="Test question?",
            statute_area="APRA",
            expected_opinions=["25-FC-118"],
            ideal_answer="Expected answer.",
            key_facts=["Fact one"],
        )
        assert entry.id == "q001"
        assert entry.statute_area == "APRA"

    def test_statute_area_optional(self) -> None:
        entry = GoldEntry(
            id="q001",
            question="Test?",
            expected_opinions=["25-FC-118"],
            ideal_answer="Answer.",
            key_facts=["Fact"],
        )
        assert entry.statute_area is None

    def test_empty_expected_opinions_rejected(self) -> None:
        with pytest.raises(ValidationError, match="expected_opinions"):
            GoldEntry(
                id="q001",
                question="Test?",
                expected_opinions=[],
                ideal_answer="Answer.",
                key_facts=["Fact"],
            )

    def test_empty_key_facts_rejected(self) -> None:
        with pytest.raises(ValidationError, match="key_facts"):
            GoldEntry(
                id="q001",
                question="Test?",
                expected_opinions=["25-FC-118"],
                ideal_answer="Answer.",
                key_facts=[],
            )

    def test_frozen(self) -> None:
        entry = GoldEntry(
            id="q001",
            question="Test?",
            expected_opinions=["25-FC-118"],
            ideal_answer="Answer.",
            key_facts=["Fact"],
        )
        with pytest.raises(ValidationError):
            entry.id = "q002"  # type: ignore[misc]


class TestGoldSet:
    def test_valid_set(self) -> None:
        gs = GoldSet(
            version="1.0",
            entries=[
                GoldEntry(
                    id="q001",
                    question="Q1?",
                    expected_opinions=["25-FC-118"],
                    ideal_answer="A1.",
                    key_facts=["F1"],
                ),
                GoldEntry(
                    id="q002",
                    question="Q2?",
                    expected_opinions=["24-FC-42"],
                    ideal_answer="A2.",
                    key_facts=["F2"],
                ),
            ],
        )
        assert len(gs.entries) == 2

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate entry ids"):
            GoldSet(
                version="1.0",
                entries=[
                    GoldEntry(
                        id="q001",
                        question="Q1?",
                        expected_opinions=["25-FC-118"],
                        ideal_answer="A1.",
                        key_facts=["F1"],
                    ),
                    GoldEntry(
                        id="q001",
                        question="Q2?",
                        expected_opinions=["24-FC-42"],
                        ideal_answer="A2.",
                        key_facts=["F2"],
                    ),
                ],
            )

    def test_empty_entries_rejected(self) -> None:
        with pytest.raises(ValidationError, match="entries"):
            GoldSet(version="1.0", entries=[])


class TestRetrievalMetrics:
    def test_construction(self) -> None:
        m = RetrievalMetrics(
            recall_at_8=0.75,
            recall_at_16=1.0,
            mrr=0.5,
            first_relevant_rank=2,
            retrieved_opinion_ids=("25-FC-118", "24-FC-42"),
        )
        assert m.recall_at_8 == 0.75
        assert m.first_relevant_rank == 2

    def test_no_relevant_found(self) -> None:
        m = RetrievalMetrics(
            recall_at_8=0.0,
            recall_at_16=0.0,
            mrr=0.0,
            first_relevant_rank=None,
            retrieved_opinion_ids=(),
        )
        assert m.first_relevant_rank is None


class TestAnswerMetrics:
    def test_construction(self) -> None:
        m = AnswerMetrics(
            citation_precision=0.8,
            citation_count=5,
            valid_citation_count=4,
            fact_coverage=0.75,
            facts_found=("fact1", "fact2", "fact3"),
            facts_missing=("fact4",),
            hedging_present=True,
            overclaim_detected=False,
            hedging_score=1.0,
        )
        assert m.citation_precision == 0.8
        assert len(m.facts_found) == 3


class TestQuestionResult:
    def test_answer_optional(self) -> None:
        r = QuestionResult(
            entry_id="q001",
            question="Test?",
            statute_area="APRA",
            retrieval=RetrievalMetrics(
                recall_at_8=1.0,
                recall_at_16=1.0,
                mrr=1.0,
                first_relevant_rank=1,
                retrieved_opinion_ids=("25-FC-118",),
            ),
            duration_seconds=1.5,
        )
        assert r.answer is None
        assert r.generated_answer is None


class TestScorecard:
    def test_round_trip(self) -> None:
        sc = Scorecard(
            run_id="20260501T120000",
            timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
            gold_set_version="1.0",
            results=[],
            summary=ScorecardSummary(
                mean_recall_at_8=0.0,
                mean_recall_at_16=0.0,
                mean_mrr=0.0,
                mean_citation_precision=0.0,
                mean_fact_coverage=0.0,
                mean_hedging_score=0.0,
                hedging_rate=0.0,
                questions_evaluated=0,
                questions_with_errors=0,
            ),
        )
        assert sc.run_id == "20260501T120000"

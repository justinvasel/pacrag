from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pac_eval.models import (
    AnswerMetrics,
    QuestionResult,
    RetrievalMetrics,
    Scorecard,
    ScorecardSummary,
)
from pac_eval.scorecard import read_scorecard, write_scorecard


def _make_scorecard(run_id: str = "20260501T120000") -> Scorecard:
    return Scorecard(
        run_id=run_id,
        timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        gold_set_version="1.0",
        results=[
            QuestionResult(
                entry_id="q001",
                question="Test question?",
                statute_area="APRA",
                retrieval=RetrievalMetrics(
                    recall_at_8=1.0,
                    recall_at_16=1.0,
                    mrr=1.0,
                    first_relevant_rank=1,
                    retrieved_opinion_ids=("25-FC-118", "24-FC-42"),
                ),
                answer=AnswerMetrics(
                    citation_precision=0.8,
                    citation_count=5,
                    valid_citation_count=4,
                    fact_coverage=0.75,
                    facts_found=("fact1", "fact2", "fact3"),
                    facts_missing=("fact4",),
                    hedging_present=True,
                    overclaim_detected=False,
                    hedging_score=1.0,
                ),
                generated_answer="Test answer with citations.",
                duration_seconds=2.5,
            ),
            QuestionResult(
                entry_id="q002",
                question="Another question?",
                statute_area=None,
                retrieval=RetrievalMetrics(
                    recall_at_8=0.5,
                    recall_at_16=0.5,
                    mrr=0.5,
                    first_relevant_rank=2,
                    retrieved_opinion_ids=("22-FC-88",),
                ),
                duration_seconds=1.0,
            ),
        ],
        summary=ScorecardSummary(
            mean_recall_at_8=0.75,
            mean_recall_at_16=0.75,
            mean_mrr=0.75,
            mean_citation_precision=0.8,
            mean_fact_coverage=0.75,
            mean_hedging_score=1.0,
            hedging_rate=1.0,
            questions_evaluated=2,
            questions_with_errors=0,
        ),
    )


class TestWriteAndRead:
    def test_round_trip(self, tmp_path: Path) -> None:
        original = _make_scorecard()
        path = write_scorecard(original, tmp_path)
        assert path.exists()
        assert path.suffix == ".parquet"

        restored = read_scorecard(path)
        assert restored.run_id == original.run_id
        assert restored.gold_set_version == original.gold_set_version
        assert len(restored.results) == len(original.results)

        r0 = restored.results[0]
        assert r0.entry_id == "q001"
        assert r0.retrieval.recall_at_8 == 1.0
        assert r0.retrieval.retrieved_opinion_ids == ("25-FC-118", "24-FC-42")
        assert r0.answer is not None
        assert r0.answer.citation_precision == 0.8
        assert r0.answer.facts_found == ("fact1", "fact2", "fact3")
        assert r0.generated_answer == "Test answer with citations."

    def test_nullable_answer(self, tmp_path: Path) -> None:
        original = _make_scorecard()
        path = write_scorecard(original, tmp_path)
        restored = read_scorecard(path)

        r1 = restored.results[1]
        assert r1.entry_id == "q002"
        assert r1.answer is None
        assert r1.generated_answer is None
        assert r1.statute_area is None

    def test_summary_preserved(self, tmp_path: Path) -> None:
        original = _make_scorecard()
        path = write_scorecard(original, tmp_path)
        restored = read_scorecard(path)

        assert restored.summary.mean_recall_at_8 == original.summary.mean_recall_at_8
        assert restored.summary.questions_evaluated == 2

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "eval"
        sc = _make_scorecard()
        path = write_scorecard(sc, out)
        assert path.exists()

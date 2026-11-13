from __future__ import annotations

from datetime import UTC, datetime

from pac_eval.diff import diff_scorecards
from pac_eval.models import (
    AnswerMetrics,
    QuestionResult,
    RetrievalMetrics,
    Scorecard,
    ScorecardSummary,
)


def _make_result(
    entry_id: str,
    recall_at_8: float = 1.0,
    fact_coverage: float = 0.8,
) -> QuestionResult:
    return QuestionResult(
        entry_id=entry_id,
        question=f"Question {entry_id}?",
        statute_area="APRA",
        retrieval=RetrievalMetrics(
            recall_at_8=recall_at_8,
            recall_at_16=1.0,
            mrr=1.0,
            first_relevant_rank=1,
            retrieved_opinion_ids=("25-FC-118",),
        ),
        answer=AnswerMetrics(
            citation_precision=1.0,
            citation_count=2,
            valid_citation_count=2,
            fact_coverage=fact_coverage,
            facts_found=("f1",),
            facts_missing=(),
            hedging_present=True,
            overclaim_detected=False,
            hedging_score=1.0,
        ),
        generated_answer="Answer.",
        duration_seconds=1.0,
    )


def _make_scorecard(
    run_id: str,
    results: list[QuestionResult],
    mean_recall_at_8: float = 1.0,
    mean_fact_coverage: float = 0.8,
) -> Scorecard:
    return Scorecard(
        run_id=run_id,
        timestamp=datetime(2026, 5, 1, tzinfo=UTC),
        gold_set_version="1.0",
        results=results,
        summary=ScorecardSummary(
            mean_recall_at_8=mean_recall_at_8,
            mean_recall_at_16=1.0,
            mean_mrr=1.0,
            mean_citation_precision=1.0,
            mean_fact_coverage=mean_fact_coverage,
            mean_hedging_score=1.0,
            hedging_rate=1.0,
            questions_evaluated=len(results),
            questions_with_errors=0,
        ),
    )


class TestDiffScorecards:
    def test_identical_no_regressions(self) -> None:
        results = [_make_result("q001")]
        a = _make_scorecard("runA", results)
        b = _make_scorecard("runB", results)
        report = diff_scorecards(a, b)
        assert "No regressions detected" in report

    def test_regression_flagged(self) -> None:
        results_a = [_make_result("q001", recall_at_8=1.0)]
        results_b = [_make_result("q001", recall_at_8=0.5)]
        a = _make_scorecard("runA", results_a, mean_recall_at_8=1.0)
        b = _make_scorecard("runB", results_b, mean_recall_at_8=0.5)
        report = diff_scorecards(a, b)
        assert "REGRESSION" in report

    def test_improvement_not_flagged(self) -> None:
        results_a = [_make_result("q001", recall_at_8=0.5)]
        results_b = [_make_result("q001", recall_at_8=1.0)]
        a = _make_scorecard("runA", results_a, mean_recall_at_8=0.5)
        b = _make_scorecard("runB", results_b, mean_recall_at_8=1.0)
        report = diff_scorecards(a, b)
        assert "REGRESSION" not in report

    def test_per_question_regression(self) -> None:
        results_a = [_make_result("q001", fact_coverage=0.8)]
        results_b = [_make_result("q001", fact_coverage=0.3)]
        a = _make_scorecard("runA", results_a, mean_fact_coverage=0.8)
        b = _make_scorecard("runB", results_b, mean_fact_coverage=0.3)
        report = diff_scorecards(a, b)
        assert "q001" in report
        assert "fact_coverage" in report

    def test_threshold_controls_sensitivity(self) -> None:
        results_a = [_make_result("q001", recall_at_8=1.0)]
        results_b = [_make_result("q001", recall_at_8=0.995)]
        a = _make_scorecard("runA", results_a, mean_recall_at_8=1.0)
        b = _make_scorecard("runB", results_b, mean_recall_at_8=0.995)
        report = diff_scorecards(a, b, threshold=0.01)
        assert "REGRESSION" not in report

    def test_entries_only_in_one_run(self) -> None:
        a = _make_scorecard("runA", [_make_result("q001"), _make_result("q002")])
        b = _make_scorecard("runB", [_make_result("q001"), _make_result("q003")])
        report = diff_scorecards(a, b)
        assert "Only in Run A" in report
        assert "q002" in report
        assert "Only in Run B" in report
        assert "q003" in report

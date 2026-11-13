from __future__ import annotations

import pytest
from helpers import make_retrieval_result
from pac_eval.metrics_retrieval import compute_retrieval_metrics


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118", "24-FC-42"])
        metrics = compute_retrieval_metrics(result, expected_opinions=["25-FC-118", "24-FC-42"])
        assert metrics.recall_at_8 == 1.0
        assert metrics.recall_at_16 == 1.0

    def test_zero_recall(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118", "24-FC-42"])
        metrics = compute_retrieval_metrics(result, expected_opinions=["99-FC-999"])
        assert metrics.recall_at_8 == 0.0
        assert metrics.recall_at_16 == 0.0

    def test_partial_recall(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118", "24-FC-42"])
        metrics = compute_retrieval_metrics(
            result, expected_opinions=["25-FC-118", "24-FC-42", "99-FC-999"]
        )
        assert metrics.recall_at_8 == pytest.approx(2 / 3)
        assert metrics.recall_at_16 == pytest.approx(2 / 3)

    def test_recall_at_8_vs_16_different(self) -> None:
        ids = [f"{i:02d}-FC-{i:03d}" for i in range(1, 11)]
        result = make_retrieval_result(opinion_ids=ids, chunks_per_opinion=1)
        metrics = compute_retrieval_metrics(result, expected_opinions=["01-FC-001", "09-FC-009"])
        assert metrics.recall_at_8 == 0.5
        assert metrics.recall_at_16 == 1.0

    def test_empty_chunks(self) -> None:
        from pac_pipeline.retrieve_models import RetrievalResult

        result = RetrievalResult(question="test", applied_filters={}, relaxed=False, chunks=[])
        metrics = compute_retrieval_metrics(result, expected_opinions=["25-FC-118"])
        assert metrics.recall_at_8 == 0.0
        assert metrics.recall_at_16 == 0.0


class TestMRR:
    def test_first_rank(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118", "24-FC-42"])
        metrics = compute_retrieval_metrics(result, expected_opinions=["25-FC-118"])
        assert metrics.mrr == 1.0
        assert metrics.first_relevant_rank == 1

    def test_second_rank(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118", "24-FC-42"])
        metrics = compute_retrieval_metrics(result, expected_opinions=["24-FC-42"])
        assert metrics.mrr == 0.5
        assert metrics.first_relevant_rank == 2

    def test_no_relevant(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118"])
        metrics = compute_retrieval_metrics(result, expected_opinions=["99-FC-999"])
        assert metrics.mrr == 0.0
        assert metrics.first_relevant_rank is None

    def test_multiple_expected_returns_first(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118", "24-FC-42", "23-FC-15"])
        metrics = compute_retrieval_metrics(result, expected_opinions=["23-FC-15", "24-FC-42"])
        assert metrics.mrr == 0.5
        assert metrics.first_relevant_rank == 2


class TestRetrievedOpinionIds:
    def test_deduplicates(self) -> None:
        result = make_retrieval_result(opinion_ids=["25-FC-118", "24-FC-42"], chunks_per_opinion=3)
        metrics = compute_retrieval_metrics(result, expected_opinions=["25-FC-118"])
        assert metrics.retrieved_opinion_ids == ("25-FC-118", "24-FC-42")

from __future__ import annotations

from pac_pipeline.retrieve_models import RetrievalResult

from pac_eval.models import RetrievalMetrics


def _recall_at_k(
    retrieved_opinion_ids: list[str],
    expected: list[str],
    k: int,
) -> float:
    top_k_ids = set(retrieved_opinion_ids[:k])
    found = sum(1 for oid in expected if oid in top_k_ids)
    return found / len(expected)


def _mrr(
    retrieved_opinion_ids: list[str],
    expected: list[str],
) -> tuple[float, int | None]:
    expected_set = set(expected)
    for rank_0, oid in enumerate(retrieved_opinion_ids):
        if oid in expected_set:
            rank = rank_0 + 1
            return 1.0 / rank, rank
    return 0.0, None


def compute_retrieval_metrics(
    result: RetrievalResult,
    expected_opinions: list[str],
) -> RetrievalMetrics:
    opinion_id_sequence = [rc.chunk.opinion_id for rc in result.chunks]

    seen: set[str] = set()
    deduplicated: list[str] = []
    for oid in opinion_id_sequence:
        if oid not in seen:
            seen.add(oid)
            deduplicated.append(oid)

    mrr_score, first_rank = _mrr(deduplicated, expected_opinions)

    return RetrievalMetrics(
        recall_at_8=_recall_at_k(deduplicated, expected_opinions, k=8),
        recall_at_16=_recall_at_k(deduplicated, expected_opinions, k=16),
        mrr=mrr_score,
        first_relevant_rank=first_rank,
        retrieved_opinion_ids=tuple(deduplicated),
    )

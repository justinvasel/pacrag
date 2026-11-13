from __future__ import annotations

import time
from datetime import UTC, datetime

import structlog
from anthropic import AsyncAnthropic
from pac_api.config import settings
from pac_pipeline.retrieve_models import RetrievedChunk

from pac_eval.metrics_answer import compute_citation_precision, compute_hedging_score
from pac_eval.metrics_judge import judge_fact_coverage
from pac_eval.metrics_retrieval import compute_retrieval_metrics
from pac_eval.models import (
    AnswerMetrics,
    GoldEntry,
    GoldSet,
    QuestionResult,
    Scorecard,
    ScorecardSummary,
)

log = structlog.get_logger()


async def _collect_answer(
    question: str,
    chunks: list[RetrievedChunk],
    client: AsyncAnthropic,
) -> str:
    from pac_api.generation import stream_answer

    parts: list[str] = []
    async for event_type, text in stream_answer(question, chunks, client):
        if event_type == "done":
            return text
        parts.append(text)
    return "".join(parts)


async def _eval_one(
    entry: GoldEntry,
    *,
    skip_generation: bool,
    with_filters: bool,
    client: AsyncAnthropic | None,
) -> QuestionResult:
    from pac_pipeline.retrieve import retrieve

    start = time.monotonic()
    error: str | None = None

    try:
        filters: dict[str, object] | None = None
        if with_filters and entry.statute_area and entry.statute_area != "both":
            filters = {"statute_areas": [entry.statute_area]}

        result = await retrieve(
            entry.question,
            filters=filters,
            skip_filter_inference=True,
            top_k=16,
        )

        retrieval_metrics = compute_retrieval_metrics(result, entry.expected_opinions)

        answer_metrics: AnswerMetrics | None = None
        generated_answer: str | None = None

        if not skip_generation and client is not None:
            top_8 = result.chunks[:8]
            generated_answer = await _collect_answer(entry.question, top_8, client)

            retrieved_ids = frozenset(rc.chunk.opinion_id for rc in result.chunks)
            precision, total, valid = compute_citation_precision(generated_answer, retrieved_ids)
            hedging_present, overclaim_detected, hedging_score = compute_hedging_score(
                generated_answer
            )

            coverage, facts_found, facts_missing = await judge_fact_coverage(
                question=entry.question,
                ideal_answer=entry.ideal_answer,
                key_facts=entry.key_facts,
                generated_answer=generated_answer,
                client=client,
                model=settings.generation_model,
            )

            answer_metrics = AnswerMetrics(
                citation_precision=precision,
                citation_count=total,
                valid_citation_count=valid,
                fact_coverage=coverage,
                facts_found=tuple(facts_found),
                facts_missing=tuple(facts_missing),
                hedging_present=hedging_present,
                overclaim_detected=overclaim_detected,
                hedging_score=hedging_score,
            )

    except Exception as e:
        from pac_pipeline.retrieve_models import RetrievalResult

        log.error("eval_question_failed", entry_id=entry.id, error=str(e))
        error = str(e)
        retrieval_metrics = compute_retrieval_metrics(
            RetrievalResult(question=entry.question, applied_filters={}, relaxed=False, chunks=[]),
            entry.expected_opinions,
        )
        answer_metrics = None
        generated_answer = None

    elapsed = time.monotonic() - start

    return QuestionResult(
        entry_id=entry.id,
        question=entry.question,
        statute_area=entry.statute_area,
        retrieval=retrieval_metrics,
        answer=answer_metrics,
        generated_answer=generated_answer,
        error=error,
        duration_seconds=round(elapsed, 2),
    )


def _compute_summary(results: list[QuestionResult]) -> ScorecardSummary:
    n = len(results)
    if n == 0:
        return ScorecardSummary(
            mean_recall_at_8=0.0,
            mean_recall_at_16=0.0,
            mean_mrr=0.0,
            mean_citation_precision=0.0,
            mean_fact_coverage=0.0,
            mean_hedging_score=0.0,
            hedging_rate=0.0,
            questions_evaluated=0,
            questions_with_errors=0,
        )

    mean_r8 = sum(r.retrieval.recall_at_8 for r in results) / n
    mean_r16 = sum(r.retrieval.recall_at_16 for r in results) / n
    mean_mrr = sum(r.retrieval.mrr for r in results) / n

    answered = [r.answer for r in results if r.answer is not None]
    n_answered = len(answered)

    mean_cp = sum(a.citation_precision for a in answered) / n_answered if n_answered else 0.0
    mean_fc = sum(a.fact_coverage for a in answered) / n_answered if n_answered else 0.0
    mean_hs = sum(a.hedging_score for a in answered) / n_answered if n_answered else 0.0
    hedging_rate = sum(1 for a in answered if a.hedging_present) / n_answered if n_answered else 0.0

    errors = sum(1 for r in results if r.error is not None)

    return ScorecardSummary(
        mean_recall_at_8=round(mean_r8, 4),
        mean_recall_at_16=round(mean_r16, 4),
        mean_mrr=round(mean_mrr, 4),
        mean_citation_precision=round(mean_cp, 4),
        mean_fact_coverage=round(mean_fc, 4),
        mean_hedging_score=round(mean_hs, 4),
        hedging_rate=round(hedging_rate, 4),
        questions_evaluated=n,
        questions_with_errors=errors,
    )


async def run_eval(
    gold_set: GoldSet,
    *,
    skip_generation: bool = False,
    with_filters: bool = False,
    concurrency: int = 3,
    limit: int | None = None,
) -> Scorecard:
    entries = list(gold_set.entries)
    if limit is not None:
        entries = entries[:limit]

    client: AsyncAnthropic | None = None
    if not skip_generation:
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    results: list[QuestionResult] = []
    for entry in entries:
        log.info("eval_question_start", entry_id=entry.id)
        result = await _eval_one(
            entry,
            skip_generation=skip_generation,
            with_filters=with_filters,
            client=client,
        )
        log.info(
            "eval_question_done",
            entry_id=entry.id,
            recall_at_8=result.retrieval.recall_at_8,
            error=result.error,
        )
        results.append(result)

    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%S")

    return Scorecard(
        run_id=run_id,
        timestamp=now,
        gold_set_version=gold_set.version,
        results=list(results),
        summary=_compute_summary(list(results)),
    )

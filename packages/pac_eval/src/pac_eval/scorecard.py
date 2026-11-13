from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pac_eval.models import (
    AnswerMetrics,
    QuestionResult,
    RetrievalMetrics,
    Scorecard,
    ScorecardSummary,
)

_SCHEMA = pa.schema(
    [
        pa.field("entry_id", pa.string()),
        pa.field("question", pa.string()),
        pa.field("statute_area", pa.string()),
        pa.field("recall_at_8", pa.float64()),
        pa.field("recall_at_16", pa.float64()),
        pa.field("mrr", pa.float64()),
        pa.field("first_relevant_rank", pa.int32()),
        pa.field("retrieved_opinion_ids", pa.string()),
        pa.field("citation_precision", pa.float64()),
        pa.field("citation_count", pa.int32()),
        pa.field("valid_citation_count", pa.int32()),
        pa.field("fact_coverage", pa.float64()),
        pa.field("facts_found", pa.string()),
        pa.field("facts_missing", pa.string()),
        pa.field("hedging_present", pa.bool_()),
        pa.field("overclaim_detected", pa.bool_()),
        pa.field("hedging_score", pa.float64()),
        pa.field("generated_answer", pa.string()),
        pa.field("error", pa.string()),
        pa.field("duration_seconds", pa.float64()),
    ]
)


def _result_to_row(r: QuestionResult) -> dict[str, object]:
    row: dict[str, object] = {
        "entry_id": r.entry_id,
        "question": r.question,
        "statute_area": r.statute_area,
        "recall_at_8": r.retrieval.recall_at_8,
        "recall_at_16": r.retrieval.recall_at_16,
        "mrr": r.retrieval.mrr,
        "first_relevant_rank": r.retrieval.first_relevant_rank,
        "retrieved_opinion_ids": json.dumps(list(r.retrieval.retrieved_opinion_ids)),
        "generated_answer": r.generated_answer,
        "error": r.error,
        "duration_seconds": r.duration_seconds,
    }

    if r.answer is not None:
        row["citation_precision"] = r.answer.citation_precision
        row["citation_count"] = r.answer.citation_count
        row["valid_citation_count"] = r.answer.valid_citation_count
        row["fact_coverage"] = r.answer.fact_coverage
        row["facts_found"] = json.dumps(list(r.answer.facts_found))
        row["facts_missing"] = json.dumps(list(r.answer.facts_missing))
        row["hedging_present"] = r.answer.hedging_present
        row["overclaim_detected"] = r.answer.overclaim_detected
        row["hedging_score"] = r.answer.hedging_score
    else:
        row["citation_precision"] = None
        row["citation_count"] = None
        row["valid_citation_count"] = None
        row["fact_coverage"] = None
        row["facts_found"] = None
        row["facts_missing"] = None
        row["hedging_present"] = None
        row["overclaim_detected"] = None
        row["hedging_score"] = None

    return row


def _row_to_result(row: dict[str, object]) -> QuestionResult:
    retrieval = RetrievalMetrics(
        recall_at_8=float(row["recall_at_8"]),  # type: ignore[arg-type]
        recall_at_16=float(row["recall_at_16"]),  # type: ignore[arg-type]
        mrr=float(row["mrr"]),  # type: ignore[arg-type]
        first_relevant_rank=(
            int(row["first_relevant_rank"])  # type: ignore[arg-type]
            if row["first_relevant_rank"] is not None
            else None
        ),
        retrieved_opinion_ids=tuple(json.loads(str(row["retrieved_opinion_ids"]))),
    )

    answer: AnswerMetrics | None = None
    if row.get("citation_precision") is not None:
        answer = AnswerMetrics(
            citation_precision=float(row["citation_precision"]),  # type: ignore[arg-type]
            citation_count=int(row["citation_count"]),  # type: ignore[arg-type]
            valid_citation_count=int(row["valid_citation_count"]),  # type: ignore[arg-type]
            fact_coverage=float(row["fact_coverage"]),  # type: ignore[arg-type]
            facts_found=tuple(json.loads(str(row["facts_found"]))),
            facts_missing=tuple(json.loads(str(row["facts_missing"]))),
            hedging_present=bool(row["hedging_present"]),
            overclaim_detected=bool(row["overclaim_detected"]),
            hedging_score=float(row["hedging_score"]),  # type: ignore[arg-type]
        )

    return QuestionResult(
        entry_id=str(row["entry_id"]),
        question=str(row["question"]),
        statute_area=str(row["statute_area"]) if row["statute_area"] is not None else None,
        retrieval=retrieval,
        answer=answer,
        generated_answer=(
            str(row["generated_answer"]) if row["generated_answer"] is not None else None
        ),
        error=str(row["error"]) if row["error"] is not None else None,
        duration_seconds=float(row["duration_seconds"]),  # type: ignore[arg-type]
    )


def write_scorecard(scorecard: Scorecard, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scorecard.run_id}.parquet"

    rows = [_result_to_row(r) for r in scorecard.results]
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)

    metadata = {
        b"run_id": scorecard.run_id.encode(),
        b"timestamp": scorecard.timestamp.isoformat().encode(),
        b"gold_set_version": scorecard.gold_set_version.encode(),
        b"summary": scorecard.summary.model_dump_json().encode(),
    }
    table = table.replace_schema_metadata({**metadata, **(table.schema.metadata or {})})

    pq.write_table(table, path)
    return path


def read_scorecard(path: Path) -> Scorecard:
    table = pq.read_table(path)
    meta = table.schema.metadata or {}

    run_id = meta[b"run_id"].decode()
    timestamp = datetime.fromisoformat(meta[b"timestamp"].decode())
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    gold_set_version = meta[b"gold_set_version"].decode()
    summary = ScorecardSummary.model_validate_json(meta[b"summary"])

    rows = table.to_pylist()
    results = [_row_to_result(r) for r in rows]

    return Scorecard(
        run_id=run_id,
        timestamp=timestamp,
        gold_set_version=gold_set_version,
        results=results,
        summary=summary,
    )

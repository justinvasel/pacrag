from __future__ import annotations

from pac_pipeline.chunk_models import Chunk
from pac_pipeline.retrieve_models import RetrievalResult, RetrievedChunk


def make_chunk(
    opinion_id: str = "25-FC-118",
    chunk_index: int = 0,
    section: str = "analysis",
    page_num: int = 1,
    text: str = "Sample chunk text.",
) -> Chunk:
    return Chunk(
        chunk_id=f"{opinion_id}#c{chunk_index}",
        opinion_id=opinion_id,
        chunk_index=chunk_index,
        section=section,
        page_num=page_num,
        text=text,
        token_count=len(text.split()),
        year=2000 + int(opinion_id[:2]),
        agency_type="city",
        statute_areas=("APRA",),
        finding="violation",
        issue_tags=("response-timeline",),
    )


def make_retrieved_chunk(
    opinion_id: str = "25-FC-118",
    chunk_index: int = 0,
    dense_rank: int | None = 1,
    lexical_rank: int | None = 2,
    fused_score: float = 0.5,
    rerank_score: float = 0.9,
    **kwargs: object,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(opinion_id=opinion_id, chunk_index=chunk_index, **kwargs),  # type: ignore[arg-type]
        dense_rank=dense_rank,
        lexical_rank=lexical_rank,
        fused_score=fused_score,
        rerank_score=rerank_score,
    )


def make_retrieval_result(
    question: str = "test question",
    opinion_ids: list[str] | None = None,
    chunks_per_opinion: int = 2,
) -> RetrievalResult:
    ids = opinion_ids or ["25-FC-118", "24-FC-42"]
    chunks: list[RetrievedChunk] = []
    for i, oid in enumerate(ids):
        for j in range(chunks_per_opinion):
            idx = i * chunks_per_opinion + j
            chunks.append(
                make_retrieved_chunk(
                    opinion_id=oid,
                    chunk_index=j,
                    dense_rank=idx + 1,
                    lexical_rank=idx + 2,
                    fused_score=1.0 / (60 + idx),
                    rerank_score=0.95 - (idx * 0.05),
                )
            )
    return RetrievalResult(
        question=question,
        applied_filters={},
        relaxed=False,
        chunks=chunks,
    )

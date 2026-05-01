from __future__ import annotations

from typing import TYPE_CHECKING

from pac_pipeline.retrieve_models import (
    RRF_K,
    TOP_K_FUSED,
    TOP_K_RERANKED,
    FusedChunk,
    RankedChunkRef,
    RetrievedChunk,
)

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


def fuse_rrf(
    dense_results: list[RankedChunkRef],
    lexical_results: list[RankedChunkRef],
    *,
    k: int = RRF_K,
    top_k: int = TOP_K_FUSED,
) -> list[FusedChunk]:
    dense_by_id = {r.chunk_id: r for r in dense_results}
    lexical_by_id = {r.chunk_id: r for r in lexical_results}

    all_ids = set(dense_by_id) | set(lexical_by_id)

    fused: list[FusedChunk] = []
    for chunk_id in all_ids:
        dense_ref = dense_by_id.get(chunk_id)
        lexical_ref = lexical_by_id.get(chunk_id)

        score = 0.0
        if dense_ref is not None:
            score += 1.0 / (k + dense_ref.rank)
        if lexical_ref is not None:
            score += 1.0 / (k + lexical_ref.rank)

        chunk = dense_ref.chunk if dense_ref is not None else lexical_ref.chunk  # type: ignore[union-attr]

        fused.append(
            FusedChunk(
                chunk=chunk,
                dense_rank=dense_ref.rank if dense_ref else None,
                lexical_rank=lexical_ref.rank if lexical_ref else None,
                fused_score=score,
            )
        )

    fused.sort(key=lambda fc: fc.fused_score, reverse=True)
    return fused[:top_k]


def rerank(
    question: str,
    candidates: list[FusedChunk],
    reranker: CrossEncoder,
    *,
    top_k: int = TOP_K_RERANKED,
) -> list[RetrievedChunk]:
    if not candidates:
        return []

    pairs = [(question, fc.chunk.text) for fc in candidates]
    scores = reranker.predict(pairs)

    scored = [
        RetrievedChunk(
            chunk=fc.chunk,
            dense_rank=fc.dense_rank,
            lexical_rank=fc.lexical_rank,
            fused_score=fc.fused_score,
            rerank_score=float(scores[i]),
        )
        for i, fc in enumerate(candidates)
    ]

    scored.sort(key=lambda rc: rc.rerank_score, reverse=True)
    return scored[:top_k]

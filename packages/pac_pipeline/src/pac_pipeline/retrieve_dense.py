from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from pac_pipeline.chunk_models import QDRANT_COLLECTION, Chunk
from pac_pipeline.retrieve_filters import to_qdrant_filter
from pac_pipeline.retrieve_models import TOP_K_PER_SOURCE, InferredFilters, RankedChunkRef

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

log = structlog.get_logger()


def _payload_to_chunk(payload: dict[str, object]) -> Chunk:
    return Chunk(
        chunk_id=str(payload["chunk_id"]),
        opinion_id=str(payload["opinion_id"]),
        chunk_index=int(payload.get("chunk_index", 0)),  # type: ignore[arg-type]
        section=str(payload.get("section", "")),
        page_num=int(payload.get("page_num", 0)),  # type: ignore[arg-type]
        text=str(payload.get("text", "")),
        token_count=int(payload.get("token_count", 0)),  # type: ignore[arg-type]
        year=int(payload["year"]),  # type: ignore[arg-type]
        agency_type=str(payload.get("agency_type", "other")),
        statute_areas=tuple(payload.get("statute_areas", ())),  # type: ignore[arg-type]
        finding=str(payload.get("finding", "")),
        issue_tags=tuple(payload.get("issue_tags", ())),  # type: ignore[arg-type]
    )


async def search_dense(
    query_vector: list[float],
    client: AsyncQdrantClient,
    *,
    filters: InferredFilters | None = None,
    top_k: int = TOP_K_PER_SOURCE,
) -> list[RankedChunkRef]:
    qdrant_filter = to_qdrant_filter(filters) if filters else None

    results = await client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    ranked: list[RankedChunkRef] = []
    for rank, point in enumerate(results.points):
        payload = point.payload or {}
        try:
            chunk = _payload_to_chunk(payload)
            ranked.append(
                RankedChunkRef(
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    score=float(point.score) if point.score is not None else 0.0,
                    chunk=chunk,
                )
            )
        except Exception as exc:
            log.warning("dense_payload_parse_error", point_id=point.id, error=str(exc))

    return ranked

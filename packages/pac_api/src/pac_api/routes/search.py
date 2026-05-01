from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, HTTPException, Query
from pac_pipeline import retrieve

from pac_api.schemas import SearchResponse, chunk_to_response

log = structlog.get_logger()
router = APIRouter()


@router.get("/search", response_model=SearchResponse)
async def search(
    question: str = Query(..., min_length=1),
    filters: str | None = Query(default=None),
    top_k: int = Query(default=10, ge=1, le=50),
) -> SearchResponse:
    parsed_filters: dict[str, object] | None = None
    if filters is not None:
        try:
            parsed_filters = json.loads(filters)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid filters JSON: {e}") from e

    result = await retrieve(question, filters=parsed_filters, top_k=top_k)

    return SearchResponse(
        question=result.question,
        applied_filters=result.applied_filters,
        relaxed=result.relaxed,
        chunks=[chunk_to_response(rc) for rc in result.chunks],
    )

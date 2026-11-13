from __future__ import annotations

import asyncio

import structlog
import tantivy
from fastapi import APIRouter, Depends
from pac_pipeline.chunk_models import QDRANT_COLLECTION
from pac_pipeline.config import settings
from qdrant_client import AsyncQdrantClient

from pac_api.db import OpinionDB
from pac_api.dependencies import get_db
from pac_api.schemas import HealthResponse

log = structlog.get_logger()
router = APIRouter()


def _build_tantivy_schema() -> tantivy.Schema:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("opinion_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("text", stored=True)
    builder.add_integer_field("year", stored=True, indexed=True, fast=True)
    builder.add_text_field("agency_type", stored=True, tokenizer_name="raw")
    builder.add_text_field("statute_areas", stored=True, tokenizer_name="raw")
    builder.add_text_field("finding", stored=True, tokenizer_name="raw")
    builder.add_text_field("issue_tags", stored=True, tokenizer_name="raw")
    return builder.build()


async def _qdrant_count(client: AsyncQdrantClient) -> int:
    try:
        info = await client.get_collection(QDRANT_COLLECTION)
        return info.points_count or 0
    except Exception:
        log.warning("qdrant_health_check_failed", exc_info=True)
        return 0


def _tantivy_count() -> int:
    try:
        schema = _build_tantivy_schema()
        index = tantivy.Index(schema, path=str(settings.tantivy_path), reuse=True)
        index.reload()
        searcher = index.searcher()
        return searcher.num_docs
    except Exception:
        log.warning("tantivy_health_check_failed", exc_info=True)
        return 0


@router.get("/healthz", response_model=HealthResponse)
async def healthz(db: OpinionDB = Depends(get_db)) -> HealthResponse:
    qdrant = AsyncQdrantClient(path=str(settings.qdrant_path))
    try:
        qdrant_count, tantivy_count = await asyncio.gather(
            _qdrant_count(qdrant),
            asyncio.to_thread(_tantivy_count),
        )
    finally:
        await qdrant.close()

    duckdb_chunks = db.get_chunk_count()
    duckdb_opinions = db.get_opinion_count()

    counts_match = qdrant_count == tantivy_count == duckdb_chunks

    return HealthResponse(
        status="ok",
        qdrant_chunks=qdrant_count,
        tantivy_chunks=tantivy_count,
        duckdb_chunks=duckdb_chunks,
        duckdb_opinions=duckdb_opinions,
        counts_match=counts_match,
    )

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache

import structlog
import tantivy
import torch
from anthropic import AsyncAnthropic
from pac_api.config import settings
from qdrant_client import AsyncQdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from pac_pipeline.retrieve_dense import search_dense
from pac_pipeline.retrieve_filters import infer_filters
from pac_pipeline.retrieve_fusion import fuse_rrf, rerank
from pac_pipeline.retrieve_lexical import search_lexical
from pac_pipeline.retrieve_models import (
    MIN_CANDIDATES_FOR_FILTERS,
    TOP_K_RERANKED,
    InferredFilters,
    RetrievalResult,
)

log = structlog.get_logger()

_EMBEDDING_CACHE_MAX = 256
_embedding_cache: dict[str, list[float]] = {}


@dataclass(frozen=True)
class RetrievalModels:
    embedder: SentenceTransformer
    reranker: CrossEncoder


@lru_cache(maxsize=1)
def get_retrieval_models() -> RetrievalModels:
    os.environ["HF_HOME"] = str(settings.models_dir.resolve())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log.info("loading_retrieval_models", device=device)

    embedder = SentenceTransformer(settings.embedding_model, device=device)
    reranker = CrossEncoder(settings.reranker_model, device=device)
    return RetrievalModels(embedder=embedder, reranker=reranker)


def _load_tantivy_index() -> tantivy.Index:
    builder = tantivy.SchemaBuilder()
    builder.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("opinion_id", stored=True, tokenizer_name="raw")
    builder.add_text_field("text", stored=True)
    builder.add_integer_field("year", stored=True, indexed=True, fast=True)
    builder.add_text_field("agency_type", stored=True, tokenizer_name="raw")
    builder.add_text_field("statute_areas", stored=True, tokenizer_name="raw")
    builder.add_text_field("finding", stored=True, tokenizer_name="raw")
    builder.add_text_field("issue_tags", stored=True, tokenizer_name="raw")
    schema = builder.build()
    return tantivy.Index(schema, path=str(settings.tantivy_path), reuse=True)


def _embed_question(question: str, embedder: SentenceTransformer) -> list[float]:
    cache_key = hashlib.sha256(question.encode()).hexdigest()

    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    vec: list[float] = embedder.encode(question, normalize_embeddings=True).tolist()

    if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX:
        oldest_key = next(iter(_embedding_cache))
        del _embedding_cache[oldest_key]

    _embedding_cache[cache_key] = vec
    return vec


async def retrieve(
    question: str,
    *,
    filters: dict[str, object] | None = None,
    skip_filter_inference: bool = False,
    top_k: int = TOP_K_RERANKED,
) -> RetrievalResult:
    models = get_retrieval_models()

    inferred = InferredFilters()
    if filters is not None:
        inferred = InferredFilters.model_validate(filters)
    elif not skip_filter_inference:
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        inferred = await infer_filters(question, client)
        log.info("filters_inferred", filters=inferred.to_applied_dict())

    query_vector = _embed_question(question, models.embedder)

    qdrant = AsyncQdrantClient(path=str(settings.qdrant_path))
    try:
        tv_index = _load_tantivy_index()

        active_filters = inferred if inferred.has_any() else None
        relaxed = False

        dense_results, lexical_results = await asyncio.gather(
            search_dense(query_vector, qdrant, filters=active_filters),
            asyncio.to_thread(search_lexical, question, tv_index, filters=active_filters),
        )

        dense_ids = {r.chunk_id for r in dense_results}
        lexical_ids = {r.chunk_id for r in lexical_results}
        total_candidates = len(dense_ids | lexical_ids)

        if active_filters is not None and total_candidates < MIN_CANDIDATES_FOR_FILTERS:
            log.info(
                "relaxing_filters",
                candidates=total_candidates,
                threshold=MIN_CANDIDATES_FOR_FILTERS,
            )
            relaxed = True
            dense_results, lexical_results = await asyncio.gather(
                search_dense(query_vector, qdrant, filters=None),
                asyncio.to_thread(search_lexical, question, tv_index, filters=None),
            )

        fused = fuse_rrf(dense_results, lexical_results)
        ranked = rerank(question, fused, models.reranker, top_k=top_k)
    finally:
        await qdrant.close()

    return RetrievalResult(
        question=question,
        applied_filters=inferred.to_applied_dict() if not relaxed else {},
        relaxed=relaxed,
        chunks=ranked,
    )

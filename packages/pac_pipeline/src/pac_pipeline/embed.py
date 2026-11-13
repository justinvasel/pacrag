from __future__ import annotations

import os
import uuid
from pathlib import Path

import structlog
import tantivy
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pac_pipeline.chunk import chunk_opinion
from pac_pipeline.chunk_models import CHUNKER_VERSION, EMBEDDING_DIM, QDRANT_COLLECTION
from pac_pipeline.config import settings
from pac_pipeline.embed_db import EmbedDB
from pac_pipeline.models import ParsedOpinion

log = structlog.get_logger()


def _ensure_qdrant_collection(client: QdrantClient) -> None:
    if client.collection_exists(QDRANT_COLLECTION):
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    for field_name, field_type in [
        ("year", PayloadSchemaType.INTEGER),
        ("agency_type", PayloadSchemaType.KEYWORD),
        ("statute_areas", PayloadSchemaType.KEYWORD),
        ("finding", PayloadSchemaType.KEYWORD),
        ("issue_tags", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=field_type,
        )


def _ensure_tantivy_index(path: Path) -> tantivy.Index:
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

    path.mkdir(parents=True, exist_ok=True)

    try:
        return tantivy.Index(schema, path=str(path), reuse=True)
    except ValueError:
        return tantivy.Index(schema, path=str(path))


def _chunk_id_to_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def embed_all(
    db_path: Path,
    *,
    opinion_id: str | None = None,
    reembed: bool = False,
    limit: int | None = None,
) -> tuple[int, int]:
    os.environ["HF_HOME"] = str(settings.models_dir.resolve())

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedder = SentenceTransformer(settings.embedding_model, device=device)

    qdrant = QdrantClient(path=str(settings.qdrant_path))
    _ensure_qdrant_collection(qdrant)

    tantivy_index = _ensure_tantivy_index(settings.tantivy_path)

    with EmbedDB(db_path) as db:
        opinions = db.get_embeddable_opinions(
            opinion_id=opinion_id,
            reembed=reembed,
            limit=limit,
            chunker_version=CHUNKER_VERSION,
        )

        if not opinions:
            log.info("no_opinions_to_embed")
            return 0, 0

        log.info("embedding_opinions", count=len(opinions))

        embedded_count = 0
        error_count = 0
        tv_writer = tantivy_index.writer()

        for oid, year, parse_path_str in tqdm(opinions, desc="Embedding"):
            try:
                parse_path = Path(parse_path_str)
                parsed = ParsedOpinion.model_validate_json(parse_path.read_text())

                metadata = db.get_metadata_for_opinion(oid)
                if metadata is None:
                    raise ValueError(f"No metadata found for {oid}")

                chunks = chunk_opinion(parsed, metadata, year=year, tokenizer=embedder.tokenizer)

                if not chunks:
                    log.warning("no_chunks_produced", opinion_id=oid)
                    db.update_embed_status(oid, "error")
                    error_count += 1
                    continue

                vectors = embedder.encode(
                    [c.text for c in chunks],
                    batch_size=32,
                    normalize_embeddings=True,
                )

                points = [
                    PointStruct(
                        id=_chunk_id_to_uuid(chunk.chunk_id),
                        vector=vectors[i].tolist(),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "opinion_id": chunk.opinion_id,
                            "chunk_index": chunk.chunk_index,
                            "section": chunk.section,
                            "page_num": chunk.page_num,
                            "text": chunk.text,
                            "token_count": chunk.token_count,
                            "year": chunk.year,
                            "agency_type": chunk.agency_type,
                            "statute_areas": list(chunk.statute_areas),
                            "finding": chunk.finding,
                            "issue_tags": list(chunk.issue_tags),
                        },
                    )
                    for i, chunk in enumerate(chunks)
                ]

                qdrant.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points,
                )

                tv_writer.delete_documents("opinion_id", oid)
                for chunk in chunks:
                    doc = tantivy.Document(
                        chunk_id=chunk.chunk_id,
                        opinion_id=chunk.opinion_id,
                        text=chunk.text,
                        year=chunk.year,
                        agency_type=chunk.agency_type,
                        statute_areas=" ".join(chunk.statute_areas),
                        finding=chunk.finding,
                        issue_tags=" ".join(chunk.issue_tags),
                    )
                    tv_writer.add_document(doc)
                tv_writer.commit()

                db.upsert_chunks(list(chunks), chunker_version=CHUNKER_VERSION)
                db.update_embed_status(oid, "ok")
                embedded_count += 1

            except Exception:
                log.exception("embedding_failed", opinion_id=oid)
                _cleanup_opinion(qdrant, tv_writer, oid)
                db.delete_chunks_for_opinion(oid)
                db.update_embed_status(oid, "error")
                error_count += 1

        tantivy_index.reload()

        return embedded_count, error_count


def _cleanup_opinion(
    qdrant: QdrantClient,
    tv_writer: tantivy.IndexWriter,
    opinion_id: str,
) -> None:
    try:
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="opinion_id", match=MatchValue(value=opinion_id))]
            ),
        )
    except Exception:
        log.warning("qdrant_cleanup_failed", opinion_id=opinion_id)
    try:
        tv_writer.delete_documents("opinion_id", opinion_id)
        tv_writer.commit()
    except Exception:
        log.warning("tantivy_cleanup_failed", opinion_id=opinion_id)

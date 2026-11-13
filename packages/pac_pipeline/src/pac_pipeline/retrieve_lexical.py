from __future__ import annotations

import structlog
import tantivy
from tantivy import Occur

from pac_pipeline.chunk_models import Chunk
from pac_pipeline.retrieve_models import TOP_K_PER_SOURCE, InferredFilters, RankedChunkRef

log = structlog.get_logger()


def _doc_to_chunk(doc: tantivy.Document) -> Chunk:
    fields = doc.to_dict()
    statute_areas_raw = str(fields["statute_areas"][0]) if fields.get("statute_areas") else ""
    issue_tags_raw = str(fields["issue_tags"][0]) if fields.get("issue_tags") else ""

    return Chunk(
        chunk_id=str(fields["chunk_id"][0]),
        opinion_id=str(fields["opinion_id"][0]),
        chunk_index=0,
        section="",
        page_num=0,
        text=str(fields["text"][0]),
        token_count=0,
        year=int(fields["year"][0]),
        agency_type=str(fields["agency_type"][0]) if fields.get("agency_type") else "other",
        statute_areas=tuple(statute_areas_raw.split()) if statute_areas_raw else (),
        finding=str(fields["finding"][0]) if fields.get("finding") else "",
        issue_tags=tuple(issue_tags_raw.split()) if issue_tags_raw else (),
    )


def _sanitize_query(text: str) -> str:
    result = text.replace("'", "").replace("\u2019", "")
    result = result.replace(":", " ")
    return result


def _build_query(
    question: str,
    index: tantivy.Index,
    filters: InferredFilters | None,
) -> tantivy.Query:
    sanitized = _sanitize_query(question)

    if filters is None or not filters.has_any():
        return index.parse_query(sanitized, default_field_names=["text"])

    subqueries: list[tuple[Occur, tantivy.Query]] = [
        (Occur.Must, index.parse_query(sanitized, default_field_names=["text"])),
    ]

    if filters.agency_type is not None:
        q = index.parse_query(filters.agency_type, default_field_names=["agency_type"])
        subqueries.append((Occur.Must, q))

    for area in filters.statute_areas:
        subqueries.append(
            (Occur.Must, index.parse_query(area, default_field_names=["statute_areas"]))
        )

    if filters.finding is not None:
        subqueries.append(
            (Occur.Must, index.parse_query(filters.finding, default_field_names=["finding"]))
        )

    for tag in filters.issue_tags:
        subqueries.append(
            (Occur.Should, index.parse_query(tag, default_field_names=["issue_tags"]))
        )

    return tantivy.Query.boolean_query(subqueries)


def search_lexical(
    question: str,
    index: tantivy.Index,
    *,
    filters: InferredFilters | None = None,
    top_k: int = TOP_K_PER_SOURCE,
) -> list[RankedChunkRef]:
    index.reload()
    searcher = index.searcher()
    query = _build_query(question, index, filters)

    search_result = searcher.search(query, limit=top_k)

    ranked: list[RankedChunkRef] = []
    for rank, (score, doc_address) in enumerate(search_result.hits):
        try:
            doc = searcher.doc(doc_address)
            chunk = _doc_to_chunk(doc)
            ranked.append(
                RankedChunkRef(
                    chunk_id=chunk.chunk_id,
                    rank=rank,
                    score=float(score),
                    chunk=chunk,
                )
            )
        except Exception as exc:
            log.warning("lexical_doc_parse_error", doc_address=str(doc_address), error=str(exc))

    return ranked

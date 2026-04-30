from __future__ import annotations

from pydantic import BaseModel

CHUNKER_VERSION: str = "0.1.0"
QDRANT_COLLECTION: str = "pac_opinions_v1"
EMBEDDING_DIM: int = 1024
MAX_CHUNK_TOKENS: int = 480
MIN_CHUNK_TOKENS: int = 250
OVERLAP_TOKENS: int = 75


class Chunk(BaseModel, frozen=True):
    chunk_id: str
    opinion_id: str
    chunk_index: int
    section: str
    page_num: int
    text: str
    token_count: int
    year: int
    agency_type: str
    statute_areas: tuple[str, ...]
    finding: str
    issue_tags: tuple[str, ...]

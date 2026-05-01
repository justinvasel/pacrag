from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import duckdb
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pac_api.db import OpinionDB
from pac_api.routes import ask, health, opinions, search
from pac_pipeline.chunk_models import Chunk
from pac_pipeline.retrieve_models import RetrievalResult, RetrievedChunk

_CREATE_OPINIONS = """
CREATE TABLE IF NOT EXISTS opinions (
    opinion_id TEXT PRIMARY KEY,
    year INTEGER NOT NULL,
    kind TEXT NOT NULL,
    sub_letter TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    parent_opinion_id TEXT,
    raw_title TEXT NOT NULL,
    agency_name TEXT NOT NULL,
    listing_url TEXT NOT NULL,
    pdf_url TEXT NOT NULL,
    pdf_path TEXT,
    pdf_sha256 TEXT,
    first_seen_at TIMESTAMP NOT NULL,
    last_checked_at TIMESTAMP NOT NULL,
    download_status TEXT NOT NULL DEFAULT 'pending',
    parse_status TEXT DEFAULT 'pending',
    parse_path TEXT,
    parsed_at TIMESTAMP,
    parser_version TEXT,
    extract_status TEXT DEFAULT 'pending',
    embed_status TEXT DEFAULT 'pending'
)
"""

_CREATE_METADATA = """
CREATE TABLE IF NOT EXISTS opinion_metadata (
    opinion_id TEXT PRIMARY KEY,
    issue_date DATE,
    counselor TEXT,
    complainant TEXT,
    public_agency TEXT NOT NULL,
    agency_type TEXT NOT NULL,
    statutes_invoked TEXT[] NOT NULL,
    statute_areas TEXT[] NOT NULL,
    issue_tags TEXT[] NOT NULL,
    tags_need_review BOOLEAN NOT NULL DEFAULT false,
    finding TEXT NOT NULL,
    summary TEXT NOT NULL,
    extracted_at TIMESTAMP NOT NULL,
    extractor_version TEXT NOT NULL
)
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    opinion_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section TEXT NOT NULL,
    page_num INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    year INTEGER NOT NULL,
    agency_type TEXT NOT NULL,
    statute_areas TEXT[] NOT NULL,
    finding TEXT NOT NULL,
    issue_tags TEXT[] NOT NULL,
    chunker_version TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
)
"""


def _seed_db(db_path: Path) -> None:
    now = datetime.now(UTC)
    conn = duckdb.connect(str(db_path))
    conn.execute(_CREATE_OPINIONS)
    conn.execute(_CREATE_METADATA)
    conn.execute(_CREATE_CHUNKS)

    opinions_data = [
        ("25-FC-118", 2025, "FC", "City of Bloomington", "violation", ["APRA"]),
        ("25-FC-002A", 2025, "FC", "Some School Board", "no_violation", ["ODL"]),
        ("24-INF-01", 2024, "INF", "State Agency", "moot", ["APRA", "ODL"]),
    ]
    for oid, year, kind, agency, finding, areas in opinions_data:
        conn.execute(
            """INSERT INTO opinions (
               opinion_id, year, kind, raw_title, agency_name,
               listing_url, pdf_url, first_seen_at, last_checked_at,
               download_status)
            VALUES (?, ?, ?, ?, ?, 'https://example.com/',
               'https://example.com/test.pdf', ?, ?, 'ok')""",
            [oid, year, kind, f"{oid}: Test", agency, now, now],
        )
        conn.execute(
            """INSERT INTO opinion_metadata (
               opinion_id, issue_date, counselor, complainant,
               public_agency, agency_type, statutes_invoked,
               statute_areas, issue_tags, tags_need_review,
               finding, summary, extracted_at, extractor_version)
            VALUES (?, ?, 'Luke H. Britt', NULL, ?, 'city',
               ['IC 5-14-3-4(b)(8)'], ?, ['denial-of-access'],
               false, ?, ?, ?, '0.4.0')""",
            [oid, date(2025, 1, 15), agency, areas, finding, f"Summary for {oid}.", now],
        )

    for i in range(3):
        conn.execute(
            """INSERT INTO chunks (chunk_id, opinion_id, chunk_index, section, page_num,
               text, token_count, year, agency_type, statute_areas, finding, issue_tags,
               chunker_version, created_at)
            VALUES (?, '25-FC-118', ?, 'analysis', 1, 'chunk text', 100, 2025,
               'city', ['APRA'], 'violation', ['denial-of-access'], '0.1.0', ?)""",
            [f"25-FC-118#c{i}", i, now],
        )
    conn.close()


@pytest.fixture
def seeded_db_path(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    _seed_db(db_path)
    return db_path


def _make_sample_chunk(
    opinion_id: str = "25-FC-118",
    chunk_index: int = 0,
    page_num: int = 1,
    section: str = "analysis",
    text: str = "The agency violated APRA by failing to respond within the statutory timeframe.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"{opinion_id}#c{chunk_index}",
            opinion_id=opinion_id,
            chunk_index=chunk_index,
            section=section,
            page_num=page_num,
            text=text,
            token_count=50,
            year=2025,
            agency_type="city",
            statute_areas=("APRA",),
            finding="violation",
            issue_tags=("denial-of-access",),
        ),
        dense_rank=chunk_index,
        lexical_rank=chunk_index + 1,
        fused_score=0.85,
        rerank_score=0.92,
    )


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        _make_sample_chunk(chunk_index=0, page_num=1),
        _make_sample_chunk(chunk_index=1, page_num=2, section="background"),
    ]


@pytest.fixture
def sample_retrieval_result(sample_chunks: list[RetrievedChunk]) -> RetrievalResult:
    return RetrievalResult(
        question="Can an agency charge fees for public records?",
        applied_filters={"statute_areas": ["APRA"]},
        relaxed=False,
        chunks=sample_chunks,
    )


@pytest.fixture
def test_app(seeded_db_path: Path) -> FastAPI:
    app = FastAPI()
    app.state.db = OpinionDB(seeded_db_path)
    app.state.anthropic_client = AsyncMock()
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(opinions.router)
    app.include_router(ask.router)
    return app


@pytest.fixture
async def client(
    test_app: FastAPI,
) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

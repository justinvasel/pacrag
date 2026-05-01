from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from anthropic import AsyncAnthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pac_pipeline import get_retrieval_models

from pac_api.config import settings
from pac_api.db import OpinionDB
from pac_api.routes import ask, health, opinions, search

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    log.info("startup_begin")

    await asyncio.to_thread(get_retrieval_models)
    log.info("retrieval_models_loaded")

    app.state.db = OpinionDB(settings.duckdb_path)
    app.state.anthropic_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    log.info("startup_complete")
    yield

    app.state.db.close()
    log.info("shutdown_complete")


app = FastAPI(title="PAC RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(opinions.router)
app.include_router(ask.router)

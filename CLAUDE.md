# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A retrieval-augmented Q&A system over Indiana Public Access Counselor advisory opinions (APRA and Open Door Law). Users describe a public-records scenario and get back analysis grounded in past PAC opinions with citations. See `handoff.md` for full project spec, milestones, and data quirks.

## Commands

```bash
# Setup
uv sync --all-packages          # install all Python workspace packages
cd apps/web && pnpm install     # install frontend deps

# Python
just lint                       # ruff check + pyright
just format                     # ruff format
just test                       # pytest across packages/ and tests/
uv run pytest packages/pac_crawler -k "test_name"  # single test

# Pipeline (stubs until wired)
just crawl                      # pac-crawl CLI
just parse                      # pac-parse CLI
just extract                    # pac-extract CLI
just embed                      # pac-embed CLI

# Services
just api                        # FastAPI on :8000 with auto-reload
just web                        # Next.js dev server (Turbopack)

# Frontend
cd apps/web && pnpm lint        # eslint
cd apps/web && pnpm typecheck   # tsc --noEmit
cd apps/web && pnpm format      # prettier

# Smoke test (requires ANTHROPIC_API_KEY in .env)
uv run python scripts/smoke_test.py
```

## Architecture

### Query pipeline (five stages, each independently testable)

```
Filter (DuckDB metadata) -> Retrieve (Qdrant dense + Tantivy BM25, parallel)
  -> RRF Fusion -> Rerank (bge-reranker-v2-m3) -> Generate (Claude Sonnet w/ citations)
```

### Indexing pipeline (offline, idempotent)

```
Crawl listing pages -> Download PDFs -> Parse (LiteParse) -> LLM-extract metadata (Haiku)
  -> Chunk by section -> Embed (bge-large-en-v1.5) -> Upsert into Qdrant + Tantivy + DuckDB
```

### Package dependency graph

```
pac_eval -> pac_pipeline -> pac_crawler
pac_api  -> pac_pipeline -> pac_crawler
```

`pac_api` owns the shared `Settings` singleton (`pac_api.config.settings`). All packages import config from there. Do not create a separate "common" package.

## Workspace Layout

- **uv workspace** (virtual root, `package = false`) with four members under `packages/`
- `apps/web/` -- Next.js 15, shadcn/ui v4 (Radix), App Router, Tailwind CSS v4, pnpm
- `data/` -- gitignored; holds PDFs, parsed JSON, Qdrant storage, Tantivy index, DuckDB catalog, HF model cache
- `scripts/` -- standalone utilities (smoke test, model warming, API key check)

## Tech Choices (Do Not Replace)

Qdrant embedded (no server), tantivy-py, DuckDB, LiteParse CLI, sentence-transformers on MPS, structlog (JSON). No LangChain, LlamaIndex, Celery, Redis, Postgres, or Docker.

## Key Conventions

- **Idempotence**: every pipeline stage is safe to rerun. DuckDB manifest is the source of truth for "has this been done."
- **Opinion ID is canonical key**: derive from listing page text, never from PDF filename (filenames are unreliable).
- **Pydantic for all cross-package types**: pyright is strict on `packages/`. No loose dicts at public APIs.
- **Logging**: structlog JSON in library code. No `print()` except in `scripts/`.
- **Tests**: pytest with pytest-recording (VCR cassettes) for HTTP/Anthropic calls. Tests must never hit live network.
- **LLM extraction validation**: validate with Pydantic, retry on failure with the error fed back. Never silently coerce or allow hallucinated metadata.
- **Citation discipline**: generated answers must cite as `(opinion_id, p. N)`. A post-processor at the API boundary rejects any citation not in the retrieved set.
- **Advisory disclaimer**: PAC opinions are non-binding. Every generated answer must include this caveat once (not repeatedly).

## Python

- Python 3.12, managed by uv
- Ruff for lint+format (line-length 100, rules: E, F, I, N, UP, B, SIM, RUF)
- Pyright strict on `packages/`
- Add deps via: `uv add --package <pkg-name> <dependency>`

## Frontend

- Next.js 15 with App Router, React 19, TypeScript strict
- shadcn/ui v4 (radix-nova style, lucide icons, CSS variables)
- Path alias: `@/*` maps to project root
- Design: navy/cream palette, Literata body type, restrained chrome, no gradients

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Other vars (`PAC_DATA_DIR`, `PAC_QDRANT_PATH`, etc.) have working defaults.

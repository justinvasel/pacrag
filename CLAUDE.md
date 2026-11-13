# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A retrieval-augmented Q&A system over Indiana Public Access Counselor advisory opinions (APRA and Open Door Law). Users describe a public-records scenario and get back analysis grounded in past PAC opinions with citations. See `handoff.md` for full project spec, milestones, and data quirks.

## Commands

```bash
# Setup
just sync                       # uv sync --all-packages (install all Python workspace packages)
cd apps/web && pnpm install     # install frontend deps

# Python
just lint                       # ruff check + pyright
just format                     # ruff format
just test                       # pytest across all packages
uv run pytest packages/pac_crawler -k "test_name"  # single test

# Indexing pipeline (run in order; each stage is idempotent)
just crawl                      # pac-crawl CLI
just parse                      # pac-parse CLI
just extract                    # pac-extract CLI
just embed                      # pac-embed CLI

# Eval (gold set: packages/pac_eval/gold_set.yaml; scorecards written to data/eval/ as Parquet)
just eval                       # pac-eval run; flags: --skip-generation, --with-filters, --limit N, --concurrency N
just eval-diff A.parquet B.parquet  # compare two scorecards, flag regressions (--threshold, default 0.01)

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
pac_eval -> pac_api, pac_pipeline
pac_api  -> pac_pipeline, pac_crawler
pac_pipeline -> pac_crawler
```

`pac_api` owns the shared `Settings` singleton (`pac_api.config.settings`), including model names (embedding, reranker, extraction, generation). `pac_pipeline` (embed, extract, retrieve) and `pac_eval` import config from there; `pac_crawler` and the pipeline CLIs read `PAC_*` env vars directly. Do not create a separate "common" package.

Caveat: `pac_pipeline` imports `pac_api.config` without declaring `pac-api` as a dependency (it can't, since `pac_api` depends on `pac_pipeline`). This works only because the workspace installs all packages together. `pac_api/__init__.py` lazy-loads `app` so importing `pac_api.config` does not build the FastAPI app.

### Eval harness (`pac_eval`)

Runs the gold set through retrieval (recall@8/16, MRR) and, unless `--skip-generation`, generation (citation precision against the retrieved set, hedging/overclaim detection, LLM-judged key-fact coverage). Results are saved as Parquet scorecards; `diff` compares two runs.

## Workspace Layout

- **uv workspace** (virtual root, `package = false`) with four members under `packages/`
- `apps/web/` -- Next.js 16, shadcn/ui v4 (Radix), App Router, Tailwind CSS v4, pnpm. `app/api/ask/route.ts` proxies to the FastAPI backend (`NEXT_PUBLIC_API_URL`, default `http://localhost:8000`)
- `data/` -- gitignored; holds PDFs, parsed JSON, Qdrant storage, Tantivy index, DuckDB catalog, HF model cache, eval scorecards
- `scripts/` -- standalone utilities (smoke test, model warming, API key check)

## Tech Choices (Do Not Replace)

Qdrant embedded (no server), tantivy-py, DuckDB, LiteParse CLI, sentence-transformers on MPS, structlog (JSON). No LangChain, LlamaIndex, Celery, Redis, Postgres, or Docker.

## Key Conventions

- **Idempotence**: every pipeline stage is safe to rerun. DuckDB manifest is the source of truth for "has this been done."
- **Opinion ID is canonical key**: derive from listing page text, never from PDF filename (filenames are unreliable).
- **Pydantic for all cross-package types**: no loose dicts at public APIs. Domain models are `frozen=True` (only the LiteParse output wrappers in `litparse.py` are not); tuple fields coerce lists from DB rows.
- **Logging**: structlog JSON in library code. No `print()` except in `scripts/`.
- **Tests**: tests must never hit live network. Current tests mock Anthropic/HTTP/model clients (`unittest.mock`, `monkeypatch`); pytest-recording is installed for VCR cassettes but no cassettes exist yet.
- **LLM extraction validation**: validate with Pydantic, retry on failure with the error fed back. Never silently coerce or allow hallucinated metadata.
- **Citation discipline**: generated answers must cite as `(opinion_id, p. N)`. A post-processor at the API boundary rejects any citation not in the retrieved set.
- **Advisory disclaimer**: PAC opinions are non-binding. Every generated answer must include this caveat once (not repeatedly).

## Python

- Python >=3.12, managed by uv (no `.python-version` pin, so uv may pick a newer interpreter; ruff and pyright target 3.12)
- Ruff for lint+format (line-length 100, rules: E, F, I, N, UP, B, SIM, RUF)
- Pyright in `standard` mode on `packages/` (including tests); `just lint` must pass with zero errors
- Pre-commit hooks run ruff, ruff-format, and basic file checks on commit
- Add deps via: `uv add --package <pkg-name> <dependency>`

## Frontend

- Next.js 16 with App Router, React 19, TypeScript strict
- shadcn/ui v4 (radix-nova style, lucide icons, CSS variables)
- Path alias: `@/*` maps to `apps/web/`
- Design: navy/cream palette, Literata body type, restrained chrome, no gradients

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Other vars (`PAC_DATA_DIR`, `PAC_QDRANT_PATH`, `PAC_TANTIVY_PATH`, `PAC_DUCKDB_PATH`, `PAC_MODELS_DIR`, `PAC_LIT_BINARY`) have working defaults.

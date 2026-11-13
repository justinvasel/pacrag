# justfile
default:
    @just --list

# Python
sync:
    uv sync --all-packages

lint:
    uv run ruff check .
    uv run pyright

format:
    uv run ruff format .

test:
    uv run pytest

# Indexing pipeline (run in order; each stage is idempotent)
crawl:
    uv run pac-crawl

parse:
    uv run pac-parse

extract:
    uv run pac-extract

embed:
    uv run pac-embed

# Eval
eval *ARGS:
    uv run pac-eval run {{ARGS}}

eval-diff A B *ARGS:
    uv run pac-eval diff {{A}} {{B}} {{ARGS}}

# Services
api:
    uv run uvicorn pac_api.main:app --reload --port 8000

# Frontend
web:
    cd apps/web && pnpm dev

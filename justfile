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
crawl *ARGS:
    uv run pac-crawl {{ARGS}}

parse *ARGS:
    uv run pac-parse {{ARGS}}

extract *ARGS:
    uv run pac-extract {{ARGS}}

embed *ARGS:
    uv run pac-embed {{ARGS}}

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

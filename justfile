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

# Pipeline (stubs for now, wired up as we go)
crawl:
    uv run pac-crawl

parse:
    uv run pac-parse

extract:
    uv run pac-extract

embed:
    uv run pac-embed

# Services
api:
    uv run uvicorn pac_api.main:app --reload --port 8000

# Frontend
web:
    cd apps/web && pnpm dev

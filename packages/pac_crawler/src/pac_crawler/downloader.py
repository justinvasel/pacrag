from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import structlog

from pac_crawler.http_client import RateLimiter, download_file
from pac_crawler.models import ListingEntry

log = structlog.get_logger()


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pdf_dest_path(data_dir: Path, year: int, opinion_id: str) -> Path:
    return data_dir / "pdfs" / str(year) / f"{opinion_id}.pdf"


async def download_opinion_pdf(
    client: httpx.AsyncClient,
    entry: ListingEntry,
    data_dir: Path,
    existing_sha256: str | None,
    limiter: RateLimiter,
) -> tuple[Path, str, str]:
    """Download a PDF. Returns (path, sha256, status).

    Status is "ok", "skipped" (hash unchanged), or "http_error".
    """
    dest = pdf_dest_path(data_dir, entry.year, entry.opinion_id)

    try:
        data = await download_file(client, entry.pdf_url, dest, limiter)
    except httpx.HTTPStatusError as e:
        log.error(
            "download_failed",
            opinion_id=entry.opinion_id,
            url=entry.pdf_url,
            status=e.response.status_code,
        )
        return dest, "", "http_error"
    except httpx.TransportError as e:
        log.error(
            "download_transport_error",
            opinion_id=entry.opinion_id,
            url=entry.pdf_url,
            error=str(e),
        )
        return dest, "", "http_error"

    sha = compute_sha256(data)

    if existing_sha256 and sha == existing_sha256:
        log.debug("download_skipped_unchanged", opinion_id=entry.opinion_id)
        return dest, sha, "skipped"

    log.info(
        "download_ok",
        opinion_id=entry.opinion_id,
        size_bytes=len(data),
        path=str(dest),
    )
    return dest, sha, "ok"


async def download_all(
    client: httpx.AsyncClient,
    entries: list[ListingEntry],
    data_dir: Path,
    existing_hashes: dict[str, str | None],
    limiter: RateLimiter,
    limit: int | None = None,
) -> list[tuple[str, Path, str, str]]:
    """Download PDFs for entries. Returns list of (opinion_id, path, sha256, status)."""
    results: list[tuple[str, Path, str, str]] = []
    to_download = entries[:limit] if limit else entries

    for entry in to_download:
        existing_sha = existing_hashes.get(entry.opinion_id)
        path, sha, status = await download_opinion_pdf(
            client, entry, data_dir, existing_sha, limiter
        )
        results.append((entry.opinion_id, path, sha, status))

    return results

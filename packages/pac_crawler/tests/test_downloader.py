from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pac_crawler.downloader import (
    compute_sha256,
    download_all,
    download_opinion_pdf,
    pdf_dest_path,
)
from pac_crawler.http_client import RateLimiter
from pac_crawler.models import ListingEntry, OpinionKind


class TestComputeSha256:
    def test_known_hash(self) -> None:
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert compute_sha256(data) == expected

    def test_empty_bytes(self) -> None:
        result = compute_sha256(b"")
        assert len(result) == 64


class TestPdfDestPath:
    def test_path_structure(self, tmp_path: Path) -> None:
        result = pdf_dest_path(tmp_path, 2025, "25-FC-118")
        assert result == tmp_path / "pdfs" / "2025" / "25-FC-118.pdf"

    def test_sub_letter_in_filename(self, tmp_path: Path) -> None:
        result = pdf_dest_path(tmp_path, 2024, "24-FC-77A")
        assert result.name == "24-FC-77A.pdf"


class TestDownloadOpinionPdf:
    @pytest.fixture
    def entry(self) -> ListingEntry:
        return ListingEntry(
            opinion_id="25-FC-118",
            year=2025,
            kind=OpinionKind.FC,
            sub_letter=None,
            is_revised=False,
            raw_title="25-FC-118: Some title",
            agency_name="Some title",
            pdf_url="https://www.in.gov/pac/files/advisory/25-FC-118.pdf",
            listing_url="https://example.com/2025/",
        )

    @pytest.fixture
    def limiter(self) -> RateLimiter:
        return RateLimiter(delay=0.0)

    async def test_successful_download(
        self, tmp_path: Path, entry: ListingEntry, limiter: RateLimiter
    ) -> None:
        pdf_data = b"%PDF-1.4 fake content"
        mock_client = AsyncMock()

        with patch(
            "pac_crawler.downloader.download_file",
            new_callable=AsyncMock,
            return_value=pdf_data,
        ):
            path, sha, status = await download_opinion_pdf(
                mock_client, entry, tmp_path, None, limiter
            )

        assert status == "ok"
        assert sha == compute_sha256(pdf_data)
        assert path == tmp_path / "pdfs" / "2025" / "25-FC-118.pdf"

    async def test_skipped_when_hash_matches(
        self, tmp_path: Path, entry: ListingEntry, limiter: RateLimiter
    ) -> None:
        pdf_data = b"%PDF-1.4 fake content"
        existing_hash = compute_sha256(pdf_data)
        mock_client = AsyncMock()

        with patch(
            "pac_crawler.downloader.download_file",
            new_callable=AsyncMock,
            return_value=pdf_data,
        ):
            _path, _sha, status = await download_opinion_pdf(
                mock_client, entry, tmp_path, existing_hash, limiter
            )

        assert status == "skipped"

    async def test_http_error(
        self, tmp_path: Path, entry: ListingEntry, limiter: RateLimiter
    ) -> None:
        import httpx

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status_code = 404

        with patch(
            "pac_crawler.downloader.download_file",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=AsyncMock(), response=mock_response
            ),
        ):
            _path, _sha, status = await download_opinion_pdf(
                mock_client, entry, tmp_path, None, limiter
            )

        assert status == "http_error"


class TestDownloadAll:
    async def test_respects_limit(self, tmp_path: Path) -> None:
        entries = [
            ListingEntry(
                opinion_id=f"25-FC-{i:03d}",
                year=2025,
                kind=OpinionKind.FC,
                sub_letter=None,
                is_revised=False,
                raw_title=f"25-FC-{i:03d}: Title",
                agency_name="Title",
                pdf_url=f"https://example.com/{i}.pdf",
                listing_url="https://example.com/2025/",
            )
            for i in range(10)
        ]
        limiter = RateLimiter(delay=0.0)
        mock_client = AsyncMock()
        pdf_data = b"fake"

        with patch(
            "pac_crawler.downloader.download_file",
            new_callable=AsyncMock,
            return_value=pdf_data,
        ):
            results = await download_all(mock_client, entries, tmp_path, {}, limiter, limit=3)

        assert len(results) == 3

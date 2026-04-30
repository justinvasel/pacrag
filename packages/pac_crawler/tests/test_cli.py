from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from pac_crawler.cli import run
from pac_crawler.manifest import ManifestDB
from pac_crawler.models import ListingEntry, OpinionKind


def _make_entries(count: int, kind: OpinionKind = OpinionKind.FC) -> list[ListingEntry]:
    prefix = "FC" if kind == OpinionKind.FC else "INF"
    return [
        ListingEntry(
            opinion_id=f"25-{prefix}-{i:03d}",
            year=2025,
            kind=kind,
            sub_letter=None,
            is_revised=False,
            raw_title=f"25-{prefix}-{i:03d}: Title {i}",
            agency_name=f"Title {i}",
            pdf_url=f"https://example.com/{prefix}-{i}.pdf",
            listing_url="https://example.com/2025/",
        )
        for i in range(count)
    ]


class TestRunListingOnly:
    async def test_populates_manifest(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.duckdb"
        entries = _make_entries(5)

        with patch(
            "pac_crawler.cli.discover_all_listings",
            new_callable=AsyncMock,
            return_value=entries,
        ):
            await run(
                data_dir=tmp_path,
                db_path=db_path,
                year_filter=None,
                kind_filter=None,
                listing_only=True,
                limit=None,
            )

        with ManifestDB(db_path) as db:
            counts = db.count_by_status()
            assert counts.get("pending", 0) == 5

    async def test_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.duckdb"
        entries = _make_entries(3)

        for _ in range(2):
            with patch(
                "pac_crawler.cli.discover_all_listings",
                new_callable=AsyncMock,
                return_value=entries,
            ):
                await run(
                    data_dir=tmp_path,
                    db_path=db_path,
                    year_filter=None,
                    kind_filter=None,
                    listing_only=True,
                    limit=None,
                )

        with ManifestDB(db_path) as db:
            counts = db.count_by_status()
            assert counts.get("pending", 0) == 3


class TestRunWithDownloads:
    async def test_downloads_pending(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.duckdb"
        entries = _make_entries(2)
        pdf_data = b"%PDF fake"

        with (
            patch(
                "pac_crawler.cli.discover_all_listings",
                new_callable=AsyncMock,
                return_value=entries,
            ),
            patch(
                "pac_crawler.downloader.download_file",
                new_callable=AsyncMock,
                return_value=pdf_data,
            ),
        ):
            await run(
                data_dir=tmp_path,
                db_path=db_path,
                year_filter=None,
                kind_filter=None,
                listing_only=False,
                limit=None,
            )

        with ManifestDB(db_path) as db:
            counts = db.count_by_status()
            assert counts.get("ok", 0) == 2

    async def test_limit_respected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.duckdb"
        entries = _make_entries(5)
        pdf_data = b"%PDF fake"

        with (
            patch(
                "pac_crawler.cli.discover_all_listings",
                new_callable=AsyncMock,
                return_value=entries,
            ),
            patch(
                "pac_crawler.downloader.download_file",
                new_callable=AsyncMock,
                return_value=pdf_data,
            ),
        ):
            await run(
                data_dir=tmp_path,
                db_path=db_path,
                year_filter=None,
                kind_filter=None,
                listing_only=False,
                limit=2,
            )

        with ManifestDB(db_path) as db:
            counts = db.count_by_status()
            assert counts.get("ok", 0) == 2
            assert counts.get("pending", 0) == 3

    async def test_second_run_skips_already_downloaded(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.duckdb"
        entries = _make_entries(3)
        pdf_data = b"%PDF fake"

        with (
            patch(
                "pac_crawler.cli.discover_all_listings",
                new_callable=AsyncMock,
                return_value=entries,
            ),
            patch(
                "pac_crawler.downloader.download_file",
                new_callable=AsyncMock,
                return_value=pdf_data,
            ),
        ):
            await run(
                data_dir=tmp_path,
                db_path=db_path,
                year_filter=None,
                kind_filter=None,
                listing_only=False,
                limit=None,
            )

        with (
            patch(
                "pac_crawler.cli.discover_all_listings",
                new_callable=AsyncMock,
                return_value=entries,
            ),
            patch(
                "pac_crawler.downloader.download_file",
                new_callable=AsyncMock,
                return_value=pdf_data,
            ) as mock_dl,
        ):
            await run(
                data_dir=tmp_path,
                db_path=db_path,
                year_filter=None,
                kind_filter=None,
                listing_only=False,
                limit=None,
            )

        mock_dl.assert_not_called()

        with ManifestDB(db_path) as db:
            counts = db.count_by_status()
            assert counts.get("ok", 0) == 3


class TestMarkMissing:
    async def test_removed_opinions_marked_missing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.duckdb"
        all_entries = _make_entries(3)
        subset = all_entries[:2]

        with patch(
            "pac_crawler.cli.discover_all_listings",
            new_callable=AsyncMock,
            return_value=all_entries,
        ):
            await run(
                data_dir=tmp_path,
                db_path=db_path,
                year_filter=None,
                kind_filter=None,
                listing_only=True,
                limit=None,
            )

        with patch(
            "pac_crawler.cli.discover_all_listings",
            new_callable=AsyncMock,
            return_value=subset,
        ):
            await run(
                data_dir=tmp_path,
                db_path=db_path,
                year_filter=None,
                kind_filter=None,
                listing_only=True,
                limit=None,
            )

        with ManifestDB(db_path) as db:
            counts = db.count_by_status()
            assert counts.get("missing", 0) == 1
            assert counts.get("pending", 0) == 2

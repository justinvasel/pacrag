from __future__ import annotations

from pathlib import Path

from pac_crawler.manifest import ManifestDB
from pac_crawler.models import ListingEntry, OpinionKind


class TestManifestDBCreation:
    def test_creates_table(self, tmp_db: Path) -> None:
        with ManifestDB(tmp_db) as db:
            tables = db._conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'opinions'"
            ).fetchall()
            assert len(tables) == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "test.duckdb"
        with ManifestDB(db_path):
            assert db_path.parent.exists()

    def test_idempotent_table_creation(self, tmp_db: Path) -> None:
        with ManifestDB(tmp_db):
            pass
        with ManifestDB(tmp_db) as db:
            tables = db._conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'opinions'"
            ).fetchall()
            assert len(tables) == 1


class TestUpsertListing:
    def test_insert_new_entry(self, tmp_db: Path, sample_entry: ListingEntry) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            rows = db._conn.execute("SELECT * FROM opinions").fetchall()
            assert len(rows) == 1

    def test_preserves_first_seen_at_on_update(
        self, tmp_db: Path, sample_entry: ListingEntry
    ) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            first_seen = db._conn.execute(
                "SELECT first_seen_at FROM opinions WHERE opinion_id = ?",
                [sample_entry.opinion_id],
            ).fetchone()

            db.upsert_listing(sample_entry)
            second_seen = db._conn.execute(
                "SELECT first_seen_at FROM opinions WHERE opinion_id = ?",
                [sample_entry.opinion_id],
            ).fetchone()

            assert first_seen == second_seen

    def test_updates_last_checked_at(self, tmp_db: Path, sample_entry: ListingEntry) -> None:
        import time

        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            first_check = db._conn.execute(
                "SELECT last_checked_at FROM opinions WHERE opinion_id = ?",
                [sample_entry.opinion_id],
            ).fetchone()

            time.sleep(0.01)
            db.upsert_listing(sample_entry)
            second_check = db._conn.execute(
                "SELECT last_checked_at FROM opinions WHERE opinion_id = ?",
                [sample_entry.opinion_id],
            ).fetchone()

            assert second_check is not None
            assert first_check is not None
            assert second_check[0] >= first_check[0]

    def test_no_duplicate_rows(self, tmp_db: Path, sample_entry: ListingEntry) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            db.upsert_listing(sample_entry)
            count = db._conn.execute("SELECT count(*) FROM opinions").fetchone()
            assert count is not None
            assert count[0] == 1

    def test_sets_pending_status(self, tmp_db: Path, sample_entry: ListingEntry) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            status = db._conn.execute(
                "SELECT download_status FROM opinions WHERE opinion_id = ?",
                [sample_entry.opinion_id],
            ).fetchone()
            assert status is not None
            assert status[0] == "pending"

    def test_revised_entry_sets_revision(self, tmp_db: Path) -> None:
        entry = ListingEntry(
            opinion_id="23-FC-100",
            year=2023,
            kind=OpinionKind.FC,
            sub_letter=None,
            is_revised=True,
            raw_title="23-FC-100 REVISED: Some title",
            agency_name="Some title",
            pdf_url="https://example.com/23-FC-100-REVISED.pdf",
            listing_url="https://example.com/2023/",
        )
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(entry)
            row = db._conn.execute(
                "SELECT revision FROM opinions WHERE opinion_id = ?",
                [entry.opinion_id],
            ).fetchone()
            assert row is not None
            assert row[0] == 1

    def test_sub_letter_sets_parent(self, tmp_db: Path, sample_entry_b: ListingEntry) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry_b)
            row = db._conn.execute(
                "SELECT parent_opinion_id FROM opinions WHERE opinion_id = ?",
                [sample_entry_b.opinion_id],
            ).fetchone()
            assert row is not None
            assert row[0] == "25-FC-002"


class TestUpsertListings:
    def test_batch_insert(
        self,
        tmp_db: Path,
        sample_entry: ListingEntry,
        sample_entry_b: ListingEntry,
    ) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listings([sample_entry, sample_entry_b])
            count = db._conn.execute("SELECT count(*) FROM opinions").fetchone()
            assert count is not None
            assert count[0] == 2


class TestUpdateDownload:
    def test_updates_fields(self, tmp_db: Path, sample_entry: ListingEntry) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            db.update_download(
                opinion_id="25-FC-118",
                pdf_path="data/pdfs/2025/25-FC-118.pdf",
                pdf_sha256="abc123",
                status="ok",
            )
            row = db._conn.execute(
                "SELECT pdf_path, pdf_sha256, download_status FROM opinions WHERE opinion_id = ?",
                ["25-FC-118"],
            ).fetchone()
            assert row is not None
            assert row[0] == "data/pdfs/2025/25-FC-118.pdf"
            assert row[1] == "abc123"
            assert row[2] == "ok"


class TestMarkMissing:
    def test_marks_unseen_as_missing(
        self,
        tmp_db: Path,
        sample_entry: ListingEntry,
        sample_entry_b: ListingEntry,
    ) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listings([sample_entry, sample_entry_b])
            count = db.mark_missing(seen_ids={sample_entry.opinion_id}, kind="FC")
            assert count == 1
            status = db._conn.execute(
                "SELECT download_status FROM opinions WHERE opinion_id = ?",
                [sample_entry_b.opinion_id],
            ).fetchone()
            assert status is not None
            assert status[0] == "missing"

    def test_does_not_mark_different_kind(
        self,
        tmp_db: Path,
        sample_entry: ListingEntry,
        sample_inf_entry: ListingEntry,
    ) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listings([sample_entry, sample_inf_entry])
            count = db.mark_missing(seen_ids=set(), kind="FC")
            assert count == 1
            inf_status = db._conn.execute(
                "SELECT download_status FROM opinions WHERE opinion_id = ?",
                [sample_inf_entry.opinion_id],
            ).fetchone()
            assert inf_status is not None
            assert inf_status[0] == "pending"

    def test_does_not_remark_already_missing(
        self,
        tmp_db: Path,
        sample_entry: ListingEntry,
    ) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            db.mark_missing(seen_ids=set(), kind="FC")
            count = db.mark_missing(seen_ids=set(), kind="FC")
            assert count == 0


class TestGetters:
    def test_get_existing_hashes(self, tmp_db: Path, sample_entry: ListingEntry) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            db.update_download("25-FC-118", "some/path.pdf", "hash123", "ok")
            hashes = db.get_existing_hashes()
            assert hashes["25-FC-118"] == "hash123"

    def test_get_existing_hashes_null(self, tmp_db: Path, sample_entry: ListingEntry) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listing(sample_entry)
            hashes = db.get_existing_hashes()
            assert hashes["25-FC-118"] is None

    def test_get_pending_downloads(
        self,
        tmp_db: Path,
        sample_entry: ListingEntry,
        sample_entry_b: ListingEntry,
    ) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listings([sample_entry, sample_entry_b])
            db.update_download("25-FC-118", "some/path.pdf", "hash123", "ok")
            pending = db.get_pending_downloads()
            assert len(pending) == 1
            assert pending[0].opinion_id == "25-FC-002A"

    def test_count_by_status(
        self,
        tmp_db: Path,
        sample_entry: ListingEntry,
        sample_entry_b: ListingEntry,
    ) -> None:
        with ManifestDB(tmp_db) as db:
            db.upsert_listings([sample_entry, sample_entry_b])
            db.update_download("25-FC-118", "some/path.pdf", "hash123", "ok")
            counts = db.count_by_status()
            assert counts["ok"] == 1
            assert counts["pending"] == 1

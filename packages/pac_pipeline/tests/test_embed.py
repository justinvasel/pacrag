from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from pac_pipeline.embed import embed_all
from pac_pipeline.embed_db import EmbedDB
from pac_pipeline.models import ParsedOpinion, ParsedPage


def _make_parsed(opinion_id: str = "25-FC-118") -> ParsedOpinion:
    full_text = (
        "STATE OF INDIANA  PUBLIC ACCESS COUNSELOR\n"
        "JENNIFER G. RUBY\n\n"
        "February 2, 2026\n\n"
        "Re: Complaint 25-FC-118\n\n"
        "BACKGROUND\n\n"
        "The complainant alleges a violation.\n\n"
        "ANALYSIS\n\n"
        "The PAC analyzes the complaint.\n\n"
        "CONCLUSION\n\n"
        "The PAC finds a violation.\n"
    )
    bg_start = full_text.index("BACKGROUND\n\n") + len("BACKGROUND\n\n")
    bg_end = full_text.index("\n\nANALYSIS")
    an_start = full_text.index("ANALYSIS\n\n") + len("ANALYSIS\n\n")
    an_end = full_text.index("\n\nCONCLUSION")
    co_start = full_text.index("CONCLUSION\n\n") + len("CONCLUSION\n\n")
    co_end = len(full_text.rstrip())

    return ParsedOpinion(
        opinion_id=opinion_id,
        full_text=full_text,
        pages=[ParsedPage(page_num=1, text=full_text, char_count=len(full_text), ocr_used=False)],
        sections={
            "BACKGROUND": (bg_start, bg_end),
            "ANALYSIS": (an_start, an_end),
            "CONCLUSION": (co_start, co_end),
        },
        parser_version="0.1.0",
    )


def _setup_parsed_files(
    tmp_path: Path,
    db_path: Path,
    opinion_ids: list[str] | None = None,
) -> None:
    if opinion_ids is None:
        opinion_ids = ["25-FC-118", "25-FC-002A"]

    for oid in opinion_ids:
        parsed = _make_parsed(oid)
        with EmbedDB(db_path) as db:
            row = db._conn.execute(
                "SELECT parse_path FROM opinions WHERE opinion_id = ?", [oid]
            ).fetchone()
        if row and row[0]:
            parse_path = Path(row[0])
            parse_path.parent.mkdir(parents=True, exist_ok=True)
            parse_path.write_text(parsed.model_dump_json(indent=2))


def _mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.tokenizer = MagicMock()
    embedder.tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))

    def fake_encode(texts: list[str], **kwargs: object) -> np.ndarray:
        return np.random.default_rng(42).random((len(texts), 1024)).astype(np.float32)

    embedder.encode.side_effect = fake_encode
    return embedder


class TestEmbedAll:
    def test_processes_eligible_opinions(
        self, tmp_db_with_extracted_opinions: Path, tmp_path: Path
    ) -> None:
        _setup_parsed_files(tmp_path, tmp_db_with_extracted_opinions)

        with (
            patch("pac_pipeline.embed.SentenceTransformer") as mock_st_cls,
            patch("pac_pipeline.embed.QdrantClient") as mock_qdrant_cls,
            patch("pac_pipeline.embed._ensure_tantivy_index") as mock_tantivy,
            patch("pac_pipeline.embed.settings") as mock_settings,
        ):
            mock_settings.models_dir = tmp_path / "models"
            mock_settings.embedding_model = "test-model"
            mock_settings.qdrant_path = tmp_path / "qdrant"
            mock_st_cls.return_value = _mock_embedder()
            mock_qdrant = MagicMock()
            mock_qdrant.collection_exists.return_value = True
            mock_qdrant_cls.return_value = mock_qdrant
            mock_tantivy_index = MagicMock()
            mock_tantivy.return_value = mock_tantivy_index

            embedded, errors = embed_all(
                db_path=tmp_db_with_extracted_opinions,
            )

        assert embedded == 2
        assert errors == 0

        with EmbedDB(tmp_db_with_extracted_opinions) as db:
            counts = db.count_by_embed_status()
            assert counts.get("ok", 0) == 2
            assert db.get_chunk_count() > 0

    def test_records_errors(self, tmp_db_with_extracted_opinions: Path, tmp_path: Path) -> None:
        with (
            patch("pac_pipeline.embed.SentenceTransformer") as mock_st_cls,
            patch("pac_pipeline.embed.QdrantClient") as mock_qdrant_cls,
            patch("pac_pipeline.embed._ensure_tantivy_index") as mock_tantivy,
            patch("pac_pipeline.embed.settings") as mock_settings,
        ):
            mock_settings.models_dir = tmp_path / "models"
            mock_settings.embedding_model = "test-model"
            mock_settings.qdrant_path = tmp_path / "qdrant"
            embedder = _mock_embedder()
            embedder.encode.side_effect = RuntimeError("Embedding failed")
            mock_st_cls.return_value = embedder
            mock_qdrant = MagicMock()
            mock_qdrant.collection_exists.return_value = True
            mock_qdrant_cls.return_value = mock_qdrant
            mock_tantivy.return_value = MagicMock()

            embedded, errors = embed_all(
                db_path=tmp_db_with_extracted_opinions,
            )

        assert embedded == 0
        assert errors == 2

    def test_idempotent(self, tmp_db_with_extracted_opinions: Path, tmp_path: Path) -> None:
        _setup_parsed_files(tmp_path, tmp_db_with_extracted_opinions)

        def run_embed() -> tuple[int, int]:
            with (
                patch("pac_pipeline.embed.SentenceTransformer") as mock_st_cls,
                patch("pac_pipeline.embed.QdrantClient") as mock_qdrant_cls,
                patch("pac_pipeline.embed._ensure_tantivy_index") as mock_tantivy,
                patch("pac_pipeline.embed.settings") as mock_settings,
            ):
                mock_settings.models_dir = tmp_path / "models"
                mock_settings.embedding_model = "test-model"
                mock_settings.qdrant_path = tmp_path / "qdrant"
                mock_st_cls.return_value = _mock_embedder()
                mock_qdrant = MagicMock()
                mock_qdrant.collection_exists.return_value = True
                mock_qdrant_cls.return_value = mock_qdrant
                mock_tantivy.return_value = MagicMock()

                return embed_all(
                    db_path=tmp_db_with_extracted_opinions,
                )

        run_embed()
        embedded, errors = run_embed()
        assert embedded == 0
        assert errors == 0

    def test_filters_by_opinion_id(
        self, tmp_db_with_extracted_opinions: Path, tmp_path: Path
    ) -> None:
        _setup_parsed_files(tmp_path, tmp_db_with_extracted_opinions, ["25-FC-118"])

        with (
            patch("pac_pipeline.embed.SentenceTransformer") as mock_st_cls,
            patch("pac_pipeline.embed.QdrantClient") as mock_qdrant_cls,
            patch("pac_pipeline.embed._ensure_tantivy_index") as mock_tantivy,
            patch("pac_pipeline.embed.settings") as mock_settings,
        ):
            mock_settings.models_dir = tmp_path / "models"
            mock_settings.embedding_model = "test-model"
            mock_settings.qdrant_path = tmp_path / "qdrant"
            mock_st_cls.return_value = _mock_embedder()
            mock_qdrant = MagicMock()
            mock_qdrant.collection_exists.return_value = True
            mock_qdrant_cls.return_value = mock_qdrant
            mock_tantivy.return_value = MagicMock()

            embedded, errors = embed_all(
                db_path=tmp_db_with_extracted_opinions,
                opinion_id="25-FC-118",
            )

        assert embedded == 1
        assert errors == 0

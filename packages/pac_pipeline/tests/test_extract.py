from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pac_pipeline.extract import build_text_payload, extract_all, extract_one
from pac_pipeline.extract_db import ExtractDB
from pac_pipeline.extract_models import OpinionMetadata
from pac_pipeline.models import ParsedOpinion, ParsedPage


def _make_parsed(
    opinion_id: str = "25-FC-118",
    *,
    with_sections: bool = True,
) -> ParsedOpinion:
    if with_sections:
        full_text = (
            "STATE OF INDIANA  PUBLIC ACCESS COUNSELOR\n"
            "JENNIFER G. RUBY\n\n"
            "February 2, 2026\n\n"
            "Re: Complaint 25-FC-118\n"
            "David Askins (Complainant) v.\n"
            "City of Bloomington (Respondent)\n\n"
            "This advisory opinion is in response to the complaint.\n\n"
            "BACKGROUND\n\n"
            "The complainant alleges that Respondent violated APRA.\n\n"
            "ANALYSIS\n\n"
            "APRA provides persons with the right to inspect and copy public records. "
            "IC 5-14-3-3(a). The agency cited IC 5-14-3-4(b)(20).\n\n"
            "CONCLUSION\n\n"
            "Based on the foregoing, the PAC finds a violation of APRA.\n\n"
            "Jennifer G. Ruby\n"
            "Public Access Counselor"
        )
        bg_start = full_text.index("BACKGROUND\n\n") + len("BACKGROUND\n\n")
        bg_end = full_text.index("\n\nANALYSIS")
        an_start = full_text.index("ANALYSIS\n\n") + len("ANALYSIS\n\n")
        an_end = full_text.index("\n\nCONCLUSION")
        co_start = full_text.index("CONCLUSION\n\n") + len("CONCLUSION\n\n")
        co_end = full_text.index("\n\nJennifer G. Ruby\nPublic Access Counselor")
        sections = {
            "BACKGROUND": (bg_start, bg_end),
            "ANALYSIS": (an_start, an_end),
            "CONCLUSION": (co_start, co_end),
        }
    else:
        full_text = (
            "STATE OF INDIANA  PUBLIC ACCESS COUNSELOR\n"
            "LUKE H. BRITT\n\n"
            "Some opinion text without standard sections.\n"
            "The agency is City of Indianapolis. IC 5-14-3-3(a) applies.\n"
        )
        sections = {}

    return ParsedOpinion(
        opinion_id=opinion_id,
        full_text=full_text,
        pages=[ParsedPage(page_num=1, text=full_text, char_count=len(full_text), ocr_used=False)],
        sections=sections,
        parser_version="0.1.0",
    )


def _make_tool_use_response(
    *,
    finding: str = "violation",
    valid: bool = True,
    invalid_issue_tags: bool = False,
) -> MagicMock:
    tool_input = {
        "issue_date": "2026-02-02",
        "complainant": "David Askins",
        "public_agency": "City of Bloomington",
        "agency_type": "city",
        "statutes_invoked": ["IC 5-14-3-4(b)(20)"],
        "statute_areas": ["APRA"],
        "issue_tags": ["denial-of-access"],
        "tags_need_review": False,
        "finding": finding,
        "summary": "The PAC found a violation of APRA by the City of Bloomington.",
    }
    if not valid:
        tool_input["agency_type"] = "invalid_agency"
    if invalid_issue_tags:
        tool_input["issue_tags"] = ["not-a-real-tag"]

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tool_123"
    tool_block.name = "extract_metadata"
    tool_block.input = tool_input

    msg = MagicMock()
    msg.content = [tool_block]
    msg.stop_reason = "tool_use"
    return msg


class TestBuildTextPayload:
    def test_with_sections(self) -> None:
        parsed = _make_parsed(with_sections=True)
        payload = build_text_payload(parsed)
        assert "[BACKGROUND]" in payload
        assert "[ANALYSIS]" in payload
        assert "[CONCLUSION]" in payload
        assert "complainant alleges" in payload.lower()
        assert "apra provides" in payload.lower()

    def test_without_sections_uses_full_text(self) -> None:
        parsed = _make_parsed(with_sections=False)
        payload = build_text_payload(parsed)
        assert "Some opinion text without standard sections" in payload

    def test_includes_caption_block(self) -> None:
        parsed = _make_parsed(with_sections=True)
        payload = build_text_payload(parsed)
        assert "David Askins" in payload
        assert "City of Bloomington" in payload


class TestExtractOne:
    def test_returns_valid_metadata(self) -> None:
        parsed = _make_parsed()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_use_response()

        result = extract_one(
            opinion_id="25-FC-118",
            parsed=parsed,
            counselor_name="Jennifer G. Ruby",
            client=mock_client,
        )

        assert isinstance(result, OpinionMetadata)
        assert result.opinion_id == "25-FC-118"
        assert result.counselor == "Jennifer G. Ruby"
        assert result.finding == "violation"
        assert result.public_agency == "City of Bloomington"
        assert result.issue_date == date(2026, 2, 2)

    def test_uses_counselor_from_argument(self) -> None:
        parsed = _make_parsed()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_use_response()

        result = extract_one(
            opinion_id="25-FC-118",
            parsed=parsed,
            counselor_name="Luke H. Britt",
            client=mock_client,
        )
        assert result.counselor == "Luke H. Britt"

    def test_handles_null_counselor(self) -> None:
        parsed = _make_parsed()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_use_response()

        result = extract_one(
            opinion_id="25-FC-118",
            parsed=parsed,
            counselor_name=None,
            client=mock_client,
        )
        assert result.counselor is None

    def test_retries_on_validation_failure(self) -> None:
        parsed = _make_parsed()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _make_tool_use_response(valid=False),
            _make_tool_use_response(valid=True),
        ]

        result = extract_one(
            opinion_id="25-FC-118",
            parsed=parsed,
            counselor_name="Jennifer G. Ruby",
            client=mock_client,
        )
        assert isinstance(result, OpinionMetadata)
        assert mock_client.messages.create.call_count == 2

    def test_retries_on_invalid_issue_tags(self) -> None:
        parsed = _make_parsed()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            _make_tool_use_response(invalid_issue_tags=True),
            _make_tool_use_response(valid=True),
        ]

        result = extract_one(
            opinion_id="25-FC-118",
            parsed=parsed,
            counselor_name="Jennifer G. Ruby",
            client=mock_client,
        )
        assert isinstance(result, OpinionMetadata)
        assert result.issue_tags == ["denial-of-access"]
        assert mock_client.messages.create.call_count == 2

    def test_raises_after_max_retries(self) -> None:
        parsed = _make_parsed()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_use_response(valid=False)

        with pytest.raises(ValueError, match="extraction failed"):
            extract_one(
                opinion_id="25-FC-118",
                parsed=parsed,
                counselor_name=None,
                client=mock_client,
            )

    def test_handles_no_sections(self) -> None:
        parsed = _make_parsed(with_sections=False)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_tool_use_response()

        result = extract_one(
            opinion_id="25-FC-100",
            parsed=parsed,
            counselor_name=None,
            client=mock_client,
        )
        assert isinstance(result, OpinionMetadata)
        mock_client.messages.create.assert_called_once()


class TestExtractAll:
    def test_processes_eligible_opinions(self, tmp_path: Path) -> None:
        db_path, data_dir = _setup_db_and_files(tmp_path)

        with (
            patch("pac_pipeline.extract.Anthropic") as mock_cls,
            patch("pac_pipeline.extract.settings") as mock_settings,
        ):
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.extraction_model = "claude-haiku-4-5-20251001"
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = _make_tool_use_response()

            extracted, errors = extract_all(
                db_path=db_path,
                data_dir=data_dir,
            )

        assert extracted == 1
        assert errors == 0

        with ExtractDB(db_path) as db:
            counts = db.count_by_extract_status()
            assert counts.get("ok", 0) == 1

    def test_records_errors(self, tmp_path: Path) -> None:
        db_path, data_dir = _setup_db_and_files(tmp_path)

        with (
            patch("pac_pipeline.extract.Anthropic") as mock_cls,
            patch("pac_pipeline.extract.settings") as mock_settings,
        ):
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.extraction_model = "claude-haiku-4-5-20251001"
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.side_effect = RuntimeError("API error")

            extracted, errors = extract_all(
                db_path=db_path,
                data_dir=data_dir,
            )

        assert extracted == 0
        assert errors == 1

        with ExtractDB(db_path) as db:
            counts = db.count_by_extract_status()
            assert counts.get("error", 0) == 1

    def test_idempotent(self, tmp_path: Path) -> None:
        db_path, data_dir = _setup_db_and_files(tmp_path)

        with (
            patch("pac_pipeline.extract.Anthropic") as mock_cls,
            patch("pac_pipeline.extract.settings") as mock_settings,
        ):
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.extraction_model = "claude-haiku-4-5-20251001"
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = _make_tool_use_response()

            extract_all(db_path=db_path, data_dir=data_dir)

        with (
            patch("pac_pipeline.extract.Anthropic") as mock_cls,
            patch("pac_pipeline.extract.settings") as mock_settings,
        ):
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.extraction_model = "claude-haiku-4-5-20251001"
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create.return_value = _make_tool_use_response()

            extracted, errors = extract_all(db_path=db_path, data_dir=data_dir)

        assert extracted == 0
        assert errors == 0


def _setup_db_and_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create a test DB with one parseable opinion and a parsed JSON file on disk."""
    from datetime import UTC, datetime

    import duckdb

    db_path = tmp_path / "test.duckdb"
    data_dir = tmp_path

    now = datetime.now(UTC)
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS opinions (
            opinion_id TEXT PRIMARY KEY,
            year INTEGER NOT NULL,
            kind TEXT NOT NULL,
            sub_letter TEXT,
            revision INTEGER NOT NULL DEFAULT 0,
            parent_opinion_id TEXT,
            raw_title TEXT NOT NULL,
            agency_name TEXT NOT NULL,
            listing_url TEXT NOT NULL,
            pdf_url TEXT NOT NULL,
            pdf_path TEXT,
            pdf_sha256 TEXT,
            first_seen_at TIMESTAMP NOT NULL,
            last_checked_at TIMESTAMP NOT NULL,
            download_status TEXT NOT NULL DEFAULT 'pending'
        )
    """)
    conn.execute(
        """
        INSERT INTO opinions (
            opinion_id, year, kind, raw_title, agency_name,
            listing_url, pdf_url, pdf_path, pdf_sha256,
            first_seen_at, last_checked_at, download_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "25-FC-118",
            2025,
            "FC",
            "25-FC-118: Alleged Violation",
            "City of Bloomington",
            "https://example.com/2025/",
            "https://example.com/25-FC-118.pdf",
            "data/pdfs/2025/25-FC-118.pdf",
            "abc123",
            now,
            now,
            "ok",
        ],
    )
    conn.execute(
        "ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parse_status TEXT DEFAULT 'pending'"
    )
    conn.execute("ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parse_path TEXT")
    conn.execute("ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMP")
    conn.execute("ALTER TABLE opinions ADD COLUMN IF NOT EXISTS parser_version TEXT")

    parsed = _make_parsed("25-FC-118", with_sections=True)
    parsed_dir = data_dir / "parsed" / "2025"
    parsed_dir.mkdir(parents=True)
    parsed_path = parsed_dir / "25-FC-118.json"
    parsed_path.write_text(parsed.model_dump_json(indent=2))

    conn.execute(
        """
        UPDATE opinions
        SET parse_status = 'ok', parse_path = ?, parsed_at = ?, parser_version = '0.1.0'
        WHERE opinion_id = '25-FC-118'
        """,
        [str(parsed_path), now],
    )
    conn.close()

    return db_path, data_dir

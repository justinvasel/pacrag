from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import duckdb
import pytest
from pac_pipeline.litparse import LiteParseError
from pac_pipeline.models import LiteParseOutput, LiteParsePage, ParsedOpinion
from pac_pipeline.parse import parse_all, parse_one

_DEFAULT_TEXT = "ANALYSIS\nSome analysis.\nCONCLUSION\nDone."


def _make_litparse_output(text: str = _DEFAULT_TEXT) -> LiteParseOutput:
    return LiteParseOutput(
        pages=[LiteParsePage(page=1, width=612, height=792, text=text, text_items=[])]
    )


class TestParseOne:
    def test_produces_valid_output(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "pdfs" / "2025" / "25-FC-118.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (_make_litparse_output(), False)
            result, out_path = parse_one("25-FC-118", 2025, pdf_path, tmp_path)

        assert isinstance(result, ParsedOpinion)
        assert result.opinion_id == "25-FC-118"
        assert result.parser_version == "0.1.0"
        assert out_path.exists()

    def test_writes_json_file(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "pdfs" / "2025" / "25-FC-118.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (_make_litparse_output(), False)
            parse_one("25-FC-118", 2025, pdf_path, tmp_path)

        output_path = tmp_path / "parsed" / "2025" / "25-FC-118.json"
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["opinion_id"] == "25-FC-118"
        restored = ParsedOpinion.model_validate(data)
        assert restored.opinion_id == "25-FC-118"

    def test_raises_on_missing_pdf(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "nonexistent.pdf"

        with pytest.raises(FileNotFoundError):
            parse_one("25-FC-118", 2025, pdf_path, tmp_path)

    def test_sections_included(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (_make_litparse_output(), False)
            result, _path = parse_one("25-FC-118", 2025, pdf_path, tmp_path)

        assert "ANALYSIS" in result.sections
        assert "CONCLUSION" in result.sections

    def test_pages_have_correct_char_count(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        text = "x" * 500

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            output = LiteParseOutput(
                pages=[LiteParsePage(page=1, width=612, height=792, text=text, text_items=[])]
            )
            mock_parse.return_value = (output, False)
            result, _path = parse_one("25-FC-118", 2025, pdf_path, tmp_path)

        assert result.pages[0].char_count == 500

    def test_ocr_flag_propagated(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (_make_litparse_output(), True)
            result, _path = parse_one("25-FC-118", 2025, pdf_path, tmp_path)

        assert result.pages[0].ocr_used is True

    def test_multi_page_full_text(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        output = LiteParseOutput(
            pages=[
                LiteParsePage(page=1, width=612, height=792, text="Page one.", text_items=[]),
                LiteParsePage(page=2, width=612, height=792, text="Page two.", text_items=[]),
            ]
        )

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (output, False)
            result, _path = parse_one("25-FC-118", 2025, pdf_path, tmp_path)

        assert "Page one." in result.full_text
        assert "Page two." in result.full_text
        assert len(result.pages) == 2


class TestParseAll:
    def test_parses_eligible_opinions(self, tmp_db_with_opinions: Path, tmp_path: Path) -> None:
        for oid, year in [("25-FC-118", 2025), ("25-FC-002A", 2025)]:
            pdf = Path(f"data/pdfs/{year}/{oid}.pdf")
            (tmp_path / pdf).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / pdf).write_bytes(b"%PDF fake")

        conn = duckdb.connect(str(tmp_db_with_opinions))
        conn.execute(
            "UPDATE opinions SET pdf_path = ? WHERE opinion_id = ?",
            [str(tmp_path / "data/pdfs/2025/25-FC-118.pdf"), "25-FC-118"],
        )
        conn.execute(
            "UPDATE opinions SET pdf_path = ? WHERE opinion_id = ?",
            [str(tmp_path / "data/pdfs/2025/25-FC-002A.pdf"), "25-FC-002A"],
        )
        conn.close()

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (_make_litparse_output(), False)
            parsed, errors = parse_all(tmp_db_with_opinions, tmp_path, year=2025)

        assert parsed == 2
        assert errors == 0

    def test_handles_errors_gracefully(self, tmp_db_with_opinions: Path, tmp_path: Path) -> None:
        pdf = tmp_path / "data/pdfs/2025/25-FC-118.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF fake")

        conn = duckdb.connect(str(tmp_db_with_opinions))
        conn.execute(
            "UPDATE opinions SET pdf_path = ? WHERE opinion_id = ?",
            [str(pdf), "25-FC-118"],
        )
        conn.execute(
            "UPDATE opinions SET pdf_path = ? WHERE opinion_id = ?",
            [str(tmp_path / "data/pdfs/2025/25-FC-002A.pdf"), "25-FC-002A"],
        )
        conn.close()

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.side_effect = [
                (_make_litparse_output(), False),
                LiteParseError("fail"),
            ]
            parsed, errors = parse_all(tmp_db_with_opinions, tmp_path, year=2025)

        assert parsed == 1
        assert errors == 1

    def test_updates_db_on_success(self, tmp_db_with_opinions: Path, tmp_path: Path) -> None:
        pdf = tmp_path / "data/pdfs/2025/25-FC-118.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF fake")

        conn = duckdb.connect(str(tmp_db_with_opinions))
        conn.execute(
            "UPDATE opinions SET pdf_path = ? WHERE opinion_id = ?",
            [str(pdf), "25-FC-118"],
        )
        conn.close()

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (_make_litparse_output(), False)
            parse_all(tmp_db_with_opinions, tmp_path, opinion_id="25-FC-118")

        conn = duckdb.connect(str(tmp_db_with_opinions))
        row = conn.execute(
            "SELECT parse_status, parse_path, parser_version FROM opinions WHERE opinion_id = ?",
            ["25-FC-118"],
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "ok"
        assert row[1] is not None
        assert row[2] is not None

    def test_updates_db_on_error(self, tmp_db_with_opinions: Path, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_db_with_opinions))
        conn.execute(
            "UPDATE opinions SET pdf_path = ? WHERE opinion_id = ?",
            [str(tmp_path / "data/pdfs/2025/25-FC-118.pdf"), "25-FC-118"],
        )
        conn.close()

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.side_effect = LiteParseError("fail")
            parse_all(tmp_db_with_opinions, tmp_path, opinion_id="25-FC-118")

        conn = duckdb.connect(str(tmp_db_with_opinions))
        row = conn.execute(
            "SELECT parse_status FROM opinions WHERE opinion_id = ?",
            ["25-FC-118"],
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "error"

    def test_idempotent(self, tmp_db_with_opinions: Path, tmp_path: Path) -> None:
        pdf = tmp_path / "data/pdfs/2025/25-FC-118.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF fake")

        conn = duckdb.connect(str(tmp_db_with_opinions))
        conn.execute(
            "UPDATE opinions SET pdf_path = ? WHERE opinion_id = ?",
            [str(pdf), "25-FC-118"],
        )
        conn.close()

        with patch("pac_pipeline.parse.parse_pdf") as mock_parse:
            mock_parse.return_value = (_make_litparse_output(), False)
            parsed1, _ = parse_all(tmp_db_with_opinions, tmp_path, opinion_id="25-FC-118")
            parsed2, _ = parse_all(tmp_db_with_opinions, tmp_path, opinion_id="25-FC-118")

        assert parsed1 == 1
        assert parsed2 == 0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pac_pipeline.litparse import (
    LiteParseError,
    _needs_ocr,
    parse_pdf,
    run_litparse,
)
from pac_pipeline.models import LiteParseOutput, LiteParsePage


def _make_litparse_json(pages: list[dict[str, object]]) -> dict[str, object]:
    return {"pages": pages}


def _make_page(page_num: int, text: str, char_count: int | None = None) -> dict[str, object]:
    return {
        "page": page_num,
        "width": 612.0,
        "height": 792.0,
        "text": text,
        "textItems": [],
    }


def _write_litparse_output(output_path: Path, pages: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_make_litparse_json(pages)))


class TestRunLitparse:
    def test_builds_correct_command_no_ocr(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        with patch("pac_pipeline.litparse.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            _write_litparse_output(output_path, [_make_page(1, "Some text")])

            run_litparse(pdf_path, output_path, use_ocr=False)

            cmd = mock_run.call_args[0][0]
            assert "--no-ocr" in cmd
            assert "--quiet" in cmd
            assert "--format" in cmd
            assert "json" in cmd

    def test_builds_correct_command_with_ocr(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        with patch("pac_pipeline.litparse.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            _write_litparse_output(output_path, [_make_page(1, "Some text")])

            run_litparse(pdf_path, output_path, use_ocr=True)

            cmd = mock_run.call_args[0][0]
            assert "--no-ocr" not in cmd

    def test_raises_on_nonzero_exit(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        with patch("pac_pipeline.litparse.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "Some error"

            with pytest.raises(LiteParseError, match="non-zero exit"):
                run_litparse(pdf_path, output_path)

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        with patch("pac_pipeline.litparse.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            output_path.write_text("not valid json {{{")

            with pytest.raises(LiteParseError, match="invalid"):
                run_litparse(pdf_path, output_path)

    def test_parses_valid_output(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        with patch("pac_pipeline.litparse.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            _write_litparse_output(
                output_path,
                [_make_page(1, "Page one"), _make_page(2, "Page two")],
            )

            result = run_litparse(pdf_path, output_path)

        assert isinstance(result, LiteParseOutput)
        assert len(result.pages) == 2
        assert result.pages[0].text == "Page one"

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "nested" / "dir" / "output.json"

        with patch("pac_pipeline.litparse.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            _write_litparse_output(output_path, [_make_page(1, "text")])

            run_litparse(pdf_path, output_path)

        assert output_path.parent.exists()

    def test_raises_on_missing_pdf(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "nonexistent.pdf"
        output_path = tmp_path / "output.json"

        with pytest.raises(FileNotFoundError):
            run_litparse(pdf_path, output_path)

    def test_raises_on_timeout(self, tmp_path: Path) -> None:
        import subprocess

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        with patch("pac_pipeline.litparse.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="lit", timeout=120)

            with pytest.raises(LiteParseError, match="timed out"):
                run_litparse(pdf_path, output_path)


class TestNeedsOcr:
    def test_below_threshold(self) -> None:
        output = LiteParseOutput(
            pages=[LiteParsePage(page=1, width=612, height=792, text="Hi", text_items=[])]
        )
        assert _needs_ocr(output) is True

    def test_above_threshold(self) -> None:
        output = LiteParseOutput(
            pages=[LiteParsePage(page=1, width=612, height=792, text="x" * 500, text_items=[])]
        )
        assert _needs_ocr(output) is False

    def test_empty_page(self) -> None:
        output = LiteParseOutput(
            pages=[LiteParsePage(page=1, width=612, height=792, text="", text_items=[])]
        )
        assert _needs_ocr(output) is True

    def test_minority_sparse_no_ocr(self) -> None:
        output = LiteParseOutput(
            pages=[
                LiteParsePage(page=1, width=612, height=792, text="x" * 500, text_items=[]),
                LiteParsePage(page=2, width=612, height=792, text="x" * 500, text_items=[]),
                LiteParsePage(page=3, width=612, height=792, text="short", text_items=[]),
            ]
        )
        assert _needs_ocr(output) is False

    def test_majority_sparse_triggers_ocr(self) -> None:
        output = LiteParseOutput(
            pages=[
                LiteParsePage(page=1, width=612, height=792, text="x" * 500, text_items=[]),
                LiteParsePage(page=2, width=612, height=792, text="Hi", text_items=[]),
                LiteParsePage(page=3, width=612, height=792, text="", text_items=[]),
            ]
        )
        assert _needs_ocr(output) is True

    def test_empty_pages_list(self) -> None:
        output = LiteParseOutput(pages=[])
        assert _needs_ocr(output) is False


class TestParsePdf:
    def test_no_ocr_when_text_sufficient(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        with patch("pac_pipeline.litparse.run_litparse") as mock_run:
            good_output = LiteParseOutput(
                pages=[LiteParsePage(page=1, width=612, height=792, text="x" * 500, text_items=[])]
            )
            mock_run.return_value = good_output

            _result, ocr_used = parse_pdf(pdf_path, output_path)

        assert ocr_used is False
        assert mock_run.call_count == 1
        assert mock_run.call_args[1]["use_ocr"] is False

    def test_ocr_fallback_when_low_text(self, tmp_path: Path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        output_path = tmp_path / "output.json"

        low_text_output = LiteParseOutput(
            pages=[LiteParsePage(page=1, width=612, height=792, text="Hi", text_items=[])]
        )
        ocr_output = LiteParseOutput(
            pages=[
                LiteParsePage(
                    page=1,
                    width=612,
                    height=792,
                    text="Full OCR text here",
                    text_items=[],
                )
            ]
        )

        with patch("pac_pipeline.litparse.run_litparse") as mock_run:
            mock_run.side_effect = [low_text_output, ocr_output]

            result, ocr_used = parse_pdf(pdf_path, output_path)

        assert ocr_used is True
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0][1]["use_ocr"] is False
        assert mock_run.call_args_list[1][1]["use_ocr"] is True
        assert result == ocr_output

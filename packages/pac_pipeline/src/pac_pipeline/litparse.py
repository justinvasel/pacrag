from __future__ import annotations

import json
import subprocess
from pathlib import Path

import structlog

from pac_pipeline.models import LiteParseOutput

log = structlog.get_logger()

LOW_CHAR_THRESHOLD: int = 50
LITPARSE_TIMEOUT: int = 120


class LiteParseError(Exception):
    def __init__(self, message: str, stderr: str = "") -> None:
        super().__init__(message)
        self.stderr = stderr


def run_litparse(
    pdf_path: Path,
    output_path: Path,
    *,
    use_ocr: bool = False,
    lit_binary: str = "lit",
    dpi: int = 150,
) -> LiteParseOutput:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        lit_binary,
        "parse",
        str(pdf_path),
        "--format",
        "json",
        "-o",
        str(output_path),
        "--quiet",
        "--dpi",
        str(dpi),
    ]
    if not use_ocr:
        cmd.append("--no-ocr")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=LITPARSE_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        raise LiteParseError(f"LiteParse timed out after {LITPARSE_TIMEOUT}s", stderr="") from e

    if result.returncode != 0:
        raise LiteParseError(
            f"LiteParse non-zero exit code {result.returncode}",
            stderr=result.stderr,
        )

    try:
        raw = json.loads(output_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise LiteParseError(f"LiteParse produced invalid output: {e}", stderr="") from e

    try:
        return LiteParseOutput.model_validate(raw)
    except Exception as e:
        raise LiteParseError(f"LiteParse output validation failed: {e}", stderr="") from e


def _needs_ocr(output: LiteParseOutput) -> bool:
    if not output.pages:
        return False
    sparse = sum(1 for p in output.pages if len(p.text.strip()) < LOW_CHAR_THRESHOLD)
    return sparse / len(output.pages) > 0.5


def parse_pdf(
    pdf_path: Path,
    output_path: Path,
    *,
    lit_binary: str = "lit",
) -> tuple[LiteParseOutput, bool]:
    log.debug("litparse_first_pass", pdf=str(pdf_path), ocr=False)
    output = run_litparse(pdf_path, output_path, use_ocr=False, lit_binary=lit_binary)

    if _needs_ocr(output):
        log.info("litparse_ocr_fallback", pdf=str(pdf_path))
        output = run_litparse(pdf_path, output_path, use_ocr=True, lit_binary=lit_binary)
        return output, True

    return output, False

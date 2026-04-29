from __future__ import annotations

from pathlib import Path

import structlog
from tqdm import tqdm

from pac_pipeline.litparse import parse_pdf
from pac_pipeline.models import PARSER_VERSION, ParsedOpinion, ParsedPage
from pac_pipeline.parse_db import ParseDB
from pac_pipeline.section_detector import detect_sections

log = structlog.get_logger()


def parse_one(
    opinion_id: str,
    year: int,
    pdf_path: Path,
    output_dir: Path,
    *,
    lit_binary: str = "lit",
) -> tuple[ParsedOpinion, Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = output_dir / "parsed" / str(year) / f"{opinion_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    litparse_output, ocr_used = parse_pdf(pdf_path, output_path, lit_binary=lit_binary)

    full_text = "\n\n".join(page.text for page in litparse_output.pages)

    pages = [
        ParsedPage(
            page_num=page.page,
            text=page.text,
            char_count=len(page.text),
            ocr_used=ocr_used,
        )
        for page in litparse_output.pages
    ]

    sections = detect_sections(full_text, opinion_id)

    parsed = ParsedOpinion(
        opinion_id=opinion_id,
        full_text=full_text,
        pages=pages,
        sections=sections,
        parser_version=PARSER_VERSION,
    )

    output_path.write_text(parsed.model_dump_json(indent=2))
    log.info(
        "parsed_opinion",
        opinion_id=opinion_id,
        pages=len(pages),
        sections=list(sections.keys()),
        ocr_used=ocr_used,
    )

    return parsed, output_path


def parse_all(
    db_path: Path,
    data_dir: Path,
    *,
    year: int | None = None,
    opinion_id: str | None = None,
    reparse: bool = False,
    lit_binary: str = "lit",
) -> tuple[int, int]:
    with ParseDB(db_path) as db:
        opinions = db.get_parseable_opinions(
            year=year,
            opinion_id=opinion_id,
            reparse=reparse,
            parser_version=PARSER_VERSION,
        )

        if not opinions:
            log.info("no_opinions_to_parse")
            return 0, 0

        log.info("parsing_opinions", count=len(opinions))

        parsed_count = 0
        error_count = 0

        for oid, op_year, pdf_path_str in tqdm(opinions, desc="Parsing"):
            try:
                pdf_path = Path(pdf_path_str)
                _result, out_path = parse_one(
                    oid,
                    op_year,
                    pdf_path,
                    data_dir,
                    lit_binary=lit_binary,
                )
                db.update_parse_status(
                    oid,
                    parse_status="ok",
                    parse_path=str(out_path),
                    parser_version=PARSER_VERSION,
                )
                parsed_count += 1
            except Exception:
                log.exception("parse_failed", opinion_id=oid)
                db.update_parse_status(oid, parse_status="error")
                error_count += 1

        return parsed_count, error_count

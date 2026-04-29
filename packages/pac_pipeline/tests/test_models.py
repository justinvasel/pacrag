from __future__ import annotations

import json

from pac_pipeline.models import (
    PARSER_VERSION,
    LiteParseOutput,
    LiteParsePage,
    LiteParseTextItem,
    ParsedOpinion,
    ParsedPage,
)


class TestLiteParseTextItem:
    def test_from_camel_case_json(self) -> None:
        data = {
            "text": "hello",
            "x": 10.0,
            "y": 20.0,
            "width": 100.0,
            "height": 12.0,
            "fontName": "g_d0_f3",
            "fontSize": 12.0,
            "confidence": 1.0,
        }
        item = LiteParseTextItem.model_validate(data)
        assert item.text == "hello"
        assert item.font_name == "g_d0_f3"
        assert item.font_size == 12.0
        assert item.confidence == 1.0

    def test_frozen(self) -> None:
        item = LiteParseTextItem(
            text="hi",
            x=0,
            y=0,
            width=10,
            height=10,
            font_name="f",
            font_size=10,
            confidence=1,
        )
        try:
            item.text = "new"  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except Exception:
            pass


class TestLiteParsePage:
    def test_from_camel_case_json(self) -> None:
        data = {
            "page": 1,
            "width": 612.0,
            "height": 792.0,
            "text": "Some text",
            "textItems": [
                {
                    "text": "Some",
                    "x": 10.0,
                    "y": 20.0,
                    "width": 50.0,
                    "height": 12.0,
                    "fontName": "g_d0_f3",
                    "fontSize": 12.0,
                    "confidence": 1.0,
                }
            ],
        }
        page = LiteParsePage.model_validate(data)
        assert page.page == 1
        assert page.text == "Some text"
        assert len(page.text_items) == 1
        assert page.text_items[0].font_name == "g_d0_f3"

    def test_empty_text_items(self) -> None:
        data = {
            "page": 1,
            "width": 612.0,
            "height": 792.0,
            "text": "",
            "textItems": [],
        }
        page = LiteParsePage.model_validate(data)
        assert page.text_items == []


class TestLiteParseOutput:
    def test_from_full_json(self) -> None:
        data = {
            "pages": [
                {
                    "page": 1,
                    "width": 612.0,
                    "height": 792.0,
                    "text": "Page one text",
                    "textItems": [],
                },
                {
                    "page": 2,
                    "width": 612.0,
                    "height": 792.0,
                    "text": "Page two text",
                    "textItems": [],
                },
            ]
        }
        output = LiteParseOutput.model_validate(data)
        assert len(output.pages) == 2
        assert output.pages[0].text == "Page one text"

    def test_roundtrip_json(self) -> None:
        data = {
            "pages": [
                {
                    "page": 1,
                    "width": 612.0,
                    "height": 792.0,
                    "text": "hello",
                    "textItems": [
                        {
                            "text": "hello",
                            "x": 0.0,
                            "y": 0.0,
                            "width": 50.0,
                            "height": 12.0,
                            "fontName": "f1",
                            "fontSize": 12.0,
                            "confidence": 0.95,
                        }
                    ],
                }
            ]
        }
        output = LiteParseOutput.model_validate(data)
        dumped = json.loads(output.model_dump_json())
        assert dumped["pages"][0]["text_items"][0]["font_name"] == "f1"


class TestParsedPage:
    def test_valid(self) -> None:
        page = ParsedPage(page_num=1, text="hello", char_count=5, ocr_used=False)
        assert page.page_num == 1
        assert page.char_count == 5
        assert page.ocr_used is False

    def test_frozen(self) -> None:
        page = ParsedPage(page_num=1, text="hello", char_count=5, ocr_used=False)
        try:
            page.text = "new"  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except Exception:
            pass


class TestParsedOpinion:
    def test_valid(self) -> None:
        opinion = ParsedOpinion(
            opinion_id="25-FC-118",
            full_text="ANALYSIS\nSome text\nCONCLUSION\nDone",
            pages=[
                ParsedPage(page_num=1, text="ANALYSIS\nSome text", char_count=18, ocr_used=False),
                ParsedPage(page_num=2, text="CONCLUSION\nDone", char_count=15, ocr_used=False),
            ],
            sections={"ANALYSIS": (9, 19), "CONCLUSION": (30, 34)},
            parser_version="0.1.0",
        )
        assert opinion.opinion_id == "25-FC-118"
        assert len(opinion.pages) == 2
        assert opinion.sections["ANALYSIS"] == (9, 19)

    def test_empty_sections(self) -> None:
        opinion = ParsedOpinion(
            opinion_id="24-INF-01",
            full_text="Some informal text",
            pages=[
                ParsedPage(page_num=1, text="Some informal text", char_count=18, ocr_used=False),
            ],
            sections={},
            parser_version="0.1.0",
        )
        assert opinion.sections == {}

    def test_json_roundtrip(self) -> None:
        opinion = ParsedOpinion(
            opinion_id="25-FC-118",
            full_text="text",
            pages=[ParsedPage(page_num=1, text="text", char_count=4, ocr_used=False)],
            sections={"ANALYSIS": (0, 4)},
            parser_version="0.1.0",
        )
        dumped = json.loads(opinion.model_dump_json())
        restored = ParsedOpinion.model_validate(dumped)
        assert restored == opinion


class TestParserVersion:
    def test_version_is_string(self) -> None:
        assert isinstance(PARSER_VERSION, str)
        assert len(PARSER_VERSION) > 0

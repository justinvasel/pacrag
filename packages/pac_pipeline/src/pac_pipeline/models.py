from __future__ import annotations

from pydantic import AliasGenerator, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

PARSER_VERSION: str = "0.1.0"

_LITPARSE_CONFIG = ConfigDict(
    frozen=True,
    alias_generator=AliasGenerator(validation_alias=to_camel),
    populate_by_name=True,
)


class LiteParseTextItem(BaseModel):
    model_config = _LITPARSE_CONFIG

    text: str
    x: float
    y: float
    width: float
    height: float
    font_name: str
    font_size: float
    confidence: float


class LiteParsePage(BaseModel):
    model_config = _LITPARSE_CONFIG

    page: int
    width: float
    height: float
    text: str
    text_items: list[LiteParseTextItem]


class LiteParseOutput(BaseModel, frozen=True):
    pages: list[LiteParsePage]


class ParsedPage(BaseModel, frozen=True):
    page_num: int
    text: str
    char_count: int
    ocr_used: bool


class ParsedOpinion(BaseModel, frozen=True):
    opinion_id: str
    full_text: str
    pages: list[ParsedPage]
    sections: dict[str, tuple[int, int]]
    parser_version: str

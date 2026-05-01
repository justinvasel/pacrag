from __future__ import annotations

from anthropic import AsyncAnthropic
from fastapi import Request

from pac_api.db import OpinionDB


async def get_db(request: Request) -> OpinionDB:
    return request.app.state.db


async def get_anthropic(request: Request) -> AsyncAnthropic:
    return request.app.state.anthropic_client

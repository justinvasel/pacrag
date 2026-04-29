from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pac_crawler.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RATE_LIMIT_DELAY,
    DEFAULT_RETRY_WAIT_MAX,
    DEFAULT_RETRY_WAIT_MIN,
    DEFAULT_TIMEOUT,
    USER_AGENT,
)

log = structlog.get_logger()


class RateLimiter:
    def __init__(self, delay: float = DEFAULT_RATE_LIMIT_DELAY) -> None:
        self._delay = delay
        self._last_request: float = 0.0

    async def acquire(self) -> None:
        elapsed = time.monotonic() - self._last_request
        wait = self._delay - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request = time.monotonic()


def build_client(
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str = USER_AGENT,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        http2=True,
        timeout=httpx.Timeout(timeout),
        headers={"User-Agent": user_agent},
        follow_redirects=True,
    )


_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
    wait=wait_exponential(min=DEFAULT_RETRY_WAIT_MIN, max=DEFAULT_RETRY_WAIT_MAX),
    reraise=True,
)
async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    limiter: RateLimiter,
) -> str:
    await limiter.acquire()
    log.debug("fetching_page", url=url)
    response = await client.get(url)
    response.raise_for_status()
    return response.text


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(DEFAULT_MAX_RETRIES),
    wait=wait_exponential(min=DEFAULT_RETRY_WAIT_MIN, max=DEFAULT_RETRY_WAIT_MAX),
    reraise=True,
)
async def download_file(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    limiter: RateLimiter,
) -> bytes:
    await limiter.acquire()
    log.debug("downloading_file", url=url, dest=str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with client.stream("GET", url) as response:
        response.raise_for_status()
        chunks: list[bytes] = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)

    data = b"".join(chunks)
    dest.write_bytes(data)
    return data

from __future__ import annotations

BASE_URL = "https://www.in.gov"
ADVISORY_INDEX_URL = f"{BASE_URL}/pac/advisory/"
INFORMAL_INDEX_URL = f"{BASE_URL}/pac/informal/"
INFORMAL_INDEX_FALLBACK = f"{BASE_URL}/pac/advisory/informal-inquiries/"

DEFAULT_RATE_LIMIT_DELAY = 0.75
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_WAIT_MIN = 2.0
DEFAULT_RETRY_WAIT_MAX = 30.0
DEFAULT_TIMEOUT = 30.0

USER_AGENT = "pac-rag-crawler/0.1.0 (Indiana PAC opinion research; github.com/justinvasel/pac-rag)"

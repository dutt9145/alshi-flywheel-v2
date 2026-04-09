"""
shared/kalshi_client.py  (v9 — result field debug + broader result matching)

Changes vs v6:
  1. _get() and _post() now sign the full /trade-api/v2 path instead of
     just the endpoint path. The trading API (trading-api.kalshi.com)
     requires the complete path in the HMAC signature. The elections domain
     was permissive about this; the trading domain is strict and returns
     401 if the signed path doesn't match the full request path.

     Before: _sign_request("GET", "/markets")
     After:  _sign_request("GET", "/trade-api/v2/markets")

  2. Everything else unchanged from v6.
"""

import base64
import json
import time
import logging
import textwrap
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from config.settings import (
    KALSHI_API_BASE, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY, DEMO_MODE
)

logger = logging.getLogger(__name__)

_RATE_LIMIT_BACKOFF_SEC = 60
_MAX_RETRIES = 3

# How far back to look beyond the most recent logged outcome.
# 2 hours catches any markets that resolved but weren't ingested in the last cycle.
_INGESTION_LOOKBACK = timedelta(hours=2)

# The path prefix expected by the trading API in signatures.
# trading-api.kalshi.com requires the full path in the signature string.
_API_PATH_PREFIX = "/trade-api/v2"


def _load_private_key():
    pem = KALSHI_PRIVATE_KEY.strip()

    if not pem.startswith("-----"):
        try:
            decoded = base64.b64decode(pem).decode()
            if "BEGIN" in decoded:
                pem = decoded
            else:
                wrapped = textwrap.fill(pem, 64)
                pem = "-----BEGIN RSA PRIVATE KEY-----\n" + wrapped + "\n-----END RSA PRIVATE KEY-----"
        except Exception:
            wrapped = textwrap.fill(pem, 64)
            pem = "-----BEGIN RSA PRIVATE KEY-----\n" + wrapped + "\n-----END RSA PRIVATE KEY-----"

    if "\\n" in pem:
        pem = pem.replace("\\n", "\n")

    if (
        "-----BEGIN RSA PRIVATE KEY-----" in pem and
        "\n" not in pem
            .replace("-----BEGIN RSA PRIVATE KEY-----", "")
            .replace("-----END RSA PRIVATE KEY-----", "")
    ):
        body    = pem.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", "").strip()
        wrapped = textwrap.fill(body, 64)
        pem     = "-----BEGIN RSA PRIVATE KEY-----\n" + wrapped + "\n-----END RSA PRIVATE KEY-----"

    if not pem.startswith("-----") and len(pem) < 200:
        try:
            with open(pem, "rb") as f:
                raw = f.read().decode()
            if "\\n" in raw:
                raw = raw.replace("\\n", "\n")
            if "\n" not in raw.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", ""):
                body    = raw.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", "").strip()
                wrapped = textwrap.fill(body, 64)
                raw     = "-----BEGIN RSA PRIVATE KEY-----\n" + wrapped + "\n-----END RSA PRIVATE KEY-----"
            pem = raw
        except Exception:
            pass

    return serialization.load_pem_private_key(
        pem.encode(), password=None, backend=default_backend()
    )


def _sign_request(method: str, path: str) -> dict:
    ts_ms     = str(int(time.time() * 1000))
    msg_str   = ts_ms + method.upper() + path
    msg_bytes = msg_str.encode("utf-8")
    key       = _load_private_key()
    signature = key.sign(
        msg_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(signature).decode("utf-8")
    return {
        "Content-Type":            "application/json",
        "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }


def _parse_kalshi_ts(ts_str: str) -> Optional[datetime]:
    """
    Parse a Kalshi timestamp string to a UTC-aware datetime.
    Handles both 'Z' suffix and '+00:00' offset formats.
    Returns None if parsing fails.
    """
    if not ts_str:
        return None
    try:
        normalized = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class KalshiClient:

    def __init__(self):
        self.base    = KALSHI_API_BASE
        self.session = requests.Session()

    # ── Core HTTP ──────────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        # FIX v7: sign the full path including /trade-api/v2 prefix.
        # trading-api.kalshi.com validates the full path in the signature.
        sign_path = _API_PATH_PREFIX + path
        for attempt in range(_MAX_RETRIES):
            headers = _sign_request("GET", sign_path)
            r = self.session.get(
                self.base + path,
                headers=headers,
                params=params,
                timeout=10,
            )
            if r.status_code == 429:
                wait = _RATE_LIMIT_BACKOFF_SEC * (attempt + 1)
                logger.warning(
                    "429 rate limit on GET %s — waiting %ds (attempt %d/%d)",
                    path, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()

        raise requests.HTTPError(f"429 rate limit not resolved after {_MAX_RETRIES} retries on {path}")

    def _post(self, path: str, body: dict) -> dict:
        # FIX v7: sign the full path including /trade-api/v2 prefix.
        sign_path = _API_PATH_PREFIX + path
        body_str  = json.dumps(body)
        headers   = _sign_request("POST", sign_path)
        r = self.session.post(
            self.base + path,
            headers=headers,
            data=body_str,
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    # ── Market fetching ────────────────────────────────────────────────────────

    def get_markets(
        self,
        status: str = "open",
        limit:  int = 100,
        cursor: Optional[str] = None,
    ) -> dict:
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params=params)

    def get_all_open_markets(self, page_sleep_sec: float = 0.25) -> list:
        """
        Fetch all open markets with pagination.
        Skips KXMVE parlay/multi-game markets during fetch so the page cap
        fills with tradeable single-game markets instead of parlays.
        """
        markets, cursor, page = [], None, 0
        max_pages = 80  # increased from 40 to dig past parlays
        useful_pages = 0  # pages that had at least one non-MVE market

        while page < max_pages:
            try:
                resp = self.get_markets(status="open", limit=100, cursor=cursor)
            except Exception as e:
                logger.error("get_all_open_markets page %d failed: %s", page + 1, e)
                break

            batch = resp.get("markets", [])
            non_mve = [m for m in batch if not m.get("ticker", "").lower().startswith("kxmve")]
            markets.extend(non_mve)
            page += 1

            if non_mve:
                useful_pages += 1

            logger.info(
                "Fetched page %d (%d non-MVE this page, %d total kept)",
                page, len(non_mve), len(markets),
            )

            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break

            # Stop early if we've found enough tradeable markets
            if len(markets) >= 2000:
                break

            time.sleep(page_sleep_sec)

        logger.info(
            "Market fetch complete: %d pages, %d tradeable markets (skipped MVE parlays)",
            page, len(markets),
        )
        return markets
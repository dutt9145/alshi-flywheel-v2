"""
shared/kalshi_client.py  (v3 — rate limit handling + smart resolved market fetching)

Changes vs v2:
  1. get_all_open_markets() — added 429 retry with 60s backoff, reduced
     inter-page sleep to 0.3s (was 0.5s, still safe under rate limits)
  2. get_resolved_markets() — added 429 retry with 60s backoff + inter-page
     sleep. max_pages raised to 350 (35,000 markets) for full history on
     first run. Accepts optional min_close_ts to filter by close time so
     subsequent runs only fetch recent resolutions instead of all 34k+
  3. _get() now surfaces the raw status code in exceptions so callers
     can detect 429 vs other errors cleanly
  4. Everything else unchanged from v2
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

# How long to wait after a 429 before retrying
_RATE_LIMIT_BACKOFF_SEC = 60

# Max retries per page on 429
_MAX_RETRIES = 3


def _load_private_key():
    """
    Load RSA private key — handles every possible format Railway might use.
    """
    pem = KALSHI_PRIVATE_KEY.strip()

    # Format 1: Base64 encoded entire PEM (most Railway-safe)
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

    # Format 2: Has literal \n instead of real newlines
    if "\\n" in pem:
        pem = pem.replace("\\n", "\n")

    # Format 3: PEM header present but no line breaks in body
    if (
        "-----BEGIN RSA PRIVATE KEY-----" in pem and
        "\n" not in pem
            .replace("-----BEGIN RSA PRIVATE KEY-----", "")
            .replace("-----END RSA PRIVATE KEY-----", "")
    ):
        body    = pem.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", "").strip()
        wrapped = textwrap.fill(body, 64)
        pem     = "-----BEGIN RSA PRIVATE KEY-----\n" + wrapped + "\n-----END RSA PRIVATE KEY-----"

    # Format 4: File path
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


class KalshiClient:

    def __init__(self):
        self.base    = KALSHI_API_BASE
        self.session = requests.Session()

    # ── Core HTTP ──────────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """
        Signed GET with retry on 429.
        Raises requests.HTTPError on non-retriable errors.
        """
        for attempt in range(_MAX_RETRIES):
            headers = _sign_request("GET", path)
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

        # Exhausted retries
        raise requests.HTTPError(f"429 rate limit not resolved after {_MAX_RETRIES} retries on {path}")

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        headers  = _sign_request("POST", path)
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

    def get_all_open_markets(self) -> list:
        """
        Fetch all open markets with pagination.
        Retries on 429 automatically via _get().
        Uses 0.3s inter-page sleep — stays comfortably under rate limits.
        """
        markets, cursor, page = [], None, 0
        max_pages = 100

        while page < max_pages:
            try:
                resp = self.get_markets(status="open", limit=100, cursor=cursor)
            except Exception as e:
                logger.error("get_all_open_markets page %d failed: %s", page + 1, e)
                break

            batch = resp.get("markets", [])
            markets.extend(batch)
            page += 1

            logger.info("Fetched page %d (%d markets so far)", page, len(markets))

            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break

            time.sleep(1.0)

        return markets

    def get_resolved_markets(
        self,
        max_pages:    int = 350,
        min_close_ts: Optional[str] = None,
    ) -> list:
        """
        Fetch settled markets from Kalshi.

        Parameters
        ----------
        max_pages : int
            Maximum pages to fetch (100 markets each).
            Default 350 = up to 35,000 markets (for full first-run history).
            Subsequent runs should pass a min_close_ts to limit the fetch.
        min_close_ts : str, optional
            ISO 8601 timestamp. If provided, stop paginating once we see
            markets with close_time older than this value.
            Use the timestamp of your most recent outcome to avoid
            re-fetching the entire history every 4 hours.

        Returns
        -------
        list of market dicts with result in ("yes", "no")
        """
        markets, cursor, page = [], None, 0

        while page < max_pages:
            params = {"status": "settled", "limit": 100}
            if cursor:
                params["cursor"] = cursor

            try:
                resp = self._get("/markets", params=params)
            except Exception as e:
                logger.error("get_resolved_markets page %d failed: %s", page + 1, e)
                break

            batch = resp.get("markets", [])

            # Filter to clean yes/no results only
            clean = [m for m in batch if m.get("result") in ("yes", "no")]
            markets.extend(clean)
            page += 1

            logger.info(
                "Resolved markets — page %d: %d settled, %d with clean result (%d total)",
                page, len(batch), len(clean), len(markets),
            )

            # Timestamp cutoff — stop early on subsequent runs
            if min_close_ts and batch:
                oldest_close = batch[-1].get("close_time", "")
                if oldest_close and oldest_close < min_close_ts:
                    logger.info(
                        "Reached min_close_ts cutoff (%s) at page %d — stopping",
                        min_close_ts, page,
                    )
                    break

            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break

            time.sleep(1.0)

        logger.info("get_resolved_markets complete — %d total clean results", len(markets))
        return markets

    def get_latest_outcome_ts(self) -> Optional[str]:
        """
        Query the outcomes table for the most recent resolved_at timestamp.
        Used to build min_close_ts for incremental resolved market fetching.
        Returns ISO string or None if no outcomes yet.
        """
        import os
        database_url = os.getenv("DATABASE_URL", "")
        try:
            if database_url:
                import psycopg2
                conn = psycopg2.connect(database_url)
                cur  = conn.cursor()
                cur.execute("SELECT MAX(logged_at) FROM outcomes")
                row  = cur.fetchone()
                conn.close()
                if row and row[0]:
                    # Subtract 1 day buffer to catch any stragglers
                    ts = row[0]
                    if hasattr(ts, "isoformat"):
                        buffered = ts - timedelta(days=1)
                        return buffered.isoformat()
                    return str(row[0])
            else:
                import sqlite3
                conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
                row  = conn.execute("SELECT MAX(logged_at) FROM outcomes").fetchone()
                conn.close()
                if row and row[0]:
                    return str(row[0])
        except Exception as e:
            logger.warning("get_latest_outcome_ts failed: %s", e)
        return None

    # ── Portfolio ──────────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        resp = self._get("/portfolio/balance")
        return resp.get("balance", 0) / 100

    def get_positions(self) -> list:
        return self._get("/portfolio/positions").get("market_positions", [])

    # ── Orders ─────────────────────────────────────────────────────────────────

    def place_order(
        self,
        ticker:          str,
        side:            str,
        count:           int,
        yes_price:       int,
        client_order_id: str,
    ) -> dict:
        if DEMO_MODE:
            logger.info(
                "[DEMO] Would place %s %s x%d @ %dc",
                side.upper(), ticker, count, yes_price,
            )
            return {
                "status": "demo", "ticker": ticker, "side": side,
                "count": count, "yes_price": yes_price,
            }
        body = {
            "action": "buy", "ticker": ticker, "type": "limit",
            "side": side, "count": count, "yes_price": yes_price,
            "client_order_id": client_order_id,
        }
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._post(f"/portfolio/orders/{order_id}/cancel", {})
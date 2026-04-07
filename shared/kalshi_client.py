"""
shared/kalshi_client.py  (v5 — incremental ingestion timestamp fix)

Changes vs v4:
  1. get_latest_outcome_ts() now returns a timezone-aware datetime object
     instead of a raw string. Previously returned psycopg2's default string
     format ("2026-04-06 05:30:00+00:00" with space separator) which failed
     string comparison against Kalshi's close_time format
     ("2026-04-06T21:00:00Z" with T separator). Space (ASCII 32) < T (ASCII 84)
     so the cutoff check never triggered and every ingestion cycle fetched
     all 35,000 resolved markets from scratch.

  2. get_resolved_markets() now parses close_time to datetime before comparing
     against min_close_ts. Both sides normalized to UTC-aware datetime objects,
     eliminating format-dependent string comparison entirely.

  3. Lookback buffer reduced from 1 day to 2 hours. The 1-day buffer caused
     ~2,400 already-processed markets to be re-fetched every 4-hour ingestion
     cycle even when the timestamp format was correct.

  4. Everything else unchanged from v4.
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
        # Replace Z with +00:00 for fromisoformat compatibility
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

    def get_all_open_markets(self, page_sleep_sec: float = 0.25) -> list:
        """
        Fetch all open markets with pagination.

        Parameters
        ----------
        page_sleep_sec : float
            Sleep between pages to stay under Kalshi's rate limit.
            Default 0.25s. Orchestrator passes SCAN_PAGE_SLEEP_SEC so this
            can be tuned centrally without touching the client.
        """
        markets, cursor, page = [], None, 0
        max_pages = 40

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

            time.sleep(page_sleep_sec)

        return markets

    def get_resolved_markets(
        self,
        max_pages:    int = 350,
        min_close_ts: Optional[datetime] = None,
    ) -> list:
        """
        Fetch settled markets from Kalshi.

        Parameters
        ----------
        max_pages : int
            Maximum pages to fetch (100 markets each).
            Default 350 = up to 35,000 markets (for full first-run history).
        min_close_ts : datetime, optional
            UTC-aware datetime cutoff. Stop paginating once markets are older
            than this value. Pass the result of get_latest_outcome_ts() for
            incremental ingestion — avoids re-fetching full history every cycle.
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
            clean = [m for m in batch if m.get("result") in ("yes", "no")]
            markets.extend(clean)
            page += 1

            logger.info(
                "Resolved markets — page %d: %d settled, %d with clean result (%d total)",
                page, len(batch), len(clean), len(markets),
            )

            # ── FIX: parse both sides to datetime before comparing ─────────
            # Previously compared raw strings — psycopg2 uses space separator
            # ("2026-04-06 05:30:00+00:00") while Kalshi uses T separator
            # ("2026-04-06T21:00:00Z"). Space (ASCII 32) < T (ASCII 84) so the
            # cutoff never triggered and every cycle fetched all 35k markets.
            if min_close_ts and batch:
                oldest_close_str = batch[-1].get("close_time", "")
                oldest_close_dt  = _parse_kalshi_ts(oldest_close_str)
                if oldest_close_dt and oldest_close_dt < min_close_ts:
                    logger.info(
                        "Reached min_close_ts cutoff (%s) at page %d — stopping incremental ingestion",
                        min_close_ts.isoformat(), page,
                    )
                    break

            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break

            time.sleep(1.0)

        logger.info("get_resolved_markets complete — %d total clean results", len(markets))
        return markets

    def get_latest_outcome_ts(self) -> Optional[datetime]:
        """
        Query outcomes table for most recent logged_at timestamp.
        Returns a UTC-aware datetime with a 2-hour lookback buffer,
        or None if no outcomes exist yet (triggers full history fetch).

        Used by orchestrator._ingest_resolved_markets() to build
        min_close_ts for incremental resolved market fetching.
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
                    ts = row[0]
                    # psycopg2 returns a datetime object for timestamptz columns
                    if isinstance(ts, datetime):
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        return ts - _INGESTION_LOOKBACK
                    # Fallback: parse string representation
                    parsed = _parse_kalshi_ts(str(ts))
                    if parsed:
                        return parsed - _INGESTION_LOOKBACK
            else:
                import sqlite3
                conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
                row  = conn.execute("SELECT MAX(logged_at) FROM outcomes").fetchone()
                conn.close()
                if row and row[0]:
                    parsed = _parse_kalshi_ts(str(row[0]))
                    if parsed:
                        return parsed - _INGESTION_LOOKBACK
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
"""
shared/kalshi_client.py  (v2 — get_resolved_markets added)

Changes vs v1:
  1. get_resolved_markets() fetches settled markets from Kalshi API
     Paginates up to 1000 markets (10 pages of 100) per nightly retrain
     Returns only markets with a definitive "yes" or "no" result
  2. Everything else unchanged
"""

import base64
import json
import time
import logging
import textwrap
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

from config.settings import (
    KALSHI_API_BASE, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY, DEMO_MODE
)

logger = logging.getLogger(__name__)


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
    if "-----BEGIN RSA PRIVATE KEY-----" in pem and "\n" not in pem.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", ""):
        body = pem.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", "").strip()
        wrapped = textwrap.fill(body, 64)
        pem = "-----BEGIN RSA PRIVATE KEY-----\n" + wrapped + "\n-----END RSA PRIVATE KEY-----"

    # Format 4: File path
    if not pem.startswith("-----") and len(pem) < 200:
        try:
            with open(pem, "rb") as f:
                raw = f.read().decode()
            if "\\n" in raw:
                raw = raw.replace("\\n", "\n")
            if "\n" not in raw.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", ""):
                body = raw.replace("-----BEGIN RSA PRIVATE KEY-----", "").replace("-----END RSA PRIVATE KEY-----", "").strip()
                wrapped = textwrap.fill(body, 64)
                raw = "-----BEGIN RSA PRIVATE KEY-----\n" + wrapped + "\n-----END RSA PRIVATE KEY-----"
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
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
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

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        headers = _sign_request("GET", path)
        r = self.session.get(self.base + path, headers=headers,
                             params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        headers  = _sign_request("POST", path)
        r = self.session.post(self.base + path, headers=headers,
                              data=body_str, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_markets(self, status: str = "open", limit: int = 100,
                    cursor: Optional[str] = None) -> dict:
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params=params)

    def get_all_open_markets(self) -> list:
        markets, cursor, page = [], None, 0
        max_pages = 100
        while page < max_pages:
            resp   = self.get_markets(status="open", limit=100, cursor=cursor)
            batch  = resp.get("markets", [])
            markets.extend(batch)
            page  += 1
            logger.info("Fetched page %d (%d markets so far)", page, len(markets))
            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break
            time.sleep(0.5)
        return markets

    def get_resolved_markets(self, max_pages: int = 10) -> list:
        """
        Fetch recently settled markets from Kalshi.

        Returns a list of market dicts that have a definitive result
        of "yes" or "no". Paginates up to max_pages * 100 markets.

        Kalshi settled markets endpoint:
            GET /markets?status=settled&limit=100
        The result field on each market is "yes", "no", or None (voided).
        """
        markets, cursor, page = [], None, 0

        while page < max_pages:
            params = {"status": "settled", "limit": 100}
            if cursor:
                params["cursor"] = cursor

            try:
                resp  = self._get("/markets", params=params)
            except Exception as e:
                logger.error("get_resolved_markets page %d failed: %s", page, e)
                break

            batch = resp.get("markets", [])

            # Keep only markets with a clean yes/no result
            clean = [
                m for m in batch
                if m.get("result") in ("yes", "no")
            ]
            markets.extend(clean)
            page += 1

            logger.info(
                "Resolved markets — page %d: %d settled, %d with clean result (%d total)",
                page, len(batch), len(clean), len(markets)
            )

            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break

            time.sleep(0.5)   # respect rate limits between pages

        logger.info("get_resolved_markets complete — %d total clean results", len(markets))
        return markets

    def get_balance(self) -> float:
        resp = self._get("/portfolio/balance")
        return resp.get("balance", 0) / 100

    def get_positions(self) -> list:
        return self._get("/portfolio/positions").get("market_positions", [])

    def place_order(self, ticker: str, side: str, count: int,
                    yes_price: int, client_order_id: str) -> dict:
        if DEMO_MODE:
            logger.info("[DEMO] Would place %s %s x%d @ %dc",
                        side.upper(), ticker, count, yes_price)
            return {"status": "demo", "ticker": ticker, "side": side,
                    "count": count, "yes_price": yes_price}
        body = {
            "action": "buy", "ticker": ticker, "type": "limit",
            "side": side, "count": count, "yes_price": yes_price,
            "client_order_id": client_order_id,
        }
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._post(f"/portfolio/orders/{order_id}/cancel", {})
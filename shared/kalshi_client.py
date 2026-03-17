"""
shared/kalshi_client.py
Authenticated Kalshi REST client with Railway-compatible private key loading.
"""

import base64
import json
import time
import logging
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
    Load RSA private key.
    Handles three formats:
      1. Proper PEM with real newlines (local .env)
      2. PEM with literal \\n (Railway environment variables)
      3. File path to a PEM file
    """
    pem = KALSHI_PRIVATE_KEY.strip()

    # Railway stores multiline env vars with literal \n — fix it
    if "\\n" in pem:
        pem = pem.replace("\\n", "\n")

    if pem.startswith("-----BEGIN"):
        pem_bytes = pem.encode()
    else:
        # Treat as file path
        with open(pem, "rb") as f:
            pem_bytes = f.read()

    return serialization.load_pem_private_key(
        pem_bytes, password=None, backend=default_backend()
    )


def _sign_request(method: str, path: str, body: str = "") -> dict:
    """Generate Kalshi RSA-PSS signed headers."""
    ts_ms     = str(int(time.time() * 1000))
    msg_str   = ts_ms + method.upper() + path + body
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
        "Content-Type":             "application/json",
        "KALSHI-ACCESS-KEY":        KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP":  ts_ms,
        "KALSHI-ACCESS-SIGNATURE":  sig_b64,
    }


class KalshiClient:
    """Thin wrapper around the Kalshi V2 trading API."""

    def __init__(self):
        self.base    = KALSHI_API_BASE
        self.session = requests.Session()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        headers = _sign_request("GET", path)
        r = self.session.get(
            self.base + path, headers=headers,
            params=params, timeout=10
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        headers  = _sign_request("POST", path, body_str)
        r = self.session.post(
            self.base + path, headers=headers,
            data=body_str, timeout=10
        )
        r.raise_for_status()
        return r.json()

    def get_markets(self, status: str = "open", limit: int = 200,
                    cursor: Optional[str] = None) -> dict:
        params = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        return self._get(f"/markets/{ticker}")

    def get_all_open_markets(self) -> list:
        markets, cursor = [], None
        while True:
            resp   = self.get_markets(status="open", limit=200, cursor=cursor)
            batch  = resp.get("markets", [])
            markets.extend(batch)
            cursor = resp.get("cursor")
            if not cursor or len(batch) < 200:
                break
        return markets

    def get_balance(self) -> float:
        resp = self._get("/portfolio/balance")
        return resp.get("balance", 0) / 100

    def get_positions(self) -> list:
        return self._get("/portfolio/positions").get("market_positions", [])

    def place_order(self, ticker: str, side: str, count: int,
                    yes_price: int, client_order_id: str) -> dict:
        if DEMO_MODE:
            logger.info(
                "[DEMO] Would place %s %s x%d @ %dc",
                side.upper(), ticker, count, yes_price
            )
            return {"status": "demo", "ticker": ticker, "side": side,
                    "count": count, "yes_price": yes_price}

        body = {
            "action":          "buy",
            "ticker":          ticker,
            "type":            "limit",
            "side":            side,
            "count":           count,
            "yes_price":       yes_price,
            "client_order_id": client_order_id,
        }
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._post(f"/portfolio/orders/{order_id}/cancel", {})
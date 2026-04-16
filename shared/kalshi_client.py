"""
shared/kalshi_client.py  (v14 — resolution debug logging)

Changes vs v13:
  1. Added DEBUG logging before the yes/no filter to catch markets with
     non-standard result values (weather, etc.)
  2. Log sample of ALL resolved tickers to see format differences.
  3. Log any markets with result values other than yes/no.

Changes vs v12:
  1. Added get_historical_markets() to fetch archived settled markets.
  2. get_resolved_markets() now also queries historical endpoint and merges
     results, ensuring old weather/sports/etc. markets get resolved.
  3. Added get_historical_cutoff() to check when markets are archived.
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
_INGESTION_LOOKBACK = timedelta(hours=2)

# The path prefix expected by the trading API in signatures.
_API_PATH_PREFIX = "/trade-api/v2"

# Weather ticker prefixes for debug logging
_WEATHER_PREFIXES = ("KXTEMP", "KXLOWT", "KXHIGH", "KXWTHR", "KXRAIN", "KXSNOW")

# Sports ticker prefixes for debug logging
_SPORTS_PREFIXES = ("KXMLB", "KXNBA", "KXNFL", "KXNHL")


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
        exclude_mve: bool = True,
    ) -> dict:
        """
        Fetch markets from Kalshi API.
        """
        params = {"status": status, "limit": limit}
        if exclude_mve:
            params["mve_filter"] = "exclude"
        if cursor:
            params["cursor"] = cursor
        return self._get("/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """
        Fetch a single market by ticker.
        """
        try:
            resp = self._get(f"/markets/{ticker}")
            return resp.get("market", {})
        except Exception as e:
            logger.error("get_market(%s) failed: %s", ticker, e)
            return {}

    def get_all_open_markets(self, page_sleep_sec: float = 0.25) -> list:
        """
        Fetch all open single-game markets with pagination.
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

    # ── Historical markets (archived) ──────────────────────────────────────────

    def get_historical_cutoff(self) -> Optional[datetime]:
        """
        Get the cutoff timestamp for historical data.
        Markets settled before this are in the historical archive.
        """
        try:
            resp = self._get("/historical/cutoff-timestamps")
            cutoffs = resp.get("cutoff_timestamps", {})
            markets_cutoff = cutoffs.get("markets")
            if markets_cutoff:
                return _parse_kalshi_ts(markets_cutoff)
        except Exception as e:
            logger.warning("get_historical_cutoff failed: %s", e)
        return None

    def get_historical_markets(
        self,
        tickers: Optional[list[str]] = None,
        min_close_ts: Optional[datetime] = None,
        max_pages: int = 50,
    ) -> list:
        """
        Fetch archived/historical markets from Kalshi.
        """
        markets, cursor, page = [], None, 0

        while page < max_pages:
            params = {"limit": 100}
            if tickers:
                params["tickers"] = ",".join(tickers[:100])
            if min_close_ts:
                params["min_close_ts"] = int(min_close_ts.timestamp())
            if cursor:
                params["cursor"] = cursor

            try:
                resp = self._get("/historical/markets", params=params)
            except Exception as e:
                logger.error("get_historical_markets page %d failed: %s", page + 1, e)
                break

            batch = resp.get("markets", [])
            
            # v14: DEBUG — log ALL result values before filtering
            if page == 0 and batch:
                unique_results = list({str(m.get("result", "NONE")) for m in batch[:50]})
                logger.info("DEBUG historical result values: %s", unique_results)
            
            # v14: DEBUG — log weather/sports markets BEFORE yes/no filter
            weather_raw = [m for m in batch if m.get("ticker", "").upper().startswith(_WEATHER_PREFIXES)]
            if weather_raw:
                logger.info(
                    "DEBUG HISTORICAL weather RAW (before filter): %s",
                    [(m.get('ticker','')[:45], m.get('result'), m.get('status')) for m in weather_raw[:5]]
                )
            
            sports_raw = [m for m in batch if m.get("ticker", "").upper().startswith(_SPORTS_PREFIXES)]
            if sports_raw and page == 0:
                logger.info(
                    "DEBUG HISTORICAL sports RAW sample: %s",
                    [(m.get('ticker','')[:45], m.get('result')) for m in sports_raw[:5]]
                )

            # Filter to those with clean results
            clean = [m for m in batch if str(m.get("result", "")).lower() in ("yes", "no")]
            
            # v14: Log markets that were FILTERED OUT (non yes/no results)
            filtered_out = [m for m in batch if str(m.get("result", "")).lower() not in ("yes", "no")]
            if filtered_out:
                non_yesno_results = list({str(m.get("result", "NONE")) for m in filtered_out})
                logger.info(
                    "DEBUG HISTORICAL filtered out %d markets with results: %s",
                    len(filtered_out), non_yesno_results[:10]
                )
            
            markets.extend(clean)
            page += 1

            # Log weather tickers found
            weather_in_batch = [
                m.get("ticker", "") for m in clean
                if m.get("ticker", "").upper().startswith(_WEATHER_PREFIXES)
            ]
            if weather_in_batch:
                logger.info(
                    "Historical markets — page %d: %d weather tickers: %s",
                    page, len(weather_in_batch), weather_in_batch[:5],
                )

            logger.info(
                "Historical markets — page %d: %d total, %d with result (%d cumulative)",
                page, len(batch), len(clean), len(markets),
            )

            if tickers:
                break

            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break

            time.sleep(0.5)

        logger.info("get_historical_markets complete — %d total results", len(markets))
        return markets

    def get_historical_by_tickers(self, tickers: list[str]) -> list:
        """
        Fetch specific tickers from historical archive.
        Batches requests to handle large ticker lists.
        """
        all_markets = []
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            try:
                params = {"tickers": ",".join(batch_tickers)}
                resp = self._get("/historical/markets", params=params)
                batch = resp.get("markets", [])
                clean = [m for m in batch if str(m.get("result", "")).lower() in ("yes", "no")]
                all_markets.extend(clean)
                logger.info(
                    "Historical batch %d-%d: %d/%d tickers resolved",
                    i, i + len(batch_tickers), len(clean), len(batch_tickers),
                )
            except Exception as e:
                logger.error("get_historical_by_tickers batch %d failed: %s", i, e)

            time.sleep(0.25)

        return all_markets

    # ── Resolution ingestion ───────────────────────────────────────────────────

    def get_resolved_markets(
        self,
        max_pages:      int = 350,
        min_settled_ts: Optional[datetime] = None,
        include_historical: bool = True,
    ) -> list:
        """
        Fetch settled markets from Kalshi — both recent and historical.
        """
        markets, cursor, page = [], None, 0
        seen_tickers = set()

        # ── Phase 1: Regular /markets?status=settled ──────────────────────────
        while page < max_pages:
            params = {"status": "settled", "limit": 100}
            if min_settled_ts:
                params["min_settled_ts"] = int(min_settled_ts.timestamp())
            if cursor:
                params["cursor"] = cursor

            try:
                resp = self._get("/markets", params=params)
            except Exception as e:
                logger.error("get_resolved_markets page %d failed: %s", page + 1, e)
                break

            batch = resp.get("markets", [])

            # v14: DEBUG — log sample of ALL tickers and results on first page
            if page == 0 and batch:
                sample_tickers = [(m.get("ticker", "")[:40], m.get("result")) for m in batch[:10]]
                logger.info("DEBUG resolved ticker+result sample: %s", sample_tickers)
                
                unique_results = list({str(m.get("result", "NONE")) for m in batch})
                logger.info("DEBUG all unique result values in batch: %s", unique_results)

            # v14: DEBUG — log weather markets BEFORE yes/no filter
            weather_raw = [m for m in batch if m.get("ticker", "").upper().startswith(_WEATHER_PREFIXES)]
            if weather_raw:
                logger.info(
                    "DEBUG weather RAW (before filter) page %d: %s",
                    page + 1,
                    [(m.get('ticker','')[:45], m.get('result'), m.get('status')) for m in weather_raw[:5]]
                )
            
            # v14: DEBUG — log sports markets BEFORE yes/no filter (first page only)
            sports_raw = [m for m in batch if m.get("ticker", "").upper().startswith(_SPORTS_PREFIXES)]
            if sports_raw and page == 0:
                logger.info(
                    "DEBUG sports RAW sample: %s",
                    [(m.get('ticker','')[:45], m.get('result')) for m in sports_raw[:5]]
                )

            clean = [m for m in batch if str(m.get("result", "")).lower() in ("yes", "no")]
            
            # v14: Log what got filtered out
            filtered_out = [m for m in batch if str(m.get("result", "")).lower() not in ("yes", "no")]
            if filtered_out and page == 0:
                non_yesno = [(m.get("ticker", "")[:40], m.get("result")) for m in filtered_out[:10]]
                logger.info("DEBUG filtered out (non yes/no): %s", non_yesno)

            markets.extend(clean)
            for m in clean:
                seen_tickers.add(m.get("ticker", ""))
            page += 1

            # Debug log weather tickers
            weather_in_batch = [
                m.get("ticker", "") for m in clean
                if m.get("ticker", "").upper().startswith(_WEATHER_PREFIXES)
            ]
            if weather_in_batch:
                logger.info(
                    "Resolved markets — page %d: %d weather tickers found: %s",
                    page, len(weather_in_batch), weather_in_batch[:5],
                )

            logger.info(
                "Resolved markets — page %d: %d settled, %d with clean result (%d total)",
                page, len(batch), len(clean), len(markets),
            )

            cursor = resp.get("cursor")
            if not cursor or len(batch) < 100:
                break

            time.sleep(1.0)

        # Summary for phase 1
        all_weather = [
            m.get("ticker", "") for m in markets
            if m.get("ticker", "").upper().startswith(_WEATHER_PREFIXES)
        ]
        all_sports = [
            m.get("ticker", "") for m in markets
            if m.get("ticker", "").upper().startswith(_SPORTS_PREFIXES)
        ]
        logger.info(
            "get_resolved_markets phase 1 complete — %d results, %d weather, %d sports",
            len(markets), len(all_weather), len(all_sports),
        )
        if all_weather:
            logger.info("Phase 1 weather sample: %s", all_weather[:5])
        if all_sports:
            logger.info("Phase 1 sports sample: %s", all_sports[:5])

        # ── Phase 2: Historical /historical/markets ───────────────────────────
        if include_historical:
            logger.info("Checking historical archive for older settlements...")
            try:
                historical_cutoff = None
                if min_settled_ts:
                    historical_cutoff = min_settled_ts - timedelta(days=30)

                historical = self.get_historical_markets(
                    min_close_ts=historical_cutoff,
                    max_pages=20,
                )

                new_from_historical = 0
                for m in historical:
                    ticker = m.get("ticker", "")
                    if ticker and ticker not in seen_tickers:
                        markets.append(m)
                        seen_tickers.add(ticker)
                        new_from_historical += 1

                historical_weather = [
                    m.get("ticker", "") for m in historical
                    if m.get("ticker", "").upper().startswith(_WEATHER_PREFIXES)
                ]
                historical_sports = [
                    m.get("ticker", "") for m in historical
                    if m.get("ticker", "").upper().startswith(_SPORTS_PREFIXES)
                ]
                logger.info(
                    "Historical phase: %d total, %d new, %d weather, %d sports",
                    len(historical), new_from_historical,
                    len(historical_weather), len(historical_sports),
                )
                if historical_weather:
                    logger.info("Historical weather sample: %s", historical_weather[:5])

            except Exception as e:
                logger.error("Historical markets fetch failed: %s", e)

        logger.info(
            "get_resolved_markets complete — %d total combined results",
            len(markets),
        )
        return markets

    def get_latest_outcome_ts(self) -> Optional[datetime]:
        """
        Query outcomes table for most recent logged_at timestamp.
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
                    if isinstance(ts, datetime):
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        return ts - _INGESTION_LOOKBACK
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
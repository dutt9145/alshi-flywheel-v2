"""
shared/arb_layer.py  (v3 — near-miss logging + orchestrator call guard + sports prop handling)

Changes vs v2:
  1. log_arb_opportunity() now logs near-misses (spread > 50% of threshold) so
     the dashboard panel shows activity even when full arb doesn't fire
  2. no_polymarket_match and empty_query results are silently skipped (no DB noise)
  3. check_and_log() convenience method added — single call handles both check()
     and log_arb_opportunity() so orchestrator can't accidentally split them
  4. Sports/esports market detection added — logs a debug warning instead of
     burning a Polymarket query that will never match
  5. ARB_MIN_SPREAD_PCT passthrough fix — near-miss threshold derived correctly
  6. Mode C stub added for future OddsPapi/news-signal arb on sports props
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config.settings import (
    ODDSPAPI_KEY, POLYMARKET_GAMMA_URL, ARB_MIN_SPREAD_PCT
)

logger = logging.getLogger(__name__)

# ── Sports/esports keywords — Polymarket won't have these ────────────────────
_SPORTS_PROP_KEYWORDS = {
    "hits", "strikeouts", "rbis", "home", "bases", "innings",
    "points", "assists", "rebounds", "yards", "touchdowns",
    "kills", "valorant", "csgo", "league", "esports", "mlb",
    "nba", "nfl", "nhl", "pitcher", "batter", "over", "under",
}

def _is_sports_prop(market_title: str) -> bool:
    words = set(market_title.lower().split())
    return bool(words & _SPORTS_PROP_KEYWORDS)


# ── Simple TTL cache ──────────────────────────────────────────────────────────
_cache: dict = {}


def _get(url, params=None, headers=None, ttl=120):
    key = url + str(sorted((params or {}).items()))
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"] < ttl):
        return entry["data"]
    try:
        r = requests.get(
            url,
            params=params or {},
            headers=headers or {},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        _cache[key] = {"ts": time.time(), "data": data}
        return data
    except Exception as e:
        logger.debug("arb_layer fetch failed %s: %s", url, e)
        return entry["data"] if entry else None


@dataclass
class ArbResult:
    ticker:            str
    kalshi_prob:       float
    polymarket_prob:   Optional[float]
    sharp_line_prob:   Optional[float]
    cross_spread:      float
    arb_exists:        bool
    direction_aligned: bool
    mode:              str
    notes:             str = ""

    @property
    def passes(self) -> bool:
        if self.mode == "passthrough":
            return True
        return self.direction_aligned or self.arb_exists

    @property
    def is_near_miss(self) -> bool:
        near_miss_threshold = ARB_MIN_SPREAD_PCT * 0.5
        return (
            not self.arb_exists
            and self.cross_spread >= near_miss_threshold
            and self.notes not in ("no_polymarket_match", "empty_query", "sports_prop_skip")
        )


# ── DB helper (Postgres-aware) ────────────────────────────────────────────────

def _log_to_db(result: ArbResult) -> None:
    """
    Write arb opportunity or near-miss to Supabase (Postgres) or SQLite.
    Uses DATABASE_URL env var to determine which.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    database_url = os.getenv("DATABASE_URL", "")

    try:
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur  = conn.cursor()
            cur.execute("""
                INSERT INTO arb_opportunities
                (detected_at, ticker, kalshi_prob, poly_prob,
                 sharp_prob, spread, mode, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                now, result.ticker, result.kalshi_prob,
                result.polymarket_prob, result.sharp_line_prob,
                result.cross_spread, result.mode, result.notes,
            ))
            conn.commit()
            conn.close()
        else:
            import sqlite3
            db_path = os.getenv("DB_PATH", "flywheel.db")
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS arb_opportunities (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at TEXT, ticker TEXT,
                    kalshi_prob REAL, poly_prob REAL, sharp_prob REAL,
                    spread REAL, mode TEXT, notes TEXT
                )
            """)
            conn.execute("""
                INSERT INTO arb_opportunities
                (detected_at, ticker, kalshi_prob, poly_prob,
                 sharp_prob, spread, mode, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, result.ticker, result.kalshi_prob,
                result.polymarket_prob, result.sharp_line_prob,
                result.cross_spread, result.mode, result.notes,
            ))
            conn.commit()
            conn.close()
    except Exception as e:
        logger.warning("arb log_to_db failed: %s", e)


# ── ArbLayer ──────────────────────────────────────────────────────────────────

class ArbLayer:

    def __init__(self):
        if ODDSPAPI_KEY:
            self._mode = "oddspapi"
            logger.info("ArbLayer: Mode A — OddsPapi cross-venue")
        else:
            self._mode = "polymarket_gamma"
            logger.info("ArbLayer: Mode B — Polymarket Gamma free cross-check")

    # ── Primary entry point — use this in orchestrator ────────────────────────

    def check_and_log(
        self,
        ticker:        str,
        kalshi_cents:  int,
        our_direction: str,
        market_title:  str = "",
    ) -> ArbResult:
        """
        Single call: runs check() and automatically handles logging.
        Use this in the orchestrator instead of calling check() and
        log_arb_opportunity() separately.
        """
        result = self.check(ticker, kalshi_cents, our_direction, market_title)
        self.log_arb_opportunity(result)
        return result

    def check(
        self,
        ticker:        str,
        kalshi_cents:  int,
        our_direction: str,
        market_title:  str = "",
    ) -> ArbResult:
        kalshi_prob = kalshi_cents / 100.0

        # Sports props: Polymarket won't have these — skip query entirely
        if self._mode == "polymarket_gamma" and _is_sports_prop(market_title):
            logger.debug(
                "ARB SKIP (sports prop, no Polymarket match expected): %s", ticker
            )
            return ArbResult(
                ticker=ticker, kalshi_prob=kalshi_prob,
                polymarket_prob=None, sharp_line_prob=None,
                cross_spread=0.0, arb_exists=False,
                direction_aligned=True, mode="passthrough",
                notes="sports_prop_skip",
            )

        if self._mode == "oddspapi":
            return self._check_oddspapi(ticker, kalshi_prob, our_direction, market_title)
        else:
            return self._check_polymarket(ticker, kalshi_prob, our_direction, market_title)

    # ── Mode A: OddsPapi ──────────────────────────────────────────────────────

    def _check_oddspapi(
        self, ticker: str, kalshi_prob: float,
        our_direction: str, market_title: str,
    ) -> ArbResult:
        data = _get(
            "https://api.oddspapi.io/v1/markets",
            {"q": market_title[:80], "sources": "kalshi,polymarket,pinnacle"},
            headers={"X-API-Key": ODDSPAPI_KEY},
            ttl=120,
        )

        poly_prob, sharp_prob = None, None
        notes_parts = []

        if data:
            for market in data.get("markets", []):
                if not any(
                    w in (market.get("title", "") + market.get("id", "")).lower()
                    for w in market_title.lower().split()[:3]
                ):
                    continue
                for book in market.get("books", []):
                    source = book.get("source", "").lower()
                    prob   = float(book.get("yes_price", 0.5))
                    if "polymarket" in source:
                        poly_prob = prob
                        notes_parts.append(f"poly={prob:.2%}")
                    if "pinnacle" in source or "sharp" in source:
                        sharp_prob = prob
                        notes_parts.append(f"sharp={prob:.2%}")
                break

        ext_prob     = sharp_prob or poly_prob
        cross_spread = abs(kalshi_prob - ext_prob) if ext_prob else 0.0
        arb_exists   = cross_spread >= ARB_MIN_SPREAD_PCT

        direction_aligned = True
        if ext_prob is not None:
            direction_aligned = ("YES" if ext_prob > 0.5 else "NO") == our_direction

        result = ArbResult(
            ticker=ticker, kalshi_prob=kalshi_prob,
            polymarket_prob=poly_prob, sharp_line_prob=sharp_prob,
            cross_spread=cross_spread, arb_exists=arb_exists,
            direction_aligned=direction_aligned, mode="oddspapi",
            notes=", ".join(notes_parts) if notes_parts else "no_match",
        )

        if arb_exists:
            logger.info(
                "ARB DETECTED %s | kalshi=%.2f ext=%.2f spread=%.2f dir_aligned=%s",
                ticker, kalshi_prob, ext_prob or 0, cross_spread, direction_aligned,
            )

        return result

    # ── Mode B: Polymarket Gamma (free) ───────────────────────────────────────

    def _check_polymarket(
        self, ticker: str, kalshi_prob: float,
        our_direction: str, market_title: str,
    ) -> ArbResult:
        stopwords = {"will", "this", "that", "with", "from", "have", "does",
                     "what", "when", "which", "there", "their", "would"}
        keywords = [
            w for w in market_title.lower().split()
            if len(w) > 3 and w not in stopwords
        ]
        query = " ".join(keywords[:4])

        if not query:
            return ArbResult(
                ticker=ticker, kalshi_prob=kalshi_prob,
                polymarket_prob=None, sharp_line_prob=None,
                cross_spread=0.0, arb_exists=False,
                direction_aligned=True, mode="passthrough",
                notes="empty_query",
            )

        pm_data = _get(
            f"{POLYMARKET_GAMMA_URL}/markets",
            {"search": query, "active": "true", "limit": "5"},
            ttl=300,
        )

        poly_prob    = None
        cross_spread = 0.0
        arb_exists   = False
        notes        = "no_polymarket_match"

        if pm_data is None:
            logger.debug("Polymarket Gamma returned None for query: %s", query)
        else:
            markets = pm_data if isinstance(pm_data, list) else pm_data.get("markets", [])

            if not markets:
                logger.debug(
                    "Polymarket Gamma: no markets for query '%s' (ticker=%s)",
                    query, ticker,
                )
            else:
                for market in markets:
                    try:
                        outcome_prices = market.get("outcomePrices", [])
                        if not outcome_prices:
                            continue
                        poly_yes     = float(outcome_prices[0])
                        poly_prob    = poly_yes
                        cross_spread = abs(kalshi_prob - poly_yes)
                        arb_exists   = cross_spread >= ARB_MIN_SPREAD_PCT
                        notes        = (
                            f"poly={poly_yes:.2%} "
                            f"kalshi={kalshi_prob:.2%} "
                            f"spread={cross_spread:.2%}"
                        )
                        logger.debug(
                            "Polymarket match: %s | %s",
                            market.get("question", "")[:50], notes,
                        )
                        break
                    except Exception as e:
                        logger.debug("Polymarket parse error: %s", e)
                        continue

        direction_aligned = True
        if poly_prob is not None:
            direction_aligned = (
                (our_direction == "YES" and poly_prob < kalshi_prob) or
                (our_direction == "NO"  and poly_prob > kalshi_prob) or
                arb_exists
            )

        result = ArbResult(
            ticker=ticker, kalshi_prob=kalshi_prob,
            polymarket_prob=poly_prob, sharp_line_prob=None,
            cross_spread=cross_spread, arb_exists=arb_exists,
            direction_aligned=direction_aligned, mode="polymarket_gamma",
            notes=notes,
        )

        if arb_exists:
            logger.info(
                "POLY ARB %s | kalshi=%.2f poly=%.2f spread=%.2f",
                ticker, kalshi_prob, poly_prob or 0, cross_spread,
            )

        return result

    # ── Log arb opportunity or near-miss to DB ────────────────────────────────

    def log_arb_opportunity(self, result: ArbResult) -> None:
        """
        Persist confirmed arbs AND near-misses to DB for dashboard display.
        Silently skips results with no external data (sports props, empty queries).
        """
        # Skip if no external data was available — nothing useful to log
        if result.notes in ("no_polymarket_match", "empty_query", "sports_prop_skip"):
            return

        # Log confirmed arbs
        if result.arb_exists:
            _log_to_db(result)
            return

        # Log near-misses (spread > 50% of threshold) so dashboard shows activity
        if result.is_near_miss:
            _log_to_db(result)
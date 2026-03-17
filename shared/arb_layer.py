"""
shared/arb_layer.py

Cross-venue arbitrage layer — the 7th consensus gate.

What it does
------------
Before any trade fires, this layer checks whether the same event is
priced differently on Kalshi vs Polymarket (and sportsbooks via OddsPapi).

A trade only passes arb_layer if:
  (a) The cross-venue spread confirms our direction (both markets agree
      the market is mispriced in the same direction), OR
  (b) A direct arb exists (spread > ARB_MIN_SPREAD_PCT) that we can
      exploit by trading the cheaper side

Three operating modes depending on which keys are present:

  Mode A — OddsPapi key present:
    Full cross-venue comparison: Kalshi vs Polymarket vs 300+ sportsbooks.
    Returns sharp-line consensus and arb flags.

  Mode B — No OddsPapi, Polymarket Gamma only:
    Free cross-check: compare Kalshi price against Polymarket gamma price
    for markets where slugs overlap.

  Mode C — No external keys:
    Passthrough — arb_layer always returns True (no veto, best-effort).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import requests

from config.settings import (
    ODDSPAPI_KEY, POLYMARKET_GAMMA_URL, ARB_MIN_SPREAD_PCT
)

logger = logging.getLogger(__name__)

_cache: dict = {}
import time

def _get(url, params=None, headers=None, ttl=120):
    key = url + str(sorted((params or {}).items()))
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"] < ttl):
        return entry["data"]
    try:
        r = requests.get(url, params=params or {}, headers=headers or {}, timeout=10)
        r.raise_for_status()
        data = r.json()
        _cache[key] = {"ts": time.time(), "data": data}
        return data
    except Exception as e:
        logger.debug("arb_layer fetch failed %s: %s", url, e)
        return entry["data"] if entry else None


@dataclass
class ArbResult:
    """Result of the cross-venue arbitrage check for one contract."""
    ticker:            str
    kalshi_prob:       float
    polymarket_prob:   Optional[float]
    sharp_line_prob:   Optional[float]   # Pinnacle or OddsPapi consensus
    cross_spread:      float             # |kalshi - best_external|
    arb_exists:        bool              # spread > ARB_MIN_SPREAD_PCT
    direction_aligned: bool              # external markets agree with our direction
    mode:              str               # 'oddspapi' | 'polymarket_gamma' | 'passthrough'
    notes:             str = ""

    @property
    def passes(self) -> bool:
        """
        Passes arb gate if:
          - Passthrough mode (no keys), OR
          - Direction is aligned (external confirms our edge), OR
          - Arb exists and we should trade it
        """
        if self.mode == "passthrough":
            return True
        return self.direction_aligned or self.arb_exists


class ArbLayer:
    """
    Checks a Kalshi market against external venues.
    Call .check(ticker, kalshi_yes_price_cents, our_direction) before executing.
    """

    def __init__(self):
        if ODDSPAPI_KEY:
            self._mode = "oddspapi"
            logger.info("ArbLayer: Mode A — OddsPapi cross-venue")
        else:
            self._mode = "polymarket_gamma"
            logger.info("ArbLayer: Mode B — Polymarket Gamma free cross-check")

    def check(
        self,
        ticker:          str,
        kalshi_cents:    int,       # Kalshi YES price in cents
        our_direction:   str,       # 'YES' or 'NO'
        market_title:    str = "",  # used to fuzzy-match Polymarket slugs
    ) -> ArbResult:
        """Run the arb check and return an ArbResult."""

        kalshi_prob = kalshi_cents / 100.0

        if self._mode == "oddspapi":
            return self._check_oddspapi(ticker, kalshi_prob, our_direction, market_title)
        elif self._mode == "polymarket_gamma":
            return self._check_polymarket(ticker, kalshi_prob, our_direction, market_title)
        else:
            return ArbResult(
                ticker=ticker, kalshi_prob=kalshi_prob,
                polymarket_prob=None, sharp_line_prob=None,
                cross_spread=0.0, arb_exists=False,
                direction_aligned=True, mode="passthrough",
            )

    # ── Mode A: OddsPapi ─────────────────────────────────────────────────────

    def _check_oddspapi(
        self, ticker: str, kalshi_prob: float,
        our_direction: str, market_title: str,
    ) -> ArbResult:
        """
        OddsPapi returns Kalshi + Polymarket + 300 sportsbooks normalized.
        We use it to find the sharp consensus probability.
        """
        data = _get(
            "https://api.oddspapi.io/v1/markets",
            {"q": market_title[:80], "sources": "kalshi,polymarket,pinnacle"},
            headers={"X-API-Key": ODDSPAPI_KEY},
            ttl=120,
        )

        poly_prob   = None
        sharp_prob  = None
        notes_parts = []

        if data:
            for market in data.get("markets", []):
                # Find best matching market
                if not any(w in (market.get("title", "") + market.get("id", "")).lower()
                           for w in market_title.lower().split()[:3]):
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

                break  # first match

        # Determine the best external reference probability
        ext_prob = sharp_prob or poly_prob
        cross_spread = abs(kalshi_prob - ext_prob) if ext_prob else 0.0
        arb_exists   = cross_spread >= ARB_MIN_SPREAD_PCT

        # Direction alignment: external market prices imply same YES/NO as us
        direction_aligned = True
        if ext_prob is not None:
            ext_direction = "YES" if ext_prob > 0.5 else "NO"
            direction_aligned = ext_direction == our_direction

        result = ArbResult(
            ticker=ticker, kalshi_prob=kalshi_prob,
            polymarket_prob=poly_prob, sharp_line_prob=sharp_prob,
            cross_spread=cross_spread, arb_exists=arb_exists,
            direction_aligned=direction_aligned, mode="oddspapi",
            notes=", ".join(notes_parts) if notes_parts else "no_match",
        )

        if arb_exists:
            logger.info(
                "ARB DETECTED %s | kalshi=%.2%% ext=%.2%% spread=%.2%% dir_aligned=%s",
                ticker, kalshi_prob, ext_prob or 0,
                cross_spread, direction_aligned,
            )

        return result

    # ── Mode B: Polymarket Gamma (free) ───────────────────────────────────────

    def _check_polymarket(
        self, ticker: str, kalshi_prob: float,
        our_direction: str, market_title: str,
    ) -> ArbResult:
        """
        Free cross-check against Polymarket Gamma API.
        Fuzzy-match on title keywords to find the same event.
        """
        keywords = [w for w in market_title.lower().split()
                    if len(w) > 3 and w not in {"will", "this", "that", "with", "from"}]
        query = " ".join(keywords[:4])

        pm_data = _get(
            f"{POLYMARKET_GAMMA_URL}/markets",
            {"search": query, "active": "true", "limit": "5"},
            ttl=300,
        )

        poly_prob    = None
        cross_spread = 0.0
        arb_exists   = False
        notes        = "no_polymarket_match"

        if pm_data:
            for market in (pm_data if isinstance(pm_data, list) else []):
                try:
                    poly_yes = float(market.get("outcomePrices", ["0.5"])[0])
                    poly_prob    = poly_yes
                    cross_spread = abs(kalshi_prob - poly_yes)
                    arb_exists   = cross_spread >= ARB_MIN_SPREAD_PCT
                    notes        = f"poly={poly_yes:.2%} spread={cross_spread:.2%}"
                    break
                except Exception:
                    continue

        direction_aligned = True
        if poly_prob is not None:
            poly_dir = "YES" if poly_prob > kalshi_prob else "NO"
            # Aligned = Polymarket agrees there's an edge in our direction
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
                "POLY ARB %s | kalshi=%.2%% poly=%.2%% spread=%.2%%",
                ticker, kalshi_prob, poly_prob or 0, cross_spread,
            )

        return result

    # ── Convenience: log arb to DB ────────────────────────────────────────────

    def log_arb_opportunity(self, result: ArbResult, db_path: str = "flywheel.db") -> None:
        """Persist detected arb opportunities for analysis."""
        if not result.arb_exists:
            return
        try:
            import sqlite3
            from datetime import datetime, timezone
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS arb_opportunities (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at  TEXT,
                    ticker       TEXT,
                    kalshi_prob  REAL,
                    poly_prob    REAL,
                    sharp_prob   REAL,
                    spread       REAL,
                    mode         TEXT,
                    notes        TEXT
                )
            """)
            conn.execute("""
                INSERT INTO arb_opportunities
                (detected_at, ticker, kalshi_prob, poly_prob, sharp_prob, spread, mode, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                result.ticker, result.kalshi_prob,
                result.polymarket_prob, result.sharp_line_prob,
                result.cross_spread, result.mode, result.notes,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug("arb log failed: %s", e)

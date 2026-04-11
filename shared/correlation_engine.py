"""
shared/correlation_engine.py  (v2 — skip unmodelable markets, crash protection)

Changes vs v1:
  1. scan() now skips MENTION, Survivor, and other unmodelable markets
     before grouping. Previously these were being correlated, passed to
     kelly_sizer, and crashing the entire orchestrator.
  2. _find_divergence() wrapped in try/except so a single bad market
     can't crash the scan loop.

Cross-market correlation engine.

Kalshi runs related markets simultaneously that often diverge in price.
Examples:
  "Will the Fed cut in March?" vs "Will the Fed cut in Q1?"
  "Lakers win tonight?" vs "Lakers cover -4.5?"
  "BTC above $100k by March 31?" vs "BTC above $90k by March 31?"

When two correlated markets diverge, one of them is mispriced.
Legging into both sides (long the cheap one, short the expensive one)
is the closest thing to riskless arbitrage available on Kalshi.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

logger = logging.getLogger(__name__)

# ── v2: Markets the correlation engine should never touch ──────────────────────
# These markets are unmodelable and cause crashes when passed to kelly_sizer.
# Check is done via substring match on ticker/event_ticker (case-insensitive).
_CORR_SKIP_SUBSTRINGS = (
    "mention",      # KXFOXNEWSMENTION, KXMLBMENTION, KXNBAMENTION, etc.
    "survivor",     # KXSURVIVORMENTION
    "roty",         # Rookie of the Year futures
    "seasonhr",     # Season-long HR totals
)


@dataclass
class CorrSignal:
    """A detected correlated market pair with price divergence."""
    event_group:        str
    cheap_ticker:       str
    cheap_direction:    str
    cheap_price_cents:  int
    expensive_ticker:   str
    expensive_direction: str
    expensive_price_cents: int
    divergence_cents:   float
    confidence:         float
    notes:              str = ""


def _ph() -> str:
    return "%s" if os.getenv("DATABASE_URL") else "?"


def _db_execute(sql: str, params: tuple = (), fetch: bool = False) -> list:
    database_url = os.getenv("DATABASE_URL", "")
    try:
        if database_url:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur  = conn.cursor()
            cur.execute(sql, params)
            result = []
            if fetch and cur.description:
                cols   = [d[0] for d in cur.description]
                result = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.commit()
            conn.close()
            return result
        else:
            import sqlite3
            conn = sqlite3.connect(os.getenv("DB_PATH", "flywheel.db"))
            conn.row_factory = sqlite3.Row
            cur  = conn.execute(sql, params)
            result = [dict(r) for r in cur.fetchall()] if fetch else []
            conn.commit()
            conn.close()
            return result
    except Exception as e:
        logger.error("CorrelationEngine DB error: %s", e)
        return []


def init_correlation_tables() -> None:
    _db_execute("""
        CREATE TABLE IF NOT EXISTS market_correlations (
            id             SERIAL PRIMARY KEY,
            ticker_a       TEXT NOT NULL,
            ticker_b       TEXT NOT NULL,
            event_group    TEXT,
            divergence     FLOAT8,
            detected_at    TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    _db_execute("""
        CREATE INDEX IF NOT EXISTS idx_correlations_ticker_a
        ON market_correlations (ticker_a)
    """)


def _get_yes_price(market: dict) -> int:
    """Extract YES midpoint in cents from a market dict."""
    try:
        bid = float(market.get("yes_bid_dollars") or 0) * 100
        ask = float(market.get("yes_ask_dollars") or 0) * 100
        if bid > 0 and ask > 0:
            return int(round((bid + ask) / 2))
    except (TypeError, ValueError):
        pass
    try:
        last = float(market.get("last_price_dollars") or 0) * 100
        if last > 0:
            return int(round(last))
    except (TypeError, ValueError):
        pass
    return 0


def _event_group(market: dict) -> str:
    """
    Extract the event group key from a market.
    Uses event_ticker if available, falls back to ticker prefix.
    """
    et = market.get("event_ticker", "").strip()
    if et:
        return et.upper()

    ticker = market.get("ticker", "")
    parts  = ticker.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}".upper()
    return parts[0].upper() if parts else ""


def _should_skip_market(market: dict) -> bool:
    """v2: Return True if this market should be excluded from correlation analysis."""
    ticker_lower = market.get("ticker", "").lower()
    et_lower     = market.get("event_ticker", "").lower()
    return any(s in ticker_lower or s in et_lower for s in _CORR_SKIP_SUBSTRINGS)


class CorrelationEngine:
    """
    Detects price divergence between correlated Kalshi markets.

    Parameters
    ----------
    min_divergence_cents : float
        Minimum price gap (in cents) to flag as a divergence opportunity.
        Default 8 cents — accounts for vig on both legs.
    max_group_size : int
        Max markets per event group to compare. Above this, skip
        (too many combinations, probably unrelated markets).
    """

    def __init__(
        self,
        min_divergence_cents: float = 8.0,
        max_group_size:       int   = 10,
    ):
        self.min_divergence_cents = min_divergence_cents
        self.max_group_size       = max_group_size
        init_correlation_tables()

    def scan(self, open_markets: list[dict]) -> list[CorrSignal]:
        """
        Group open markets by event, then find divergent pairs.
        Returns list of CorrSignal sorted by divergence descending.

        v2: Skips unmodelable markets (MENTION, Survivor, etc.) that
        previously crashed the orchestrator when passed to kelly_sizer.
        """
        # Group markets by event, skipping unmodelable ones
        groups: dict[str, list[dict]] = {}
        skipped = 0
        for market in open_markets:
            # v2: skip unmodelable markets before grouping
            if _should_skip_market(market):
                skipped += 1
                continue

            price = _get_yes_price(market)
            if price == 0:
                continue
            group = _event_group(market)
            if not group:
                continue
            groups.setdefault(group, []).append(market)

        if skipped > 0:
            logger.debug(
                "CorrelationEngine skipped %d unmodelable markets", skipped
            )

        # Only process groups with 2+ markets
        signals: list[CorrSignal] = []
        for group, markets in groups.items():
            if len(markets) < 2:
                continue
            if len(markets) > self.max_group_size:
                continue

            # v2: wrap in try/except so one bad group can't crash the scan
            try:
                group_signals = self._find_divergence(group, markets)
                signals.extend(group_signals)
            except Exception as e:
                logger.warning(
                    "CorrelationEngine: error processing group %s: %s",
                    group, e,
                )

        # Sort by divergence — biggest opportunity first
        signals.sort(key=lambda s: s.divergence_cents, reverse=True)

        if signals:
            logger.info(
                "CorrelationEngine found %d divergent pairs across %d groups",
                len(signals), len([g for g in groups if len(groups[g]) >= 2]),
            )

        return signals

    def _find_divergence(
        self, group: str, markets: list[dict]
    ) -> list[CorrSignal]:
        """
        For each pair in the group, detect if they imply contradictory
        probabilities at a detectable spread.
        """
        signals = []

        for m_a, m_b in combinations(markets, 2):
            price_a = _get_yes_price(m_a)
            price_b = _get_yes_price(m_b)

            if price_a == 0 or price_b == 0:
                continue

            ticker_a = m_a.get("ticker", "")
            ticker_b = m_b.get("ticker", "")

            title_a   = m_a.get("title", "").lower()
            title_b   = m_b.get("title", "").lower()
            nested    = self._detect_nested_thresholds(title_a, title_b, price_a, price_b)
            timeframe = self._detect_timeframe_mismatch(title_a, title_b, price_a, price_b)
            raw_div   = abs(price_a - price_b)

            if nested and nested["divergence"] >= self.min_divergence_cents:
                sig = self._build_signal(group, m_a, price_a, m_b, price_b, nested)
                if sig:
                    signals.append(sig)
                    self._log_divergence(sig)

            elif timeframe and timeframe["divergence"] >= self.min_divergence_cents:
                sig = self._build_signal(group, m_a, price_a, m_b, price_b, timeframe)
                if sig:
                    signals.append(sig)
                    self._log_divergence(sig)

            elif raw_div >= self.min_divergence_cents * 1.5:
                sig = CorrSignal(
                    event_group          = group,
                    cheap_ticker         = ticker_b if price_b < price_a else ticker_a,
                    cheap_direction      = "YES",
                    cheap_price_cents    = min(price_a, price_b),
                    expensive_ticker     = ticker_a if price_a > price_b else ticker_b,
                    expensive_direction  = "NO",
                    expensive_price_cents= max(price_a, price_b),
                    divergence_cents     = raw_div,
                    confidence           = min(0.6, raw_div / 30.0),
                    notes                = f"Raw price divergence {raw_div}¢ in group {group}",
                )
                signals.append(sig)
                self._log_divergence(sig)

        return signals

    def _detect_nested_thresholds(
        self, title_a: str, title_b: str, price_a: int, price_b: int
    ) -> Optional[dict]:
        """
        Detect if two markets are nested threshold markets.
        e.g. "BTC above $100k" and "BTC above $90k" — the $90k must be >= $100k.
        """
        import re

        def extract_number(title: str) -> Optional[float]:
            numbers = re.findall(r"\$?([\d,]+(?:k|m)?)\b", title)
            for n in numbers:
                try:
                    clean = n.replace(",", "").replace("k", "000").replace("m", "000000")
                    return float(clean)
                except ValueError:
                    pass
            return None

        num_a = extract_number(title_a)
        num_b = extract_number(title_b)

        if num_a is None or num_b is None or num_a == num_b:
            return None

        if num_a < num_b:
            if price_a < price_b:
                divergence = price_b - price_a
                return {
                    "type":       "nested_threshold",
                    "divergence": divergence,
                    "cheap":      "a",
                    "expensive":  "b",
                    "notes":      f"Nested threshold: easier market (${num_a:,.0f}) underpriced vs harder (${num_b:,.0f})",
                }
        else:
            if price_b < price_a:
                divergence = price_a - price_b
                return {
                    "type":       "nested_threshold",
                    "divergence": divergence,
                    "cheap":      "b",
                    "expensive":  "a",
                    "notes":      f"Nested threshold: easier market (${num_b:,.0f}) underpriced vs harder (${num_a:,.0f})",
                }

        return None

    def _detect_timeframe_mismatch(
        self, title_a: str, title_b: str, price_a: int, price_b: int
    ) -> Optional[dict]:
        """
        Detect if one market has a broader timeframe than the other.
        """
        broad_terms  = ["q1", "q2", "q3", "q4", "year", "annual", "2025", "2026"]
        narrow_terms = ["january", "february", "march", "april", "may", "june",
                        "july", "august", "september", "october", "november", "december",
                        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

        a_is_broad  = any(t in title_a for t in broad_terms)
        b_is_broad  = any(t in title_b for t in broad_terms)
        a_is_narrow = any(t in title_a for t in narrow_terms)
        b_is_narrow = any(t in title_b for t in narrow_terms)

        if a_is_broad and b_is_narrow and price_a < price_b:
            divergence = price_b - price_a
            return {
                "type":       "timeframe_mismatch",
                "divergence": divergence,
                "cheap":      "a",
                "expensive":  "b",
                "notes":      "Broader timeframe market priced below narrower — buy broad",
            }

        if b_is_broad and a_is_narrow and price_b < price_a:
            divergence = price_a - price_b
            return {
                "type":       "timeframe_mismatch",
                "divergence": divergence,
                "cheap":      "b",
                "expensive":  "a",
                "notes":      "Broader timeframe market priced below narrower — buy broad",
            }

        return None

    def _build_signal(
        self,
        group: str,
        m_a: dict, price_a: int,
        m_b: dict, price_b: int,
        detection: dict,
    ) -> Optional[CorrSignal]:
        ticker_a = m_a.get("ticker", "")
        ticker_b = m_b.get("ticker", "")
        cheap    = detection["cheap"]
        divergence = detection["divergence"]

        cheap_ticker        = ticker_a if cheap == "a" else ticker_b
        cheap_price         = price_a  if cheap == "a" else price_b
        expensive_ticker    = ticker_b if cheap == "a" else ticker_a
        expensive_price     = price_b  if cheap == "a" else price_a

        confidence = min(0.85, divergence / 20.0 * 0.85)

        logger.info(
            "[CORR] %s | cheap=%s@%d¢ expensive=%s@%d¢ div=%.1f¢ conf=%.2f",
            group, cheap_ticker, cheap_price,
            expensive_ticker, expensive_price,
            divergence, confidence,
        )

        return CorrSignal(
            event_group          = group,
            cheap_ticker         = cheap_ticker,
            cheap_direction      = "YES",
            cheap_price_cents    = cheap_price,
            expensive_ticker     = expensive_ticker,
            expensive_direction  = "NO",
            expensive_price_cents= expensive_price,
            divergence_cents     = divergence,
            confidence           = confidence,
            notes                = detection.get("notes", ""),
        )

    def _log_divergence(self, sig: CorrSignal) -> None:
        p = _ph()
        try:
            _db_execute(
                f"""
                INSERT INTO market_correlations
                    (ticker_a, ticker_b, event_group, divergence, detected_at)
                VALUES ({p}, {p}, {p}, {p}, {p})
                """,
                (
                    sig.cheap_ticker, sig.expensive_ticker,
                    sig.event_group, sig.divergence_cents,
                    datetime.now(timezone.utc),
                ),
            )
        except Exception as e:
            logger.warning("Correlation log failed: %s", e)